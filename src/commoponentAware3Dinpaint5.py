import torch
import numpy as np
from tqdm import tqdm
from pytorch3d.renderer import TexturesUV
from pytorch3d.ops import knn_points
import json
import os

class ComponentAware3DInpaintingGPU:
    def __init__(self, mesh, device, max_cos_map, avg_cos_map, face2label=None):
        """
        GAWP (Geometric-Aware Wavefront Propagation) 纹理补全算法
        创新点：精英置信度排序 + 语义物理隔离 + 动态波前生长
        """
        self.mesh = mesh
        self.device = device
        
        # 1. 预处理置信度场 (取单通道并扁平化)
        max_cos = self._to_tensor(max_cos_map)
        avg_cos = self._to_tensor(avg_cos_map)
        if max_cos.ndim == 3: max_cos = max_cos[..., 0]
        if avg_cos.ndim == 3: avg_cos = avg_cos[..., 0]
        
        self.max_cos_map = max_cos.reshape(-1)
        self.avg_cos_map = avg_cos.reshape(-1)
        
        # 2. 加载部件标签 lookup table
        self.face_label_tensor = None
        if face2label is not None:
            if isinstance(face2label, str) and os.path.exists(face2label):
                with open(face2label, "r") as f: face2label = json.load(f)
            face2label = {int(k): int(v) for k, v in face2label.items()} if isinstance(face2label, dict) else {}
            if len(face2label) > 0:
                max_fid = max(face2label.keys())
                self.face_label_tensor = torch.full((max_fid + 1,), -1, dtype=torch.int32, device=self.device)
                ids = torch.tensor(list(face2label.keys()), dtype=torch.long, device=self.device)
                lbls = torch.tensor(list(face2label.values()), dtype=torch.int32, device=self.device)
                self.face_label_tensor[ids] = lbls

    def _to_tensor(self, x):
        if not torch.is_tensor(x):
            return torch.from_numpy(np.array(x)).to(self.device).float()
        return x.to(self.device).float()

    def set_texture_map(self, texture):
        new_map = texture.permute(1, 2, 0).to(self.device)
        new_tex = TexturesUV([new_map], self.mesh.textures.faces_uvs_padded(), 
                             self.mesh.textures.verts_uvs_padded(), sampling_mode="nearest")
        self.mesh.textures = new_tex

    @torch.no_grad()
    def UV_pos_render(self, texture_dim):
        """完全使用 Kaolin 渲染位置图"""
        import kaolin as kal
        verts = self.mesh.verts_packed()
        faces = self.mesh.faces_packed()
        verts_uv = self.mesh.textures.verts_uvs_padded()[0]
        faces_uv = self.mesh.textures.faces_uvs_padded()[0]
        
        uv_face_attr = torch.index_select(verts_uv, 0, faces_uv.view(-1)).view(faces.shape[0], faces_uv.shape[1], 2).unsqueeze(0)
        face_vertices_world = kal.ops.mesh.index_vertices_by_faces(verts.unsqueeze(0), faces)
        face_vertices_z = torch.zeros_like(face_vertices_world[:, :, :, -1])
        
        uv_pos, face_idx = kal.render.mesh.rasterize(texture_dim, texture_dim, face_vertices_z, 
                                                    uv_face_attr * 2 - 1, face_features=face_vertices_world)
        uv_pos = torch.clamp(uv_pos, -1, 1) / 2 + 0.5
        uv_pos[face_idx == -1] = 0
        return uv_pos, face_idx

    def _safe_knn(self, p1, p2, K, chunk_size=40000):
        """分片 KNN，防止大面积补全时显存溢出 (OOM)"""
        N = p1.shape[1]
        all_dists, all_idxs = [], []
        for i in range(0, N, chunk_size):
            p1_chunk = p1[:, i : min(i + chunk_size, N), :]
            dists, idxs, _ = knn_points(p1_chunk, p2, K=K)
            all_dists.append(dists)
            all_idxs.append(idxs)
        return torch.cat(all_dists, dim=1), torch.cat(all_idxs, dim=1)

    @torch.no_grad()
    def update_texture(self, position_map, face_idx, 
                       cos_threshold=0.2, k_neighbors=60, smooth_scale=0.5):
        """
        核心补全算法
        """
        print(f"\n[GAWP] Initializing Wavefront Growth... K={k_neighbors}")
        print('cos_threshold', cos_threshold)
        texture = self.mesh.textures.maps_padded()[0] # (H, W, 3)
        h, w, c = texture.shape
        points = position_map.reshape(-1, 3)
        colors = texture.reshape(-1, 3).clone()
        
        # 1. 确定初始待修复区域 (Mask)
        valid_geo = (points[:, 0] != 0)
        # 黑色像素或投影角度过大的像素均视为待修复
        fill_mask = valid_geo & ((colors.mean(dim=1) < 0.01) | (self.max_cos_map < cos_threshold))
        
        
        # 2. 初始化置信度场 (Score Map)
        score_map = torch.zeros(len(points), device=self.device)
        known_mask = (~fill_mask) & valid_geo
        # 初始分数来自拍摄视角的 Cosine 值
        score_map[known_mask] = self.avg_cos_map[known_mask]
        
        # 3. 准备语义标签
        texel_comp = torch.zeros(len(points), dtype=torch.int32, device=self.device)
        if self.face_label_tensor is not None:
            f_idx = torch.clamp(face_idx.reshape(-1).long(), 0, len(self.face_label_tensor)-1)
            texel_comp = self.face_label_tensor[f_idx]

        # 4. 传播循环 (有序扩张)
        initial_unfilled_count = fill_mask.sum().item()
        pbar = tqdm(total=initial_unfilled_count, desc="Propagating")
        
        batch_size = 2000 # 每一轮填色的数量，越小越稳，越大越快
        DECAY = 0.98     # 置信度随传播代数衰减

        while fill_mask.sum() > 0:
            u_idx = torch.nonzero(fill_mask, as_tuple=True)[0]
            k_idx = torch.nonzero((~fill_mask) & valid_geo, as_tuple=True)[0]
            if len(k_idx) == 0: break
            
            # KNN 搜索 (分片防爆版)
            K_val = min(k_neighbors, len(k_idx))
            p1 = points[u_idx].unsqueeze(0)
            p2 = points[k_idx].unsqueeze(0)
            d_sq, i_knn = self._safe_knn(p1, p2, K=K_val, chunk_size=25000)
            
            dists = torch.sqrt(d_sq.squeeze(0) + 1e-8)
            g_neighbor = k_idx[i_knn.squeeze(0)] # 映射回全局索引
            
            # --- 创新点：基于邻居置信度的精英排序 ---
            neighbor_scores = score_map[g_neighbor]
            layer_score = neighbor_scores.mean(dim=1) # 计算待修复点的潜在分数
            sort_order = torch.argsort(layer_score, descending=True)
            
            # 选取本轮“信誉最好”的一批点进行处理
            n_proc = min(batch_size, len(u_idx))
            top_idx = sort_order[:n_proc]
            b_target = u_idx[top_idx]
            b_dists = dists[top_idx]
            b_neighbor = g_neighbor[top_idx]
            
            # --- 权重计算 ---
            # 高斯空间距离权重
            sigma = b_dists.mean(dim=1, keepdim=True) + 1e-5
            w_dist = torch.exp(-(b_dists**2) / (2 * (sigma * smooth_scale)**2))
            
            # 语义防火墙：完全屏蔽异类部件的颜色
            w_comp = torch.zeros_like(w_dist)
            target_c = texel_comp[b_target].unsqueeze(1)
            neighbor_c = texel_comp[b_neighbor]
            w_comp[target_c == neighbor_c] = 1.0

            # 最终权重合成
            total_w = w_dist * w_comp
            sum_w = total_w.sum(dim=1, keepdim=True)
            
            # --- 填色逻辑 (带 Fallback) ---
            # 正常点：有同类邻居，按高斯加权混色
            valid_b = (sum_w.squeeze() > 1e-6)
            if valid_b.any():
                v_target = b_target[valid_b]
                v_neighbor = b_neighbor[valid_b]
                v_norm_w = total_w[valid_b] / (sum_w[valid_b] + 1e-12)
                
                neighbor_colors = colors[v_neighbor]
                new_c = (neighbor_colors * v_norm_w.unsqueeze(-1)).sum(dim=1)

                # 计算原始颜色的“信任权重” (0~1)
                # 越接近阈值的点，越不信任原色，更多地混合补全色
                # 假设阈值是 0.06，cos 值为 0.05 的点会获得一部分原色保留，实现平滑过渡
                orig_conf = torch.clamp(self.avg_cos_map[v_target] / cos_threshold, 0, 1).unsqueeze(-1)

                # 执行 Alpha 混合：Result = Confidence * Original + (1 - Confidence) * Inpainted
                colors[v_target] = orig_conf * colors[v_target] + (1 - orig_conf) * new_c
                
                fill_mask[v_target] = False
                score_map[v_target] = layer_score[top_idx][valid_b] * DECAY
                pbar.update(valid_b.sum().item())

            # 孤儿点：周围全是异类，为了不留黑洞，直接降级取最近邻
            # if (~valid_b).any():
            #     orphan_target = b_target[~valid_b]
            #     orphan_neighbor_nearest = b_neighbor[~valid_b, 0]
            #     colors[orphan_target] = colors[orphan_neighbor_nearest]
            #     fill_mask[orphan_target] = False
            #     pbar.update(len(orphan_target))
            
            torch.cuda.empty_cache()
            
        pbar.close()
        return colors.reshape(h, w, 3).cpu().numpy(), (~fill_mask).cpu().numpy().reshape(h, w)

    def __call__(self, texture_input, **kwargs):
        from PIL import Image
        import torchvision.transforms as T
        
        # 1. 统一加载纹理为 Tensor [C, H, W]
        if isinstance(texture_input, str):
            tex = T.ToTensor()(Image.open(texture_input).convert("RGB"))
        else:
            tex = texture_input
            
        self.set_texture_map(tex)
        
        # 2. 几何渲染
        pos_map, f_idx = self.UV_pos_render(tex.shape[1])
        
        # 3. 运行有序波前补全
        c_out, r_mask = self.update_texture(pos_map, f_idx, **kwargs)
        
        # 4. 返回结果 (包含置信度补全后的纹理、位置图、以及补全标记)
        return torch.from_numpy(c_out).to(self.device), pos_map, r_mask