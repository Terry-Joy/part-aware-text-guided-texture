import torch
import numpy as np
from tqdm import tqdm
from pytorch3d.renderer import TexturesUV
from pytorch3d.ops import knn_points
import json

class ComponentAware3DInpaintingGPU:
    def __init__(self, mesh, device, max_cos_map, avg_cos_map, face2label=None):
        self.mesh = mesh
        self.device = device
        
        # ==================== 【关键修复点】 ====================
        # 1. 先转 Tensor
        max_cos = self._to_tensor(max_cos_map)
        avg_cos = self._to_tensor(avg_cos_map)
        
        # 2. 如果是 (H, W, 3) 或 (H, W, C)，只取第一个通道
        if max_cos.ndim == 3 and max_cos.shape[-1] >= 1:
            max_cos = max_cos[..., 0]
        if avg_cos.ndim == 3 and avg_cos.shape[-1] >= 1:
            avg_cos = avg_cos[..., 0]
            
        # 3. 现在可以安全 Flatten 了 (N,)
        self.max_cos_map = max_cos.reshape(-1)
        self.avg_cos_map = avg_cos.reshape(-1)
        # =======================================================
        
        # (后面 face2label 的代码不用动)
        self.face_label_tensor = None
        if face2label is not None:
            if isinstance(face2label, str):
                with open(face2label, "r") as f:
                    face2label = json.load(f)
            face2label = {int(k): int(v) for k, v in face2label.items()} if face2label else {}
            
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
        """完全使用 Kaolin 在 GPU 上渲染位置图"""
        import kaolin as kal
        verts = self.mesh.verts_packed()
        faces = self.mesh.faces_packed()
        verts_uv = self.mesh.textures.verts_uvs_padded()[0]
        faces_uv = self.mesh.textures.faces_uvs_padded()[0]
        
        uv_face_attr = torch.index_select(verts_uv, 0, faces_uv.view(-1)).view(faces.shape[0], faces_uv.shape[1], 2).unsqueeze(0)
        face_vertices_world = kal.ops.mesh.index_vertices_by_faces(verts.unsqueeze(0), faces)
        face_vertices_z = torch.zeros_like(face_vertices_world[:, :, :, -1])
        
        uv_position, face_idx = kal.render.mesh.rasterize(
            texture_dim, texture_dim, face_vertices_z, 
            uv_face_attr * 2 - 1, face_features=face_vertices_world
        )
        uv_position = torch.clamp(uv_position, -1, 1) / 2 + 0.5
        uv_position[face_idx == -1] = 0
        return uv_position, face_idx

    @torch.no_grad()
    def update_texture(self, position_map, face_idx, 
                       cos_threshold=0.10, 
                       k_neighbors=60, 
                       smooth_scale=2.0, 
                       component_weight_same=1.0, 
                       component_weight_diff=0.0): 
        
        print(f"\n[Start GPU Inpainting] Mode: Strict Component Isolation (Epoch-based)")
        
        # 1. 准备数据
        texture = self.mesh.textures.maps_padded()[0]
        h, w, c = texture.shape
        points = position_map.reshape(-1, 3)
        colors = texture.reshape(-1, 3).clone()
        
        # 2. 计算 Mask
        luminance = colors.mean(dim=1)
        black_mask = (luminance < 0.01)
        low_cos_mask = self.max_cos_map < cos_threshold
        valid_mask = (points[:, 0] != 0)
        
        fill_mask = valid_mask & (black_mask | low_cos_mask)
        
        # 3. 准备 Component Map
        texel_component = torch.zeros(len(points), dtype=torch.int32, device=self.device)
        if self.face_label_tensor is not None:
            face_idx_flat = face_idx.reshape(-1).long()
            valid_f_mask = face_idx_flat >= 0
            safe_idx = torch.clamp(face_idx_flat, 0, len(self.face_label_tensor)-1)
            texel_component[valid_f_mask] = self.face_label_tensor[safe_idx[valid_f_mask]]

        # 4. 循环准备
        total_holes = fill_mask.sum().item()
        if total_holes == 0:
            return colors.reshape(h, w, 3).cpu().numpy(), np.zeros((h, w), dtype=bool)

        pbar = tqdm(total=total_holes, desc="Inpainting Progress")
        
        # ========================== 核心循环 (Epoch机制) ==========================
        # 只要还有洞，且上一轮有进度，就继续
        
        epoch = 0
        while fill_mask.sum() > 0:
            epoch += 1
            # 获取所有当前需要填补的索引
            all_unfilled = torch.nonzero(fill_mask, as_tuple=True)[0]
            known_idx = torch.nonzero((~fill_mask) & valid_mask, as_tuple=True)[0] # 这一轮已知的点
            
            if len(known_idx) == 0: break
            
            # 【关键修改】打乱顺序！防止死磕同一个难补的区域
            # 这样由于随机性，总能碰上位于边界的好填的点
            perm = torch.randperm(len(all_unfilled), device=self.device)
            all_unfilled = all_unfilled[perm]
            
            # 本轮（Epoch）更新了多少个点
            updated_in_this_epoch = 0
            
            # 按 Batch 处理所有未填补的点
            batch_size = 2000 # 显存够大可以设大点，比如 10000
            
            for i in range(0, len(all_unfilled), batch_size):
                batch_target_idx = all_unfilled[i : i + batch_size]
                
                # --- KNN ---
                p1 = points[batch_target_idx].unsqueeze(0)
                p2 = points[known_idx].unsqueeze(0) # 注意：随着 known 变多，这里会变慢，但逻辑正确
                
                # K 不能太大，否则显存爆炸；也不能太小，否则找不到同部件
                K_curr = min(k_neighbors, len(known_idx))
                dists_sq, idxs, _ = knn_points(p1, p2, K=K_curr)
                
                dists = torch.sqrt(dists_sq.squeeze(0) + 1e-8)
                global_neighbor_idx = known_idx[idxs.squeeze(0)]
                
                # --- 严格隔离权重计算 ---
                sigma = dists.mean(dim=1, keepdim=True) + 1e-5
                w_dist = torch.exp(-(dists**2) / (2 * (sigma * smooth_scale)**2))
                
                w_comp = torch.zeros_like(w_dist)
                if self.face_label_tensor is not None:
                    target_comp = texel_component[batch_target_idx].unsqueeze(1)
                    neighbor_comp = texel_component[global_neighbor_idx]
                    same_mask = (target_comp == neighbor_comp)
                    w_comp[same_mask] = 1.0
                else:
                    w_comp.fill_(1.0)

                # --- 筛选有效更新 ---
                total_w = w_dist * w_comp
                sum_w = total_w.sum(dim=1, keepdims=True)
                
                valid_update_mask = (sum_w > 1e-6).squeeze()
                
                if valid_update_mask.sum() > 0:
                    # 提取有效数据
                    valid_batch_idx = batch_target_idx[valid_update_mask]
                    valid_total_w = total_w[valid_update_mask]
                    valid_sum_w = sum_w[valid_update_mask]
                    valid_neighbors = global_neighbor_idx[valid_update_mask]
                    
                    # 归一化并计算颜色
                    norm_w = valid_total_w / valid_sum_w
                    neighbor_colors = colors[valid_neighbors]
                    new_colors = (neighbor_colors * norm_w.unsqueeze(-1)).sum(dim=1)
                    
                    # 写入颜色
                    colors[valid_batch_idx] = new_colors
                    
                    # 【重要】在这里不立即更新 fill_mask，而在 Epoch 结束统一更新？
                    # 不，为了加速传播，立即更新 fill_mask 是可以的，
                    # 但为了逻辑简单，我们在 loop 内部直接更新，
                    # 下一个 batch 虽然看不到本轮新填的（因为 known_idx 还没变），但下一轮 Epoch 能看到。
                    fill_mask[valid_batch_idx] = False 
                    
                    count = valid_update_mask.sum().item()
                    updated_in_this_epoch += count
                    pbar.update(count)
            
            # --- Epoch 结束检查 ---
            # --- Epoch 结束检查：是否卡死了？ ---
            if updated_in_this_epoch == 0:
                if not fallback_triggered:
                    # 【第一阶段 -> 第二阶段】：解开封印
                    print(f"\n[Warning] Epoch {epoch} stuck! Strict isolation failed for remaining holes.")
                    print(f" -> Switching to FALLBACK MODE: Relaxing semantic constraints...")
                    
                    current_w_diff = 1.0 # 或者 0.5，给异类一个机会
                    fallback_triggered = True
                    
                    # 不要 break，带着新权重继续下一轮 loop
                    continue 
                else:
                    # 【第二阶段也卡死了】：那真是神仙难救
                    print(f"Warning: Epoch {epoch} made no progress even in fallback mode. Stopping.")
                    break
        
        pbar.close()
        print("Inpainting Done.")
        
        colors_out = colors.reshape(h, w, 3).cpu().numpy()
        red_mask = np.zeros((h, w), dtype=bool) 
        
        return colors_out, red_mask

    def __call__(self, texture_input, **kwargs):
        # 兼容旧接口
        from PIL import Image
        import torchvision.transforms as T
        if isinstance(texture_input, str):
            texture = T.ToTensor()(Image.open(texture_input))
        else:
            texture = texture_input
            
        self.set_texture_map(texture)
        position_map, face_idx = self.UV_pos_render(texture.shape[1])
        
        # position_map 和 face_idx 已经在 GPU 上了
        colors_out, red_mask = self.update_texture(position_map, face_idx, **kwargs)
        
        return torch.from_numpy(colors_out).to(self.device), position_map, red_mask