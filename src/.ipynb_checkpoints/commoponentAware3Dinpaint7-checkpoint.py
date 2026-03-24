import os
import json
import torch
import numpy as np
import kaolin as kal
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from pytorch3d.renderer import TexturesUV
from pytorch3d.ops import knn_points
from scipy.ndimage import binary_dilation
from .renderer.voronoi import voronoi_solve

# 假设你的 utils 里有这个，如果没有，下面我会提供一个兼容语义的 GPU 版本
# from .utils import construct_sparse_L 

class ComponentAware3DInpaintingGPU2:
    def __init__(self, mesh, device, max_cos_map, avg_cos_map, face2label=None):
        self.mesh = mesh
        self.device = device
        
        # 1. Max Cos Map: 用于定义哪些点是"没颜色"的 (和黑色点一起作为初始空洞)
        if not torch.is_tensor(max_cos_map):
            max_cos_map = torch.from_numpy(np.array(max_cos_map)).float()
        if max_cos_map.ndim == 3: 
            max_cos_map = max_cos_map[..., 0]
        self.max_cos_map = max_cos_map.to(self.device).reshape(-1)

        # 2. Face Label: 用于构建语义矩阵，防止渗色
        self.face_label_tensor = None
        if face2label is not None:
            if isinstance(face2label, str) and os.path.exists(face2label):
                with open(face2label, "r") as f:
                    face2label = json.load(f)
            face2label = {int(k): int(v) for k, v in face2label.items()} if face2label else {}
            if len(face2label) > 0:
                max_fid = max(face2label.keys())
                self.face_label_tensor = torch.full((max_fid + 1,), -1, dtype=torch.int32, device=self.device)
                ids = torch.tensor(list(face2label.keys()), dtype=torch.long, device=self.device)
                lbls = torch.tensor(list(face2label.values()), dtype=torch.int32, device=self.device)
                self.face_label_tensor[ids] = lbls

    def set_texture_map(self, texture):
        if texture.shape[0] == 3:
            new_map = texture.permute(1, 2, 0).to(self.device)
        else:
            new_map = texture.to(self.device)
        new_tex = TexturesUV([new_map], self.mesh.textures.faces_uvs_padded(), 
                             self.mesh.textures.verts_uvs_padded(), sampling_mode="nearest")
        self.mesh.textures = new_tex

    @torch.no_grad()
    def UV_pos_render(self, texture_dim):
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
    def construct_semantic_sparse_L(self, target_points, all_points, target_comp, all_comp, k_neighbors=60, smooth_scale=2.0):
        """
        构建稀疏矩阵 L_invalid，并在这里加入【语义隔离】
        """
        # 1. GPU KNN
        p1 = target_points.unsqueeze(0)
        p2 = all_points.unsqueeze(0)
        dists_sq, idxs, _ = knn_points(p1, p2, K=k_neighbors)
        
        dists = torch.sqrt(dists_sq.squeeze(0) + 1e-8)
        neighbor_indices = idxs.squeeze(0)

        # 2. 几何权重 (Gaussian)
        sigma = dists.mean(dim=1, keepdim=True) + 1e-5
        weights = torch.exp(-(dists**2) / (2 * (sigma * smooth_scale)**2))

        # 3. 【语义隔离核心】(Semantic Shield)
        # 如果有标签，把异类邻居的权重强制置零！
        if self.face_label_tensor is not None:
            neighbor_comp = all_comp[neighbor_indices] 
            t_comp = target_comp.unsqueeze(1)
            
            # 只有同类才保留权重
            semantic_mask = (t_comp == neighbor_comp).float()
            
            # Fallback: 如果孤岛（周围全是异类），则放行，否则该点永远无法被填色
            is_isolated = (semantic_mask.sum(dim=1, keepdim=True) == 0)
            semantic_mask[is_isolated.expand_as(semantic_mask)] = 1.0 
            
            weights = weights * semantic_mask

        # 4. 构建稀疏矩阵 indices 和 values
        # 注意：这里我们【不要】做归一化 (row normalize)，因为 MVPaint 的 while 循环里会动态做归一化
        N_target = target_points.shape[0]
        N_total = all_points.shape[0]
        
        row_idx = torch.arange(N_target, device=self.device).unsqueeze(1).repeat(1, k_neighbors).reshape(-1)
        col_idx = neighbor_indices.reshape(-1)
        values = weights.reshape(-1) # 原始权重
        
        indices = torch.stack([row_idx, col_idx])
        L_sparse = torch.sparse_coo_tensor(indices, values, (N_target, N_total))
        
        return L_sparse

    @torch.no_grad()
    def update_texture(self, position_map, face_idx, k_neighbors=60):
        print(f"\n[Start GPU Inpainting] Mode: Dynamic Iterative Propagation (MVPaint Style)")
        
        texture = self.mesh.textures.maps_padded()[0]
        h, w, c = texture.shape
        points = position_map.reshape(-1, 3)
        colors = texture.reshape(-1, 3).clone()
        
        # =========================================================
        # 1. 定义初始空洞 (Holes) 与 有效源 (Sources)
        # =========================================================
        luminance = colors.mean(dim=1)
        
        # [核心修复 1]：用 face_idx 精准判断是不是 Mesh 上的有效像素，避免坐标误杀
        f_idx_flat = face_idx.reshape(-1).long()
        valid_pixel = (f_idx_flat >= 0)
        
        # 哪些点是"有颜色"且可靠的？
        is_reliable = (self.max_cos_map >= 0.8) & (luminance >= 0.01)
        
        # 哪些洞需要被填？(属于网格，且颜色不可靠)
        fill_mask = valid_pixel & (~is_reliable)
        invalid_index = torch.nonzero(fill_mask, as_tuple=True)[0]
        
        # 准备返回的 red_mask
        red_mask_np = fill_mask.reshape(h, w).cpu().numpy()
        
        if len(invalid_index) == 0:
            return colors.reshape(h, w, 3).cpu().numpy(), red_mask_np

        # =========================================================
        # 2. 准备 Semantic Component
        # =========================================================
        texel_comp = torch.full((len(points),), -1, dtype=torch.int32, device=self.device)
        if self.face_label_tensor is not None:
            valid_f = f_idx_flat >= 0
            safe_f = torch.clamp(f_idx_flat, 0, len(self.face_label_tensor)-1)
            texel_comp[valid_f] = self.face_label_tensor[safe_f[valid_f]]

        # =========================================================
        # 3. 构建稀疏矩阵 (L_invalid)
        # =========================================================
        print(f" -> Building Semantic Sparse Matrix for {len(invalid_index)} pixels...")
        target_points = points[invalid_index]
        target_comp = texel_comp[invalid_index]
        
        L_invalid = self.construct_semantic_sparse_L(
            target_points, points, target_comp, texel_comp, k_neighbors=k_neighbors
        )

        # =========================================================
        # 4. 核心 While 循环 (解决背景污染的关键)
        # =========================================================
        
        # [核心修复 2]：默认所有人都是 0 (未填色)，包括背景点！
        colored_count = torch.zeros(len(points), 1, device=self.device)
        
        # 只有【有效】且【可靠】的网格点，才是合法的初始颜料源！
        source_mask = valid_pixel & is_reliable
        colored_count[source_mask] = 1.0
        
        total_colored = colored_count.sum()
        coloring_round = 0
        stage = "uncolored"
        
        print(" -> Running Iterative Propagation...")
        pbar = tqdm(miniters=100)
        
        while stage == "uncolored" or coloring_round > 0:
            # A. 分子：加权颜色和 (只算已填色的邻居)
            weighted_color_sum = torch.sparse.mm(L_invalid, colors * colored_count)
            
            # B. 分母：权重和 (只算已填色的邻居的权重)
            valid_weight_sum = torch.sparse.mm(L_invalid, colored_count)
            
            # C. 更新条件
            can_update_mask = (valid_weight_sum > 1e-6)
            
            # D. 计算新颜色
            new_colors = weighted_color_sum / (valid_weight_sum + 1e-8)
            
            # E. 更新颜色
            current_hole_colors = colors[invalid_index]
            colors[invalid_index] = torch.where(can_update_mask, new_colors, current_hole_colors)
            
            # F. 更新 colored_count
            new_filled_mask = can_update_mask.float()
            current_status = colored_count[invalid_index]
            colored_count[invalid_index] = torch.max(current_status, new_filled_mask)
            
            # G. 检查进度
            new_total_colored = colored_count.sum()
            color_num = new_total_colored - total_colored
            
            if color_num > 0:
                total_colored = new_total_colored
                coloring_round += 1
            else:
                stage = "colored"
                coloring_round -= 1
                
            pbar.update(1)
            if coloring_round > 10000:
                print("Break: Iteration > 10000")
                break
                
        pbar.close()
        print("Inpainting Done.")
        return colors.reshape(h, w, 3).cpu().numpy(), red_mask_np

    def __call__(self, texture_input, **kwargs):
        if isinstance(texture_input, str):
            rgb = transforms.ToTensor()(Image.open(texture_input).convert("RGB"))
        elif isinstance(texture_input, torch.Tensor):
            rgb = texture_input
        else:
             raise ValueError("texture_input must be a path or tensor")
            
        self.set_texture_map(rgb)
        position_map, face_idx = self.UV_pos_render(rgb.shape[1])
        
        # 核心更新
        colors_out, red_mask = self.update_texture(position_map, face_idx, **kwargs)
        
        # Voronoi 后处理
        result_tex_rgb = torch.from_numpy(colors_out).to(self.device)
        bkgd_mask = position_map.sum(-1) == 0
        fore_mask = (1 - bkgd_mask.int()).squeeze()
        bigger_region = binary_dilation(fore_mask.cpu().numpy(), iterations=8)
        
        # 加补全
        result_tex_rgb = voronoi_solve(result_tex_rgb, position_map.squeeze()[..., 0])
        
        bigger_region = torch.from_numpy(bigger_region).unsqueeze(-1).to(self.device)
        result_tex_rgb = torch.where(bigger_region > 0, result_tex_rgb, 1) # 背景白

        # 可视化 mask
        red_vis_tex = result_tex_rgb.clone()
        red_vis_tex_flat = red_vis_tex.reshape(-1, 3)
        mask_flat = torch.from_numpy(red_mask.reshape(-1)).to(self.device)
        red_vis_tex_flat[mask_flat] = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        red_vis_tex = red_vis_tex_flat.reshape(red_vis_tex.shape)
        
        return result_tex_rgb, position_map, red_vis_tex