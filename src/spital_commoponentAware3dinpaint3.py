import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from PIL import Image
from torchvision import transforms
from pytorch3d.renderer import TexturesUV
from sklearn.neighbors import NearestNeighbors
import kaolin as kal
from .renderer.voronoi import voronoi_solve  # 你的 voronoi solve

class ComponentSparse3DInpainting:
    def __init__(self, mesh, device, max_cos_map, face2label=None):
        """
        Args:
            mesh: pytorch3d Meshes 对象
            device: torch.device
            max_cos_map: torch.Tensor [H,W] 或 np.array，最大可见性/可靠性
            face2label: dict {face_id: component_id} 或 None
        """
        self.mesh = mesh
        self.device = device
        self.max_cos_map = max_cos_map

        if isinstance(face2label, str):
            import json
            with open(face2label, "r") as f:
                face2label = json.load(f)
            self.face2label = {int(k): int(v) for k, v in face2label.items()}
        elif isinstance(face2label, dict):
            self.face2label = {int(k): int(v) for k, v in face2label.items()}
        else:
            self.face2label = None

    def set_texture_map(self, texture):
        """绑定纹理到 mesh"""
        new_map = texture.permute(1,2,0).to(self.device)
        new_tex = TexturesUV(
            [new_map],
            self.mesh.textures.faces_uvs_padded(),
            self.mesh.textures.verts_uvs_padded(),
            sampling_mode="nearest"
        )
        self.mesh.textures = new_tex

    @torch.no_grad()
    def UV_pos_render(self, texture_dim):
        """渲染 UV 空间位置"""
        verts = self.mesh.verts_packed()
        faces = self.mesh.faces_packed()
        verts_uv = self.mesh.textures.verts_uvs_padded()[0]
        faces_uv = self.mesh.textures.faces_uvs_padded()[0]
        uv_face_attr = torch.index_select(verts_uv, 0, faces_uv.view(-1)).view(
            faces.shape[0], faces_uv.shape[1], 2).unsqueeze(0)
        face_vertices_world = kal.ops.mesh.index_vertices_by_faces(verts.unsqueeze(0), faces)
        face_vertices_z = torch.zeros_like(face_vertices_world[:, :, :, -1], device=verts.device)
        uv_position, face_idx = kal.render.mesh.rasterize(
            texture_dim, texture_dim, face_vertices_z,
            uv_face_attr * 2 - 1, face_features=face_vertices_world
        )
        uv_position = torch.clamp(uv_position, -1, 1) / 2 + 0.5
        uv_position[face_idx == -1] = 0
        return uv_position, face_idx

    def construct_sparse_matrix(self, points, colors, fill_mask, texel_component, k_neighbors=60,
                                component_weight_same=1.0, component_weight_diff=0.1):
        """
        构建稀疏矩阵 L 用于一次性颜色插值
        """
        known_idx = np.where(~fill_mask)[0]
        unknown_idx = np.where(fill_mask)[0]

        if len(unknown_idx) == 0 or len(known_idx) == 0:
            return colors, unknown_idx

        nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, len(known_idx)), algorithm='auto')
        nbrs.fit(points[known_idx])
        distances, indices = nbrs.kneighbors(points[unknown_idx])
        distances = np.maximum(distances, 1e-8)
        distance_weights = 1.0 / distances  # 可以调指数

        # 部件权重
        known_comp = texel_component[known_idx][indices]
        unknown_comp = texel_component[unknown_idx][:, None]
        comp_weight = np.where(known_comp == unknown_comp, component_weight_same, component_weight_diff)

        total_weight = distance_weights * comp_weight
        norm_weight = total_weight / (total_weight.sum(axis=1, keepdims=True) + 1e-12)

        # 插值
        colors[unknown_idx] = (colors[known_idx][indices] * norm_weight[..., None]).sum(axis=1)
        return colors, unknown_idx

    @torch.no_grad()
    def update_texture(self, position_map, cos_threshold=0.10, k_neighbors=60,
                       component_weight_same=1.0, component_weight_diff=0.1):
        texture = self.mesh.textures.maps_padded()[0]
        tex_np = texture.cpu().numpy()
        h, w = tex_np.shape[:2]

        points = position_map.reshape(-1, 3).cpu().numpy()
        colors = tex_np.reshape(-1, 3).copy()

        if torch.is_tensor(self.max_cos_map):
            max_cos_flat = self.max_cos_map.detach().cpu().numpy()
        else:
            max_cos_flat = np.array(self.max_cos_map)
        if max_cos_flat.ndim == 3 and max_cos_flat.shape[2] > 1:
            max_cos_flat = max_cos_flat[..., 0]
        max_cos_flat = max_cos_flat.reshape(-1)

        black_mask = np.all(colors == 0, axis=1)
        low_cos_mask = max_cos_flat < cos_threshold
        valid_mask = points[:, 0] != 0
        fill_mask = valid_mask & (black_mask | low_cos_mask)
        print(f"[update_texture] {int(fill_mask.sum())}/{int(valid_mask.sum())} texels need filling")

        texel_component = np.zeros(len(points), dtype=np.int32)
        # 部件信息
        if self.face2label is not None:
            face_idx = getattr(self, "face_idx", None)
            if face_idx is not None:
                face_idx_flat = face_idx.cpu().numpy().reshape(-1)
                max_face_id = int(face_idx_flat.max()) if face_idx_flat.size > 0 else 0
                face_label_arr = np.zeros(max_face_id + 1, dtype=np.int32)
                for fid, cid in self.face2label.items():
                    if fid <= max_face_id:
                        face_label_arr[fid] = cid
                valid_face_mask = face_idx_flat >= 0
                texel_component[valid_face_mask] = face_label_arr[face_idx_flat[valid_face_mask]]

        colors, unknown_idx = self.construct_sparse_matrix(
            points, colors, fill_mask, texel_component,
            k_neighbors=k_neighbors,
            component_weight_same=component_weight_same,
            component_weight_diff=component_weight_diff
        )

        colors = colors.reshape(h, w, 3)

        # colors: numpy (H, W, 3) 或 torch tensor [H,W,3] (已在你的代码中)
        colors_tensor = torch.from_numpy(colors).to(self.device) if not torch.is_tensor(colors) else colors.to(self.device)
        # mask 从 position_map 中取出第一通道并确保是二维 (H,W)
        mask = position_map[..., 0].to(self.device)  # 可能是 shape (1,H,W) 或 (H,W)
        if mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        elif mask.dim() != 2:
            # 保险起见，压扁第一个为 2D（若 mask 形状不规整，可打印以调试）
            mask = mask.reshape(mask.shape[-2], mask.shape[-1])

        # debug 打印（出错时打开）
        # print("voronoi input shapes:", colors_tensor.shape, mask.shape)

        colors_voronoi = voronoi_solve(
            colors_tensor,  # [H, W, C] tensor on device
            mask            # [H, W] tensor on device
        )

        # 返回 numpy 供后续处理（与你原来流程一致）
        colors = colors_voronoi.cpu().numpy()

        fill_mask = fill_mask.reshape(h, w)
        return colors, fill_mask

    def __call__(self, texture_input):
        if isinstance(texture_input, str):
            texture = transforms.ToTensor()(Image.open(texture_input))
        else:
            texture = texture_input
        self.set_texture_map(texture)
        position_map, face_idx = self.UV_pos_render(texture.shape[1])
        self.face_idx = face_idx  # 保存供构建部件稀疏矩阵用
        colors_out, fill_mask = self.update_texture(position_map)
        colors_out_torch = torch.from_numpy(colors_out).to(self.device)

        # 红色可视化
        red_vis_tex = colors_out_torch.clone()
        red_mask_torch = torch.from_numpy(fill_mask).to(self.device)
        red_vis_tex[red_mask_torch > 0] = torch.tensor([1.0, 0.0, 0.0], device=self.device)

        return colors_out_torch, position_map, red_vis_tex
