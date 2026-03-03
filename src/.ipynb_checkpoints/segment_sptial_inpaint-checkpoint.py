import os
from PIL import Image
import numpy as np
import torch
from .renderer.project import UVProjection as UVP
from .utils import *
from torchvision import transforms
import kaolin as kal
from scipy.ndimage import binary_dilation
from .renderer.voronoi import voronoi_solve
from pytorch3d.renderer import TexturesUV

class SpatialAware3DInpainting:
    def __init__(self, mesh, device, max_cos_map):
        self.mesh = mesh
        self.device = device
        self.max_cos_map = max_cos_map

    def set_texture_map(self, texture):
        new_map = texture.permute(1, 2, 0)
        new_map = new_map.to(self.device)
        new_tex = TexturesUV(
            [new_map],
            self.mesh.textures.faces_uvs_padded(),
            self.mesh.textures.verts_uvs_padded(),
            sampling_mode="nearest"
        )
        self.mesh.textures = new_tex

    @torch.no_grad()
    def UV_pos_render(self, texture_dim):
        verts = self.mesh.verts_packed()
        faces = self.mesh.faces_packed()
        verts_uv = self.mesh.textures.verts_uvs_padded()[0]
        faces_uv = self.mesh.textures.faces_uvs_padded()[0]
        uv_face_attr = torch.index_select(verts_uv, 0, faces_uv.view(-1)).view(faces.shape[0], faces_uv.shape[1],
                                                                               2).unsqueeze(0)
        face_vertices_world = kal.ops.mesh.index_vertices_by_faces(verts.unsqueeze(0), faces)
        face_vertices_z = torch.zeros_like(face_vertices_world[:, :, :, -1], device=verts.device)
        uv_position, face_idx = kal.render.mesh.rasterize(texture_dim, texture_dim, face_vertices_z,
                                                          uv_face_attr * 2 - 1, face_features=face_vertices_world)
        uv_position = torch.clamp(uv_position, -1, 1) / 2 + 0.5
        uv_position[face_idx == -1] = 0
        return uv_position, face_idx

    # @torch.no_grad()
    # def update_colored_points(self, colored_points):
    #     points = colored_points[:, :3]
    #     colors = colored_points[:, 3:]  
    #     color_mask = colors.sum(axis=1) != 0

    #     if color_mask.sum() != len(points):
    #         colors, invalid_index = self.knn_color_completion(points, colors, color_mask)

    #     return np.concatenate([points, colors], 1)

    @torch.no_grad()
    def update_colored_points(self, colored_points, color_mask):
        """
        输入:
            colored_points: np.ndarray, shape [M, 6] -> [x,y,z, r,g,b]
                这里只会传入需要补全/更新的那部分点。
            color_mask: np.ndarray, shape [M], bool
                True 表示该点已有颜色，False 表示该点需要补全颜色。
        返回:
            np.ndarray, shape [M, 6] -> [x,y,z, r,g,b] （已补全，但不做红色标记）
        """
        # 保持原来逻辑：只做补全，不在这里涂红
        points = colored_points[:, :3]
        colors = colored_points[:, 3:].copy()  # copy一份，numpy

        if np.all(color_mask):
            return colored_points  # 全有色，直接返回

        colors, invalid_index = self.knn_color_completion(points, colors, color_mask)
        # 返回拼回去的点+色
        return np.concatenate([points, colors], 1)



    def knn_color_completion(self, points, colors, color_mask, n_neighbors=60):
        """
        Completes missing colors for points using k-Nearest Neighbors (kNN).

        Args:
            points (np.ndarray): Array of shape (N, 3), where N is the number of points, representing point cloud coordinates.
            colors (np.ndarray): Array of shape (N, 3), representing the RGB colors of the points.
            color_mask (np.ndarray): Boolean array of shape (N,), indicating which points have valid colors.
            n_neighbors (int): Number of neighbors to consider for kNN. Default is 60.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Updated colors of shape (N, 3) with missing values completed.
                - Indices of points that had missing colors initially (before completion).
        """

        normals = get_normals(points)
        normals = torch.tensor(normals, dtype=torch.float32)
        normals = torch.nn.functional.normalize(normals, p=2, dim=1)

        points = torch.tensor(points, dtype=torch.float32)
        colors = torch.tensor(colors, dtype=torch.float32)
        color_mask = torch.tensor(color_mask, dtype=torch.bool)
        invalid_index = torch.where(color_mask == False)[0]

        invalid_index_ori = deepcopy(invalid_index)
        # 将已有颜色的点和无色的点分开
        unknown_points = points[~color_mask]

        unknown_normals = normals[~color_mask]

        # 使用 NearestNeighbors 找到无色点的 k 近邻
        knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')

        knn.fit(points)
        distances, indices = knn.kneighbors(unknown_points)
        distances = distances[:, 1:]
        indices = indices[:, 1:]

        edge_neighbors_normals = index_select(normals, torch.LongTensor(indices), 0)
        cos = torch.einsum('ec,enc->en', unknown_normals, edge_neighbors_normals)
        cos = cos / 2 + 0.5
        cos = torch.where(cos < 0.5, 1e-8, cos)
        cos = torch.where(cos > 0.9, 10, cos)

        # cos = torch.where(cos < 0.3, 1e-4, cos)  # 原来0.5太严
        # cos = torch.where(cos > 0.85, 5, cos)    # 原来10太激进


        distances = torch.from_numpy(distances)
        distance_score = torch.nn.functional.normalize((1 / distances), p=2, dim=1)

        weight = cos * distance_score
        # weight = distance_score  # 只用距离权重

        colored_count = torch.ones_like(colors[:, 0])  # [V]
        colored_count[invalid_index] = 0
        L_invalid = construct_sparse_L(indices, weight, m=invalid_index.shape[0], n=colors.shape[0])
        # 根据 k 近邻插值颜色
        total_colored = colored_count.sum()
        coloring_round = 0
        stage = "uncolored"
        from tqdm import tqdm
        pbar = tqdm(miniters=100)
        while stage == "uncolored" or coloring_round > 0:
            new_color = torch.matmul(L_invalid, colors * colored_count[:, None])  # [IV, 3] 邻接点贡献和
            new_count = torch.matmul(L_invalid, colored_count)[:, None]  # [IV, 1] 有几个邻接点是有贡献的
            colors[invalid_index] = torch.where(new_count > 0, new_color / new_count, colors[invalid_index])
            colored_count[invalid_index] = (new_count[:, 0] > 0).float()
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
                print("coloring_round > 10000, break")
                break

        return colors.numpy(), invalid_index_ori.numpy()

    @torch.no_grad()
    def update_texture(self, position_map, cos_threshold=0.17):
        """
        Updates the texture map by interpolating missing colors using the position map.

        Args:
            position_map (torch.Tensor): Tensor of shape (H, W, 3), representing the 3D positions of points in the UV space.

        Returns:
            np.ndarray: Updated texture map of shape (H, W, 3) with completed colors.
        """
        texture = self.mesh.textures.maps_padded()[0]
        texture_map_np = texture.cpu().numpy()

        h, w = texture_map_np.shape[:2]
        points = position_map.reshape(-1, 3).cpu().numpy()
        texture = texture_map_np.reshape(-1, 3)


        if not hasattr(self, "max_cos_map") or self.max_cos_map is None:
            raise RuntimeError("self.max_cos_map not set - please compute it before calling update_texture()")
        # 将 max_cos_map 转为 numpy 平坦数组
        if torch.is_tensor(self.max_cos_map):
            max_cos_np = self.max_cos_map.detach().cpu().numpy()
        else:
            max_cos_np = np.array(self.max_cos_map)
        # 只取第一个通道（如果有多个通道）
        if max_cos_np.ndim == 3 and max_cos_np.shape[2] > 1:
            max_cos_np = max_cos_np[..., 0]

        max_cos_flat = max_cos_np.reshape(-1)

        # --- 构建填充掩码 fill_mask ---
        # 有效的 uv_position 点（position_map 中非 0）
        valid_pos_mask = points[:, 0] != 0
        valid_points = points[valid_pos_mask]
        valid_colors = texture[valid_pos_mask]
        valid_max_cos = max_cos_flat[valid_pos_mask]

        # 颜色为黑（all zero）
        color_black_mask = np.all(valid_colors == 0, axis=1)

        # 低可见性掩码（只考虑有效点）
        low_cos_mask = valid_max_cos < cos_threshold

        # 需要补全的点（长度 = N_valid）
        fill_mask_valid = color_black_mask | low_cos_mask

        # debug 打印
        num_fill = int(fill_mask_valid.sum())
        tot = fill_mask_valid.shape[0]
        print(f"[update_texture] Filling {num_fill} / {tot} texels ({num_fill/tot*100:.2f}%) "
              f"(black={int(color_black_mask.sum())}, low_cos={int(low_cos_mask.sum())})")

        # 若没有需要补全的点，直接返回原始纹理
        if num_fill == 0:
            return texture_map_np, fill_mask.reshape(h, w)

        colored_points = np.concatenate([valid_points, valid_colors], 1)

        color_mask = ~fill_mask_valid

        # 调用 update_colored_points
        updated_colored_points = self.update_colored_points(colored_points, color_mask=color_mask)

        # 写回补全结果
        colored_points[:, 3:] = updated_colored_points[:, 3:]

        # 将更新后的颜色放回原始 flat texture
        texture[valid_pos_mask] = colored_points[:, 3:]
        colors = texture.reshape(h, w, 3)

        # 创建 H*W 的 fill_mask
        fill_mask = np.zeros(h * w, dtype=bool)
        # 只在有效点里填充 fill_mask_valid
        fill_mask[valid_pos_mask] = fill_mask_valid
        # reshape 回 H, W
        fill_mask = fill_mask.reshape(h, w)

        # 返回更新后的纹理和要补全的地方用于红色点
        return colors, fill_mask

    def __call__(self, texture_input):
        """
        Processes a given texture and updates it with spatial-aware inpainting.

        Args:
            texture_path (str): Path to the input texture image.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Resultant texture map as a tensor of shape (H, W, 3) after inpainting.
                - Position map of shape (H, W, 3), representing the 3D positions in UV space.
        """
            # 如果传的是路径：
        if isinstance(texture_input, str):
            result_tex_rgb = transforms.ToTensor()(Image.open(texture_input))
        # 如果传的是 tensor：
        elif isinstance(texture_input, torch.Tensor):
            result_tex_rgb = texture_input
        else:
            raise ValueError("texture_input must be a path or tensor")

        self.set_texture_map(result_tex_rgb)
        position_map, face_idx = self.UV_pos_render(result_tex_rgb.shape[1])

        result_tex_rgb, fill_mask = self.update_texture(position_map)
        result_tex_rgb = torch.from_numpy(result_tex_rgb).to(self.device)

        bkgd_mask = position_map.sum(-1) == 0
        fore_mask = (1 - bkgd_mask.int()).squeeze()
        bigger_region = binary_dilation(fore_mask.cpu().numpy(), iterations=8)

        result_tex_rgb = voronoi_solve(result_tex_rgb, position_map.squeeze()[..., 0])
        bigger_region = torch.from_numpy(bigger_region).unsqueeze(-1).to(self.device)
        result_tex_rgb = torch.where(bigger_region > 0, result_tex_rgb, 1)

        # ===== 红色高亮可视化补全区域 =====
        red_vis_tex = result_tex_rgb.clone()
        fill_mask_torch = torch.from_numpy(fill_mask).to(self.device)
        red_vis_tex[fill_mask_torch > 0] = torch.tensor([1.0, 0.0, 0.0], device=self.device)


        return result_tex_rgb, position_map, red_vis_tex

