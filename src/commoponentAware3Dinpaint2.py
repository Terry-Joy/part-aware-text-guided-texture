
# ### 1️⃣ `k_neighbors`（每个未染色点选最近邻数量）

# * 作用：控制每个点参考的有色点数量，影响加权平均的稳定性。
# * 建议：

#   * 小模型或小纹理图：`30~60`
#   * 大模型或高分辨率纹理：`60~150`
# * 原则：

#   * 太小可能导致颜色噪点。
#   * 太大可能导致边缘细节被拉平（尤其部件边界）。

# ---

# ### 2️⃣ `batch_size`（每轮染色处理未填充点的数量）

# * 作用：控制每轮迭代处理多少点。
# * 建议：

#   * 500~1000 为默认安全值。
#   * 高分辨率纹理可以增大到 `2000`，但内存消耗增加。
# * 原则：

#   * 较小批次 → 每轮颜色更局部，迭代更细粒度。
#   * 较大批次 → 更快收敛，但可能边缘被拉平。

# ---

# ### 3️⃣ `cos_threshold`（判定补全点的阈值）

# * 作用：决定哪些 texel 被认为是需要补全的（黑色或 max_cos < 阈值）。
# * 建议：

#   * 默认 `0.17`
#   * 如果你希望更严格 → 提高到 `0.2~0.25`
#   * 如果希望更多点自动补全 → 降到 `0.1~0.15`

# ---

# ### 4️⃣ `distance_exp`（距离权重指数）

# * 作用：控制距离权重衰减速度。
# * 建议：

#   * 默认 `1.0`
#   * 对边缘细节重要时可以略微增大到 `1.2~1.5`，让近邻贡献更大。
#   * 对整体平滑优先时保持 `1.0`。

# ---

# ### 5️⃣ 部件权重（component_weight_same / component_weight_diff）

# * 作用：保证同部件内颜色优先传播。
# * 建议：

#   * 同部件：`1.0`
#   * 不同部件：`0.05~0.1`（弱化跨部件传播）
# * 原则：

#   * 不同部件权重太大 → 可能越界传播，破坏边界。
#   * 不同部件权重太小 → 有点颜色可能不自然。

# ---

# ### 6️⃣ `max_iter`（最大迭代轮数）

# * 作用：控制补全过程的迭代次数。
# * 建议：

#   * 10~20 次通常足够。
#   * 可根据未填充点比例动态结束（未填充点 < 1% 就提前终止）。

# ---

# ### 7️⃣ 排序策略

# * **两层排序**：

#   1. KNN 找距离最近的已知点。
#   2. 对每批未填充点，根据邻域平均 cos 从大到小排序染色。
# * 原则：

#   * 先染最可靠的（邻域 cos 大）→ 避免暗点拉低颜色。
#   * 分批染色（batch_size）比一次性全体染色更稳。

# ---

# ### 8️⃣ 综合建议

# * 初始尝试参数：

# ```python
# k_neighbors = 60
# batch_size = 500
# cos_threshold = 0.17
# distance_exp = 1.0
# component_weight_same = 1.0
# component_weight_diff = 0.1
# max_iter = 15
# ```

# * 如果看到边缘部件颜色拉低 → 增大 `component_weight_same` 或降低 `batch_size`。
# * 如果整体颜色太暗 → 调整 `cos_threshold` 或排序策略。

# ---

# 我可以帮你写一段**自动根据未填充点比例自适应 batch_size 和 max_iter** 的逻辑，让补全速度和质量兼顾，这样就不用手动调太多参数。

# 你希望我帮你写吗？



import torch
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from pytorch3d.renderer import TexturesUV
import json

class ComponentAware3DInpainting:
    def __init__(self, mesh, device, max_cos_map, avg_cos_map, face2label=None):
        """
        Args:
            mesh: pytorch3d Meshes 对象
            device: torch.device
            max_cos_map: torch.Tensor [H,W]，最大cos map
            avg_cos_map: torch.Tensor [H,W]，平均cos map
            face2label: dict {face_id: component_id} 或 None
        """
        self.mesh = mesh
        self.device = device
        self.max_cos_map = max_cos_map
        self.avg_cos_map = avg_cos_map
        # 读取 face2label
        if isinstance(face2label, str):
            with open(face2label, "r") as f:
                self.face2label = json.load(f)
            # 确保 key/value 为 int
            self.face2label = {int(k): int(v) for k, v in self.face2label.items()}
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
        import kaolin as kal
        verts = self.mesh.verts_packed()
        faces = self.mesh.faces_packed()
        verts_uv = self.mesh.textures.verts_uvs_padded()[0]
        faces_uv = self.mesh.textures.faces_uvs_padded()[0]

        uv_face_attr = torch.index_select(verts_uv, 0, faces_uv.view(-1)).view(faces.shape[0], faces_uv.shape[1],2).unsqueeze(0)
        face_vertices_world = kal.ops.mesh.index_vertices_by_faces(verts.unsqueeze(0), faces)
        face_vertices_z = torch.zeros_like(face_vertices_world[:,:,:,-1], device=verts.device)
        uv_position, face_idx = kal.render.mesh.rasterize(
            texture_dim, texture_dim, face_vertices_z,
            uv_face_attr * 2 - 1, face_features=face_vertices_world
        )
        uv_position = torch.clamp(uv_position,-1,1)/2 + 0.5
        uv_position[face_idx==-1] = 0
        return uv_position, face_idx

    @torch.no_grad()
    def update_texture(self, position_map, face_idx,
                       cos_threshold=0.10,
                       k_neighbors=100,
                       max_iter=20,
                       distance_exp=1.0,
                       component_weight_same=1.0,
                       component_weight_diff=0.05,
                       batch_size=500):
        """两层排序 + 加权迭代纹理补全（修正版）"""
        print()
        # texture: (H, W, C)
        texture = self.mesh.textures.maps_padded()[0]
        tex_np = texture.cpu().numpy()  # (H, W, C)
        h, w = tex_np.shape[0], tex_np.shape[1]

        colors = tex_np  # (H, W, 3)
        points = position_map.reshape(-1, 3).cpu().numpy()
        colors_flat = colors.reshape(-1, 3).copy()  # (H*W, 3)

        # max_cos_flat
        max_cos_np = self.max_cos_map.detach().cpu().numpy() if torch.is_tensor(self.max_cos_map) else np.array(self.max_cos_map)
        if max_cos_np.ndim == 3 and max_cos_np.shape[2] > 1:
            max_cos_np = max_cos_np[..., 0]
        max_cos_flat = max_cos_np.reshape(-1)

        # avg_cos_flat
        avg_cos_np = self.avg_cos_map.detach().cpu().numpy() if torch.is_tensor(self.avg_cos_map) else np.array(self.avg_cos_map)
        if avg_cos_np.ndim == 3 and avg_cos_np.shape[2] > 1:
            avg_cos_np = avg_cos_np[..., 0]
        avg_cos_flat = avg_cos_np.reshape(-1)
        print('????')
        # 待补全 mask: 黑色或者 max_cos < 阈值
        black_mask = np.all(colors_flat == 0, axis=1)
        low_cos_mask = max_cos_flat < cos_threshold
        valid_mask = (points[:, 0] != 0)
        # fill_mask = valid_mask & low_cos_mask
        fill_mask = valid_mask & (black_mask | low_cos_mask)

        print(f"[update_texture] {int(fill_mask.sum())} / {int(valid_mask.sum())} texels need filling")

        # --- build texel_component from face_idx + face2label ---
        # face_idx may be tensor shape [1,H,W] or [H,W]
        if torch.is_tensor(face_idx):
            face_idx_np = face_idx.cpu().numpy()
        else:
            face_idx_np = np.array(face_idx)
        # squeeze leading dim if present
        if face_idx_np.ndim == 3 and face_idx_np.shape[0] == 1:
            face_idx_np = face_idx_np[0]
        face_idx_flat = face_idx_np.reshape(-1)  # length H*W

        # initialize
        texel_component = np.zeros(len(points), dtype=np.int32)
        if self.face2label is not None:
            # build array lookup for speed
            # determine max face id
            max_face_id = int(face_idx_flat.max()) if face_idx_flat.size > 0 else 0
            # create default 0 array
            face_label_arr = np.zeros(max_face_id + 1, dtype=np.int32)
            # fill with provided mapping (if some face ids > max_face_id, we clamp)
            for fid, cid in self.face2label.items():
                fid_i = int(fid)
                if fid_i <= max_face_id:
                    face_label_arr[fid_i] = int(cid)
            # for face_idx == -1 (background) we'll keep 0
            valid_face_mask = face_idx_flat >= 0
            texel_component[valid_face_mask] = face_label_arr[face_idx_flat[valid_face_mask]]
        else:
            texel_component[:] = 0

        # working copies
        points_t = points
        colors_t = colors_flat.copy()
        fill_mask_t = fill_mask.copy()

        for iter_idx in tqdm(range(max_iter), desc="Inpainting"):
            unfilled_idx = np.where(fill_mask_t)[0]
            if unfilled_idx.size == 0:
                break

            known_idx = np.where((~fill_mask_t) & valid_mask)[0]
            if known_idx.size == 0:
                print("No known points left to propagate.")
                break

            # Step1: find knn among known points
            nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, len(known_idx)), algorithm='auto')
            nbrs.fit(points_t[known_idx])
            distances, indices = nbrs.kneighbors(points_t[unfilled_idx])  # shapes: (N_unfilled, k)
            distances = np.maximum(distances, 1e-8)
            # distance weight (higher = more influence)
            # distance_weights = 1.0 / (distances ** distance_exp)  # (N_unfilled, k)

            # Step2: neighbor avg cos (for sorting)
            # note: avg_cos_flat[known_idx][indices] -> (N_unfilled, k)
            neighbor_avg_cos = avg_cos_flat[known_idx][indices].mean(axis=1)  # (N_unfilled,)
            sort_order = np.argsort(-neighbor_avg_cos)  # descending
            unfilled_idx_sorted = unfilled_idx[sort_order]
            distances_sorted = distances[sort_order]
            indices_sorted = indices[sort_order]

            # Step3: batch update
            for batch_start in range(0, len(unfilled_idx_sorted), batch_size):
                batch_idx = unfilled_idx_sorted[batch_start:batch_start + batch_size]
                batch_distances = distances_sorted[batch_start:batch_start + batch_size]  # (B,k)
                distance_weights = 1.0 / (batch_distances ** distance_exp + 1e-12)

                batch_indices = indices_sorted[batch_start:batch_start + batch_size]  # (B,k)

                # 部件权重：同部件 -> component_weight_same, else -> component_weight_diff
                batch_comp_weights = np.ones_like(batch_distances, dtype=np.float32) * component_weight_diff
                unfilled_comp = texel_component[batch_idx][:, None]  # (B,1)
                known_comp = texel_component[known_idx][batch_indices]  # (B,k)
                same_mask = (known_comp == unfilled_comp)
                batch_comp_weights[same_mask] = component_weight_same

                # final numeric total weights = distance_weight * comp_weight
                # total_weights = distance_weights * batch_comp_weights  # (B, k)
                total_weights = distance_weights * batch_comp_weights

                # normalize per-row to avoid scale issues (optional, but stable)
                row_sum = (total_weights.sum(axis=1, keepdims=True) + 1e-12)
                norm_weights = total_weights / row_sum  # (B,k)

                neighbor_colors = colors_t[known_idx][batch_indices]  # (B,k,3)
                weighted_color = (neighbor_colors * norm_weights[..., None]).sum(axis=1)  # (B,3)

                colors_t[batch_idx] = weighted_color
                fill_mask_t[batch_idx] = False


        print('!!!!')
        # reshape back to (H, W, 3)
        colors_out = colors_t.reshape(h, w, 3)
        red_mask = np.zeros((h, w), dtype=bool)
        red_mask[fill_mask.reshape(h, w)] = True
        return colors_out, red_mask


    def __call__(self, texture_input, **kwargs):
        """输入纹理 -> 输出补全结果（注意返回 face_idx）"""
        from PIL import Image
        import torchvision.transforms as T
        if isinstance(texture_input, str):
            texture = T.ToTensor()(Image.open(texture_input))
        else:
            texture = texture_input

        # set texture
        self.set_texture_map(texture)
        # UV render must return (position_map, face_idx)
        position_map, face_idx = self.UV_pos_render(texture.shape[1])
        colors_out, red_mask = self.update_texture(position_map, face_idx, **kwargs)
        colors_out_torch = torch.from_numpy(colors_out).float().to(self.device)
        

        return colors_out_torch, position_map, red_mask