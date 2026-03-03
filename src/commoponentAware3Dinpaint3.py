
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

    # @torch.no_grad()
    # def update_texture(self, position_map, face_idx,
    #                    cos_threshold=0.17,
    #                    k_neighbors=60,
    #                    max_iter=5,  
    #                    distance_exp=1.0,
    #                    component_weight_same=1.0,
    #                    component_weight_diff=0.05,
    #                    batch_size=500): 
    #     """
    #     CPU 5步杀版（修复 Shape Bug）：锁定总数，固定5轮填完
    #     """
    #     print("\n[Start CPU Inpainting - Fixed 5 Steps]...")
        
    #     # 1. 准备数据
    #     texture = self.mesh.textures.maps_padded()[0]
    #     tex_np = texture.cpu().numpy()
    #     h, w = tex_np.shape[:2]

    #     points = position_map.reshape(-1, 3).cpu().numpy()
    #     colors = tex_np.reshape(-1, 3).copy() 

    #     # ==================== 【修复点】正确处理 Cos Map 的维度 ====================
    #     # 必须先转 numpy，判断是不是 3 通道，取第一通道后再 reshape
        
    #     # 处理 Max Cos
    #     if torch.is_tensor(self.max_cos_map):
    #         max_cos_np = self.max_cos_map.detach().cpu().numpy()
    #     else:
    #         max_cos_np = np.array(self.max_cos_map)
    #     # 如果是 (H, W, 3) 或 (H, W, C)，只取第一个通道
    #     if max_cos_np.ndim == 3 and max_cos_np.shape[2] >= 1:
    #         max_cos_np = max_cos_np[..., 0]
    #     max_cos_flat = max_cos_np.reshape(-1) # 现在的长度应该是 N，而不是 3N

    #     # 处理 Avg Cos
    #     if torch.is_tensor(self.avg_cos_map):
    #         avg_cos_np = self.avg_cos_map.detach().cpu().numpy()
    #     else:
    #         avg_cos_np = np.array(self.avg_cos_map)
    #     if avg_cos_np.ndim == 3 and avg_cos_np.shape[2] >= 1:
    #         avg_cos_np = avg_cos_np[..., 0]
    #     avg_cos_flat = avg_cos_np.reshape(-1)
    #     # =======================================================================

    #     # 2. 计算初始 Mask
    #     black_mask = np.all(colors == 0, axis=1) # (N,)
    #     low_cos_mask = max_cos_flat < cos_threshold # (N,) 现在这里不会报错了
    #     valid_mask = (points[:, 0] != 0) # (N,)
        
    #     fill_mask = valid_mask & (black_mask | low_cos_mask)
        
    #     # 准备部件信息
    #     texel_component = np.zeros(len(points), dtype=np.int32)
    #     if self.face2label is not None:
    #         # 搬运原本的部件逻辑
    #         if torch.is_tensor(face_idx):
    #             face_idx_np = face_idx.cpu().numpy()
    #         else:
    #             face_idx_np = np.array(face_idx)
    #         if face_idx_np.ndim == 3 and face_idx_np.shape[0] == 1:
    #             face_idx_np = face_idx_np[0]
    #         face_idx_flat = face_idx_np.reshape(-1)
            
    #         max_face_id = int(face_idx_flat.max()) if face_idx_flat.size > 0 else 0
    #         face_label_arr = np.zeros(max_face_id + 1, dtype=np.int32)
    #         for fid, cid in self.face2label.items():
    #             if fid <= max_face_id:
    #                 face_label_arr[fid] = int(cid)
            
    #         valid_face_mask = face_idx_flat >= 0
    #         texel_component[valid_face_mask] = face_label_arr[face_idx_flat[valid_face_mask]]
        
    #     # 3. 准备循环变量
    #     points_t = points
    #     colors_t = colors.copy()
    #     fill_mask_t = fill_mask.copy()
        
    #     # 【核心逻辑】算出总数，除以 5，向上取整
    #     initial_total_unfilled = np.sum(fill_mask_t)
    #     if initial_total_unfilled == 0:
    #         print("Nothing to fill.")
    #         return colors_t.reshape(h, w, 3), np.zeros((h, w), dtype=bool)

    #     print(f"Total holes: {initial_total_unfilled}")
        
    #     fixed_step_size = int(np.ceil(initial_total_unfilled / 5.0))
        
    #     pbar = tqdm(total=initial_total_unfilled, desc="5-Step Inpainting")
        
    #     # ==================== 核心循环 ====================
    #     while np.sum(fill_mask_t) > 0:
            
    #         # A. 找学生和老师
    #         unfilled_idx = np.where(fill_mask_t)[0]
    #         known_idx = np.where((~fill_mask_t) & valid_mask)[0]
            
    #         if len(known_idx) == 0:
    #             print("No known points left!")
    #             break
                
    #         # B. 建学校
    #         nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, len(known_idx)), algorithm='auto')
    #         nbrs.fit(points_t[known_idx])
            
    #         # C. 找邻居
    #         distances, indices = nbrs.kneighbors(points_t[unfilled_idx])
    #         distances = np.maximum(distances, 1e-8)
            
    #         # D. 排序
    #         neighbor_avg_cos = avg_cos_flat[known_idx][indices].mean(axis=1)
    #         sort_order = np.argsort(-neighbor_avg_cos) 
            
    #         unfilled_idx_sorted = unfilled_idx[sort_order]
    #         distances_sorted = distances[sort_order]
    #         indices_sorted = indices[sort_order]
            
    #         # E. 切片
    #         n_process = min(fixed_step_size, len(unfilled_idx_sorted))
            
    #         batch_idx = unfilled_idx_sorted[:n_process]
    #         batch_distances = distances_sorted[:n_process]
    #         batch_indices = indices_sorted[:n_process]
            
    #         # F. 计算权重
    #         distance_weights = 1.0 / (batch_distances ** distance_exp + 1e-12)
            
    #         batch_comp_weights = np.ones_like(batch_distances, dtype=np.float32) * component_weight_diff
    #         if self.face2label is not None:
    #             unfilled_comp = texel_component[batch_idx][:, None]
    #             known_comp = texel_component[known_idx][batch_indices]
    #             same_mask = (known_comp == unfilled_comp)
    #             batch_comp_weights[same_mask] = component_weight_same
            
    #         total_weights = distance_weights * batch_comp_weights
    #         row_sum = (total_weights.sum(axis=1, keepdims=True) + 1e-12)
    #         norm_weights = total_weights / row_sum
            
    #         # G. 填色
    #         neighbor_colors = colors_t[known_idx][batch_indices]
    #         weighted_color = (neighbor_colors * norm_weights[..., None]).sum(axis=1)
            
    #         colors_t[batch_idx] = weighted_color
    #         fill_mask_t[batch_idx] = False
            
    #         # H. 更新进度
    #         pbar.update(n_process)
            
    #     pbar.close()
    #     print("\nInpainting Done.")
        
    #     # 返回结果
    #     colors_out = colors_t.reshape(h, w, 3)
    #     red_mask = np.zeros((h, w), dtype=bool)
    #     red_mask[fill_mask.reshape(h, w)] = True
        
    #     return colors_out, red_mask
    @torch.no_grad()
    def update_texture(self, position_map, face_idx,
                       cos_threshold=0.10,
                       k_neighbors=60,
                       max_iter=5,
                       distance_exp=1.0,
                       component_weight_same=1.0,
                       component_weight_diff=0.05,
                       batch_size=500): 
        """
        分层迭代版：基于平均信誉度衰减 (Layered Average Decay)
        """
        print("\n[Start Layered-Propagation Inpainting]...")
        
        # 1. 准备数据
        texture = self.mesh.textures.maps_padded()[0]
        tex_np = texture.cpu().numpy()
        h, w = tex_np.shape[:2]

        points = position_map.reshape(-1, 3).cpu().numpy()
        colors = tex_np.reshape(-1, 3).copy() 

        # Cos 处理
        if torch.is_tensor(self.max_cos_map):
            max_cos_np = self.max_cos_map.detach().cpu().numpy()
        else:
            max_cos_np = np.array(self.max_cos_map)
        if max_cos_np.ndim == 3: max_cos_np = max_cos_np[..., 0]
        max_cos_flat = max_cos_np.reshape(-1)

        if torch.is_tensor(self.avg_cos_map):
            avg_cos_np = self.avg_cos_map.detach().cpu().numpy()
        else:
            avg_cos_np = np.array(self.avg_cos_map)
        if avg_cos_np.ndim == 3: avg_cos_np = avg_cos_np[..., 0]
        avg_cos_flat = avg_cos_np.reshape(-1)

        # 2. Mask
        black_mask = np.all(colors == 0, axis=1)
        low_cos_mask = max_cos_flat < cos_threshold
        valid_mask = (points[:, 0] != 0)
        fill_mask = valid_mask & (black_mask | low_cos_mask)
        
        # ==================== 【关键：分层积分卡】 ====================
        # score_map 用于记录当前的“代际分数”
        score_map = np.zeros(len(points), dtype=np.float32)
        
        # 第 0 层 (老祖宗) 的分数 = 它的几何属性 (0~1)
        # 这就是你说的“第一层肯定是按 Cos”
        known_mask_init = (~fill_mask) & valid_mask
        score_map[known_mask_init] = avg_cos_flat[known_mask_init] # 或者 max_cos
        
        # ==========================================================

        # 部件准备 (省略，保持原样)
        texel_component = np.zeros(len(points), dtype=np.int32)
        if self.face2label is not None:
             # (搬运省略的代码)
             pass

        # 循环变量
        points_t = points
        colors_t = colors.copy()
        fill_mask_t = fill_mask.copy()
        
        initial_total = np.sum(fill_mask_t)
        if initial_total == 0: return colors_t.reshape(h, w, 3), np.zeros((h, w), dtype=bool)

        fixed_step_size = int(np.ceil(initial_total / 5.0))
        pbar = tqdm(total=initial_total, desc="Layered Propagation")

        # 【衰减系数】：每一层打几折？建议 0.9 或 0.95
        DECAY_FACTOR = 0.95

        while np.sum(fill_mask_t) > 0:
            unfilled_idx = np.where(fill_mask_t)[0]
            known_idx = np.where((~fill_mask_t) & valid_mask)[0]
            
            if len(known_idx) == 0: break
            
            # KNN
            nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, len(known_idx)), algorithm='auto')
            nbrs.fit(points_t[known_idx])
            distances, indices = nbrs.kneighbors(points_t[unfilled_idx])
            distances = np.maximum(distances, 1e-8)
            
            # ==================== 【怎么算后面的平均值？】 ====================
            # 1. 拿到所有老师的分数
            neighbor_scores = score_map[known_idx][indices] # (N, k)
            
            # 2. 计算平均分 (这就是你想要的：综合考虑所有邻居的层级)
            #    也可以加上距离权重，离得近的分数占比大，公式更平滑:
            #    avg_score = sum(score / dist) / sum(1 / dist)
            #    简单起见，直接算 mean 也很稳：
            current_layer_score = neighbor_scores.mean(axis=1)
            
            # 3. 排序：分数高的（也就是层级靠前的）先填
            sort_order = np.argsort(-current_layer_score)
            # ==============================================================
            
            unfilled_idx_sorted = unfilled_idx[sort_order]
            distances_sorted = distances[sort_order]
            indices_sorted = indices[sort_order]
            
            # 拿到排好序的分数，作为这一批点的“底分”
            sorted_scores = current_layer_score[sort_order]
            
            # 切片
            n_process = min(fixed_step_size, len(unfilled_idx_sorted))
            batch_idx = unfilled_idx_sorted[:n_process]
            batch_distances = distances_sorted[:n_process]
            batch_indices = indices_sorted[:n_process]
            batch_base_scores = sorted_scores[:n_process]
            
            # 计算权重 (这里回归到看距离，保证平滑，不看 Score)
            distance_weights = 1.0 / (batch_distances ** distance_exp + 1e-12)
            batch_comp_weights = np.ones_like(batch_distances, dtype=np.float32) * component_weight_diff
            if self.face2label is not None:
                unfilled_comp = texel_component[batch_idx][:, None]
                known_comp = texel_component[known_idx][batch_indices]
                same_mask = (known_comp == unfilled_comp)
                batch_comp_weights[same_mask] = component_weight_same
            
            total_weights = distance_weights * batch_comp_weights
            norm_weights = total_weights / (total_weights.sum(axis=1, keepdims=True) + 1e-12)
            
            # 填色
            neighbor_colors = colors_t[known_idx][batch_indices]
            weighted_color = (neighbor_colors * norm_weights[..., None]).sum(axis=1)
            colors_t[batch_idx] = weighted_color
            fill_mask_t[batch_idx] = False
            
            # ==================== 【关键：传给下一代】 ====================
            # 这一批点填好了，它们是第 N 层。
            # 它们的得分 = 它们刚才算出来的平均分 * 衰减系数
            # 这样，它们的“孩子”（下一轮的邻居）算出来的分就会更低，自然排到后面
            score_map[batch_idx] = batch_base_scores * DECAY_FACTOR
            # ==========================================================
            
            pbar.update(n_process)
            
        pbar.close()
        # (省略 Padding)
        print("Done.")
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