import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载数据
data_path = "path/to/your/intermediate_dir/cpna_entropy_analysis.pt" # 🌟 改成你的实际路径
data = torch.load(data_path)

print("--- 基础信息校验 ---")
print(f"包含键值: {list(data.keys())}")
print(f"Part Mask 形状: {data['part_mask'].shape}") # 预期 (V, 64, 64)
print(f"轨迹步数: {len(data['trajectory'])}")

# 2. 验证初始化对齐 (Entropy Reduction)
print("\n--- CPNA 初始化对齐效果验证 ---")
z_T_raw = data['z_T_raw']      # 对齐前
z_T_aligned = data['z_T_aligned']  # 对齐后

def calc_cross_view_var(latent, mask):
    # 只计算非背景区域的跨视角方差
    # latent: (V, 4, H, W)
    # mask: (V, H, W)
    fore_mask = (mask > 0).unsqueeze(1) # (V, 1, H, W)
    # 计算 6 个视角在每个像素位置的方差，然后对前景求平均
    var_map = torch.var(latent, dim=0) # (4, H, W)
    avg_var = var_map[fore_mask[0].repeat(4, 1, 1)].mean()
    return avg_var.item()

var_raw = calc_cross_view_var(z_T_raw, data['part_mask'])
var_aligned = calc_cross_view_var(z_T_aligned, data['part_mask'])

print(f"对齐前 (Raw) 跨视角方差: {var_raw:.6f}")
print(f"对齐后 (Aligned) 跨视角方差: {var_aligned:.6f}")
print(f"🔥 熵减幅度: {((var_raw - var_aligned) / var_raw * 100):.2