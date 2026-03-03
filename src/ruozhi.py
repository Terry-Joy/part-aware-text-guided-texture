# 1. 把“地图”拿出来
part_mask = cpna_data["part_mask"]  # shape: (6, 64, 64)

# 2. 把 t=200 时的预测终点拿出来
z_0_pred = cpna_data["trajectory"][200]["z_0_pred"] # shape: (6, 4, 64, 64)

# 3. 终极一刀：只切出车门！
door_mask = (part_mask == 2)
door_pixels = z_0_pred.permute(1, 0, 2, 3)[:, door_mask] # 瞬间拿到 6 个视角里所有车门像素！

# 4. 然后你就可以算方差、画 t-SNE 了...