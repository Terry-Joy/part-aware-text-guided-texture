import torch

def verify_cpna_data(path="./cpna_analysis_data.pt"):
    try:
        data = torch.load(path, map_location='cpu')
        print(f"🚀 成功加载数据包: {path}\n")
    except Exception as e:
        print(f"❌ 加载失败，文件可能不存在或损坏: {e}")
        return

    # --- 1. 顶层键值检查 ---
    expected_keys = ['part_mask', 'z_T_raw', 'z_T_aligned', 'trajectory', 'init_stats']
    found_keys = list(data.keys())
    print(f"【1. 目录索引】")
    for key in expected_keys:
        status = "✅ 存在" if key in data else "⚠️ 缺失"
        print(f" - {key:12}: {status}")

    # --- 2. Part Mask 维度校验 ---
    print(f"\n【2. 寻宝地图 (Part Mask)】")
    if 'part_mask' in data:
        mask = data['part_mask']
        # 预期形状为 (V, H, W) [cite: 1, 6]
        print(f" - Shape: {mask.shape} (预期: (V, 96, 96))")
        print(f" - 包含部件数: {len(torch.unique(mask))}")
    else:
        print(" - ⚠️ 警告: 没有 Mask 数据，后续无法切分部件特征")

    # --- 3. 初始化噪声对齐检查 ---
    print(f"\n【3. 初始化 $z_T$ 状态】")
    for k in ['z_T_raw', 'z_T_aligned']:
        if k in data and data[k] is not None:
            # 预期形状为 (V, 4, H, W) 
            print(f" - {k:12}: {data[k].shape} (预期: (V, 4, 96, 96))")
        else:
            print(f" - {k:12}: ⚠️ 缺失")

    # --- 4. 轨迹数据深度扫描 ---
    print(f"\n【4. 轨迹录制轨迹 (Trajectory)】")
    traj = data.get('trajectory', {})
    if not traj:
        print(" - ⚠️ 轨迹列表为空！")
    else:
        timesteps = sorted(list(traj.keys()), reverse=True)
        print(f" - 总计捕获步数: {len(timesteps)}")
        
        # 检查每一时刻的 Shape 是否统一
        all_zt_ok = True
        all_z0_ok = True
        for t in timesteps:
            step = traj[t]
            # 每一时刻应包含 z_t 和 z_0_pred [cite: 94, 98]
            if 'z_t' in step and step['z_t'].shape[1] != 4: all_zt_ok = False
            if 'z_0_pred' in step and step['z_0_pred'].shape[1] != 4: all_z0_ok = False
        
        print(f" - $z_t$ 通道数自洽性: {'✅ 正常(4通道)' if all_zt_ok else '❌ 通道数不对'}")
        print(f" - $z_{{0|t}}$ 通道数自洽性: {'✅ 正常(4通道)' if all_z0_ok else '❌ 通道数不对'}")
        print(f" - 时间轴跨度: 从 t={timesteps[0]} 到 t={timesteps[-1]}")

    # --- 5. 初始统计量校验 ---
    print(f"\n【5. CPNA 均值方差快照 (Init Stats)】")
    stats = data.get('init_stats')
    if stats:
        pids = list(stats.keys())
        print(f" - 捕获部件数量: {len(pids)}")
        sample_pid = pids[0]
        # 预期 mu_global 形状为 (4, 1) [cite: 3, 6]
        mu_g_shape = stats[sample_pid]['mu_global'].shape
        print(f" - 样例部件(PID={sample_pid}) 全局均值形状: {mu_g_shape} (预期: (4, 1))")
    else:
        print(" - ⚠️ 缺失初始化统计量，无法分析具体如何对齐")

if __name__ == "__main__":
    verify_cpna_data("./cpna_analysis_data.pt")