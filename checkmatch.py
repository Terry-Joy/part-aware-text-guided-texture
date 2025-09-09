import os
import numpy as np
from PIL import Image, ImageDraw

def check_non_transparent_match(combined_mask_path, render_dir, H, W, output_dir, alpha_threshold=10, dot_size=1):
    """
    检查渲染图像的非透明区域是否与组合mask分割后的非透明区域位置一致，
    并保存不匹配位置的点位图像（使用PIL实现）
    
    参数:
        combined_mask_path: 组合mask图像路径 (4×H×W)
        render_dir: 包含渲染图像的目录 (render_0000.png 到 render_0003.png)
        H, W: 单个视图的高度和宽度
        output_dir: 输出目录
        alpha_threshold: Alpha通道阈值（0-255）
        dot_size: 标记点的大小（像素）
    
    返回:
        各视图的匹配结果
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载组合mask图像
    combined_mask_img = Image.open(combined_mask_path).convert('RGBA')
    combined_mask = np.array(combined_mask_img)
    
    # 调整尺寸确保一致
    if combined_mask.shape[0] != H or combined_mask.shape[1] != 4 * W:
        combined_mask = combined_mask[:H, :4*W]
    
    # 分割组合mask为4个独立视图
    mask_views = np.split(combined_mask, 4, axis=1)
    
    print("开始检测4个视图的非透明像素匹配...")
    print(f"图像尺寸: {H}x{W}，Alpha阈值: {alpha_threshold}")
    print(f"组合mask: {combined_mask_path}")
    print(f"渲染图像目录: {render_dir}")
    print(f"输出目录: {output_dir}")
    
    results = []
    
    for i in range(4):
        # 获取当前视图的mask
        mask_view = mask_views[i]
        
        # 加载渲染图像
        render_path = os.path.join(render_dir, f"render_{i:04d}.png")
        render_img = Image.open(render_path).convert('RGBA')
        render_arr = np.array(render_img)
        
        # 调整尺寸确保一致
        if render_arr.shape[0] != H or render_arr.shape[1] != W:
            render_arr = render_arr[:H, :W]
            mask_view = mask_view[:H, :W]
        
        # 提取非透明掩码
        # Mask视图的非透明掩码
        mask_alpha = mask_view[..., 3] if mask_view.ndim == 3 else mask_view
        mask_non_transparent = mask_alpha > alpha_threshold
        
        # 渲染图像的非透明掩码
        render_non_transparent = render_arr[..., 3] > alpha_threshold
        
        # 计算匹配情况
        render_non_trans_count = np.sum(render_non_transparent)
        match_count = np.sum(render_non_transparent & mask_non_transparent)
        
        if render_non_trans_count > 0:
            match_percent = (match_count / render_non_trans_count) * 100
        else:
            match_percent = 100.0
        
        # 找出不匹配的位置
        mismatch_mask = render_non_transparent & ~mask_non_transparent
        mismatch_positions = np.argwhere(mismatch_mask)
        
        # 使用PIL创建不匹配位置图像
        # 1. 创建透明背景图像
        mismatch_img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(mismatch_img)
        
        # 2. 绘制不匹配点
        red = (255, 0, 0, 255)  # 红色，完全不透明
        for y, x in mismatch_positions:
            # 绘制小方块代替点
            draw.rectangle(
                [x - dot_size//2, y - dot_size//2, 
                 x + dot_size//2, y + dot_size//2],
                fill=red
            )
        
        # 3. 创建合成图像：渲染图像 + 不匹配点
        composite_img = Image.alpha_composite(render_img, mismatch_img)
        
        # 保存图像
        mismatch_path = os.path.join(output_dir, f"mismatch_view_{i}.png")
        composite_img.save(mismatch_path)
        
        # 保存原始渲染图像用于参考
        render_img.save(os.path.join(output_dir, f"render_view_{i}.png"))
        
        # 保存mask视图
        Image.fromarray(mask_view).save(os.path.join(output_dir, f"mask_view_{i}.png"))
        
        results.append({
            "view_id": i,
            "render_non_trans_count": int(render_non_trans_count),
            "match_count": int(match_count),
            "match_percent": float(match_percent),
            "mismatch_count": int(render_non_trans_count - match_count),
            "mismatch_positions": mismatch_positions,
            "mismatch_image": mismatch_path
        })
    
    # 计算整体统计数据
    total_non_trans = sum(r["render_non_trans_count"] for r in results)
    total_matches = sum(r["match_count"] for r in results)
    overall_percent = (total_matches / total_non_trans) * 100 if total_non_trans > 0 else 100.0
    
    # 生成报告
    report_path = os.path.join(output_dir, "match_report.txt")
    with open(report_path, "w") as f:
        f.write("非透明区域匹配检测报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"总非透明像素: {total_non_trans}\n")
        f.write(f"匹配像素数: {total_matches}\n")
        f.write(f"总体匹配率: {overall_percent:.2f}%\n\n")
        
        f.write("各视图详细统计:\n")
        f.write("-" * 50 + "\n")
        for res in results:
            f.write(f"视图 {res['view_id']}:\n")
            f.write(f"  非透明像素: {res['render_non_trans_count']}\n")
            f.write(f"  匹配像素: {res['match_count']} ({res['match_percent']:.2f}%)\n")
            f.write(f"  不匹配像素: {res['mismatch_count']}\n")
            f.write(f"  不匹配位置图像: {os.path.basename(res['mismatch_image'])}\n")
            f.write("-" * 50 + "\n")
    
    print(f"报告已生成: {report_path}")
    
    return {
        "total_non_trans": total_non_trans,
        "total_matches": total_matches,
        "overall_percent": overall_percent,
        "view_results": results
    }

# 使用示例
if __name__ == "__main__":
    # 图像尺寸参数
    H = 1024  # 单个视图高度
    W = 1024  # 单个视图宽度
    

    COMBINED_MASK = "./results/MVD_27Jul2025-142016/intermediate/mask.png"  # 4×H×W的图像
    RENDER_DIR = "./results/MVD_27Jul2025-142016/intermediate"  # 包含render_0000.png到render_0003.png
    OUTPUT_DIR = "./results/MVD_27Jul2025-142016/intermediate/output"
    # 路径设置
    
    # 执行检测
    result = check_non_transparent_match(
        combined_mask_path=COMBINED_MASK,
        render_dir=RENDER_DIR,
        H=H,
        W=W,
        output_dir=OUTPUT_DIR,
        alpha_threshold=10,  # Alpha通道阈值
        dot_size=1           # 点的大小（像素）
    )
    
    # 打印结果摘要
    print("\n检测完成:")
    print(f"总非透明像素: {result['total_non_trans']}")
    print(f"匹配像素数: {result['total_matches']}")
    print(f"总体匹配率: {result['overall_percent']:.2f}%")
    
    print("\n各视图匹配率:")
    for res in result["view_results"]:
        print(f"视图 {res['view_id']}: {res['match_percent']:.2f}% (不匹配: {res['mismatch_count']}像素)")
        print(f"  不匹配位置图像: {res['mismatch_image']}")
    # COMBINED_MASK = "./config/MVD_26Jul2025-224257/intermediate/mask.png"  # 4×H×W的图像