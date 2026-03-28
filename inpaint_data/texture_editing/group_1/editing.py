import cv2
import numpy as np

def process_image(input_path, output_path, target_rgb, tolerance=30):
    """
    读取图片，将指定颜色改为黑色，其他改为白色
    :param input_path: 输入图片路径
    :param output_path: 输出图片路径
    :param target_rgb: 目标颜色，格式为 (R, G, B)
    :param tolerance: 颜色容差（0-255），越大匹配范围越广
    """
    # 1. 读取图片 (OpenCV 默认是 BGR 格式)
    img = cv2.imread(input_path)
    if img is None:
        print("错误：找不到图片，请检查路径！")
        return

    # 2. 将目标 RGB 转换为 OpenCV 的 BGR
    target_bgr = np.array([target_rgb[2], target_rgb[1], target_rgb[0]])

    # 3. 设置颜色的上下限（根据容差）
    lower_bound = np.clip(target_bgr - tolerance, 0, 255)
    upper_bound = np.clip(target_bgr + tolerance, 0, 255)

    # 4. 创建掩码：匹配到的颜色部分为白色(255)，其余为黑色(0)
    mask = cv2.inRange(img, lower_bound, upper_bound)

    # 5. 根据你的需求进行反转：
    # 我们希望匹配到的颜色变黑色(0)，其他变白色(255)
    # 所以直接把 mask 取反即可
    result = cv2.bitwise_not(mask)

    # 6. 保存结果
    cv2.imwrite(output_path, result)
    print(f"处理完成！图片已保存至: {output_path}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 比如你想把图片里的“红色”变成黑色，其他变白色
    # 红色 RGB 通常是 (255, 0, 0)
    my_target_color = (31, 119, 180) 
    
    process_image(
        input_path='314d5728e6ac496abdf1fe7268ca9ae0_segment_6views_modify_render_concat.png',  # 换成你的图片文件名
        output_path='314d5728e6ac496abdf1fe7268ca9ae0_segment_6views_modify_render_concat_edit.png', 
        target_rgb=my_target_color,
        tolerance=0  # 如果颜色抓得不准，就把这个值调大点
    )