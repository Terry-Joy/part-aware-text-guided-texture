import cv2
import numpy as np

# ================= 配置区域（修改这里） =================
INPUT_PATH = '2fd7954d31dc47d7a48f9e0cb8faeeba_concat.png'           # 输入图片路径
OUTPUT_PATH = 'vase_2.png'         # 输出图片路径
TARGET_COLOR = (0, 0, 0)         # 目标颜色 (B, G, R) 格式
THRESHOLD = 0                     # 颜色容差 (10-50)
# =======================================================

def main():
    # 读取图片
    img = cv2.imread(INPUT_PATH)
    if img is None:
        print(f"错误：无法读取图片 {INPUT_PATH}")
        return
    
    # 创建颜色范围
    target = np.array(TARGET_COLOR, dtype=np.uint8)
    lower = np.maximum(target - THRESHOLD, 0)
    upper = np.minimum(target + THRESHOLD, 255)
    
    # 创建掩膜
    mask = cv2.inRange(img, lower, upper)
    
    # 生成结果：匹配区域黑色，其他白色
    output = np.ones_like(img) * 255
    output[mask > 0] = 0
    
    # 保存
    cv2.imwrite(OUTPUT_PATH, output)
    print(f"完成！匹配像素数：{np.sum(mask > 0)}")
    print(f"结果已保存：{OUTPUT_PATH}")

if __name__ == "__main__":
    main()