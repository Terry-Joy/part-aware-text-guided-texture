from PIL import Image, ImageDraw

# 加载两张图片
image1 = Image.open('b77fcf18510f410c915456193111a8db_big_mask_768.png')  # 左边的图片
# image2 = Image.open('b77fcf18510f410c915456193111a8db_big_mask_768.png')
image2 = Image.open('b77fcf18510f410c915456193111a8db_labelcolor_768x768_concat copy 6.png')  # 右边的图片

# 确保两张图片大小一致
if image1.size != image2.size:
    raise ValueError("两张图片尺寸不一致，请确保它们具有相同的宽度和高度。")

# 获取图片的宽度和高度
width, height = image1.size

# 计算每个子图的宽度
num_subimages = 6
subimage_width = width // num_subimages

# 转换为相同模式（RGBA）
# image1 = image1.convert('RGBA')
image1 = image1.convert('RGB')
image2 = image2.convert('RGB')  # 第二张图片只关心 RGB 颜色，不需要透明度

# 初始化统计变量
total_mismatch_count = 0
total_pixels = 0

# 遍历每一行（6 个子图）
for i in range(num_subimages):
    # 提取第一张图片的第 i 个子图
    subimage1 = image1.crop((i * subimage_width, 0, (i + 1) * subimage_width, height))
    
    # 提取第二张图片的第 i 个子图
    subimage2 = image2.crop((i * subimage_width, 0, (i + 1) * subimage_width, height))
    
    # 创建一个全黑的画布，用于标记不匹配的像素
    result_subimage = Image.new('RGB', (subimage_width, height), color=(0, 0, 0))  # 全黑背景
    draw = ImageDraw.Draw(result_subimage)
    
    # 初始化当前子图的不匹配像素计数器
    mismatch_count = 0
    
    # 获取子图的宽度和高度
    subimage_width, subimage_height = subimage1.size
    total_pixels += subimage_width * subimage_height
    
    # 遍历每个像素
    for x in range(subimage_width):
        for y in range(subimage_height):
            # 获取两张子图的像素值
            pixel1 = subimage1.getpixel((x, y))
            pixel2 = subimage2.getpixel((x, y))
            
            # 判断第一张子图是否为白色（非黑色前景）
            # if pixel1[0] == 255 and pixel1[1] == 255 and pixel1[2] == 255 and pixel1[3] > 0:
            if not (pixel1[0] == 0 and pixel1[1] == 0 and pixel1[2] == 0):
                # 判断第二张子图是否为黑色背景
                if pixel2[0] == 0 and pixel2[1] == 0 and pixel2[2] == 0:  # 如果是黑色，则标记为不匹配
                    draw.point((x, y), fill=(255, 0, 0))  # 用红色标记不匹配的像素
                    mismatch_count += 1  # 统计不匹配的像素数量
            
            if pixel1[0] == 0 and pixel1[1] == 0 and pixel1[2] == 0:
                if not (pixel2[0] == 0 and pixel2[1] == 0 and pixel2[2] == 0):
                    draw.point((x, y), fill=(255, 0, 0))  # 用红色标记不匹配的像素
                    mismatch_count += 1  # 统计不匹配的像素数量
    
    # 打印当前子图的不匹配像素个数和百分比
    subimage_total_pixels = subimage_width * subimage_height
    subimage_mismatch_percentage = (mismatch_count / subimage_total_pixels) * 100
    print(f"子图 {i + 1} 的不匹配像素个数: {mismatch_count}")
    print(f"子图 {i + 1} 的不匹配像素百分比: {subimage_mismatch_percentage:.2f}%")
    
    # 更新总不匹配像素计数
    total_mismatch_count += mismatch_count
    
    # 保存当前子图的结果图像
    result_subimage.save(f'result_mismatch_{i + 1}.png')

# 计算总的不匹配像素百分比
total_mismatch_percentage = (total_mismatch_count / total_pixels) * 100

# 打印总的不匹配像素个数和百分比
print(f"总的不匹配像素个数: {total_mismatch_count}")
print(f"总的不匹配像素百分比: {total_mismatch_percentage:.2f}%")


# # 示例调用
# if __name__ == "__main__":
#     ratio = compare_masks("b77fcf18510f410c915456193111a8db_labelcolor_768x768_concat.png", "mask.png")
#     print("最终匹配度:", ratio, "%")
