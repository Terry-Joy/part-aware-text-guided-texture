import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision.transforms import Resize, ToTensor
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def load_source_images(source_folder):
    """加载源多视角图像文件夹"""
    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"源图像文件夹不存在: {source_folder}")
    
    # 获取所有源图像路径并按数字排序
    source_paths = sorted(glob.glob(os.path.join(source_folder, "*.png"))) + \
                  sorted(glob.glob(os.path.join(source_folder, "*.jpg"))) + \
                  sorted(glob.glob(os.path.join(source_folder, "*.jpeg")))
    
    # 按文件名中的数字排序
    source_paths = sorted(source_paths, key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    
    if not source_paths:
        raise ValueError(f"在文件夹 {source_folder} 中未找到图像文件")
    
    source_images = []
    transform = ToTensor()
    
    for img_path in source_paths:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img)  # 转换为张量 [0,1]
        source_images.append(img_tensor)
    
    return torch.stack(source_images), source_paths

def process_generated_image(generated_path, num_views):
    """处理包含多个视角的水平拼接生成图像"""
    if not os.path.exists(generated_path):
        raise FileNotFoundError(f"生成图像文件不存在: {generated_path}")
    
    # 加载水平拼接图像
    full_img = Image.open(generated_path).convert("RGB")
    width, height = full_img.size
    
    # 计算每个视角的宽度
    view_width = (width - 1024) // num_views
    if view_width <= 0:
        raise ValueError(f"图像宽度 {width} 不足以容纳 {num_views} 个视角")
    
    # 分割每个视角
    views = []
    view_images = []  # 保存每个视角的PIL图像
    
    for i in range(num_views):
        # 计算裁剪区域 (左、上、右、下)
        left = i * view_width
        top = 0
        right = (i + 1) * view_width
        bottom = height
        
        # 裁剪
        cropped = full_img.crop((left, top, right, bottom))
        view_images.append(cropped)
        views.append(ToTensor()(cropped))
    
    return torch.stack(views), view_images

def save_comparison_images(generated_images, source_paths, output_dir):
    """保存每个视角的对比图像（单独保存）"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 确保图像数量匹配
    num_views = min(len(generated_images), len(source_paths))
    
    # 为每个视角保存单独的生成图像和源图像
    for i in range(num_views):
        # 保存生成图像
        gen_path = os.path.join(output_dir, f"generated_view_{i:02d}.png")
        generated_images[i].save(gen_path)
        print(f"已保存生成视角 {i}: {gen_path}")
        
        # 保存源图像
        src_img = Image.open(source_paths[i])
        src_path = os.path.join(output_dir, f"source_view_{i:02d}.png")
        src_img.save(src_path)
        print(f"已保存源视角 {i}: {src_path}")
    
    print(f"已保存 {num_views} 个视角的对比图像到 {output_dir}")

def calculate_metrics(generated_path, source_folder, num_views, output_dir="comparison_results"):
    """
    计算生成图像与源图像之间的指标并保存对比图像
    
    参数:
    generated_path: str, 包含多视角的水平拼接图像路径
    source_folder: str, 源多视角图像文件夹路径
    num_views: int, 视角数量
    output_dir: str, 对比图像输出目录
    
    返回:
    metrics: dict, 包含FID、KID和CLIP分数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    try:
        # 加载源图像 - 每个视角一个独立图像
        source_images, source_paths = load_source_images(source_folder)
        source_images = source_images.to(device)
        print(f"成功加载源图像: {len(source_images)}张, 形状: {source_images.shape}")
        
        # 加载生成图像 - 单张水平拼接图像
        generated_images, generated_pil_images = process_generated_image(generated_path, num_views)
        generated_images = generated_images.to(device)
        print(f"成功加载生成图像: {len(generated_images)}个视角, 形状: {generated_images.shape}")
        
        # 保存对比图像（单独保存）
        save_comparison_images(generated_pil_images, source_paths, output_dir)
        
        # 检查数量是否匹配
        if len(generated_images) != len(source_images):
            print(f"警告: 生成视角数({len(generated_images)})与源视角数({len(source_images)})不匹配")
        
        # 确保图像在[0,1]范围内
        generated_images = generated_images.clamp(0, 1)
        source_images = source_images.clamp(0, 1)
        
        # 打印图像统计信息用于调试
        print(f"生成图像统计: min={generated_images.min().item():.4f}, max={generated_images.max().item():.4f}, mean={generated_images.mean().item():.4f}")
        print(f"源图像统计: min={source_images.min().item():.4f}, max={source_images.max().item():.4f}, mean={source_images.mean().item():.4f}")
        
        # 为FID/KID调整尺寸
        fid_size = (299, 299)
        fid_resize = Resize(fid_size)
        gen_fid = fid_resize(generated_images)
        src_fid = fid_resize(source_images)
        
        # 转换为uint8格式 [0, 255]
        gen_uint8 = (gen_fid * 255).byte()
        src_uint8 = (src_fid * 255).byte()
        
        # 计算FID
        print("计算FID...")
        fid = FrechetInceptionDistance(feature=2048).to(device)
        fid.update(src_uint8, real=True)
        fid.update(gen_uint8, real=False)
        fid_score = fid.compute().item()
        print(f"FID计算完成: {fid_score:.4f}")
        
        # 计算KID
        print("计算KID...")
        kid = KernelInceptionDistance(subset_size=min(50, len(generated_images))).to(device)
        kid.update(src_uint8, real=True)
        kid.update(gen_uint8, real=False)
        kid_mean, kid_std = kid.compute()
        kid_score = kid_mean.item() * 1000  # 按论文要求放大1000倍
        print(f"KID计算完成: {kid_score:.4f} (x10^3)")
        
        # 计算CLIP分数
        print("加载CLIP模型...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        print("准备CLIP输入...")
        # 准备CLIP输入 - 使用原始尺寸
        gen_inputs = clip_processor(images=generated_images, return_tensors="pt", padding=True).to(device)
        src_inputs = clip_processor(images=source_images, return_tensors="pt", padding=True).to(device)
        
        print("提取图像特征...")
        # 提取图像特征
        with torch.no_grad():
            gen_features = clip_model.get_image_features(**gen_inputs)
            src_features = clip_model.get_image_features(**src_inputs)
        
        # 计算余弦相似度
        gen_features = F.normalize(gen_features, dim=-1)
        src_features = F.normalize(src_features, dim=-1)
        clip_scores = torch.sum(gen_features * src_features, dim=-1)
        clip_score = torch.mean(clip_scores).item()
        print(f"CLIP分数计算完成: {clip_score:.4f}")
        
        # 保存每个视角的CLIP分数
        view_clip_scores = []
        for i in range(len(generated_images)):
            # 提取单个视角的特征
            with torch.no_grad():
                gen_feat = clip_model.get_image_features(
                    **clip_processor(images=generated_images[i:i+1], return_tensors="pt", padding=True).to(device)
                )
                src_feat = clip_model.get_image_features(
                    **clip_processor(images=source_images[i:i+1], return_tensors="pt", padding=True).to(device)
                )
            
            # 计算单个视角的相似度
            gen_feat = F.normalize(gen_feat, dim=-1)
            src_feat = F.normalize(src_feat, dim=-1)
            view_score = torch.sum(gen_feat * src_feat, dim=-1).item()
            view_clip_scores.append(view_score)
            print(f"视角 {i} CLIP分数: {view_score:.4f}")
        
        return {
            "FID": fid_score,
            "KID": kid_score,
            "CLIP_score": clip_score,
            "view_CLIP_scores": view_clip_scores
        }
    
    except Exception as e:
        import traceback
        print(f"计算指标时出错: {e}")
        print(traceback.format_exc())
        return None

# 使用示例
if __name__ == "__main__":
    # 配置参数
    num_views = 3  # 视角数量
    generated_path = "source_images/4_view_segment_textured_views_rgb.jpg"  # 水平拼接生成图像
    source_folder = "source_images/3_source"  # 源图像文件夹
    output_dir = "comparison_results"  # 对比图像输出目录
    
    print("开始计算指标...")
    metrics = calculate_metrics(generated_path, source_folder, num_views, output_dir)
    
    if metrics is not None:
        print("\n评估结果:")
        print(f"FID: {metrics['FID']:.4f} (值越低越好)")
        print(f"KID: {metrics['KID']:.4f} (x10^3, 值越低越好)")
        print(f"整体CLIP分数: {metrics['CLIP_score']:.4f} (值越高越好)")
        
        # 打印每个视角的CLIP分数
        print("\n各视角CLIP分数:")
        for i, score in enumerate(metrics['view_CLIP_scores']):
            print(f"视角 {i}: {score:.4f}")
    else:
        print("指标计算失败")