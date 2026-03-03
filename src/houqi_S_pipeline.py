import torch.nn.functional as F
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
from IPython.display import display
import numpy as np
import json
import copy
import math
import random
import torchvision.transforms as T
import torch
import inspect
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import Compose, Resize, GaussianBlur, InterpolationMode

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDPMScheduler, DDIMScheduler, UniPCMultistepScheduler
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import (
    BaseOutput,
    randn_tensor,
    numpy_to_pil,
    pt_to_pil,
    # make_image_grid,
    is_accelerate_available,
    is_accelerate_version,
    is_compiled_module,
    logging,
    randn_tensor,
    replace_example_docstring
)

from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.models.attention_processor import Attention, AttentionProcessor

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from .renderer.project import UVProjection as UVP


from .syncmvd.attention7 import SamplewiseAttnProcessor2_0, replace_attention_processors
from .syncmvd.prompt import *
from .syncmvd.step import step_tex
from .utils import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import cv2

# from .segment_sptial_inpaint import SpatialAware3DInpainting
# from .commoponentAware3Dinpaint2 import ComponentAware3DInpainting
# from .spital_commoponentAware3dinpaint3 import ComponentSparse3DInpainting

# # get_device
# if torch.cuda.is_available():
#     device = torch.device("cuda:2")
#     torch.cuda.set_device(device)
# else:
#     device = torch.device("cpu")

# Background colors
color_constants = {"black": [-1, -1, -1], "white": [1, 1, 1], "maroon": [0, -1, -1],
                   "red": [1, -1, -1], "olive": [0, 0, -1], "yellow": [1, 1, -1],
                   "green": [-1, 0, -1], "lime": [-1, 1, -1], "teal": [-1, 0, 0],
                   "aqua": [-1, 1, 1], "navy": [-1, -1, 0], "blue": [-1, -1, 1],
                   "purple": [0, -1, 0], "fuchsia": [1, -1, 1]}
color_names = list(color_constants.keys())


# Used to generate depth or normal conditioning images
@torch.no_grad()
def get_conditioning_images(uvp, output_size, render_size=768, blur_filter=5, cond_type="normal", img_path=""):
    verts, normals, depths, cos_maps, texels, fragments = uvp.render_geometry(
        image_size=render_size)
    masks = normals[..., 3][:, None, ...]
    original_mask = masks.clone().detach()
    # print('normal shape is: ', normals.shape)
    # print('original_mask', original_mask.shape)
    masks = Resize((output_size//8,)*2, antialias=True)(masks)
    # print('output_size is: ', output_size)
    normals_transforms = Compose([
        Resize((output_size,)*2,
               interpolation=InterpolationMode.BILINEAR, antialias=True),
        GaussianBlur(blur_filter, blur_filter//3+1)]
    )

    if cond_type == "normal":
        view_normals = uvp.decode_view_normal(
            normals).permute(0, 3, 1, 2) * 2 - 1
        conditional_images = normals_transforms(view_normals)
    # Some problem here, depth controlnet don't work when depth is normalized
    # But it do generate using the unnormalized form as below
    elif cond_type == "depth":
        view_depths = uvp.decode_normalized_depth(depths).permute(0, 3, 1, 2)
        conditional_images = normals_transforms(view_depths)
    elif cond_type == "segment":
        # # 确保已经加载了面标签
        # if uvp.new_face_labels is None:
        #     face2label_path = os.path.join(img_path, "labels.json")
        #     uvp.load_face_labels(face2label_path)

        # pix_to_face = fragments.pix_to_face[..., 0]  # [batch_size, H, W]
        # num_views = pix_to_face.shape[0]
        # device = pix_to_face.device

        # # 创建标签到颜色的映射
        # unique_labels = sorted(set(uvp.new_face_labels.values()))
        # num_unique_labels = len(unique_labels)
        # torch.manual_seed(42)
        # random_colors = torch.rand(num_unique_labels, 3, device=device)
        # random_colors = torch.clamp(random_colors, min=0.3, max=1.0)
        # label_to_color = {label: random_colors[idx] for idx, label in enumerate(unique_labels)}
        # label_to_color[-1] = torch.tensor([0.0, 0.0, 0.0], device=device)  # 默认颜色

        # # 生成每个视角的分割图
        # segment_images = []
        # for view_idx in range(num_views):
        #     # 获取当前视图的像素到面映射
        #     view_pix_to_face = pix_to_face[view_idx]  # [H, W]

        #     # 创建空的RGB图像
        #     h, w = view_pix_to_face.shape
        #     rgb_image = torch.zeros((h, w, 3), device=device)

        #     # 将面ID映射到标签和颜色
        #     for face_id in torch.unique(view_pix_to_face):
        #         if face_id == -1:  # 背景
        #             continue

        #         # 获取面标签
        #         label = uvp.new_face_labels.get(face_id.item(), -1)
        #         color = label_to_color[label]

        #         # 应用颜色到所有属于该面的像素
        #         mask = (view_pix_to_face == face_id)
        #         rgb_image[mask] = color

        #     segment_images.append(rgb_image.permute(2, 0, 1))  # 转换为 [C, H, W]

        # conditional_images = torch.stack(segment_images)
        # conditional_images = normals_transforms(conditional_images)

        segment_images = []
        for i in range(5):  # 假设有4张图像
            if (i == 1):
                continue
            img_name = f"segment_copy_{i:04d}.png"
            img_path_full = os.path.join(img_path, img_name)

            # 打开图像并转换为RGB
            img = Image.open(img_path_full).convert("RGB")
            # 转换为张量并归一化到[0,1]
            img_tensor = T.functional.to_tensor(img)
            segment_images.append(img_tensor)

        # 堆叠张量 [4, 3, H, W]
        conditional_images = torch.stack(segment_images)
        conditional_images = normals_transforms(conditional_images)
    return conditional_images, masks, original_mask

def apply_global_style_injection(latents, part_mask, eps=1e-6):
    """
    计算多视角的全局部件风格(Mean/Std)，并注入到每个视角的局部部件中。
    Args:
        latents: (V, 4, H, W)
        part_mask: (V, H, W)  部件ID掩码
    """
    if part_mask.device != latents.device:
        part_mask = part_mask.to(latents.device)

    unique_ids = torch.unique(part_mask)
    print('unique_ids', len(unique_ids))
    # 我们需要修改 latents，建议 clone 一份或者原地修改
    # 这里为了安全演示，先 clone，你可以根据显存情况决定是否 inplace
    aligned_latents = latents.clone() 

    for pid in unique_ids:
        # ==========================================
        # 1. 第一步：计算全局风格 (Global Style)
        # ==========================================
        # 找出所有视角中，属于这个部件的所有像素
        # mask_all_views.shape -> (V, H, W)
        mask_all_views = (part_mask == pid)
        
        # 如果这个部件在所有视角里都没出现（或者极少），跳过
        if mask_all_views.sum() <= 1:
            continue

        # 提取所有像素：(V, 4, H, W) -> 选出 N 个像素 -> (4, N)
        # 这一步把 6 个视角的头全部拼在了一起！
        global_pixels = latents.permute(1, 0, 2, 3)[:, mask_all_views] 
        
        # 计算全局均值和方差 (4, 1)
        mu_global = global_pixels.mean(dim=1, keepdim=True)
        sigma_global = global_pixels.std(dim=1, keepdim=True) + eps
        
        # ==========================================
        # 2. 第二步：注入到每个视角 (Injection)
        # ==========================================
        for v in range(latents.shape[0]):
            # 获取当前视角、当前部件的掩码
            local_mask = (part_mask[v] == pid)
            
            # 如果当前视角没看到这个部件，跳过
            if local_mask.sum() <= 1:
                continue
            
            # 提取局部像素 (4, N_local)
            local_pixels = latents[v, :, local_mask]
            
            # 计算局部统计量 (用于去风格化)
            mu_local = local_pixels.mean(dim=1, keepdim=True)
            sigma_local = local_pixels.std(dim=1, keepdim=True) + eps
            
            # 核心公式：
            # (局部 - 局部均值) / 局部方差  <-- 变成标准态
            # * 全局方差 + 全局均值        <-- 注入全局风格
            injected_pixels = (local_pixels - mu_local) / sigma_local
            # injected_pixels = (local_pixels - mu_local) / sigma_local * sigma_global + mu_global
            
            # 写回
            aligned_latents[v, :, local_mask] = injected_pixels
            
    return aligned_latents

def apply_global_style_injection_with_viz(latents, part_mask, output_dir="viz_results", eps=1e-6, adain_first=1):
    """
    改进版：带全量可视化与自动化证据收集的 Part-AdaIN。
    修复了 ValueError: I/O operation on closed file 报错。
    """
    device = latents.device
    num_views = latents.shape[0]
    os.makedirs(output_dir, exist_ok=True)

    unique_ids = torch.unique(part_mask)
    aligned_latents = latents.clone()

    for pid in unique_ids:
        pid_val = int(pid.item())
        # 0 代表背景或无效区域，通常跳过，或者你可以根据需求保留
        if pid_val == 0: continue 

        # 创建部件专属文件夹
        part_dir = os.path.join(output_dir, f"part_{pid_val}")
        os.makedirs(part_dir, exist_ok=True)

        # 1. 准备全局数据
        mask_all_views = (part_mask == pid)
        # 如果这个部件在所有视角中总像素太少，跳过可视化
        if mask_all_views.sum() <= num_views: continue

        # 计算全局统计量 (Global Reference)
        global_pixels = latents.permute(1, 0, 2, 3)[:, mask_all_views] # (4, N_total)
        mu_global = global_pixels.mean(dim=1, keepdim=True)
        sigma_global = global_pixels.std(dim=1, keepdim=True) + eps

        # 收集绘图数据
        pixels_before = []
        pixels_after = []

        # 2. 开启文件记录，并执行对齐 (解决 I/O 报错的关键：所有写操作都在 with 块内)
        with open(os.path.join(part_dir, "stats.txt"), "w") as f:
            f.write(f"--- Statistics for Part ID: {pid_val} ---\n")
            f.write(f"Global Target Mean: {mu_global.flatten().tolist()}\n")
            f.write(f"Global Target Std: {sigma_global.flatten().tolist()}\n\n")

            for v in range(num_views):
                # 保存掩码图 (论文配图用：部件黑，背景白)
                m_img = (part_mask[v] != pid).float().cpu().numpy() * 255
                cv2.imwrite(os.path.join(part_dir, f"view_{v}_mask.png"), m_img.astype(np.uint8))

                local_mask = (part_mask[v] == pid)
                if local_mask.sum() <= 1:
                    pixels_before.append(torch.zeros((4, 1)).to(device))
                    pixels_after.append(None)
                    continue
                
                # 提取原始局部像素
                lp_before = latents[v, :, local_mask]
                pixels_before.append(lp_before)
                
                mu_local = lp_before.mean(dim=1, keepdim=True)
                sigma_local = lp_before.std(dim=1, keepdim=True) + eps
                
                # 记录对齐前的数值
                f.write(f"View {v} [BEFORE] -> Mu: {mu_local.flatten().tolist()}, Sigma: {sigma_local.flatten().tolist()}\n")
                
                # 执行 AdaIN 核心对齐公式
                # (x - mu_local) / sigma_local * sigma_global + mu_global
                if adain_first == 1:
                    lp_after = (lp_before - mu_local) / sigma_local * sigma_global + mu_global
                else:
                    lp_after = (lp_before - mu_local) / sigma_local
                
                # 写回对齐后的 latents
                aligned_latents[v, :, local_mask] = lp_after
                pixels_after.append(lp_after)
                
                # 验证对齐后的数值（理论上应该非常接近全局值）
                mu_val_after = lp_after.mean(dim=1).flatten().tolist()
                f.write(f"View {v} [AFTER ] -> Mu: {mu_val_after}\n\n")

        # ==========================================
        # 可视化 A: KDE/Histogram (证明统计分布重合)
        # ==========================================
        plt.figure(figsize=(12, 5))
        # 通道 0 的分布对比
        plt.subplot(1, 2, 1)
        for v in range(num_views):
            if pixels_before[v].shape[1] > 1:
                plt.hist(pixels_before[v][0].cpu().numpy(), bins=50, density=True, alpha=0.3, label=f'View {v}')
        plt.title(f"Part {pid_val}: Latent Dist. BEFORE")
        plt.xlabel("Value")
        plt.legend()

        plt.subplot(1, 2, 2)
        for v in range(num_views):
            if pixels_after[v] is not None:
                plt.hist(pixels_after[v][0].detach().cpu().numpy(), bins=50, density=True, alpha=0.3, label=f'View {v}')
        plt.title(f"Part {pid_val}: Latent Dist. AFTER")
        plt.xlabel("Value")
        plt.legend()
        plt.savefig(os.path.join(part_dir, "dist_comparison.png"))
        plt.close()

        # ==========================================
        # 可视化 B: Energy Heatmap (证明物理能量一致)
        # ==========================================
        def get_part_heatmap(lt, msk_flag):
            # 计算 L2 能量能量
            energy = torch.sqrt(torch.mean(lt**2, dim=0)) 
            # 仅显示该部件区域，非部件区域置为背景
            energy_np = energy.cpu().numpy()
            msk_np = msk_flag.cpu().numpy()
            return np.where(msk_np, energy_np, np.nan) # 非部件区域设为透明/NaN

        fig, axes = plt.subplots(2, num_views, figsize=(num_views*4, 8))
        for v in range(num_views):
            # 对齐前热力图
            im1 = axes[0, v].imshow(get_part_heatmap(latents[v], part_mask[v] == pid), cmap='inferno')
            axes[0, v].set_title(f"View {v} Original Energy")
            axes[0, v].axis('off')
            # 对齐后热力图
            im2 = axes[1, v].imshow(get_part_heatmap(aligned_latents[v], part_mask[v] == pid), cmap='inferno')
            axes[1, v].set_title(f"View {v} Aligned Energy")
            axes[1, v].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(part_dir, "energy_heatmap.png"))
        plt.close()

        # ==========================================
        # 可视化 C: t-SNE (证明语义特征空间去视角化)
        # ==========================================
        try:
            all_pts_before, all_pts_after, labels = [], [], []
            for v in range(num_views):
                if pixels_after[v] is not None and pixels_after[v].shape[1] > 10:
                    pts_b = pixels_before[v].t() # (N, 4)
                    pts_a = pixels_after[v].t()  # (N, 4)
                    # 采样点
                    num_samples = min(pts_b.size(0), 150)
                    idx = torch.randperm(pts_b.size(0))[:num_samples]
                    all_pts_before.append(pts_b[idx].cpu().numpy())
                    all_pts_after.append(pts_a[idx].detach().cpu().numpy())
                    labels.extend([v] * num_samples)
            
            X_b = np.concatenate(all_pts_before, axis=0)
            X_a = np.concatenate(all_pts_after, axis=0)
            
            tsne = TSNE(n_components=2, init='pca', learning_rate='auto')
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            feat_2d_b = tsne.fit_transform(X_b)
            plt.scatter(feat_2d_b[:,0], feat_2d_b[:,1], c=labels, cmap='tab10', s=15, alpha=0.7)
            plt.title("t-SNE: Features Clustered by View (Original)")
            
            plt.subplot(1, 2, 2)
            feat_2d_a = tsne.fit_transform(X_a)
            plt.scatter(feat_2d_a[:,0], feat_2d_a[:,1], c=labels, cmap='tab10', s=15, alpha=0.7)
            plt.title("t-SNE: Semantic Mixing (Aligned)")
            
            plt.savefig(os.path.join(part_dir, "tsne_analysis.png"))
            plt.close()
        except Exception as e:
            print(f"t-SNE for part {pid_val} failed (possibly too few pixels): {e}")

    return aligned_latents

def apply_low_freq_style_injection(latents, part_mask, kernel_size=3, sigma=0.5, eps=1e-6):
    """
    只针对低频分量进行多视角部件风格对齐。
    Args:
        latents: (V, 4, H, W)
        part_mask: (V, H, W)
        kernel_size: 高斯核大小，用于提取低频。
        sigma: 高斯模糊的强度，越大则融合的频率越低。
    """
    if part_mask.device != latents.device:
        part_mask = part_mask.to(latents.device)

    # ==========================================
    # 0. 频率分离：提取低频成分
    # ==========================================
    # 使用简单的高斯模糊作为低通滤波器
    # 为了保持边缘，你可以尝试 kernel_size=3, sigma=0.5~1.0
    padding = kernel_size // 2
    # 构建高斯核并应用模糊
    low_freq_latents = gaussian_blur(latents, kernel_size, sigma) 
    high_freq_latents = latents - low_freq_latents  # 保护起来的高频纹理

    unique_ids = torch.unique(part_mask)
    # 对齐后的低频部分
    aligned_low_freq = low_freq_latents.clone() 

    for pid in unique_ids:
        mask_all_views = (part_mask == pid)
        if mask_all_views.sum() <= 1:
            continue

        # ==========================================
        # 1. 第一步：在低频层计算全局风格
        # ==========================================
        # 提取低频像素的全局统计量
        global_pixels_low = low_freq_latents.permute(1, 0, 2, 3)[:, mask_all_views]
        mu_global = global_pixels_low.mean(dim=1, keepdim=True)
        sigma_global = global_pixels_low.std(dim=1, keepdim=True) + eps
        
        # ==========================================
        # 2. 第二步：低频对齐 (只改意图，不改细节)
        # ==========================================
        for v in range(latents.shape[0]):
            local_mask = (part_mask[v] == pid)
            if local_mask.sum() <= 1:
                continue
            
            local_pixels_low = low_freq_latents[v, :, local_mask]
            
            # 计算局部低频统计量
            mu_local = local_pixels_low.mean(dim=1, keepdim=True)
            sigma_local = local_pixels_low.std(dim=1, keepdim=True) + eps
            
            # 执行 AdaIN：把当前视角的色调往全局平均色调上拉
            # 这样就不会因为高频的“噪点”或“纹理”冲突而产生爆炸梯度
            injected_pixels_low = (local_pixels_low - mu_local) / sigma_local * sigma_global + mu_global
            
            aligned_low_freq[v, :, local_mask] = injected_pixels_low

    # ==========================================
    # 3. 结果合并：低频(已对齐) + 高频(原汁原味)
    # ==========================================
    return aligned_low_freq + high_freq_latents

def gaussian_blur(x, k=3, s=0.5):
    """
    x: (V, 4, H, W) -> 通常是 Half (FP16)
    k: kernel_size
    s: sigma
    """
    # 1. 构造 1D 高斯核
    grid = torch.arange(k, device=x.device).float() - (k - 1) / 2
    kernel_1d = torch.exp(-grid.pow(2) / (2 * s**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # 2. 构造 2D 高斯核
    kernel_2d = kernel_1d.view(k, 1) * kernel_1d.view(1, k)
    
    # 3. 关键修复：把 kernel 的精度转成和输入 x 一致 (比如从 Float 转成 Half)
    kernel = kernel_2d.view(1, 1, k, k).repeat(4, 1, 1, 1).to(x.dtype)
    
    # 4. 卷积计算
    return F.conv2d(x, kernel, padding=k//2, groups=4)

def apply_reference_style_injection(latents, part_mask, source_view_idx=2, eps=1e-6):
    """
    修改版：提取指定视角(source_view_idx)的部件风格，并注入到其他所有视角中。
    Args:
        latents: (V, 4, H, W)
        part_mask: (V, H, W)  部件ID掩码
        source_view_idx: int, 作为风格参考的视角索引 (例如 1 代表视角2)
    """
    if part_mask.device != latents.device:
        part_mask = part_mask.to(latents.device)

    unique_ids = torch.unique(part_mask)
    # print('unique_ids', len(unique_ids))
    
    # 建议 clone，防止梯度原地修改报错
    aligned_latents = latents.clone() 

    for pid in unique_ids:
        # ==========================================
        # 1. 第一步：获取【源视角】的风格 (Reference Style)
        # ==========================================
        
        # 获取源视角下，该部件的 Mask
        # source_mask shape: (H, W)
        source_mask = (part_mask[source_view_idx] == pid)
        
        # 关键检查：如果源视角(比如背面)压根没看到这个部件(比如脸)，
        # 那就没法提取风格，只能跳过这个部件的处理
        if source_mask.sum() <= 1:
            # print(f"Warning: Source view {source_view_idx} does not contain part {pid}. Skipping.")
            continue

        # 提取源视角的像素：(4, H, W) -> 选出 N 个像素 -> (4, N)
        # 注意这里只取 source_view_idx 这一张图的数据
        ref_pixels = latents[source_view_idx, :, source_mask] 
        
        # 计算源视角的均值和方差 (4, 1) -> 这就是我们要推广的标准
        mu_ref = ref_pixels.mean(dim=1, keepdim=True)
        sigma_ref = ref_pixels.std(dim=1, keepdim=True) + eps
        
        # ==========================================
        # 2. 第二步：注入到所有其他视角 (Injection)
        # ==========================================
        for v in range(latents.shape[0]):
            # 如果是源视角自己，可以选择跳过（其实不跳过也行，自己注入自己等于没变）
            # if v == source_view_idx:
            #     continue

            # 获取当前视角、当前部件的掩码
            local_mask = (part_mask[v] == pid)
            
            # 如果当前视角没看到这个部件，跳过
            if local_mask.sum() <= 1:
                continue
            
            # 提取局部像素 (4, N_local)
            local_pixels = latents[v, :, local_mask]
            
            # 计算局部统计量 (用于抹除该视角原有的风格)
            mu_local = local_pixels.mean(dim=1, keepdim=True)
            sigma_local = local_pixels.std(dim=1, keepdim=True) + eps
            
            # 核心公式 (AdaIN)：
            # 1. (local - mu_local) / sigma_local   <-- 归一化，抹除自我风格
            # 2. * sigma_ref + mu_ref               <-- 反归一化，穿上源视角的“衣服”
            injected_pixels = (local_pixels - mu_local) / sigma_local * sigma_ref + mu_ref
            
            # 写回
            aligned_latents[v, :, local_mask] = injected_pixels
            
    return aligned_latents

# Revert time 0 background to time t to composite with time t foreground
@torch.no_grad()
def composite_rendered_view(scheduler, backgrounds, foregrounds, masks, t):
    composited_images = []
    for i, (background, foreground, mask) in enumerate(zip(backgrounds, foregrounds, masks)):
        if t > 0:
            alphas_cumprod = scheduler.alphas_cumprod[t]
            noise = torch.normal(0, 1, background.shape,
                                 device=background.device)
            background = (1-alphas_cumprod) * noise + \
                alphas_cumprod * background
        composited = foreground * mask + background * (1-mask)
        composited_images.append(composited)
    composited_tensor = torch.stack(composited_images)
    return composited_tensor


def get_attn_layer_sizes(H, W):
    """
    给定输入图像分辨率，返回各层 attention 的空间尺寸
    """
    H_lat, W_lat = H // 8, W // 8  # latent 空间大小
    sizes = {
        "down_blocks.0": (H_lat, W_lat),
        "down_blocks.1": (H_lat // 2, W_lat // 2),
        "down_blocks.2": (H_lat // 4, W_lat // 4),
        "down_blocks.3": (H_lat // 8, W_lat // 8),
        "mid_block": (H_lat // 8, W_lat // 8),
        "up_blocks.3": (H_lat // 2, W_lat // 2),
        "up_blocks.2": (H_lat // 4, W_lat // 4),
        "up_blocks.1": (H_lat // 8, W_lat // 8),
        "up_blocks.0": (H_lat, W_lat),
    }
    return sizes



def load_segment_mask(mask_path, input_resolution, num_views=4, device="cpu"):
    """
    加载多视角语义 mask，并生成各层 tokens 及用于全局 AdaIN 的 96x96 mask
    
    Returns:
        mask_tokens_per_layer (dict): { (h,w): tensor[num_views, seq_len] }
        global_mask_96 (torch.Tensor): tensor[num_views, 96, 96]
    """
    mask = Image.open(mask_path)
    H_total, W_total = mask.size[1], mask.size[0]
    H, W = input_resolution

    # 1. 拆分视角
    view_w = W_total // num_views
    if view_w * num_views != W_total:
        raise ValueError(f"宽度 {W_total} 不能整除 num_views={num_views}")

    masks = []
    for i in range(num_views):
        left = i * view_w
        right = (i+1) * view_w
        mask_i = mask.crop((left, 0, right, H_total))
        masks.append(mask_i)

    # 2. 预处理：转换唯一类别 ID
    processed_masks = []
    for mask_i in masks:
        arr = np.array(mask_i)
        if arr.ndim == 3:  # RGB 标签转唯一 ID
#             h, w, c = arr.shape
#             print(f"  形状: {arr.shape}")
            
#             # 获取所有唯一的RGB组合
#             unique_colors = np.unique(arr.reshape(-1, 3), axis=0)
#             print(f"  独特颜色数量: {len(unique_colors)}")
            
            arr = arr[:, :, 0].astype(np.int32) * 256 * 256 + \
                  arr[:, :, 1].astype(np.int32) * 256 + \
                  arr[:, :, 2].astype(np.int32)
        else:
            arr = arr.astype(np.int32)
        processed_masks.append(Image.fromarray(arr))

    # 3. 生成用于 Attention 层的 tokens (Flattened)
    attn_layer_sizes = get_attn_layer_sizes(H, W)
    mask_tokens_per_layer = {}
    
    for name, (h, w) in attn_layer_sizes.items():
        layer_tokens = []
        for p_mask in processed_masks:
            # 使用最近邻插值保持类别 ID 不变
            m_resized = p_mask.resize((w, h), Image.NEAREST)
            # 转为 tensor 并展平
            t_resized = torch.from_numpy(np.array(m_resized)).long().to(device)
            layer_tokens.append(t_resized.flatten().unsqueeze(0)) # (1, seq_len)
        
        mask_tokens_per_layer[(h, w)] = torch.cat(layer_tokens, dim=0)

    # 4. 生成用于全局 AdaIN 的 96x96 mask (2D Spatial)
    global_96_list = []
    for p_mask in processed_masks:
        m_96 = p_mask.resize((96, 96), Image.NEAREST)
        t_96 = torch.from_numpy(np.array(m_96)).long().to(device)
        global_96_list.append(t_96.unsqueeze(0)) # (1, 96, 96)
    
    global_mask_96 = torch.cat(global_96_list, dim=0) # (num_views, 96, 96)

    unique_ids = torch.unique(global_mask_96)
    count = len(unique_ids)
    
    # print(f"Found {count} unique Part IDs.")
    # print(f"ID List: {unique_ids.tolist()}") # 也可以看看具体是哪些ID
    return mask_tokens_per_layer, global_mask_96


# Split into micro-batches to use less memory in each unet prediction
# But need more investigation on reducing memory usage
# Assume it has no possitive effect and use a large "max_batch_size" to skip splitting
def split_groups(attention_mask, max_batch_size, ref_view=[]):
    group_sets = []
    group = set()
    ref_group = set()
    idx = 0
    while idx < len(attention_mask):
        new_group = group | set([idx])
        new_ref_group = (ref_group | set(
            attention_mask[idx] + ref_view)) - new_group
        if len(new_group) + len(new_ref_group) <= max_batch_size:
            group = new_group
            ref_group = new_ref_group
            idx += 1
        else:
            assert len(group) != 0, "Cannot fit into a group"
            group_sets.append((group, ref_group))
            group = set()
            ref_group = set()
    if len(group) > 0:
        group_sets.append((group, ref_group))

    group_metas = []
    for group, ref_group in group_sets:
        in_mask = sorted(list(group | ref_group))
        out_mask = []
        group_attention_masks = []
        for idx in in_mask:
            if idx in group:
                out_mask.append(in_mask.index(idx))
            group_attention_masks.append(
                [in_mask.index(idxx) for idxx in attention_mask[idx] if idxx in in_mask])
        ref_attention_mask = [in_mask.index(idx) for idx in ref_view]
        group_metas.append(
            [in_mask, out_mask, group_attention_masks, ref_attention_mask])

#     每个 group_meta 里会包含：

# in_mask → 当前 group 和对应 ref 的总索引

# out_mask → 当前 group query 的索引

# group_attention_masks → 每个 query 对应哪些 key 参与 attention

# ref_attention_mask → ref 的索引

# 这就是你在显存紧张时控制 attention 范围的核心逻辑。
    return group_metas


'''

    MultiView-Diffusion Stable-Diffusion Pipeline
    Modified from a Diffusers StableDiffusionControlNetPipeline
    Just mimic the pipeline structure but did not follow any API convention

'''


class StableSyncMVDPipeline(StableDiffusionControlNetPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel]],
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = False,
        gpu_id: int = 0
    ):
        # vae = vae.to('cuda:1')
        # for param in vae.parameters():
        # 	param.data = param.data.to('cuda:1')
        # for buffer in vae.buffers():
        # 	buffer.data = buffer.data.to('cuda:1')
        super().__init__(
            vae, text_encoder, tokenizer, unet,
            controlnet, scheduler, safety_checker,
            feature_extractor, requires_safety_checker
        )
        # print('gpu_id is: ', gpu_id)
        # self.device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
        self.scheduler = DDPMScheduler.from_config(self.scheduler.config)
        self.model_cpu_offload_seq = "vae->text_encoder->unet->vae"
        self.gpu_id = gpu_id
        torch.cuda.set_device(f"cuda:{gpu_id}")
        self.enable_model_cpu_offload(gpu_id=gpu_id)
        self.enable_vae_slicing()
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor)

    @property
    def _execution_device(self):
        """
        返回当前执行设备，根据 self.gpu_id 选择 GPU，
        如果不可用则回退到 CPU
        """
        if hasattr(self, "gpu_id") and torch.cuda.is_available():
            return torch.device(f"cuda:{self.gpu_id}")
        else:
            return torch.device("cpu")

    def initialize_pipeline(
            self,
            mesh_path=None,
            mesh_transform=None,
            mesh_autouv=None,
            camera_azims=None,
            camera_centers=None,
            top_cameras=True,
            ref_views=[],
            latent_size=None,
            render_rgb_size=None,
            texture_size=None,
            texture_rgb_size=None,
            max_batch_size=24,
            logging_config=None,
            use_adjacent_segment=False,
            use_adjacent_baseline=False,
            segment_img_path="",
            face2label_path="",
            yizhi=False,
    ):
        # Make output dir
        output_dir = logging_config["output_dir"]

        self.result_dir = f"{output_dir}/results"
        self.intermediate_dir = f"{output_dir}/intermediate"

        dirs = [output_dir, self.result_dir, self.intermediate_dir]
        for dir_ in dirs:
            if not os.path.isdir(dir_):
                os.mkdir(dir_)

        # Define the cameras for rendering
        self.camera_poses = []
        self.attention_mask = []
        self.centers = camera_centers

        cam_count = len(camera_azims)
        front_view_diff = 360
        back_view_diff = 360
        front_view_idx = 0
        back_view_idx = 0
        for i, azim in enumerate(camera_azims):
            if azim < 0:
                azim += 360
            self.camera_poses.append((0, azim))
            self.attention_mask.append(
                [(cam_count+i-1) % cam_count, i, (i+1) % cam_count])
            if abs(azim) < front_view_diff:
                # find min
                front_view_idx = i
                front_view_diff = abs(azim)
            if abs(azim - 180) < back_view_diff:
                back_view_idx = i
                back_view_diff = abs(azim - 180)
        # front_view_idx是离0最近的一个视角的id，back是180度最近
        # Add two additional cameras for painting the top surfaces
        # if top_cameras:
        #     self.camera_poses.append((30, 0))
        #     self.camera_poses.append((30, 180))

        #     self.attention_mask.append([front_view_idx, cam_count])
        #     self.attention_mask.append([back_view_idx, cam_count+1])

        # Reference view for attention (all views attend the the views in this list)
        # A forward view will be used if not specified
        if len(ref_views) == 0:
            ref_views = [front_view_idx]

        self.segment_mask_token = None
        print('use_adjacent_segment, yizhi', use_adjacent_segment, yizhi)
        # calculate segment attention mask tokens
        if use_adjacent_segment or use_adjacent_baseline:
            print('use segment attention')
            self.segment_mask_token, self.global_mask_96 = load_segment_mask(
                segment_img_path, (latent_size*8, latent_size*8), num_views=len(self.camera_poses), device=self._execution_device)

        # Calculate in-group attention mask
        self.group_metas = split_groups(
            self.attention_mask, max_batch_size, ref_views)
        # Set up pytorch3D for projection between screen space and UV space
        # uvp is for latent and uvp_rgb for rgb color
        self.uvp = UVP(texture_size=texture_size, render_size=latent_size,
                       sampling_mode="nearest", channels=4, device=self._execution_device)
        # print('mesh_transform["scale"]', mesh_transform["scale"])
        if mesh_path.lower().endswith(".obj"):
            self.uvp.load_mesh(
                mesh_path, scale_factor=mesh_transform["scale"] or 1, autouv=mesh_autouv)
        elif mesh_path.lower().endswith(".glb"):
            self.uvp.load_glb_mesh(
                mesh_path, scale_factor=mesh_transform["scale"] or 1, autouv=mesh_autouv)
        else:
            assert False, "The mesh file format is not supported. Use .obj or .glb."
        # set up camera settings
        # camera_pose (elev, azim, radius)
        self.uvp.set_cameras_and_render_settings(
            self.camera_poses, centers=camera_centers, camera_distance=2.0)
        # uvp, verts shape is:  torch.Size([10, 96, 96, 4])
        self.uvp_rgb = UVP(texture_size=texture_rgb_size, render_size=render_rgb_size,
                           sampling_mode="nearest", channels=3, device=self._execution_device)
        self.uvp_rgb.mesh = self.uvp.mesh.clone()
        self.uvp_rgb.set_cameras_and_render_settings(
            self.camera_poses, centers=camera_centers, camera_distance=2.0)
        # verts shape is:  torch.Size([10, 1536, 1536, 4])

        # 渲染几何体并计算余弦角度权重
        _, _, _, cos_maps, _, _ = self.uvp_rgb.render_geometry()
        # verts shape is:  torch.Size([10, 1536, 1536, 4])

        self.uvp_rgb.calculate_cos_angle_weights(cos_maps, fill=False)
        # Save some VRAM
        del _, cos_maps
        self.uvp.to("cpu")
        self.uvp_rgb.to("cpu")

        # self.vae = copy.deepcopy(self.vae).to(self._execution_device)
        # conv_in = self.vae.encoder.conv_in
        # conv_in.weight = nn.Parameter(conv_in.weight.data.to(self._execution_device))
        # if conv_in.bias is not None:
        # 	conv_in.bias = nn.Parameter(conv_in.bias.data.to(self._execution_device))

        # 强制将所有参数和缓冲区移动到目标设备
        # for param in self.vae.parameters():
        #     param.data = param.data.to(self._execution_device)
        # for buffer in self.vae.buffers():
        #     buffer.data = buffer.data.to(self._execution_device)

        # create color_images and extend it
        color_images = torch.FloatTensor([color_constants[name] for name in color_names]).reshape(
            -1, 3, 1, 1).to(dtype=self.text_encoder.dtype, device=self._execution_device)
        color_images = torch.ones(
            (1, 1, latent_size*8, latent_size*8),
            device=self._execution_device,
            dtype=self.text_encoder.dtype
        ) * color_images
        color_images = ((0.5*color_images)+0.5)
        # print(f"color_images device: {color_images.device}")
        # print(f"VAE device: {next(self.vae.parameters()).device}")
        # print(f"VAE dtype: {next(self.vae.parameters()).dtype}")
        # conv_in = self.vae.encoder.conv_in
        # print(f"Conv_in weight device: {conv_in.weight.device}")
        # encode color img
        color_latents = encode_latents(self.vae, color_images)

        self.color_latents = {color[0]: color[1] for color in zip(
            color_names, [latent for latent in color_latents])}
        self.vae = self.vae.to("cpu")

        print("Done Initialization")

    '''
        Modified from a StableDiffusion ControlNet pipeline
        Multi ControlNet not supported yet
    '''
    @torch.no_grad()
    def __call__(
        self,
        prompt: str = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: str = None,

        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        return_dict: bool = False,
        callback: Optional[Callable[[
            int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        max_batch_size=6,

        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_guess_mode: bool = False,
        controlnet_conditioning_scale: Union[float, List[float]] = 0.7,
        controlnet_conditioning_end_scale: Union[float, List[float]] = 0.9,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 0.99,
        guidance_rescale: float = 0.0,

        mesh_path: str = None,
        mesh_transform: dict = None,
        mesh_autouv=False,
        camera_azims=None,
        camera_centers=None,
        top_cameras=True,
        texture_size=1536,  # init latent texture
        render_rgb_size=1024,
        texture_rgb_size=1024,
        multiview_diffusion_end=0.8,
        exp_start=0.0,
        exp_end=6.0,
        shuffle_background_change=0.4,
        shuffle_background_end=0.99,  # 0.4

        use_directional_prompt=True,

        ref_attention_end=0.2,

        logging_config=None,
        cond_type="depth",
        img_path="",

        early_use_adjacent_baseline=False,
        early_use_adjacent_segment=True,
        early_use_ref=True,
        early_segment_weight=-1.0,
        early_ref_weight=1.0,

        late_use_adjacent_baseline=False,
        late_use_adjacent_segment=False,
        late_use_ref=False,
        segment_img_path="",
        late_segment_weight=-0.8,
        late_ref_weight=1.0,
        face2label_path="",
        use_semantic_anchor=False,
        anchor_weight=0.5,

        # last yizhi
        yizhi=False,
        yizhi_start=0.0,
        yizhi_end=1.0,
        
        adain_first=0, 
        adain_early=False,
        adain_time=0.2,
    ):
        # Setup pipeline settings
        self.initialize_pipeline(
            mesh_path=mesh_path,
            mesh_transform=mesh_transform,
            mesh_autouv=mesh_autouv,
            camera_azims=camera_azims,
            camera_centers=camera_centers,
            top_cameras=top_cameras,
            ref_views=[],
            latent_size=height//8,  # 96
            render_rgb_size=render_rgb_size,  # 1024
            texture_size=texture_size,  # 1536
            texture_rgb_size=texture_rgb_size,  # 1024

            max_batch_size=max_batch_size,

            logging_config=logging_config,
            use_adjacent_segment=early_use_adjacent_segment or late_use_adjacent_segment,
            use_adjacent_baseline=early_use_adjacent_baseline,
            segment_img_path=segment_img_path,
            face2label_path=face2label_path,
            yizhi=yizhi,
        )
        print('nega', negative_prompt)
        # 获取函数签名
        self.face2label_path = face2label_path
        signature = locals()
        # 打印所有参数

        # 获取训练步数，条件网络的缩放因子，日志间隔，视图快速预览，纹理快速预览
        num_timesteps = self.scheduler.config.num_train_timesteps
        initial_controlnet_conditioning_scale = controlnet_conditioning_scale
        log_interval = logging_config.get("log_interval", 10)
        view_fast_preview = logging_config.get("view_fast_preview", True)
        tex_fast_preview = logging_config.get("tex_fast_preview", True)
        # 假如是编译模块，就直接用其原始模块
        controlnet = self.controlnet._orig_mod if is_compiled_module(
            self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(
                control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(
                control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            # mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            mult = 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                control_guidance_end
            ]

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            torch.zeros((1, 3, height, width), device=self._execution_device),
            callback_steps,
            negative_prompt,
            None,
            None,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, list):
            assert len(prompt) == 1 and len(
                negative_prompt) == 1, "Only implemented for 1 (negative) prompt"
        assert num_images_per_prompt == 1, "Only implemented for 1 image per-prompt"
        batch_size = len(self.uvp.cameras)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
        # 	controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = controlnet_guess_mode or global_pool_conditions

        # 3. Encode input prompt
        prompt, negative_prompt = prepare_directional_prompt(
            prompt, negative_prompt)

        text_encoder_lora_scale = (
            cross_attention_kwargs.get(
                "scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=text_encoder_lora_scale,
        )
        # print('len(prompt_embeds)', len(prompt_embeds))
        # 分割嵌入向量
        negative_prompt_embeds, prompt_embeds = torch.chunk(prompt_embeds, 2)
        prompt_embed_dict = dict(
            zip(direction_names, [emb for emb in prompt_embeds]))
        # print('prompt_embed_dict is: ', prompt_embed_dict)
        negative_prompt_embed_dict = dict(
            zip(direction_names, [emb for emb in negative_prompt_embeds]))

        # (4. Prepare image) This pipeline use internal conditional images from Pytorch3D
        # 将uvp转入gpu并获取对应的条件图像和mask，
        # print('self._execution_device', self._execution_device)
        self.uvp.to(self._execution_device)
        conditioning_images, masks, origin_mask = get_conditioning_images(
            self.uvp, height, cond_type=cond_type, img_path=img_path)
        conditioning_images = conditioning_images.to(self._execution_device)
        # print('heigth shape is: ', height)
        # print('conditioning_images, mask', conditioning_images.shape, masks.shape)
        conditioning_images = conditioning_images.type(prompt_embeds.dtype)
        # 归一化 [0, 1]
        cond = (conditioning_images/2+0.5).permute(0, 2, 3, 1).cpu().numpy()
        # cond = (conditioning_images/2).permute(0, 2, 3, 1).cpu().numpy()

        # print('self.uvp.cond depth is: ', cond.shape) # (10, 768, 768, 3)
        # 将多个条件图像合并
        cond = np.concatenate([img for img in cond], axis=1)
        # print('concatenate cond shape is: ', cond.shape)
        numpy_to_pil(cond)[0].save(f"{self.intermediate_dir}/cond.jpg")

        # print('mask shape is:', masks.shape)
        # origin_mask = np.concatenate([img.detach().cpu().numpy() for img in masks], axis=1)
        # numpy_to_pil(origin_mask)[0].save(f"{self.intermediate_dir}/mask.jpg")
        # 存储 1024 x 1024
        origin_mask = origin_mask.cpu()
        # masks = []
        # for i in range(2):
        #     # 提取单张掩码 [1, 512, 512] → [512, 512]
        #     mask = origin_mask[i].squeeze(0).numpy() * 255
        #     mask = mask.astype(np.uint8)
        #     masks.append(mask)

        # # 3. 创建透明RGB图像
        # transparent_masks = []
        # for mask in masks:
        #     # 创建RGBA图像
        #     rgba = Image.fromarray(mask)
        #     # 转为RGBA模式
        #     rgba = rgba.convert("RGBA")

        #     # 提取所有通道数据
        #     rgba_data = np.array(rgba)
        #     # 当R通道<128时设为完全透明（也可以根据掩码值自定义阈值）
        #     rgba_data[..., 3] = np.where(rgba_data[..., 0] < 128, 0, 255)

        #     # 对于值>0的区域设为白色（RGB=255）
        #     rgba_data[..., :3] = np.where(
        #         rgba_data[..., [0]] > 0,
        #         255,
        #         rgba_data[..., :3]
        #     )

        #     transparent_masks.append(Image.fromarray(rgba_data, "RGBA"))

        # # 4. 创建1×4网格
        # grid_width = 6 * 768
        # grid_height = 768
        # grid_image = Image.new("RGBA", (grid_width, grid_height))

        # # 5. 拼接所有掩码
        # x_offset = 0
        # for mask_img in transparent_masks:
        #     grid_image.paste(mask_img, (x_offset, 0))
        #     x_offset += 768

        # # 6. 保存结果
        # grid_image.save(f"{self.intermediate_dir}/mask.png")

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        # 获取UNet的输入通道数
        num_channels_latents = self.unet.config.in_channels
        # 生成latents, zt
        latents = self.prepare_latents(
            batch_size,  # 相机个数
            num_channels_latents,  # 输入通道数
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            None,
        )
        # 设置噪声纹理
        latent_tex = self.uvp.set_noise_texture()
        # 渲染带纹理的视图
        noise_views = self.uvp.render_textured_views()
        # 获取前景
        foregrounds = [view[:-1] for view in noise_views]
        masks = [view[-1:] for view in noise_views]
        # latents加噪乘 (1 - mask) + mask * foreground
        composited_tensor = composite_rendered_view(
            self.scheduler, latents, foregrounds, masks, timesteps[0]+1)
        latents = composited_tensor.type(latents.dtype)  # (10, 4, 96, 96)
        if adain_first != 0:
            # latents = apply_reference_style_injection(latents, self.global_mask_96)
            self.viz_dir = self.intermediate_dir + "part_viz_dir"
            latents = apply_global_style_injection_with_viz(latents, self.global_mask_96, self.viz_dir, adain_first)
        self.uvp.to("cpu")
        # print('latents shape is: ', latents.shape)

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []

        for i in range(len(timesteps)):
            # 每个时间步保留哪些网络张量列表
            keeps = [
                1.0 - float(i / len(timesteps) <
                            s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(
                controlnet, ControlNetModel) else keeps)

        # 8. Denoising loop
        # 预热步骤的数量
        num_warmup_steps = len(timesteps) - \
            num_inference_steps * self.scheduler.order
        intermediate_results = []
        # 随机选背景颜色
        # background_colors = [random.choice(
        #     list(color_constants.keys())) for i in range(len(self.camera_poses))]
        background_colors = ["black" for i in range(len(self.camera_poses))]

        # 存储controlnet输出的形状
        dbres_sizes_list = []
        mbres_size_list = []

        positive_prompt_embeds = [azim_prompt(
            prompt_embed_dict, pose) for pose in self.camera_poses]
        positive_prompt_embeds = torch.stack(
            positive_prompt_embeds, axis=0)

        negative_prompt_embeds = [azim_neg_prompt(
            negative_prompt_embed_dict, pose) for pose in self.camera_poses]
        negative_prompt_embeds = torch.stack(
            negative_prompt_embeds, axis=0)
        
        
        if view_fast_preview:
            # 假设你的初始噪声变量名是 latents (形状 V, 4, H, W)
            # 我们可以直接调用你的预览逻辑
            init_decoded_results = []

            # 将当前的初始噪声（对齐后的）转为预览图
            # 注意：这里 latent_images 通常是一个 batch，如果你的框架里是分步存储的，直接传入 latents 即可
            images = latent_preview(latents.to(self._execution_device))

            # 水平拼接所有视角的雪花图
            init_strip = np.concatenate([img for img in images], axis=1)

            # 转换为 PIL 并保存，命名为 step_init 代表 T=1000
            numpy_to_pil(init_strip)[0].save(
                f"{self.intermediate_dir}/step_init_T1000.jpg"
            )
            print(f">>> 已保存 T=1000 初始雪花图至: {self.intermediate_dir}/step_init_T1000.jpg")
        
        # 使用progress_bar跟踪进度
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # mix prompt embeds according to azim angle
                # 根据相机姿势混合正向和负向提示嵌入

                # expand the latents if we are doing classifier free guidance
                latent_model_input = self.scheduler.scale_model_input(
                    latents, t)

                '''
                    Use groups to manage prompt and results
                    Make sure negative and positive prompt does not perform attention together
                '''
                # 将正向和负向提示嵌入分组
                prompt_embeds_groups = {"positive": positive_prompt_embeds}
                result_groups = {}
                if do_classifier_free_guidance:
                    prompt_embeds_groups["negative"] = negative_prompt_embeds

                for prompt_tag, prompt_embeds in prompt_embeds_groups.items():
                    if prompt_tag == "positive" or not guess_mode:
                        # controlnet(s) inference
                        control_model_input = latent_model_input
                        controlnet_prompt_embeds = prompt_embeds

                        # 如果controlnet_keep是列表，则按元素乘conditioning_scale
                        if isinstance(controlnet_keep[i], list):
                            cond_scale = [
                                c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                        else:
                            controlnet_cond_scale = controlnet_conditioning_scale
                            if isinstance(controlnet_cond_scale, list):
                                controlnet_cond_scale = controlnet_cond_scale[0]
                            cond_scale = controlnet_cond_scale * \
                                controlnet_keep[i]

                        # Split into micro-batches according to group meta info
                        # Ignore this feature for now
                        down_block_res_samples_list = []
                        mid_block_res_sample_list = []
                        # 分割成小批次
                        # print('control_model_input shape is: ', control_model_input.shape)
                        # print('self._execution_device', self._execution_device)
                        model_input_batches = [torch.index_select(control_model_input, dim=0, index=torch.tensor(
                            meta[0], device=self._execution_device)) for meta in self.group_metas]
                        prompt_embeds_batches = [torch.index_select(controlnet_prompt_embeds, dim=0, index=torch.tensor(
                            meta[0], device=self._execution_device)) for meta in self.group_metas]
                        conditioning_images_batches = [torch.index_select(conditioning_images, dim=0, index=torch.tensor(
                            meta[0], device=self._execution_device)) for meta in self.group_metas]
                        # 对每个小批次调用controlnet推理
                        for model_input_batch, prompt_embeds_batch, conditioning_images_batch \
                                in zip(model_input_batches, prompt_embeds_batches, conditioning_images_batches):
                            down_block_res_samples, mid_block_res_sample = self.controlnet(
                                model_input_batch,
                                t,
                                encoder_hidden_states=prompt_embeds_batch,
                                controlnet_cond=conditioning_images_batch,
                                conditioning_scale=cond_scale,
                                guess_mode=guess_mode,
                                return_dict=False,
                            )
                            down_block_res_samples_list.append(
                                down_block_res_samples)
                            mid_block_res_sample_list.append(
                                mid_block_res_sample)

                        ''' For the ith element of down_block_res_samples, concat the ith element of all mini-batch result '''
                        model_input_batches = prompt_embeds_batches = conditioning_images_batches = None
                        # 猜测模式下，记录down_block_res_samples和mid_block_res_sample的形状
                        if guess_mode:
                            for dbres in down_block_res_samples_list:
                                dbres_sizes = []
                                for res in dbres:
                                    dbres_sizes.append(res.shape)
                                dbres_sizes_list.append(dbres_sizes)

                            for mbres in mid_block_res_sample_list:
                                mbres_size_list.append(mbres.shape)

                    else:
                        # Infered ControlNet only for the conditional batch.
                        # To apply the output of ControlNet to both the unconditional and conditional batches,
                        # add 0 to the unconditional batch to keep it unchanged.
                        # We copy the tensor shapes from a conditional batch
                        # 如果是负向提示，生成零张量以保持无条件的批次不变，从条件批次复制张量形状
                        down_block_res_samples_list = []
                        mid_block_res_sample_list = []
                        for dbres_sizes in dbres_sizes_list:
                            down_block_res_samples_list.append([torch.zeros(
                                shape, device=self._execution_device, dtype=latents.dtype) for shape in dbres_sizes])
                        for mbres in mbres_size_list:
                            mid_block_res_sample_list.append(torch.zeros(
                                mbres, device=self._execution_device, dtype=latents.dtype))
                        dbres_sizes_list = []
                        mbres_size_list = []

                    '''
                    
                        predict the noise residual, split into mini-batches
                        Downblock res samples has n samples, we split each sample into m batches
                        and re group them into m lists of n mini batch samples.
                    
                    '''
                    # 存储每个小批次的噪声预测结果
                    noise_pred_list = []
                    # 将 latent_model_input 和 prompt_embeds 按照 group_metas 中的索引分割成小批次。
                    model_input_batches = [torch.index_select(latent_model_input, dim=0, index=torch.tensor(
                        meta[0], device=self._execution_device)) for meta in self.group_metas]
                    prompt_embeds_batches = [torch.index_select(prompt_embeds, dim=0, index=torch.tensor(
                        meta[0], device=self._execution_device)) for meta in self.group_metas]

                    for model_input_batch, prompt_embeds_batch, down_block_res_samples_batch, mid_block_res_sample_batch, meta \
                            in zip(model_input_batches, prompt_embeds_batches, down_block_res_samples_list, mid_block_res_sample_list, self.group_metas):
                        # if t > num_timesteps * (1 - 0.4):
                        #     # self attention reuse, 根据t动态设置attention的处理策略
                        #     replace_attention_processors(
                        #         self.unet, SamplewiseAttnProcessor2_0, attention_mask=meta[
                        #             2], ref_attention_mask=meta[3],
                        #         use_adjacent_baseline=False, use_adjacent_segment=True, segment_mask_tokens=self.segment_mask_token, segment_weight=-1.0, use_ref = False, ref_weight=1, resolution=height)
                        # else:
                        #     replace_attention_processors(
                        #         self.unet, SamplewiseAttnProcessor2_0, attention_mask=meta[
                        #             2], ref_attention_mask=meta[3],
                        #         use_adjacent_baseline=False, use_adjacent_segment=True, segment_mask_tokens=self.segment_mask_token, segment_weight=-0.8, use_ref = False, ref_weight=0, resolution=height)

                        # 遍历每个小批次并进行噪声预测
                        use_adjacent_baseline = early_use_adjacent_baseline or early_use_adjacent_segment
                        use_adjacent_segment = early_use_adjacent_segment or late_use_adjacent_segment
                        use_ref = early_use_ref or late_use_ref
                        if not (use_adjacent_baseline or use_adjacent_segment or use_ref):
                            pass
                        else:
                            if not yizhi:
                                replace_attention_processors(
                                    self.unet,
                                    SamplewiseAttnProcessor2_0,
                                    attention_mask=meta[2],
                                    ref_attention_mask=meta[3],
                                    use_adjacent_baseline=early_use_adjacent_baseline,
                                    use_adjacent_segment=early_use_adjacent_segment,
                                    segment_mask_tokens=self.segment_mask_token,
                                    segment_weight=early_segment_weight,
                                    use_ref=early_use_ref,
                                    ref_weight=early_ref_weight,
                                    resolution=height,
                                )
                            else:
                                start_step = num_timesteps * (1 - yizhi_start)
                                end_step = num_timesteps * (1 - yizhi_end)
                                if t < start_step and t > end_step:
                                    # 1. 计算当前进度 ratio (0.0 -> 1.0)
                                    # 现在的 t 在 (start_step, end_step) 之间
                                    # start_step 是大数(比如800)，end_step 是小数(比如200)
                                    # 我们希望从 0.0 开始，所以用 (start - t)
                                    raw_ratio = (start_step - t) / (start_step - end_step)
    
                                    # 2. 限制 ratio 在 [0.0, 1.0] 之间，这是为了计算平滑过渡
                                    # 超过 1.0 的部分（即 0.7 之后）我们先把它卡在 1.0
                                    active_ratio = torch.clamp(raw_ratio, 0.0, 1.0)

                                    # 3. 计算 S 型平滑起步 (0.3 到 0.7 这段路走得丝滑)
                                    smooth_ratio = active_ratio * active_ratio * (3 - 2 * active_ratio)

                                    # 4. 计算最终权重
                                    # 如果 t 还在 [start, end] 之间，权重随 S 曲线增加
                                    # 如果 t 已经跑到了 end 之后 (也就是 0.7 之后)，smooth_ratio 固定为 1.0
                                    # 此时 yizhi_weight 就一直保持为 early_segment_weight
                                    yizhi_weight = early_segment_weight * smooth_ratio

                                    replace_attention_processors(
                                        self.unet,
                                        SamplewiseAttnProcessor2_0,
                                        attention_mask=meta[2],
                                        ref_attention_mask=meta[3],
                                        use_adjacent_baseline=early_use_adjacent_baseline,
                                        use_adjacent_segment=early_use_adjacent_segment,
                                        segment_mask_tokens=self.segment_mask_token,
                                        segment_weight=early_segment_weight,
                                        use_ref=early_use_ref,
                                        ref_weight=early_ref_weight,
                                        yizhi=yizhi,
                                        yizhi_weight=yizhi_weight,
                                        resolution=height,
                                    )
                                elif t > start_step:
                                    replace_attention_processors(
                                        self.unet,
                                        SamplewiseAttnProcessor2_0,
                                        attention_mask=meta[2],
                                        ref_attention_mask=meta[3],
                                        use_adjacent_baseline=early_use_adjacent_baseline,
                                        use_adjacent_segment=early_use_adjacent_segment,
                                        segment_mask_tokens=self.segment_mask_token,
                                        segment_weight=early_segment_weight,
                                        use_ref=early_use_ref,
                                        ref_weight=early_ref_weight,
                                        yizhi=False,
                                        yizhi_weight=0,
                                        resolution=height,
                                    )
                                else:
                                    replace_attention_processors(
                                        self.unet,
                                        SamplewiseAttnProcessor2_0,
                                        attention_mask=meta[2],
                                        ref_attention_mask=meta[3],
                                        use_adjacent_baseline=early_use_adjacent_baseline,
                                        use_adjacent_segment=early_use_adjacent_segment,
                                        segment_mask_tokens=self.segment_mask_token,
                                        segment_weight=early_segment_weight,
                                        use_ref=early_use_ref,
                                        ref_weight=early_ref_weight,
                                        yizhi=yizhi,
                                        yizhi_weight=early_segment_weight,
                                        resolution=height,
                                    )

                            # 至少有一个特殊机制开启 → 替换 processor
                            # if t > num_timesteps * (1 - ref_attention_end):
                            #     replace_attention_processors(
                            #         self.unet,
                            #         SamplewiseAttnProcessor2_0,
                            #         attention_mask=meta[2],
                            #         ref_attention_mask=meta[3],
                            #         use_adjacent_baseline=early_use_adjacent_baseline,
                            #         use_adjacent_segment=early_use_adjacent_segment,
                            #         segment_mask_tokens=self.segment_mask_token,
                            #         segment_weight=early_segment_weight,
                            #         use_ref=early_use_ref,
                            #         ref_weight=early_ref_weight,
                            #         resolution=height,
                            #     )
                            # else:
                            #     replace_attention_processors(
                            #         self.unet,
                            #         SamplewiseAttnProcessor2_0,
                            #         attention_mask=meta[2],
                            #         ref_attention_mask=meta[3],
                            #         use_adjacent_baseline=late_use_adjacent_baseline,
                            #         use_adjacent_segment=late_use_adjacent_segment,
                            #         segment_mask_tokens=self.segment_mask_token,
                            #         segment_weight=late_segment_weight,
                            #         use_ref=late_use_ref,
                            #         ref_weight=late_ref_weight,
                            #         resolution=height,
                            #     )

                        # # if segment mask
                        #     replace_attention_processors(
                        #         self.unet, SamplewiseAttnProcessor2_0, attention_mask=meta[2], ref_attention_mask=meta[3],
                        #         use_adjacent_baseline=True, use_adjacent_segment=False, segment_mask_tokens=None, segment_weight=0.0, use_ref_weight=1)
                        noise_pred = self.unet(
                            model_input_batch,
                            t,
                            encoder_hidden_states=prompt_embeds_batch,
                            cross_attention_kwargs=cross_attention_kwargs,
                            down_block_additional_residuals=down_block_res_samples_batch,
                            mid_block_additional_residual=mid_block_res_sample_batch,
                            return_dict=False,
                        )[0]
                        noise_pred_list.append(noise_pred)
                    # 根据 group_metas 中的索引重新组合噪声预测结果。
                    # 将所有小批次的噪声预测结果拼接成一个完整的张量。
                    noise_pred_list = [torch.index_select(noise_pred, dim=0, index=torch.tensor(
                        meta[1], device=self._execution_device)) for noise_pred, meta in zip(noise_pred_list, self.group_metas)]
                    noise_pred = torch.cat(noise_pred_list, dim=0)
                    down_block_res_samples_list = None
                    mid_block_res_sample_list = None
                    noise_pred_list = None
                    model_input_batches = prompt_embeds_batches = down_block_res_samples_batches = mid_block_res_sample_batches = None

                    result_groups[prompt_tag] = noise_pred
                # 获取正向噪声预测
                positive_noise_pred = result_groups["positive"]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred = result_groups["negative"] + guidance_scale * (
                        positive_noise_pred - result_groups["negative"])

                # 调整噪声预测
                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                self.uvp.to(self._execution_device)
                # compute the previous noisy sample x_t -> x_t-1
                # Multi-View step or individual step
                current_exp = ((exp_end-exp_start) * i /
                               num_inference_steps) + exp_start
                if t > (1-multiview_diffusion_end)*num_timesteps:
                    # 使用多视图步骤
                    step_results = step_tex(
                        scheduler=self.scheduler,
                        uvp=self.uvp,
                        model_output=noise_pred,
                        timestep=t,
                        sample=latents,
                        texture=latent_tex,
                        return_dict=True,
                        main_views=[],
                        exp=current_exp,
                        **extra_step_kwargs
                    )

                    # Z_{0|t}
                    pred_original_sample = step_results["pred_original_sample"]
                    latents = step_results["prev_sample"]  # Z
                    latent_tex = step_results["prev_tex"]  # W

                    # Composit latent foreground with random color background
                    background_latents = [self.color_latents[color]
                                          for color in background_colors]
                    composited_tensor = composite_rendered_view(
                        self.scheduler, background_latents, latents, masks, t)
                    # 前景和背景合成的结果
                    latents = composited_tensor.type(latents.dtype)

                    intermediate_results.append(
                        (latents.to("cpu"), pred_original_sample.to("cpu")))
                else:
                    # 单视角去噪
                    step_results = self.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs, return_dict=True)

                    # Z_{0|t}
                    pred_original_sample = step_results["pred_original_sample"]
                    latents = step_results["prev_sample"]  # Z
                    latent_tex = None

                    intermediate_results.append(
                        (latents.to("cpu"), pred_original_sample.to("cpu")))

                del noise_pred, result_groups

                # Update pipeline settings after one step:
                # 1. Annealing ControlNet scale
                # 当前时间步的比例用以更新controlnet_conditioning_scale
                if (1-t/num_timesteps) < control_guidance_start[0]:
                    # 在起始阶段
                    controlnet_conditioning_scale = initial_controlnet_conditioning_scale
                elif (1-t/num_timesteps) > control_guidance_end[0]:
                    # 在结束阶段
                    controlnet_conditioning_scale = controlnet_conditioning_end_scale
                else:
                    # 否则按比例插值
                    alpha = ((1-t/num_timesteps) - control_guidance_start[0]) / (
                        control_guidance_end[0] - control_guidance_start[0])
                    controlnet_conditioning_scale = alpha * initial_controlnet_conditioning_scale + \
                        (1-alpha) * controlnet_conditioning_end_scale

                # 2. Shuffle background colors; only black and white used after certain timestep
                if (1-t/num_timesteps) < shuffle_background_change:
                    # [0, shuffle_background_change) 时随机选颜色
                    # background_colors = [random.choice(
                    #     list(color_constants.keys())) for i in range(len(self.camera_poses))]
                    background_colors = ["black" for i in range(len(self.camera_poses))]
                elif (1-t/num_timesteps) < shuffle_background_end:
                    # [shuffle_background_change, shuffle_background_end] 时随机选黑白
                    # background_colors = [random.choice(
                    #     ["black", "white"]) for i in range(len(self.camera_poses))]
                    background_colors = ["black" for i in range(len(self.camera_poses))]
                else:
                    # [shuffle_background_end, 1], 不变
                    background_colors = background_colors

                # Logging at "log_interval" intervals and last step
                # Choose to uses color approximation or vae decoding
                if i % log_interval == log_interval-1 or t == 1:
                    # i是log_interval的倍数减1或者当前时间步是1（last step）, 记录
                    if view_fast_preview:
                        # 检查是否启用快速预览模式
                        # decoded_results 用于存储解码后的图像。
                        decoded_results = []
                        for latent_images in intermediate_results[-1]:
                            # 使用 latent_preview 函数生成预览图像，fast decode
                            images = latent_preview(
                                latent_images.to(self._execution_device))
                            # 将生成的图像沿水平方向拼接成一个长图。
                            images = np.concatenate(
                                [img for img in images], axis=1)
                            decoded_results.append(images)
                        # 将所有拼接后的图像沿垂直方向拼接成一个最终结果图像
                        result_image = np.concatenate(decoded_results, axis=0)
                        # 使用 numpy_to_pil 函数将结果图像转换为 PIL 图像，并保存到指定路径。
                        numpy_to_pil(result_image)[0].save(
                            f"{self.intermediate_dir}/step_{i:02d}.jpg")
                    else:
                        decoded_results = []
                        for latent_images in intermediate_results[-1]:
                            images = decode_latents(
                                self.vae, latent_images.to(self._execution_device))

                            images = np.concatenate(
                                [img for img in images], axis=1)

                            decoded_results.append(images)
                        result_image = np.concatenate(decoded_results, axis=0)
                        numpy_to_pil(result_image)[0].save(
                            f"{self.intermediate_dir}/step_{i:02d}.jpg")

                    # 判断t是否大于或等于(1-multiview_diffusion_end)*num_timesteps
                    if not t < (1-multiview_diffusion_end)*num_timesteps:
                        if tex_fast_preview:
                            tex = latent_tex.clone()
                            # 记录纹理颜色
                            texture_color = latent_preview(tex[None, ...])
                            numpy_to_pil(texture_color)[0].save(
                                f"{self.intermediate_dir}/texture_{i:02d}.jpg")
                        else:
                            # 暂存保存纹理图像
                            # tex = latent_tex.clone()
                            # # 记录纹理颜色
                            # texture_color = latent_preview(tex[None, ...])
                            # numpy_to_pil(texture_color)[0].save(
                            #     f"{self.intermediate_dir}/texture_{i:02d}.jpg")

                            self.uvp_rgb.to(self._execution_device)
                            result_tex_rgb, result_tex_rgb_output = get_rgb_texture(
                                self.vae, self.uvp_rgb, pred_original_sample)
                            numpy_to_pil(result_tex_rgb_output)[0].save(
                                f"{self.intermediate_dir}/texture_{i:02d}.png")
                            self.uvp_rgb.to("cpu")
                    
                    if adain_early and t > num_timesteps * (1 - adain_time) :
                        latents = apply_global_style_injection(latents, self.global_mask_96)   
                        # latents = apply_reference_style_injection(latents, self.global_mask_96)

                self.uvp.to("cpu")

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    # i是最后一个时间步
                    # 当前迭代步 i + 1 大于预热步骤数 num_warmup_steps，并且 i + 1 是调度器阶数 self.scheduler.order 的倍数。
                    # 更新进度条
                    progress_bar.update()
                    # 调用回调函数
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

                # Signal the program to skip or end
                import select
                import sys
                if select.select([sys.stdin], [], [], 0)[0]:
                    userInput = sys.stdin.readline().strip()
                    if userInput == "skip":
                        return None
                    elif userInput == "end":
                        exit(0)

        self.uvp.to(self._execution_device)
        self.uvp_rgb.to(self._execution_device)
        mesh = self.uvp_rgb.mesh

        # print cos_maps statistics
        for i, cos_map in enumerate(self.uvp_rgb.cos_maps):
            print(f"[cos_maps] Camera {i}: shape={cos_map.shape}, min={cos_map.min().item():.6f}, max={cos_map.max().item():.6f}")


        cos_maps_stack = torch.stack(self.uvp_rgb.cos_maps, dim=0)  # [N_views, H, W, C]
        max_cos_map, _ = torch.max(cos_maps_stack, dim=0)            # [H, W, C]
        avg_cos_map = torch.mean(cos_maps_stack, dim=0)

        # origin output, no inpaint
        result_tex_rgb, result_tex_rgb_output, result_views, inpaint_mask = get_unconsistent_image(
            self.vae, self.uvp_rgb, latents)

        # 原始不一致多视角图片保存
        result_views_rgb = torch.cat(result_views, axis=-1)  # 拼接成 [3, H, N*W]
        result_views_rgb = result_views_rgb.permute(1, 2, 0).cpu().numpy()[None, ...]
        result_v = numpy_to_pil(result_views_rgb)[0]
        result_save_path = f"{self.result_dir}/result_views_rgb.jpg"
        result_v.save(result_save_path)


        # 不一致的纹理渲染图与mesh保存
        tex_to_save = result_tex_rgb.permute(1,2,0).detach().cpu().clone()
        uv_tex = tex_to_save  # (H, W, 3) float
        # mask = inpaint_mask  # 假设 shape = (1,H,W) or (H,W)
        # mask = mask.to(uv_tex.device)
        # if mask.dim() == 3:
        #     mask = mask.squeeze(0)     # 变成 (H, W)
        # mask = mask > 0.5              # bool mask (H, W)
        # # 扩展成 (H, W, 3)
        # mask3 = mask.unsqueeze(-1).expand_as(uv_tex)
        # # 红色
        # red_color = torch.tensor([1.0, 0.0, 0.0]).view(1,1,3)
        # uv_tex[mask3] = red_color.expand_as(uv_tex)[mask3]
        # self.uvp.save_mesh(f"{self.result_dir}/textured_with_mask.obj", uv_tex)

    

        # print('uv_tex shape is: ', uv_tex.shape)
        # self.uvp_rgb.set_texture_map(uv_tex.permute(2,0,1))
        # unconsistent_textured_views = self.uvp_rgb.render_textured_views()
        # unconsistent_textured_views_rgb = torch.cat(
        #     unconsistent_textured_views, axis=-1)[:-1, ...]
        # unconsistent_textured_views_rgb = unconsistent_textured_views_rgb.permute(
        #     1, 2, 0).cpu().numpy()[None, ...]
        # unconsistent_v = numpy_to_pil(unconsistent_textured_views_rgb)[0]
        # unconsistent_save_path = f"{self.result_dir}/unconsistent_textured_views_rgb.jpg"
        # unconsistent_v.save(unconsistent_save_path)
        # print("✅ 已保存不一致纹理的 mesh 与渲染图：",
        #     f"{self.result_dir}/textured_with_mask.obj",
        #     unconsistent_save_path)

        # 1. 提取 Mask 并复制到 CPU (避免污染原变量 inpaint_mask)
#         _temp_mask = inpaint_mask.detach().cpu().clone() 
        
#         # 2. 压缩维度，确保变成 (H, W)
#         if _temp_mask.dim() == 3:
#             _temp_mask = _temp_mask.squeeze(0)
        
#         # 3. 转换为 Numpy 数组
#         # 逻辑：大于 0.5 的地方设为 255 (白)，否则为 0 (黑)
#         mask_np_gray = (_temp_mask.numpy() > 0.5).astype(np.uint8) * 255

#         # 4. 堆叠成 3 通道 RGB 图片 (H, W, 3)
#         # 这样出来的图就是标准的黑白 RGB 图片
#         mask_preview_rgb = np.stack([mask_np_gray] * 3, axis=-1)

#         # 5. 保存图片
#         save_img_path = f"{self.result_dir}/texture_mask_only.png"
#         Image.fromarray(mask_preview_rgb).save(save_img_path)
        
#         print(f"Saved mask texture image to: {save_img_path}")

        # never buquanyong
        mask = (max_cos_map < 0.0001)  # shape [H, W, C]，与 tex_to_save 相同
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)  # 确保是 [H, W, 1]
        mask = mask.expand_as(tex_to_save)  # 扩展到 [H, W, 3]
        # tex_to_save[mask] = 0.0  # 置黑
        # never buquanyong end

        self.uvp.save_mesh(f"{self.result_dir}/textured.obj", tex_to_save)
        # self.uvp.save_mesh(f"{self.result_dir}/textured.obj",
        #                    result_tex_rgb.permute(1, 2, 0))

        # set origin texture
        print('result_tex_rgb_first is: ', result_tex_rgb.shape)
        self.uvp_rgb.set_texture_map(result_tex_rgb)
        original_textured_views = self.uvp_rgb.render_textured_views()
        original_textured_views_rgb = torch.cat(
            original_textured_views, axis=-1)[:-1, ...]
        original_textured_views_rgb = original_textured_views_rgb.permute(
            1, 2, 0).cpu().numpy()[None, ...]
        origin_v = numpy_to_pil(original_textured_views_rgb)[0]
        origin_save_path = f"{self.result_dir}/original_textured_views_rgb.jpg"
        origin_v.save(origin_save_path)
        print('result_tex_rgb_first is: ', result_tex_rgb.shape)


        # SpatialAware3DInpainting + 可见性map
        # s3i = SpatialAware3DInpainting(mesh, self._execution_device, max_cos_map)
        # result_tex_rgb, postion_map, red_vis_tex = s3i(result_tex_rgb)

        # # max_cos map + avg_cos map + part_id + ComponentAware3DInpainting
    
        # max_cos map + part_id + Spatial+ComponentAware3DInpainting
        # c3i = ComponentSparse3DInpainting(mesh, self._execution_device, max_cos_map, self.face2label_path)
        # result_tex_rgb, postion_map, red_mask = c3i(result_tex_rgb)



        # # === 🟥 红色可视化版本 ===
        # c3i = ComponentAware3DInpainting(mesh, self._execution_device, max_cos_map, avg_cos_map, self.face2label_path)
        # result_tex_rgb, postion_map, red_mask = c3i(result_tex_rgb)
        # inpaint_dir = f"{self.result_dir}/after_inpainting"
        # # 如果目录不存在，就创建
        # os.makedirs(inpaint_dir, exist_ok=True)

        # #  保存红色纹理为 mesh
        # print('result_tex_rgb is: ', result_tex_rgb.shape)
        # # print('red_vis_tex is: ', red_vis_tex.shape)
        # red_mask_rgb = np.stack([red_mask]*3, axis=-1)  # (H, W, 3)
        # red_vis_tex = torch.from_numpy(red_mask_rgb).float().to(self._execution_device)
        # self.uvp.save_mesh(f"{inpaint_dir}/textured_red.obj", red_vis_tex)

        # # 设置纹理为红色的
        # self.uvp_rgb.set_texture_map(red_vis_tex.permute(2, 0, 1))
        # red_textured_views = self.uvp_rgb.render_textured_views()
        # red_textured_views_rgb = torch.cat(red_textured_views, axis=-1)[:-1, ...]
        # red_textured_views_rgb = red_textured_views_rgb.permute(1, 2, 0).cpu().numpy()[None, ...]
        # red_v = numpy_to_pil(red_textured_views_rgb)[0]
        # red_v.save(f"{inpaint_dir}/inpaint_textured_views_red.jpg")

        # print("✅ 已保存红色可视化 mesh 与渲染图：",
        #     f"{inpaint_dir}/textured_red.obj",
        #     f"{inpaint_dir}/inpaint_textured_views_red.jpg")
        # # === 🟥 红色可视化版本 end ===


        # === 原始 inpaint 保存 ===
        # self.uvp.save_mesh(f"{self.result_dir}/after_inpainting/textured.obj", result_tex_rgb)

        # self.uvp_rgb.set_texture_map(result_tex_rgb.permute(2, 0, 1))
        # # print('result_tex_rgb_after is: ', result_tex_rgb.shape)
        # inpaint_textured_views = self.uvp_rgb.render_textured_views()
        # inpaint_textured_views_rgb = torch.cat(inpaint_textured_views, axis=-1)[:-1, ...]
        # inpaint_textured_views_rgb = inpaint_textured_views_rgb.permute(
        #     1, 2, 0).cpu().numpy()[None, ...]
        # inpaint_v = numpy_to_pil(inpaint_textured_views_rgb)[0]
        # # print('diaomao, save textured views')

        # save_path = f"{self.result_dir}/after_inpainting/inpaint_textured_views_rgb.jpg"
        # print("save to: ", save_path)
        # os.makedirs(self.result_dir, exist_ok=True)
        # inpaint_v.save(save_path)
        # print("saved successfully?", os.path.exists(save_path))

        # === 原始 inpaint end ===

        # display(v)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        self.uvp.to("cpu")
        self.uvp_rgb.to("cpu")

        # return result_tex_rgb, inpaint_textured_views, inpaint_v
        return result_tex_rgb, None, None
