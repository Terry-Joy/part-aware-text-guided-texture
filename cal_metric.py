import os
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- 工具函数 ----------------
def load_image_fill_alpha(path_or_img, size=(256,256), background_color=(255,255,255)):
    """读取 RGBA 图像或 PIL Image，透明区域填白底"""
    if isinstance(path_or_img, str):
        img = Image.open(path_or_img)
    else:
        img = path_or_img
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, background_color)
        bg.paste(img, mask=img.split()[3])
        img = bg
    else:
        img = img.convert("RGB")
    img = img.resize(size, Image.BICUBIC)
    tensor = torch.tensor(np.array(img)).permute(2,0,1).float() / 255.0
    return tensor

def split_horizontal_concat(img, n_views=6):
    """将水平拼接 1x6 的图拆成 6 张单视角图"""
    w, h = img.size
    view_w = w // n_views
    views = []
    for i in range(n_views):
        left = i * view_w
        right = left + view_w
        view = img.crop((left, 0, right, h))
        views.append(view)
    return views

def tensorize_image_list(img_list, size=(256,256)):
    """将 PIL 图片列表转换为 RGB tensor batch"""
    tensors = [load_image_fill_alpha(img, size=size) for img in img_list]
    return torch.stack(tensors).to(device)

def recursive_find_images(root_dir):
    return list(Path(root_dir).rglob("textured_views_rgb.jpg"))

def compute_clip_batch(model, preprocess, imgs_gt, imgs_gen):
    """CLIP similarity"""
    imgs_gt_proc = torch.stack([preprocess(img).to(device) for img in imgs_gt])
    imgs_gen_proc = torch.stack([preprocess(img).to(device) for img in imgs_gen])
    with torch.no_grad():
        gt_features = model.encode_image(imgs_gt_proc)
        gen_features = model.encode_image(imgs_gen_proc)
        gt_features /= gt_features.norm(dim=-1, keepdim=True)
        gen_features /= gen_features.norm(dim=-1, keepdim=True)
        scores = (gt_features * gen_features).sum(dim=-1)
    return scores.cpu().numpy()

# ---------------- 主函数 ----------------
def main(gt_dir, gen_dir, output_json="metrics_summary.json", resize_fid_kid=(299,299)):
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    results = {}
    all_attention_names = set()

    for model_id_dir in tqdm(sorted(Path(gen_dir).iterdir())):
        if not model_id_dir.is_dir(): continue
        model_id = model_id_dir.name
        results[model_id] = {}

        # ---------------- 读取 GT ----------------
        gt_path = Path(gt_dir) / model_id / f"{model_id}_origin_render_6views_concat.png"
        if not gt_path.exists():
            print(f"GT missing for {model_id}")
            continue
        gt_img = Image.open(gt_path)
        gt_views = split_horizontal_concat(gt_img, n_views=6)
        gt_fid_tensor = tensorize_image_list(gt_views, size=resize_fid_kid)
        gt_clip_tensor = gt_views  # 直接传 PIL Image 给 CLIP preprocess

        # ---------------- 遍历 attention 文件夹 ----------------
        for attn_dir in model_id_dir.iterdir():
            if not attn_dir.is_dir(): continue
            attn_type = attn_dir.name
            all_attention_names.add(attn_type)
            results[model_id][attn_type] = {}

            gen_images_paths = recursive_find_images(attn_dir)
            if len(gen_images_paths)==0: continue

            for img_path in gen_images_paths:
                relative_path = img_path.relative_to(model_id_dir)
                exp_name = "_".join(relative_path.parts[:-1])

                gen_img = Image.open(img_path)
                gen_views = split_horizontal_concat(gen_img, n_views=6)
                gen_fid_tensor = tensorize_image_list(gen_views, size=resize_fid_kid)
                gen_clip_tensor = gen_views

                # ---------------- FID ----------------
                fid_metric = FrechetInceptionDistance(feature=2048).to(device)
                gt_uint8 = (gt_fid_tensor.clamp(0,1)*255).to(torch.uint8)
                gen_uint8 = (gen_fid_tensor.clamp(0,1)*255).to(torch.uint8)
                fid_metric.update(gt_uint8, real=True)
                fid_metric.update(gen_uint8, real=False)
                fid = fid_metric.compute().item()

                # ---------------- KID ----------------
                kid_metric = KernelInceptionDistance(subset_size=6).to(device)
                kid_metric.update(gt_uint8, real=True)
                kid_metric.update(gen_uint8, real=False)
                kid_mean, kid_std = kid_metric.compute()
                kid = kid_mean.item()  # 取均值

                # ---------------- CLIP ----------------
                clip_score = compute_clip_batch(clip_model, clip_preprocess, gt_clip_tensor, gen_clip_tensor).mean()

                results[model_id][attn_type][exp_name] = {
                    "FID": fid,
                    "KID": kid,
                    "CLIP": float(clip_score)
                }

    # ---------------- 汇总 ----------------
    summary = {}
    attention_global_avg = {attn: {"FID":[],"KID":[],"CLIP":[]} for attn in all_attention_names}
    attention_global_best = {attn: {"FID":[],"KID":[],"CLIP":[]} for attn in all_attention_names}

    for model_id, attn_dict in results.items():
        summary[model_id] = {}
        for attn_type, exp_dict in attn_dict.items():
            metrics_array = np.array([[v["FID"], v["KID"], v["CLIP"]] for v in exp_dict.values()])
            exp_names = list(exp_dict.keys())
            sorted_idx = np.argsort(metrics_array[:,0])
            best_idx = sorted_idx[0]
            worst_idx = sorted_idx[-1]
            second_idx = sorted_idx[1] if len(sorted_idx)>1 else best_idx

            best_metrics = metrics_array[best_idx].tolist()
            mean_metrics = metrics_array.mean(axis=0).tolist()

            summary[model_id][attn_type] = {
                "best": {exp_names[best_idx]: exp_dict[exp_names[best_idx]]},
                "second": {exp_names[second_idx]: exp_dict[exp_names[second_idx]]},
                "worst": {exp_names[worst_idx]: exp_dict[exp_names[worst_idx]]},
                "mean": mean_metrics
            }

            attention_global_avg[attn_type]["FID"].append(mean_metrics[0])
            attention_global_avg[attn_type]["KID"].append(mean_metrics[1])
            attention_global_avg[attn_type]["CLIP"].append(mean_metrics[2])

            attention_global_best[attn_type]["FID"].append(best_metrics[0])
            attention_global_best[attn_type]["KID"].append(best_metrics[1])
            attention_global_best[attn_type]["CLIP"].append(best_metrics[2])

    # ---------------- 全局平均 ----------------
    attention_summary = {}
    for attn_type in all_attention_names:
        attention_summary[attn_type] = {
            "avg_of_mean_all_model_ids": {
                "FID": float(np.mean(attention_global_avg[attn_type]["FID"])),
                "KID": float(np.mean(attention_global_avg[attn_type]["KID"])),
                "CLIP": float(np.mean(attention_global_avg[attn_type]["CLIP"]))
            },
            "avg_of_best_all_model_ids": {
                "FID": float(np.mean(attention_global_best[attn_type]["FID"])),
                "KID": float(np.mean(attention_global_best[attn_type]["KID"])),
                "CLIP": float(np.mean(attention_global_best[attn_type]["CLIP"]))
            }
        }

    # ---------------- 保存 JSON ----------------
    with open(output_json,"w") as f:
        json.dump({"details": results, "summary": summary, "attention_global": attention_summary}, f, indent=2)
    print(f"Metrics saved to {output_json}")

# ---------------- 调用 ----------------
if __name__=="__main__":
    gt_dir = "/home/zhangtianle/text-to-texture/MVPaint/syncmvd_seg_exp/syncmvd_ablation_1/syncmvd_ablation_1_img"
    gen_dir = "exp/ablation_1"
    main(gt_dir, gen_dir)
