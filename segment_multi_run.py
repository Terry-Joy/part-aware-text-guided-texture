import os
from os.path import join, isdir, abspath, dirname, basename, splitext
from IPython.display import display
from datetime import datetime
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDPMScheduler
# pipeline2 测试
from src.pipeline2 import StableSyncMVDPipeline
import yaml
import argparse
import shutil


def parse_config(config_path):
    """解析YAML配置文件并返回配置字典"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Run Multi-View Diffusion experiment')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    args = parser.parse_args()
    
    # 解析配置文件
    opt = parse_config(args.config)
    
    # 处理网格文件路径
    if opt.get('mesh_config_relative', False):
        mesh_path = join(dirname(args.config), opt['mesh'])
    else:
        mesh_path = abspath(opt['mesh'])
    
    # 设置输出目录
    output_root = opt.get('output', dirname(args.config))
    output_name_components = []
    
    if 'prefix' in opt and opt['prefix']:
        output_name_components.append(opt['prefix'])
    
    if opt.get('use_mesh_name', False):
        mesh_name = splitext(basename(mesh_path))[0].replace(" ", "_")
        output_name_components.append(mesh_name)
    
    timeformat = opt.get('timeformat', '%d%b%Y-%H%M%S')
    if timeformat:
        output_name_components.append(datetime.now().strftime(timeformat))
    
    output_name = "_".join(output_name_components) or "output"
    output_dir = join(output_root, output_name)
    
    if not isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    else:
        print(f"Output directory already exists: {output_dir}. Using time string to avoid conflict.")
        output_name = f"{output_name}_{datetime.now().strftime('%H%M%S')}"
        output_dir = join(output_root, output_name)
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving results to: {output_dir}")
    
    # 保存使用的配置文件
    shutil.copy(args.config, join(output_dir, "config.yaml"))
    
    # 设置日志配置
    logging_config = {
        "output_dir": output_dir,
        "log_interval": opt.get('log_interval', 10),
        "view_fast_preview": opt.get('view_fast_preview', True),
        "tex_fast_preview": opt.get('tex_fast_preview', True),
    }
    
    # 加载ControlNet模型
    cond_type = opt.get('cond_type', 'depth').lower()
    if cond_type == "normal":
        model_path = opt.get('normal_controlnet_path', "lllyasviel/control_v11p_sd15_normalbae")
        controlnet = ControlNetModel.from_pretrained(
            model_path, 
            variant="fp16", 
            torch_dtype=torch.float16
        )
    elif cond_type == "depth":
        model_path = opt.get('depth_controlnet_path', "lllyasviel/control_v11f1p_sd15_depth")
        controlnet = ControlNetModel.from_pretrained(
            model_path, 
            variant="fp16", 
            torch_dtype=torch.float16
        )
    elif cond_type == "segment":
        model_path = opt.get('segment_controlnet_path', "/mnt/lab/data/zhangtianle/train_controlnet/PartObjaverse-Tiny_mesh_checkpoint/20250627_175533/checkpoint-17000/controlnet")
        controlnet = ControlNetModel.from_pretrained(
            model_path, 
            variant="fp16", 
            torch_dtype=torch.float16
        )
    else:
        raise ValueError(f"Unsupported condition type: {cond_type}. Supported types: 'normal', 'depth'")
    
    # 创建Stable Diffusion管道
    sd_path = opt.get('stable_diffusion_path', "runwayml/stable-diffusion-v1-5")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        sd_path, 
        controlnet=controlnet, 
        torch_dtype=torch.float16
    )
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    # pipe = pipe.to('cuda:1')
    # 创建多视图扩散管道
    syncmvd = StableSyncMVDPipeline(**pipe.components)
    
    # 运行生成过程
    result_tex_rgb, textured_views, v = syncmvd(
        prompt=opt['prompt'],
        height=opt.get('latent_view_size', 128) * 8,
        width=opt.get('latent_view_size', 128) * 8,
        num_inference_steps=opt.get('steps', 30),
        guidance_scale=opt.get('guidance_scale', 15.5),
        negative_prompt=opt.get('negative_prompt', 'oversmoothed, blurry, depth of field, out of focus, low quality, bloom, glowing effect.'),
        generator=torch.manual_seed(opt.get('seed', 0)),
        max_batch_size=48,
        controlnet_guess_mode=opt.get('guess_mode', False),
        controlnet_conditioning_scale=opt.get('conditioning_scale', 0.7),
        controlnet_conditioning_end_scale=opt.get('conditioning_scale_end', 0.9),
        control_guidance_start=opt.get('control_guidance_start', 0.0),
        control_guidance_end=opt.get('control_guidance_end', 0.99),
        guidance_rescale=opt.get('guidance_rescale', 0.0),
        use_directional_prompt=True,
        mesh_path=mesh_path,
        mesh_transform={"scale": opt.get('mesh_scale', 2.0)},
        mesh_autouv=not opt.get('keep_mesh_uv', False),
        camera_azims=opt.get('camera_azims', [0, 90, 180, 270]),
        top_cameras=not opt.get('no_top_cameras', False),
        texture_size=opt.get('latent_tex_size', 512),
        render_rgb_size=opt.get('rgb_view_size', 1536),
        texture_rgb_size=opt.get('rgb_tex_size', 1024),
        multiview_diffusion_end=opt.get('mvd_end', 0.8),
        exp_start=opt.get('mvd_exp_start', 0.0),
        exp_end=opt.get('mvd_exp_end', 6.0),
        ref_attention_end=opt.get('ref_attention_end', 0.2),
        shuffle_background_change=opt.get('shuffle_bg_change', 0.4),
        shuffle_background_end=opt.get('shuffle_bg_end', 0.8),
        logging_config=logging_config,
        cond_type=cond_type,
        img_path=opt.get('img_path', ""),
        early_use_adjacent_baseline=opt.get('early_use_adjacent_baseline', True),
        early_use_adjacent_segment=opt.get('early_use_adjacent_segment', False),
        early_use_ref=opt.get('early_use_ref', True),
        early_semgent_weight=opt.get('early_semgent_weight', -1.0),
        early_ref_weight=opt.get('early_ref_weight', 1.0),

        late_use_adjacent_baseline=opt.get('late_use_adjacent_baseline', False),
        late_use_adjacent_segment=opt.get('late_use_adjacent_segment', False),
        late_use_ref=opt.get('late_use_ref', False),
        segment_img_path=opt.get('segment_img_path', ""),
        late_segment_weight=opt.get('late_segment_weight', -0.8),
        late_ref_weight=opt.get('late_ref_weight', 1.0),
    )
    
    # 显示结果
    display(v)

if __name__ == "__main__":
    main()