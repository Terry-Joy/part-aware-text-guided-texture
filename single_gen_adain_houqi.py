import os
from os.path import join, isdir, abspath, dirname, basename, splitext
from IPython.display import display
from datetime import datetime
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDPMScheduler
# pipeline2 测试
# from src.houqi_pipeline import StableSyncMVDPipeline
# from src.global_average_pipeline import StableSyncMVDPipeline
from src.houqi_vis_pipeline import StableSyncMVDPipeline
import yaml
import argparse
import shutil
# 告诉驱动：哪怕你很新，也别给我用 TF32，我要原本的精度！
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

# # 告诉驱动：别瞎选最快算法，选那个结果固定的算法！
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


def parse_config(config_path):
    """解析YAML配置文件并返回配置字典"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def make_experiment_dir(opt, mesh_path):
    from os.path import join, splitext, basename
    import os
    from datetime import datetime

    mesh_name = splitext(basename(mesh_path))[0]

    # --- 原有逻辑：构建早晚期模块列表 ---
    early_modules = []
    if opt.get('early_use_adjacent_segment', False) or opt.get('use_segment', False):
        early_modules.append('seg')
    if opt.get('early_use_ref', False) or opt.get('use_ref', False):
        early_modules.append('ref')
    if opt.get('early_use_adjacent_baseline', False) or opt.get('use_adjacent', False):
        early_modules.append('adj')

    late_modules = []
    if opt.get('late_use_adjacent_segment', False) or opt.get('use_segment', False):
        late_modules.append('seg')
    if opt.get('late_use_ref', False) or opt.get('use_ref', False):
        late_modules.append('ref')
    if opt.get('late_use_adjacent_baseline', False) or opt.get('use_adjacent', False):
        late_modules.append('adj')

    exp_parts = []
    if early_modules:
        exp_parts.append("early_" + "_".join(early_modules))
    if late_modules:
        exp_parts.append("late_" + "_".join(late_modules))
    exp_type = "_".join(exp_parts) if exp_parts else "none"

    # --- 原有逻辑：权重字符串 ---
    weight_vals = []
    for mod in early_modules:
        if mod == 'seg':
            weight_vals.append(str(opt.get('early_segment_weight', 'None')))
        elif mod == 'ref':
            weight_vals.append(str(opt.get('early_ref_weight', 'None')))
    for mod in late_modules:
        if mod == 'seg':
            weight_vals.append(str(opt.get('late_segment_weight', 'None')))
        elif mod == 'ref':
            weight_vals.append(str(opt.get('late_ref_weight', 'None')))

    weight_str = "_".join(weight_vals) if weight_vals else None

    # ref_attention_end
    ref_end = opt.get('ref_attention_end', None)
    denoise_time_str = f"denoise_time_{ref_end}" if ref_end is not None else None

    # seed
    seed_dir = f"seed_{opt.get('seed', 0)}"

    # 输出根目录
    output_root = opt.get('output', "exp/mul_gen_mask")

    # 时间戳
    timeformat = opt.get('timeformat', '%d%b%Y-%H%M%S')
    time_str = datetime.now().strftime(timeformat)

    # ==========================================
    # 组合路径 (修改部分在这里)
    # ==========================================
    path_parts = [output_root, mesh_name, exp_type]
    
    if denoise_time_str:
        path_parts.append(denoise_time_str)
    
    if weight_str:
        path_parts.append(weight_str)

    # --- [新增 1] Semantic Anchor 标识 ---
    if opt.get('use_semantic_anchor', False):
        # 格式示例: Anchor_w0.5
        anchor_w = opt.get('anchor_weight', 0.0)
        path_parts.append(f"Anchor_w{anchor_w}")

    # --- [新增 2] Yizhi (抑制) 标识 ---
    if opt.get('yizhi', False):
        # 格式示例: Yizhi_0.0_1.0
        y_start = opt.get('yizhi_start', 0.0)
        y_end = opt.get('yizhi_end', 1.0)
        path_parts.append(f"Yizhi_{y_start}_{y_end}")

    # 继续原有的路径组合
    path_parts.append(seed_dir)
    path_parts.append(time_str)
    
    guidance_str = f"gs_{opt.get('guidance_scale', 15.5)}"
    path_parts.append(guidance_str)

    output_dir = join(*path_parts)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Run Multi-View Diffusion experiment')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    args = parser.parse_args()
    
    # 解析配置文件
    opt = parse_config(args.config)
    gpu_id = opt.get('gpu_id', 0)
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
    output_dir = make_experiment_dir(opt, mesh_path)
    # join(output_root, output_name)
    
    # if not isdir(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    # else:
    #     print(f"Output directory already exists: {output_dir}. Using time string to avoid conflict.")
    #     output_name = f"{output_name}_{datetime.now().strftime('%H%M%S')}"
    #     output_dir = join(output_root, output_name)
    #     os.makedirs(output_dir, exist_ok=True)
    
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
    syncmvd = StableSyncMVDPipeline(**pipe.components, gpu_id=gpu_id)
    print('seed', opt.get('seed', 0))
    # 运行生成过程
    syncmvd(
        prompt=opt['prompt'],
        # negative_prompt=opt['negative_prompt'],
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
        early_segment_weight=opt.get('early_segment_weight', -1.0),
        early_ref_weight=opt.get('early_ref_weight', 1.0),

        late_use_adjacent_baseline=opt.get('late_use_adjacent_baseline', False),
        late_use_adjacent_segment=opt.get('late_use_adjacent_segment', False),
        late_use_ref=opt.get('late_use_ref', False),
        segment_img_path=opt.get('segment_img_path', ""),
        late_segment_weight=opt.get('late_segment_weight', -0.8),
        late_ref_weight=opt.get('late_ref_weight', 1.0),
        # 补全才有下面这个
        face2label_path=opt.get('face2label_path', ""),
        yizhi=opt.get('yizhi', False),
        yizhi_start=opt.get('yizhi_start', 0.0),
        yizhi_end=opt.get('yizhi_end', 1.0),
        
        adain_first=opt.get('adain_first', 0),
        adain_early=opt.get('adain_early', False),
        adain_time=opt.get('adain_time', 0.3),
    )
    
    # 显示结果
    # display(v)

if __name__ == "__main__":
    main()