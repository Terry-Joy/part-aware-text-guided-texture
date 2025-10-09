import os
import yaml
from pathlib import Path

# === 用户可配置路径 ===
mesh_dir = Path("/home/zhangtianle/text-to-texture/MVPaint/syncmvd_exp_glb")
img_dir = Path("/home/zhangtianle/text-to-texture/MVPaint/syncmvd_seg_exp/syncmvd_ablation_1/syncmvd_ablation_1_img")
prompt_dir = Path("/home/zhangtianle/text-to-texture/MVPaint/syncmvd_seg_exp/syncmvd_ablation_1/syncmvd_ablation_1_prompt")
output_dir = Path("config/ablation_1")
output_dir.mkdir(exist_ok=True)

seeds = [1]  # 每个 obj 的 seed 数量
gpu_num = 4  # GPU 数量

# 早晚段 segment 权重消融
segment_weights = [-0.8, -0.5, 0.0, 0.5, 0.8]

# ref_weight 消融
ref_weights = [1.0]

# adj_baseline 开关
adj_baseline_on = True  # 如果开，则早晚都开；否则早晚都关

# ref_attention_end 消融（默认值先）
ref_attention_end_values = [0.2]

# guidance_scale 消融（默认值先）
guidance_scales = [15.5]

# 遍历 obj —— 根据 img_dir 生成
all_obj_ids = [p.name for p in img_dir.iterdir() if p.is_dir()]
selected_ids = all_obj_ids

configs = []

# === 基础字段（不会变的全局配置） ===
base_config = {
    # 模型路径配置
    "normal_controlnet_path": "lllyasviel/control_v11p_sd15_normalbae",
    "depth_controlnet_path": "lllyasviel/control_v11f1p_sd15_depth",
    "segment_controlnet_path": "/mnt/lab/data/zhangtianle/train_controlnet/PartObjaverse-Tiny_mesh_checkpoint/20250627_175533/checkpoint-17000/controlnet",
    "stable_diffusion_path": "/home/zhangtianle/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/f03de327dd89b501a01da37fc5240cf4fdba85a1",

    # 网格设置
    "mesh_config_relative": True,
    "use_mesh_name": False,
    "mesh_scale": 2.0,
    "keep_mesh_uv": False,

    # 输出设置
    "output": "./exp/ablation_1/",
    "prefix": "MVD",
    "timeformat": "%d%b%Y-%H%M%S",

    # 扩散设置
    "steps": 30,

    # ControlNet 设置
    "cond_type": "depth",
    "guess_mode": False,
    "conditioning_scale": 0.7,
    "conditioning_scale_end": 0.9,
    "control_guidance_start": 0.0,
    "control_guidance_end": 0.99,
    "guidance_rescale": 0.0,

    # 视图设置
    "camera_azims": [0, 45, 90, 180, 225, 270],
    "no_top_cameras": False,
    "latent_view_size": 96,
    "latent_tex_size": 512,
    "rgb_view_size": 1024,
    "rgb_tex_size": 1024,

    # 多视图扩散设置
    "mvd_end": 0.8,
    "mvd_exp_start": 0.0,
    "mvd_exp_end": 6.0,
    "shuffle_bg_change": 0.4,
    "shuffle_bg_end": 0.8,

    # 日志设置
    "log_interval": 10,
    "tex_fast_preview": True,
    "view_fast_preview": True,
}

for obj_idx, obj_id in enumerate(selected_ids):
    mesh_path = mesh_dir / obj_id / f"{obj_id}.obj"
    img_path = img_dir / obj_id / f"{obj_id}_segment_6views_modify_render_concat.png"
    prompt_path = prompt_dir / f"{obj_id}_prompt.json"

    with open(prompt_path, 'r') as f:
        prompt_data = yaml.safe_load(f)
    prompts = prompt_data.get("prompt", ["A 3D model"])  # 这里可能是列表

    gpu_id = obj_idx % gpu_num

    for seed in seeds:
        for ref_attention_end in ref_attention_end_values:
            ref_str = str(ref_attention_end).replace('.', 'p')
            for guidance_scale in guidance_scales:
                gs_str = str(guidance_scale).replace('.', 'p')

                for prompt_idx, prompt_text in enumerate(prompts, start=1):
                    prompt_tag = f"prompt{prompt_idx}"

                    # 1. segment_only
                    for early_w in segment_weights:
                        for late_w in segment_weights:
                            cfg_name = f"{obj_id}_seed{seed}_seg_e{early_w}_l{late_w}_ref{ref_str}_gs{gs_str}_{prompt_tag}.yaml"
                            cfg_path = output_dir / cfg_name
                            config = {
                                **base_config,
                                "mesh": str(mesh_path),
                                "segment_img_path": str(img_path),
                                "prompt": prompt_text,  # ✅ 不加额外引号
                                "seed": seed,
                                "gpu_id": gpu_id,
                                "guidance_scale": guidance_scale,
                                "ref_attention_end": ref_attention_end,
                                "early_use_adjacent_baseline": False,
                                "early_use_adjacent_segment": True,
                                "early_use_ref": False,
                                "early_semgent_weight": early_w,
                                "early_ref_weight": 0.0,
                                "late_use_adjacent_baseline": False,
                                "late_use_adjacent_segment": True,
                                "late_use_ref": False,
                                "late_segment_weight": late_w,
                                "late_ref_weight": 0.0
                            }
                            with open(cfg_path, 'w') as f:
                                yaml.dump(config, f, sort_keys=False)
                            configs.append(cfg_path)

                    # 2. ref_only
                    for ref_w in ref_weights:
                        cfg_name = f"{obj_id}_seed{seed}_ref{int(ref_w*10)}_ref{ref_str}_gs{gs_str}_{prompt_tag}.yaml"
                        cfg_path = output_dir / cfg_name
                        config = {
                            **base_config,
                            "mesh": str(mesh_path),
                            "segment_img_path": str(img_path),
                            "prompt": prompt_text,
                            "seed": seed,
                            "gpu_id": gpu_id,
                            "guidance_scale": guidance_scale,
                            "ref_attention_end": ref_attention_end,
                            "early_use_adjacent_baseline": False,
                            "early_use_adjacent_segment": False,
                            "early_use_ref": True,
                            "early_semgent_weight": 0.0,
                            "early_ref_weight": ref_w,
                            "late_use_adjacent_baseline": False,
                            "late_use_adjacent_segment": False,
                            "late_use_ref": True,
                            "late_segment_weight": 0.0,
                            "late_ref_weight": ref_w
                        }
                        with open(cfg_path, 'w') as f:
                            yaml.dump(config, f, sort_keys=False)
                        configs.append(cfg_path)

                    # 3. adj_baseline_only
                    adj_on = adj_baseline_on
                    cfg_name = f"{obj_id}_seed{seed}_adj_ref{ref_str}_gs{gs_str}_{prompt_tag}.yaml"
                    cfg_path = output_dir / cfg_name
                    config = {
                        **base_config,
                        "mesh": str(mesh_path),
                        "segment_img_path": str(img_path),
                        "prompt": prompt_text,
                        "seed": seed,
                        "gpu_id": gpu_id,
                        "guidance_scale": guidance_scale,
                        "ref_attention_end": ref_attention_end,
                        "early_use_adjacent_baseline": adj_on,
                        "early_use_adjacent_segment": False,
                        "early_use_ref": False,
                        "early_semgent_weight": 0.0,
                        "early_ref_weight": 0.0,
                        "late_use_adjacent_baseline": adj_on,
                        "late_use_adjacent_segment": False,
                        "late_use_ref": False,
                        "late_segment_weight": 0.0,
                        "late_ref_weight": 0.0
                    }
                    with open(cfg_path, 'w') as f:
                        yaml.dump(config, f, sort_keys=False)
                    configs.append(cfg_path)

                    # 4. all_off
                    cfg_name = f"{obj_id}_seed{seed}_off_ref{ref_str}_gs{gs_str}_{prompt_tag}.yaml"
                    cfg_path = output_dir / cfg_name
                    config = {
                        **base_config,
                        "mesh": str(mesh_path),
                        "segment_img_path": str(img_path),
                        "prompt": prompt_text,
                        "seed": seed,
                        "gpu_id": gpu_id,
                        "guidance_scale": guidance_scale,
                        "ref_attention_end": ref_attention_end,
                        "early_use_adjacent_baseline": False,
                        "early_use_adjacent_segment": False,
                        "early_use_ref": False,
                        "early_semgent_weight": 0.0,
                        "early_ref_weight": 0.0,
                        "late_use_adjacent_baseline": False,
                        "late_use_adjacent_segment": False,
                        "late_use_ref": False,
                        "late_segment_weight": 0.0,
                        "late_ref_weight": 0.0
                    }
                    with open(cfg_path, 'w') as f:
                        yaml.dump(config, f, sort_keys=False)
                    configs.append(cfg_path)

print(f"总共生成 {len(configs)} 个 config 文件，保存在 {output_dir}")
