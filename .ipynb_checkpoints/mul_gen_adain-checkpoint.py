import os
import json
import yaml
from pathlib import Path

# === 用户可配置路径 ===
mesh_dir = Path("/root/autodl-tmp/part-aware-text-guided-texture/lact_nvs_exp/lact_nvs_exp_glb")
img_dir = Path("/root/autodl-tmp/part-aware-text-guided-texture/lact_nvs_exp/lact_nvs_exp_glb_segment_render")
prompt_dir = Path("/root/autodl-tmp/part-aware-text-guided-texture/lact_nvs_exp/lact_nvs_exp_prompts")

# 生成的 yaml 存放目录
output_dir = Path("config/adain_yizhi/exp")
output_dir.mkdir(parents=True, exist_ok=True)

# face2label 映射 JSON 路径
face2label_json = "/root/autodl-tmp/part-aware-text-guided-texture/lact_nvs_exp/lact_nvs_exp_face2labels.json"

# === 其它超参数 ===
seeds = [1, 3]  # 每个 obj 的 seed 数量
gpu_num = 4     # (脚本中暂未用到，保留变量)

# ref_attention_end 消融
ref_attention_end_values = [0.2]

# guidance_scale 消融
guidance_scales = [15.5]

# 遍历 obj —— 根据 img_dir 生成
all_obj_ids = [p.name for p in img_dir.iterdir() if p.is_dir()]
selected_ids = all_obj_ids

# 读取 face2label 映射
if not Path(face2label_json).exists():
    print(f"[WARN] face2label JSON 文件不存在：{face2label_json} ，脚本将继续但所有 face2label_path 会为空字符串。")
    face2label_map = {}
else:
    with open(face2label_json, "r", encoding="utf-8") as f:
        face2label_map = json.load(f)

configs = []

# === 基础配置 (Common Config) ===
base_config = {
    "normal_controlnet_path": "lllyasviel/control_v11p_sd15_normalbae",
    "depth_controlnet_path": "lllyasviel/control_v11f1p_sd15_depth",
    "segment_controlnet_path": "/mnt/lab/data/zhangtianle/train_controlnet/PartObjaverse-Tiny_mesh_checkpoint/20250627_175533/checkpoint-17000/controlnet",
    "stable_diffusion_path": "/mnt/lab/zhangtianle/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/f03de327dd89b501a01da37fc5240cf4fdba85a1",
    
    "mesh_config_relative": True,
    "use_mesh_name": False,
    "mesh_scale": 2.0,
    "keep_mesh_uv": False,
    
    "output": "./exp/global_average_exp/result", # 结果输出路径
    "prefix": "MVD",
    "timeformat": "%d%b%Y-%H%M%S",
    
    "steps": 30,
    "cond_type": "depth",
    "guess_mode": False,
    "conditioning_scale": 0.7,
    "conditioning_scale_end": 0.9,
    "control_guidance_start": 0.0,
    "control_guidance_end": 0.99,
    "guidance_rescale": 0.0,
    
    "camera_azims": [0, 60, 120, 180, 240, 300],
    "no_top_cameras": False,
    
    "latent_view_size": 96,
    "latent_tex_size": 512,
    "rgb_view_size": 1024,
    "rgb_tex_size": 1024,
    
    "mvd_end": 0.8,
    "mvd_exp_start": 0.0,
    "mvd_exp_end": 6.0,
    "shuffle_bg_change": 0.4,
    "shuffle_bg_end": 0.8,
    
    "log_interval": 10,
    "tex_fast_preview": True,
    "view_fast_preview": True,

    # 默认值，会被实验组覆盖
    "early_use_ref": False,
    "early_ref_weight": 0.0,
    "late_use_ref": False,
    "late_ref_weight": 0.0,
    "use_semantic_anchor": False,
    "anchor_weight": 0.5,
}

# === 定义 6 组实验配置 ===
# 这里定义每个实验组特有的参数变化
experiment_groups = [
    # (1) Baseline: Adj only, No Yizhi
    {
        "name_suffix": "base_noyizhi",
        "params": {
            "late_use_adjacent_baseline": True,
            "late_use_adjacent_segment": False,
            "early_use_adjacent_baseline": True,
            "early_use_adjacent_segment": False,
            "early_semgent_weight": 0.0, # Baseline 不需要 segment weight，设为0或默认
            "late_segment_weight": 0.0,
            "yizhi": False,
            "yizhi_start": 0.0, # 默认值
            "yizhi_end": 1.0    # 默认值
        }
    },
    # (2) Segment: Seg only, W=3.0, No Yizhi
    {
        "name_suffix": "seg_w3_noyizhi",
        "params": {
            "late_use_adjacent_baseline": False,
            "late_use_adjacent_segment": True,
            "early_use_adjacent_baseline": False,
            "early_use_adjacent_segment": True,
            "early_semgent_weight": 3.0,
            "late_segment_weight": 3.0,
            "yizhi": False,
            "yizhi_start": 0.0,
            "yizhi_end": 1.0
        }
    },
    # (3) Baseline + Yizhi (0.3-0.7), W=3.0 (Weight在这里作为抑制的基准或者衰减目标)
    {
        "name_suffix": "base_w3_yizhi0307",
        "params": {
            "late_use_adjacent_baseline": True,
            "late_use_adjacent_segment": False,
            "early_use_adjacent_baseline": True,
            "early_use_adjacent_segment": False,
            "early_semgent_weight": 3.0,
            "late_segment_weight": 3.0,
            "yizhi": True,
            "yizhi_start": 0.3,
            "yizhi_end": 0.7
        }
    },
    # (4) Segment + Yizhi (0.3-0.7), W=3.0
    {
        "name_suffix": "seg_w3_yizhi0307",
        "params": {
            "late_use_adjacent_baseline": False,
            "late_use_adjacent_segment": True,
            "early_use_adjacent_baseline": False,
            "early_use_adjacent_segment": True,
            "early_semgent_weight": 3.0,
            "late_segment_weight": 3.0,
            "yizhi": True,
            "yizhi_start": 0.3,
            "yizhi_end": 0.7
        }
    },
    # (5) Baseline + Yizhi (0.7-1.0), W=5.0
    {
        "name_suffix": "base_w5_yizhi0710",
        "params": {
            "late_use_adjacent_baseline": True,
            "late_use_adjacent_segment": False,
            "early_use_adjacent_baseline": True,
            "early_use_adjacent_segment": False,
            "early_semgent_weight": 5.0,
            "late_segment_weight": 5.0,
            "yizhi": True,
            "yizhi_start": 0.7,
            "yizhi_end": 1.0
        }
    },
    # (6) Segment + Yizhi (0.7-1.0), W=5.0
    {
        "name_suffix": "seg_w5_yizhi0710",
        "params": {
            "late_use_adjacent_baseline": False,
            "late_use_adjacent_segment": True,
            "early_use_adjacent_baseline": False,
            "early_use_adjacent_segment": True,
            "early_semgent_weight": 5.0,
            "late_segment_weight": 5.0,
            "yizhi": True,
            "yizhi_start": 0.7,
            "yizhi_end": 1.0
        }
    },
]

# === 主循环生成 ===

for obj_idx, obj_id in enumerate(selected_ids):
    mesh_path = mesh_dir / obj_id / f"{obj_id}.obj"
    img_path = img_dir / obj_id / f"{obj_id}_segment_6views_modify_render_concat.png"
    prompt_path = prompt_dir / f"{obj_id}_prompt.json"

    # 读取 Prompt
    if not prompt_path.exists():
        print(f"[WARN] Prompt file missing: {prompt_path}, skipping.")
        continue
        
    with open(prompt_path, 'r') as f:
        prompt_data = yaml.safe_load(f)
    prompts = prompt_data.get("prompt", ["A 3D model"]) 

    # 获取 Face2Label
    raw_face_path = face2label_map.get(obj_id)
    face2label_path_for_config = ""
    if raw_face_path:
        candidate = Path(raw_face_path)
        if not candidate.exists():
            print(f"[WARN] face2label exists in map but file missing: {raw_face_path}")
            face2label_path_for_config = raw_face_path
        else:
            face2label_path_for_config = str(candidate.resolve())
    else:
        # print(f"[WARN] No face2label map for {obj_id}")
        pass

    # GPU 分配 (默认0)
    gpu_id = obj_idx % 4

    for seed in seeds:
        for ref_attention_end in ref_attention_end_values:
            ref_str = str(ref_attention_end).replace('.', 'p')
            
            for guidance_scale in guidance_scales:
                gs_str = str(guidance_scale).replace('.', 'p')

                for prompt_idx, prompt_text in enumerate(prompts, start=1):
                    # 如果只有一个 prompt，可以简化 tag
                    prompt_tag = f"prompt{prompt_idx}"
                    
                    # === 遍历 6 组实验 ===
                    for exp in experiment_groups:
                        exp_suffix = exp["name_suffix"]
                        exp_params = exp["params"]
                        
                        # 构建文件名
                        # 格式: {obj}_s{seed}_{suffix}_ref{ref}_gs{gs}.yaml
                        cfg_name = f"{obj_id}_seed{seed}_{exp_suffix}_ref{ref_str}_gs{gs_str}.yaml"
                        cfg_path = output_dir / cfg_name
                        
                        # 构建最终 Config
                        current_config = base_config.copy()
                        
                        # 1. 填入 OBJ 相关动态路径
                        current_config.update({
                            "mesh": str(mesh_path),
                            "segment_img_path": str(img_path),
                            "prompt": prompt_text,
                            "face2label_path": face2label_path_for_config,
                        })
                        
                        # 2. 填入 循环变量
                        current_config.update({
                            "seed": seed,
                            "gpu_id": gpu_id,
                            "guidance_scale": guidance_scale,
                            "ref_attention_end": ref_attention_end,
                        })
                        
                        # 3. 填入 实验组特定参数 (覆盖前面的默认值)
                        current_config.update(exp_params)
                        
                        # 写入文件
                        with open(cfg_path, 'w') as f:
                            yaml.dump(current_config, f, sort_keys=False)
                        
                        configs.append(cfg_path)

print(f"总共生成 {len(configs)} 个 config 文件，保存在 {output_dir}")