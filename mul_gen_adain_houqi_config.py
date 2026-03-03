import os
import json
import yaml
from pathlib import Path

# === 用户可配置路径 ===
mesh_dir = Path("/root/autodl-tmp/bishetest/")
img_dir = Path("/root/autodl-tmp/bishetest/")
prompt_dir = Path("/root/autodl-tmp/bishetest/")

# 生成的 yaml 存放目录
output_dir = Path("config/bishe_6views_ours_gd7_5/gd7_5_adain_yizhi_0_5_3_0_norefattn_no_adj_no_adain") # 改个名防止混淆
output_dir.mkdir(parents=True, exist_ok=True)

# face2label 映射 JSON 路径
face2label_json = "/root/autodl-tmp/part-aware-text-guided-texturelact_nvs_exp/lact_nvs_exp_face2labels.json"

# === 其它超参数 ===
seeds = [1, 3]  # 每个 obj 的 seed 数量
gpu_num = 8

# ref_attention_end 消融
ref_attention_end_values = [0.2]

# guidance_scale 消融
guidance_scales = [7.5]

# 遍历 obj
all_obj_ids = [p.name for p in img_dir.iterdir() if p.is_dir()]
selected_ids = all_obj_ids

# 读取 face2label 映射
if not Path(face2label_json).exists():
    print(f"[WARN] face2label JSON 文件不存在：{face2label_json} ，脚本将继续。")
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
    "stable_diffusion_path": "/root/autodl-tmp/checkpoint/f03de327dd89b501a01da37fc5240cf4fdba85a1",
    
    "mesh_config_relative": True,
    "use_mesh_name": False,
    "mesh_scale": 2.0,
    "keep_mesh_uv": False,
    
    "output": "./exp/gd7_5_adain_yizhi_0_5_3_0_norefattn_no_adj_no_adain/", 
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

    # 默认实验参数
    "early_use_ref": False,
    "early_ref_weight": 0.0,
    "late_use_ref": False,
    "late_ref_weight": 0.0,
    "use_semantic_anchor": False,
    "anchor_weight": 0.5,
    
    # === 新增 AdaIN 默认参数 ===
    "adain_first": 0,
    "adain_early": False,
    "adain_time": 0.3,
}

# === 定义 6 组基础实验配置 ===
experiment_groups = [
    # (1) Baseline
    {
        "name_suffix": "base_noyizhi",
        "params": {
            "late_use_adjacent_baseline": False,
            "late_use_adjacent_segment": False,
            "early_use_adjacent_baseline": False,
            "early_use_adjacent_segment": False,
            "early_segment_weight": 0.0,
            "late_segment_weight": 0.0,
            "yizhi": False,
            "yizhi_start": 0.0,
            "yizhi_end": 1.0
        }
    },
    # (2) Segment W=3.0 No Yizhi
    # {
    #     "name_suffix": "seg_w3_noyizhi",
    #     "params": {
    #         "late_use_adjacent_baseline": False,
    #         "late_use_adjacent_segment": True,
    #         "early_use_adjacent_baseline": False,
    #         "early_use_adjacent_segment": True,
    #         "early_segment_weight": 3.0,
    #         "late_segment_weight": 3.0,
    #         "yizhi": False,
    #         "yizhi_start": 0.0,
    #         "yizhi_end": 1.0
    #     }
    # },
    # (3) Baseline + Yizhi (0.3-0.7) W=2.5
    # {
    #     "name_suffix": "base_w3_yizhi0507",
    #     "params": {
    #         "late_use_adjacent_baseline": True,
    #         "late_use_adjacent_segment": False,
    #         "early_use_adjacent_baseline": True,
    #         "early_use_adjacent_segment": False,
    #         "early_segment_weight": 3.0,
    #         "late_segment_weight": 3.0,
    #         "yizhi": True,
    #         "yizhi_start": 0.5,
    #         "yizhi_end": 0.7
    #     }
    # },
    # (4) Segment + Yizhi (0.3-0.7) W=3
    # {
    #     "name_suffix": "seg_w3_yizhi0307",
    #     "params": {
    #         "late_use_adjacent_baseline": False,
    #         "late_use_adjacent_segment": True,
    #         "early_use_adjacent_baseline": False,
    #         "early_use_adjacent_segment": True,
    #         "early_segment_weight": 3.0,
    #         "late_segment_weight": 3.0,
    #         "yizhi": True,
    #         "yizhi_start": 0.3,
    #         "yizhi_end": 0.7
    #     }
    # },
    # (5) Baseline + Yizhi (0.7-1.0) W=5
    # {
    #     "name_suffix": "base_w5_yizhi0710",
    #     "params": {
    #         "late_use_adjacent_baseline": True,
    #         "late_use_adjacent_segment": False,
    #         "early_use_adjacent_baseline": True,
    #         "early_use_adjacent_segment": False,
    #         "early_segment_weight": 5.0,
    #         "late_segment_weight": 5.0,
    #         "yizhi": True,
    #         "yizhi_start": 0.7,
    #         "yizhi_end": 1.0
    #     }
    # },
    # (6) Segment + Yizhi (0.7-1.0) W=5
    # {
    #     "name_suffix": "seg_w5_yizhi0710",
    #     "params": {
    #         "late_use_adjacent_baseline": False,
    #         "late_use_adjacent_segment": True,
    #         "early_use_adjacent_baseline": False,
    #         "early_use_adjacent_segment": True,
    #         "early_segment_weight": 5.0,
    #         "late_segment_weight": 5.0,
    #         "yizhi": True,
    #         "yizhi_start": 0.7,
    #         "yizhi_end": 1.0
    #     }
    # },
]

# === 定义 AdaIN 变体 ===
adain_variations = [
    # 1. 原始设置 (不做 AdaIN 操作)
    # {
    #     "suffix": "no_adain",
    #     "params": {
    #         "adain_first": 0,
    #         "adain_early": False,
    #         "adain_time": 0.3
    #     }
    # },
    # 2. AdaIN First (只做第一步/初始化)
    {
        "suffix": "adain_first",
        "params": {
            "adain_first": 0,
            "adain_early": False,
            "adain_time": 0.3
        }
    },
    # 3. AdaIN Early (前期循环做)
    # {
    #     "suffix": "adain_early",
    #     "params": {
    #         "adain_first": True, # 通常 early 包含了 first，或者这两个参数是独立的
    #         "adain_early": True,
    #         "adain_time": 0.3
    #     }
    # }
        # 3. AdaIN Early (前期循环做)
    # {
    #     "suffix": "move_style_first",
    #     "params": {
    #         "adain_first": 2, # 通常 early 包含了 first，或者这两个参数是独立的
    #         "adain_early": False,
    #         "adain_time": 0.3
    #     }
    # },
    # {
    #     "suffix": "No_adain",
    #     "params": {
    #         "adain_first": 0, # 通常 early 包含了 first，或者这两个参数是独立的
    #         "adain_early": False,
    #         "adain_time": 0.3
    #     }
    # }
]

# === 主循环生成 ===

for obj_idx, obj_id in enumerate(selected_ids):
    mesh_path = mesh_dir / obj_id / "glb"/ f"{obj_id}.obj"
    img_path = img_dir / obj_id / "segment_6views_render"/ f"{obj_id}_segment_6views_modify_render_concat.png"
    prompt_path = prompt_dir / obj_id / f"{obj_id}_prompt.json"

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

    # GPU 分配
    gpu_id = obj_idx % gpu_num

    for seed in seeds:
        for ref_attention_end in ref_attention_end_values:
            ref_str = str(ref_attention_end).replace('.', 'p')
            
            for guidance_scale in guidance_scales:
                gs_str = str(guidance_scale).replace('.', 'p')

                for prompt_idx, prompt_text in enumerate(prompts, start=1):
                    # 如果只有一个 prompt，可以简化 tag
                    if prompt_idx == 2: 
                        continue
                    prompt_tag = f"prompt_{prompt_idx}"
                    
                    # === 遍历 6 组基础实验 ===
                    for exp in experiment_groups:
                        exp_suffix = exp["name_suffix"]
                        exp_params = exp["params"]
                        
                        # === 遍历 3 组 AdaIN 变体 ===
                        for adain_var in adain_variations:
                            adain_suffix = adain_var["suffix"]
                            adain_params = adain_var["params"]

                            # 构建文件名
                            # 格式: {obj}_s{seed}_{exp}_{adain}.yaml
                            cfg_name = f"{obj_id}_seed{seed}_{prompt_tag}_{exp_suffix}_{adain_suffix}.yaml"
                            cfg_path = output_dir / cfg_name
                            
                            # 构建最终 Config
                            current_config = base_config.copy()
                            
                            # === 关键修改在这里 ===
                            # 强制给 prompt 加上双引号
                            # 这样 yaml.dump 输出时，会自动在外面套上单引号，变成 '"..."' 的形式
                            formatted_prompt = f'"{prompt_text}"'

                            # 1. 基础动态路径
                            current_config.update({
                                "mesh": str(mesh_path),
                                "segment_img_path": str(img_path),
                                "prompt": formatted_prompt,  # 这里用处理过的 prompt
                                "face2label_path": face2label_path_for_config,
                            })
                            
                            # 2. 循环变量
                            current_config.update({
                                "seed": seed,
                                "gpu_id": gpu_id,
                                "guidance_scale": guidance_scale,
                                "ref_attention_end": ref_attention_end,
                            })
                            
                            # 3. 实验组参数
                            current_config.update(exp_params)
                            
                            # 4. AdaIN 参数
                            current_config.update(adain_params)
                            
                            # 写入文件
                            with open(cfg_path, 'w') as f:
                                yaml.dump(current_config, f, sort_keys=False)
                            
                            configs.append(cfg_path)

print(f"总共生成 {len(configs)} 个 config 文件，保存在 {output_dir}")