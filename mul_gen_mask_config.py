import os
import yaml
from pathlib import Path

BASE_YAML = "config/test_mul_depth_config.yaml"  # 你的模板 YAML
MESH_DIR = "/home/zhangtianle/text-to-texture/MVPaint/syncmvd_exp_glb"       # dir/id/id.obj
OUTPUT_YAML_DIR = "config/multi_gen_mask"

os.makedirs(OUTPUT_YAML_DIR, exist_ok=True)

# 获取所有 id
id_dirs = [d for d in Path(MESH_DIR).iterdir() if d.is_dir()]
id_dirs.sort()

# GPU 分配规则，例如按数量循环分配
num_gpus = 4  # 例如 0,1,2,3
for i, id_dir in enumerate(id_dirs):
    id_name = id_dir.name
    mesh_path = str(id_dir / f"{id_name}.obj")
    gpu_id = i % num_gpus

    # 读取模板 YAML
    with open(BASE_YAML, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 修改 mesh 和 gpu_id
    config["mesh"] = mesh_path
    config["gpu_id"] = gpu_id
    config["output"] = "exp/mul_gen_mask"

    # 保存新的 YAML
    out_yaml = Path(OUTPUT_YAML_DIR) / f"{id_name}.yaml"
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    
    print(f"生成 YAML: {out_yaml}")
