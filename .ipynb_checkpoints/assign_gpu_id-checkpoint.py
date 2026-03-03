import os
import re
import yaml
from collections import defaultdict

# === 配置 ===
root_dir = "./config/ablation_1_reshalf"  # ← 改成你的yaml目录
gpu_list = [0, 1, 2, 3]

# === 1. 收集所有yaml路径 ===
yaml_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".yaml")]
if not yaml_files:
    print("❌ No YAML files found.")
    exit()

# === 2. 提取 model_id ===
pattern = re.compile(r"^([0-9a-fA-F]+)_")  # 匹配形如 49d0b382def54ff1b544cc6293ea1b22_ 开头
model_to_files = defaultdict(list)

for path in yaml_files:
    fname = os.path.basename(path)
    m = pattern.match(fname)
    if not m:
        print(f"[Skip] {fname}: cannot extract model id")
        continue
    model_id = m.group(1)
    model_to_files[model_id].append(path)

# === 3. 平均分配到GPU ===
unique_models = list(model_to_files.keys())
unique_models.sort()

gpu_assignment = {}
for i, model_id in enumerate(unique_models):
    gpu_assignment[model_id] = gpu_list[i % len(gpu_list)]

print("=== GPU assignment ===")
for m, g in gpu_assignment.items():
    print(f"{m} -> GPU {g}")

# === 4. 修改每个yaml中的 gpu_id ===
for model_id, paths in model_to_files.items():
    gpu_id = gpu_assignment[model_id]
    for yaml_path in paths:
        with open(yaml_path, "r") as f:
            try:
                data = yaml.safe_load(f)
            except Exception as e:
                print(f"[Error] Failed to load {yaml_path}: {e}")
                continue

        # 修改 gpu_id
        if "gpu_id" in data:
            data["gpu_id"] = gpu_id
        else:
            # 没有gpu_id字段就加一个
            data["gpu_id"] = gpu_id

        # 保存
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, sort_keys=False, allow_unicode=True)

        print(f"[Updated] {yaml_path} -> gpu_id: {gpu_id}")

print("✅ All YAML files updated.")
