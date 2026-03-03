import os
import yaml

# 要修改的目录
config_dir = "config/ablation_1_reshalf"

# 指定要写入的路径
new_sd_path = "/root/autodl-tmp/checkpoint/f03de327dd89b501a01da37fc5240cf4fdba85a1"  # 可改成你想要的

for fname in os.listdir(config_dir):
    if not fname.endswith(".yaml"):
        continue

    fpath = os.path.join(config_dir, fname)
    with open(fpath, "r") as f:
        data = yaml.safe_load(f)

    # 如果 key 不存在，也创建它
    old_path = data.get("stable_diffusion_path", None)
    data["stable_diffusion_path"] = new_sd_path

    if old_path != new_sd_path:
        print(f"✅ Updated {fname}: {old_path} -> {new_sd_path}")
    else:
        print(f"⏩ Skipped (already set): {fname}")

    # 写回 YAML
    with open(fpath, "w") as f:
        yaml.dump(data, f, sort_keys=False)

print("🎯 All done!")
