import os
import yaml

# 需要修改的目录
root_dir = "./config/ablation_1_reshalf"  # ← 改成你的yaml所在目录
new_prefix = "/root/autodl-tmp/part-aware-text-guided-texture"

for dirpath, _, filenames in os.walk(root_dir):
    for fname in filenames:
        if not fname.endswith(".yaml"):
            continue
        fpath = os.path.join(dirpath, fname)

        with open(fpath, "r") as f:
            try:
                data = yaml.safe_load(f)
            except Exception as e:
                print(f"[Error] Failed to read {fpath}: {e}")
                continue

        modified = False

        # --- 修改 mesh ---
        if "mesh" in data and isinstance(data["mesh"], str):
            parts = data["mesh"].split("/syncmvd_")
            if len(parts) > 1:
                data["mesh"] = f"{new_prefix}/syncmvd_{parts[1]}"
                modified = True

        # --- 修改 segment_img_path ---
        if "segment_img_path" in data and isinstance(data["segment_img_path"], str):
            parts = data["segment_img_path"].split("/syncmvd_")
            if len(parts) > 1:
                data["segment_img_path"] = f"{new_prefix}/syncmvd_{parts[1]}"
                modified = True

        if modified:
            with open(fpath, "w") as f:
                yaml.dump(data, f, sort_keys=False, allow_unicode=True)
            print(f"[Updated] {fpath}")
        else:
            print(f"[Skip] {fpath} (no change)")
