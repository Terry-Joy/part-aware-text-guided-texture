import os
import yaml

# ✅ 配置目录
config_dir = "config/ablation_1_reshalf"

# ✅ 替换路径模板
# 其中 {id} 会被替换成提取的ID
segment_path_template = (
    "/root/autodl-tmp/part-aware-text-guided-texture/"
    "syncmvd_seg_exp/syncmvd_ablation_1/syncmvd_ablation_1_img/"
    "{id}/{id}_segment_6views_modify_render_concat.png"
)

for fname in os.listdir(config_dir):
    if not fname.endswith(".yaml"):
        continue

    fpath = os.path.join(config_dir, fname)

    # 从文件名提取ID（即第一个'_'前的部分）
    id_ = fname.split("_")[0]

    with open(fpath, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        print(f"⚠️ 跳过：{fname} 格式异常")
        continue

    # 替换 segment_img_path
    new_path = segment_path_template.format(id=id_)
    data["segment_img_path"] = new_path

    # 写回文件
    with open(fpath, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"✅ 已修改 {fname} ：{new_path}")
