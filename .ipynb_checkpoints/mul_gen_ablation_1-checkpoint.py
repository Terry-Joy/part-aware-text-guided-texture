import subprocess
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os

# 你要处理的 yaml 文件夹
YAML_DIR = "config/multi_gen_mask"
yaml_files = sorted(Path(YAML_DIR).glob("*.yaml"))

# 解析 YAML 并按 GPU 分组
yaml_groups = {}
for yml in yaml_files:
    with open(yml, "r") as f:
        cfg = yaml.safe_load(f)
    gpu_id = cfg.get("gpu_id", 0)
    yaml_groups.setdefault(gpu_id, []).append(yml)

print(f"Found {len(yaml_files)} YAMLs across {len(yaml_groups)} GPUs")


def worker(gpu_id, yml_list):
    for yml in yml_list:
        log_file = yml.with_suffix(".log")
        print(f"[GPU {gpu_id}] Running {yml} (logging -> {log_file})")

        with open(log_file, "w") as f:
            try:
                subprocess.run(
                    ["python", "single_gen_mask.py", "--config", str(yml)],
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
                )
            except subprocess.CalledProcessError as e:
                print(f"[GPU {gpu_id}] Error running {yml}: {e}")

        print(f"[GPU {gpu_id}] Finished {yml}")


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=len(yaml_groups)) as executor:
        for gpu_id, yml_list in yaml_groups.items():
            executor.submit(worker, gpu_id, yml_list)
