import subprocess
import yaml
from pathlib import Path
from multiprocessing import Process, Manager
import os
from tqdm import tqdm

YAML_DIR = "config/ablation_1_reshalf"
yaml_files = sorted(Path(YAML_DIR).glob("*.yaml"))

# 按 GPU 分组
yaml_groups = {}
for yml in yaml_files:
    with open(yml, "r") as f:
        cfg = yaml.safe_load(f)
    gpu_id = cfg.get("gpu_id", 0)
    yaml_groups.setdefault(gpu_id, []).append(yml)

total_files = len(yaml_files)
print(f"Found {total_files} YAMLs across {len(yaml_groups)} GPUs")

manager = Manager()
lock = manager.Lock()
# 总进度条
pbar_total = tqdm(total=total_files, desc="Total Progress", ncols=100, position=0)

def worker(gpu_id, yml_list, lock, pbar_total, position):
    # 每个 GPU 单独进度条
    pbar_gpu = tqdm(total=len(yml_list), desc=f"GPU {gpu_id}", ncols=100, position=position)
    for idx, yml in enumerate(yml_list):
        log_file = yml.with_suffix(".log")
        print(f"[GPU {gpu_id}] ({idx+1}/{len(yml_list)}) Running {yml}")
        with open(log_file, "w") as f:
            subprocess.run(
                ["python", "single_gen_texture.py", "--config", str(yml)],
                check=True,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=os.environ,
            )
        print(f"[GPU {gpu_id}] Finished {yml}")
        # 更新 GPU 和总进度条
        pbar_gpu.update(1)
        with lock:
            pbar_total.update(1)
    pbar_gpu.close()

if __name__ == "__main__":
    processes = []
    for i, (gpu_id, yml_list) in enumerate(yaml_groups.items()):
        p = Process(target=worker, args=(gpu_id, yml_list, lock, pbar_total, i + 1))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    pbar_total.close()
