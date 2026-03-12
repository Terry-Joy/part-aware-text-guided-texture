import subprocess
import yaml
from pathlib import Path
from multiprocessing import Process, Manager
import os
from tqdm import tqdm

# ================= 配置区域 =================
YAML_DIR = "config/bishe_6views_ours_gd7_5/gd7_5_adain_yizhi_0_5_3_0_norefattn_full_design_start_0.0"
# 定义记录成功历史的文件路径
SUCCESS_LOG_PATH = Path(YAML_DIR) / "success_history.txt"
# ===========================================

def load_success_history(log_path):
    """读取已经成功的任务列表"""
    processed = set()
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                # 使用 resolve() 获取绝对路径，确保匹配准确
                processed.add(line.strip())
    return processed

def worker(gpu_id, yml_list, lock, pbar_total, position, success_log_path):
    # 每个 GPU 单独进度条
    pbar_gpu = tqdm(total=len(yml_list), desc=f"GPU {gpu_id}", ncols=100, position=position)
    
    for idx, yml in enumerate(yml_list):
        log_file = yml.with_suffix(".log")
        # print(f"[GPU {gpu_id}] ({idx+1}/{len(yml_list)}) Running {yml.name}")
        
        try:
            with open(log_file, "w") as f:
                subprocess.run(
                    ["python", "single_gen_adain_houqi.py", "--config", str(yml)],
                    check=True, # 只有返回码为0才算成功，否则抛出异常
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=os.environ,
                )
            
            # --- 任务成功后的处理 ---
            # 记录到成功日志 (加锁防止多进程写入冲突)
            with lock:
                with open(success_log_path, "a", encoding="utf-8") as f_log:
                    # 写入绝对路径
                    f_log.write(str(yml.resolve()) + "\n")
                # 更新总进度条
                pbar_total.update(1)
            
            # print(f"[GPU {gpu_id}] Finished {yml.name}")

        except subprocess.CalledProcessError:
            print(f"\n[GPU {gpu_id}] FAILED: {yml.name}. Check log: {log_file}")
            # 失败了不记录到 success_log，也不更新 total_bar (或者你可以选择更新但不记录)
            # 这里选择不抛出异常，继续执行下一个任务
        except Exception as e:
            print(f"\n[GPU {gpu_id}] Error running {yml.name}: {e}")

        # 更新 GPU 独立进度条
        pbar_gpu.update(1)

    pbar_gpu.close()

if __name__ == "__main__":
    # 1. 读取历史记录
    processed_files = load_success_history(SUCCESS_LOG_PATH)
    print(f"Loaded {len(processed_files)} previously completed tasks.")

    # 2. 扫描文件并过滤
    all_yaml_files = sorted(Path(YAML_DIR).glob("*.yaml"))
    
    yaml_files_to_run = []
    for yml in all_yaml_files:
        # 比较绝对路径
        if str(yml.resolve()) not in processed_files:
            yaml_files_to_run.append(yml)
    
    total_files = len(yaml_files_to_run)
    skipped_count = len(all_yaml_files) - total_files
    
    print(f"Total YAMLs found: {len(all_yaml_files)}")
    print(f"Skipping: {skipped_count}")
    print(f"Remaining to run: {total_files}")

    if total_files == 0:
        print("All tasks completed! Nothing to run.")
        exit(0)

    # 3. 按 GPU 分组 (只分剩余的任务)
    yaml_groups = {}
    for yml in yaml_files_to_run:
        with open(yml, "r") as f:
            cfg = yaml.safe_load(f)
        gpu_id = cfg.get("gpu_id", 0)
        yaml_groups.setdefault(gpu_id, []).append(yml)

    print(f"Distributing tasks across {len(yaml_groups)} GPUs")

    # 4. 启动进程
    manager = Manager()
    lock = manager.Lock()
    
    # 总进度条 (只显示剩余任务数)
    pbar_total = tqdm(total=total_files, desc="Total Progress", ncols=100, position=0)

    processes = []
    # 这里 enumerate 的 i 用于控制进度条显示的行号
    for i, (gpu_id, yml_list) in enumerate(yaml_groups.items()):
        p = Process(
            target=worker, 
            args=(gpu_id, yml_list, lock, pbar_total, i + 1, SUCCESS_LOG_PATH)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    pbar_total.close()
    print("\nBatch processing finished.")