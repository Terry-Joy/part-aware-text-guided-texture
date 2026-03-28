#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import shutil
import subprocess
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区域 =================
BASE_CONFIG = 'editing_config/314d5728e6ac496abdf1fe7268ca9ae0_seed1_prompt_1_base_w3_yizhi0507_adain_first.yaml'
PROMPTS_FILE = './editing_config/314d5728e6ac496abdf1fe7268ca9ae0.txt'
OUTPUT_DIR = 'editing_config_batch/314d5728e6ac496abdf1fe7268ca9ae0/xiaorong'
MAIN_SCRIPT = 'single_gen_texture_editing.py'
MAX_PARALLEL = 2
LOG_FILE = './editing_config_batch/314d5728e6ac496abdf1fe7268ca9ae0/batch_log.txt'
SEEDS_PER_PROMPT = 2
SEED_RANGE = (1, 10000)

# 环境变量设置（关键！）
ENV_VARS = {
    'CUDA_VISIBLE_DEVICES': '0',      # 指定 GPU 0
    'OMP_NUM_THREADS': '1',           # 修复 libgomp 警告
    'OPENBLAS_NUM_THREADS': '1',      # 避免线程冲突
    'MKL_NUM_THREADS': '1',           # 避免线程冲突
}
# ===========================================

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_yaml(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

def load_prompts():
    with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

def generate_seeds(num_seeds):
    return random.sample([1, 3], num_seeds)

def get_completed_tasks():
    completed = set()
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if '✓' in line:
                    try:
                        parts = line.split('|')
                        if len(parts) >= 3:
                            seed = parts[1].strip().replace('seed=', '')
                            prompt = parts[2].strip()
                            completed.add((prompt, seed))
                    except:
                        pass
    return completed

def log_message(message):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] {message}\n"
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_line)
    print(message)

def create_config(prompt, seed, idx):
    config = load_yaml(BASE_CONFIG)
    config['prompt'] = prompt
    config['seed'] = seed
    
    base_name = Path(BASE_CONFIG).stem
    new_name = f"{base_name}_batch{idx:03d}_seed{seed}.yaml"
    new_path = Path(OUTPUT_DIR) / new_name
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    save_yaml(config, new_path)
    return new_path, prompt, seed

def run_task(config_path, prompt, seed, task_id):
    log_message(f"\n[任务 {task_id}] 开始 | seed={seed} | {prompt[:40]}...")
    log_message(f"配置文件：{config_path}")
    
    start_time = time.time()
    try:
        # 合并环境变量
        env = os.environ.copy()
        env.update(ENV_VARS)
        
        result = subprocess.run(
            ['python', MAIN_SCRIPT, '--config', str(config_path)],
            capture_output=False,
            text=True,
            env=env  # 传入环境变量
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            log_message(f"✓ [任务 {task_id}] 完成 | seed={seed} | ({elapsed:.1f}s)")
            return True
        else:
            log_message(f"✗ [任务 {task_id}] 失败 | seed={seed} | ({elapsed:.1f}s)")
            return False
    except Exception as e:
        log_message(f"✗ [任务 {task_id}] 异常 | seed={seed} | {e}")
        return False

def main():
    # 确保所有目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    # 设置全局环境变量
    for key, value in ENV_VARS.items():
        os.environ[key] = value
    
    # 加载提示词
    all_prompts = load_prompts()
    completed = get_completed_tasks()
    
    # 为每个 prompt 生成种子
    all_tasks = []
    for prompt in all_prompts:
        seeds = generate_seeds(SEEDS_PER_PROMPT)
        for seed in seeds:
            if (prompt, str(seed)) not in completed:
                all_tasks.append((prompt, seed))
    
    if len(all_tasks) == 0:
        log_message("所有任务已完成！如需重新运行，请删除 batch_log.txt")
        return
    
    log_message("=" * 70)
    log_message(f"批量任务启动 | Prompts: {len(all_prompts)} | Seeds/Prompt: {SEEDS_PER_PROMPT}")
    log_message(f"总任务数：{len(all_prompts) * SEEDS_PER_PROMPT} | 待运行：{len(all_tasks)} | 并行：{MAX_PARALLEL}")
    log_message(f"GPU: {ENV_VARS['CUDA_VISIBLE_DEVICES']} | OMP_THREADS: {ENV_VARS['OMP_NUM_THREADS']}")
    log_message("=" * 70)
    
    # 生成配置文件
    config_files = []
    for idx, (prompt, seed) in enumerate(all_tasks):
        config_path, _, _ = create_config(prompt, seed, idx)
        config_files.append((config_path, prompt, seed))
    
    log_message(f"\n已生成 {len(config_files)} 个配置文件")
    log_message("=" * 70)
    log_message("开始执行任务...")
    log_message("=" * 70)
    
    # 并行执行
    success_count = 0
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = {
            executor.submit(run_task, cfg_path, prompt, seed, idx+1): idx
            for idx, (cfg_path, prompt, seed) in enumerate(config_files)
        }
        
        for future in as_completed(futures):
            if future.result():
                success_count += 1
    
    # 统计结果
    log_message("\n" + "=" * 70)
    log_message(f"批量完成！成功：{success_count}/{len(all_tasks)}")
    log_message(f"总计：{len(all_prompts) * SEEDS_PER_PROMPT} 任务 | 成功率：{success_count/len(all_tasks)*100:.1f}%")
    log_message("=" * 70)

if __name__ == "__main__":
    main()