#!/usr/bin/env python3
import argparse
import yaml
import os
import sys
import copy
import traceback
import torch.multiprocessing as mp  # 使用 torch 的多进程模块
from queue import Empty

# 假设 pipeline 在 src 目录下，根据实际情况调整引用
try:
    from src.texture_inpainting_pipeline import TextureInpaintingPipeline
except ImportError:
    print("Warning: Could not import TextureInpaintingPipeline. Ensure src is in python path.")
    # Mock class for testing
    class TextureInpaintingPipeline:
        def __init__(self, config): pass
        def run(self): pass

def load_base_config(config_path):
    """加载基础YAML配置文件"""
    defaults = {
        "gpu_id": 0, 
        "mesh_path": "", 
        "mesh_transform": {"scale": 2.0},
        "mesh_autouv": False,
        "texture_path": "",
        "camera_azims": [0, 60, 120, 180, 240, 300],
        "camera_elev": [0, 0, 0, 0, 0, 0],
        "camera_centers": None,
        "texture_size": 1024,
        "render_rgb_size": 1024,
        "texture_rgb_size": 1024,
        "inpainting_method": "component5",
        "face2label_path": None, 
        "output_dir": "",
        "log_interval": 10,
        "view_fast_preview": True,
        "tex_fast_preview": True,
        "timeformat": "%d%b%Y-%H%M%S"
    }

    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
            
    return config

def parse_arguments():
    parser = argparse.ArgumentParser(description='多GPU批量纹理补全工具')
    parser.add_argument('--root_dir', type=str, required=True,
                        help='包含results文件夹的根目录路径')
    parser.add_argument('--labels_dir', type=str, required=True,
                        help='包含label json文件的根目录路径')
    parser.add_argument('--base_config', type=str, default=None,
                        help='基础配置文件路径 (可选)')
    parser.add_argument('--gpu_ids', type=str, default="0",
                        help='使用的GPU列表，用逗号分隔')
    # [新增] 指定日志文件路径
    parser.add_argument('--log_file', type=str, default="processed_tasks.txt",
                        help='记录已完成任务的日志文件路径')
    return parser.parse_args()

def find_model_id_and_label_path(result_dir, labels_root_dir):
    norm_result_dir = os.path.normpath(result_dir)
    parts = norm_result_dir.split(os.sep)
    found_model_id = None
    
    for part in reversed(parts):
        if part in ['results', 'result', '.', '..']:
            continue
            
        candidate_path = os.path.join(labels_root_dir, part)
        if os.path.isdir(candidate_path):
            found_model_id = part
            break
    
    if found_model_id:
        label_json_path = os.path.join(labels_root_dir, found_model_id, "labels.json")
        return label_json_path, found_model_id
    
    return None, None

def load_processed_set(log_file):
    """加载已处理的任务列表到内存集合中"""
    processed = set()
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                path = line.strip()
                if path:
                    processed.add(os.path.abspath(path)) # 存储绝对路径以防歧义
    return processed

def gpu_worker(gpu_id, task_queue, base_config, labels_dir, lock, log_file):
    """
    [修改] 增加了 lock 和 log_file 参数
    """
    print(f"[Worker Start] 进程启动，绑定 GPU ID: {gpu_id}")
    
    while True:
        try:
            result_dir = task_queue.get(timeout=3) 
        except Empty:
            break

        # ==================== 任务处理逻辑 ====================
        mesh_path = os.path.join(result_dir, "textured.obj")
        texture_path = os.path.join(result_dir, "textured.png")
        output_dir = os.path.join(result_dir, "after_component5_inpaint")
        
        if not (os.path.exists(mesh_path) and os.path.exists(texture_path)):
            print(f"[Worker GPU {gpu_id}] 跳过无效目录: {result_dir}")
            continue

        label_path, model_id = find_model_id_and_label_path(result_dir, labels_dir)
        
        if not label_path or not os.path.exists(label_path):
            print(f"[Worker GPU {gpu_id}] [Error] 无法找到对应的 label 文件: {model_id}")
            continue

        os.makedirs(output_dir, exist_ok=True)
        print(f"[Worker GPU {gpu_id}] 开始处理 Model: {model_id}")

        try:
            current_config = copy.deepcopy(base_config)
            current_config["gpu_id"] = gpu_id 
            current_config["mesh_path"] = mesh_path
            current_config["texture_path"] = texture_path
            current_config["output_dir"] = output_dir
            current_config["face2label_path"] = label_path

            pipeline = TextureInpaintingPipeline(current_config)
            pipeline.run()
            
            print(f"[Worker GPU {gpu_id}] 完成 Model: {model_id}")

            # [新增] 成功完成后，加锁写入日志
            # 使用绝对路径确保一致性
            abs_path = os.path.abspath(result_dir)
            with lock:
                with open(log_file, "a", encoding='utf-8') as f:
                    f.write(abs_path + "\n")
                    f.flush() # 确保立即写入磁盘

        except Exception as e:
            print(f"[Worker GPU {gpu_id}] 错误: 处理 {result_dir} 失败. \nReason: {e}")
            traceback.print_exc()
        # ====================================================

    print(f"[Worker End] GPU {gpu_id} 任务队列已空，进程退出。")

def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    args = parse_arguments()
    
    if not os.path.exists(args.root_dir):
        print(f"错误: 任务根目录不存在 {args.root_dir}")
        sys.exit(1)
        
    if not os.path.exists(args.labels_dir):
        print(f"错误: Label根目录不存在 {args.labels_dir}")
        sys.exit(1)

    try:
        gpu_list = [int(x.strip()) for x in args.gpu_ids.split(',')]
    except ValueError:
        print("错误: gpu_ids 格式不正确")
        sys.exit(1)

    print(f"可用 GPU 列表: {gpu_list}")
    
    # ==================== 【修改开始】 ====================
    # 强制将日志文件路径设置在 root_dir 下
    # 这样 processed_tasks.txt 就会出现在你的数据目录里
    final_log_file = os.path.join(args.root_dir, os.path.basename(args.log_file))
    print(f"日志文件路径: {final_log_file}")
    # ==================== 【修改结束】 ====================

    # 加载已完成记录 (传入新的路径)
    processed_set = load_processed_set(final_log_file)
    print(f"已加载 {len(processed_set)} 条历史成功记录。")

    base_config = load_base_config(args.base_config)

    task_queue = mp.Queue()
    task_count = 0
    skipped_count = 0
    
    print(f"正在扫描目录: {args.root_dir} ...")
    for root, dirs, files in os.walk(args.root_dir):
        if os.path.basename(root) == 'results':
            if os.path.exists(os.path.join(root, "textured.obj")):
                # 检查是否已处理
                abs_root = os.path.abspath(root)
                if abs_root in processed_set:
                    skipped_count += 1
                    continue
                
                task_queue.put(root)
                task_count += 1
    
    print(f"扫描结束。")
    print(f"  - 待处理任务: {task_count}")
    print(f"  - 跳过已完成: {skipped_count}")

    if task_count == 0:
        print("没有新任务需要处理。")
        return

    # 创建进程锁
    file_lock = mp.Lock()

    processes = []
    for gpu_id in gpu_list:
        # 注意：这里传入的是 final_log_file
        p = mp.Process(target=gpu_worker, args=(gpu_id, task_queue, base_config, args.labels_dir, file_lock, final_log_file))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n所有并行任务已完成。")

if __name__ == "__main__":
    main()
    
    #     parser = argparse.ArgumentParser(description='多GPU批量纹理补全工具')
    # parser.add_argument('--root_dir', type=str, required=True,
    #                     help='包含results文件夹的根目录路径')
    # parser.add_argument('--labels_dir', type=str, required=True,
    #                     help='包含label json文件的根目录路径')
    # parser.add_argument('--base_config', type=str, default=None,
    #                     help='基础配置文件路径 (可选)')
    # parser.add_argument('--gpu_ids', type=str, default="0",
    #                     help='使用的GPU列表，用逗号分隔')
    # # [新增] 指定日志文件路径
    # parser.add_argument('--log_file', type=str, default="processed_tasks.txt",
    #                     help='记录已完成任务的日志文件路径')
    
# python mul_texture_inpaint.py --root_dir ../exp_adain_05_30_norefattn_real_jiaocha_6views_25 --labels_dir ../bishetest --gpu_ids "0,1,2,3,4,5,6,7" 

# python mul_texture_inpaint.py --root_dir exp/gd7_5_adain_yizhi_0_5_3_0_norefattn_no_adain/ --labels_dir ../bishetest/ --gpu_ids "0,1,2,3,4,5,6"

# python mul_texture_inpaint.py --root_dir exp/gd7_5_adain_yizhi_0_5_3_0_norefattn_full_design/ --labels_dir ../bishetest/ --gpu_ids "0,1,2,3,4,5,6"

# python mul_texture_inpaint.py --root_dir exp/gd7_5_adain_yizhi_0_5_3_0_norefattn_full_design_start_0.0/ --labels_dir ../bishetest/ --gpu_ids "0,1,2,3,4,5,6"

# python mul_texture_inpaint.py --root_dir exp/gd7_5_adain_yizhi_0_5_3_0_norefattn_full_design_start_0.3/ --labels_dir ../bishetest/ --gpu_ids "0,1,2,3,4,5,6"

# python mul_texture_inpaint.py --root_dir exp/gd7_5_adain_yizhi_0_5_3_0_norefattn_full_design_start_0.7/ --labels_dir ../bishetest/ --gpu_ids "0,1,2,3,4,5,6"