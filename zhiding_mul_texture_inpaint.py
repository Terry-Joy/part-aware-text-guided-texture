#!/usr/bin/env python3
import argparse
import yaml
import os
import sys
import copy
import traceback
import torch.multiprocessing as mp  # 使用 torch 的多进程模块
from queue import Empty

# ==========================================================
# 【核心配置区】直接在这里指定： "文件夹路径": "对应的Model_ID"
# ==========================================================
TASKS_CONFIG = {
    # "inpaint_data/07fe989ba464464fae6943445703afca_prompt1_seed3_full_design": "07fe989ba464464fae6943445703afca",
    # "inpaint_data/07fe989ba464464fae6943445703afca_prompt1_seed3_no_adj": "07fe989ba464464fae6943445703afca",
    # "inpaint_data/07fe989ba464464fae6943445703afca_prompt1_seed3_noyizhi": "07fe989ba464464fae6943445703afca",
    # "inpaint_data/07fe989ba464464fae6943445703afca_prompt1_seed3_no_adain": "07fe989ba464464fae6943445703afca",
    # "inpaint_data/honglvshuibei_8cdaa8d8fbc34606a734f3dbd1e63e12/full_design": "8cdaa8d8fbc34606a734f3dbd1e63e12", 
    # "inpaint_data/honglvshuibei_8cdaa8d8fbc34606a734f3dbd1e63e12/no_adj": "8cdaa8d8fbc34606a734f3dbd1e63e12",
    # "inpaint_data/honglvshuibei_8cdaa8d8fbc34606a734f3dbd1e63e12/no_yizhi": "8cdaa8d8fbc34606a734f3dbd1e63e12",
    # "inpaint_data/honglvshuibei_8cdaa8d8fbc34606a734f3dbd1e63e12/no_adain": "8cdaa8d8fbc34606a734f3dbd1e63e12",
    #     "inpaint_data/xiaorong_sofa/zhen_full_design_jia_no_yizhi": "0401a13ba90d4f5c8c07954abcd6922c", 
    # "inpaint_data/xiaorong_sofa/zhen_no_yizhi_jia_full_design": "0401a13ba90d4f5c8c07954abcd6922c",
    # "inpaint_data/xiaorong_sofa/no_adj": "0401a13ba90d4f5c8c07954abcd6922c",
    # "inpaint_data/caomei3/no_adain": "6e95a8cb47114f2285be08a70e8990b6",
    # "inpaint_data/caomei/no_yizhi": "6e95a8cb47114f2285be08a70e8990b6", 
    # "inpaint_data/caomei/no_adj": "6e95a8cb47114f2285be08a70e8990b6", 
    # "inpaint_data/caomei/no_adain": "6e95a8cb47114f2285be08a70e8990b6", 

    # "inpaint_data/user_study/01b52fada7f94115b0d6431cfb0d5b21":"01b52fada7f94115b0d6431cfb0d5b21",
    # "inpaint_data/user_study/2ef369ebae66464f9d2479057a6bf6d2":"2ef369ebae66464f9d2479057a6bf6d2",
    # "inpaint_data/user_study/8d3d39516f7b40d5aa19e85226940899":"8d3d39516f7b40d5aa19e85226940899",
    # "inpaint_data/user_study/07aa44e2f864476f998e3ada75ccfa45":"07aa44e2f864476f998e3ada75ccfa45",
    # "inpaint_data/user_study/5ae212805ceb43f6981243db2eb8c501":"5ae212805ceb43f6981243db2eb8c501",
    # "inpaint_data/user_study/aa1a732d4fdc44e0a20cee312a929d77":"aa1a732d4fdc44e0a20cee312a929d77",
    # "inpaint_data/user_study/1bb0cf7261174670ad1134093875e1d1":"1bb0cf7261174670ad1134093875e1d1",
    # "inpaint_data/user_study/77d56cc994f94d8280c38009688e7413":"77d56cc994f94d8280c38009688e7413"
    "inpaint_data/texture_editing/group_1/green/ours": "0baad2ab0df5463fa55a4b03ba9caa09",
    "inpaint_data/texture_editing/group_1/green/xiaorong": "0baad2ab0df5463fa55a4b03ba9caa09",
    "inpaint_data/texture_editing/group_1/marron": "0baad2ab0df5463fa55a4b03ba9caa09", 
    "inpaint_data/texture_editing/group_1/red": "0baad2ab0df5463fa55a4b03ba9caa09",    
    "inpaint_data/texture_editing/group_1/yellow": "0baad2ab0df5463fa55a4b03ba9caa09",

    "inpaint_data/texture_editing/group_2/leaf_ours": "0baad2ab0df5463fa55a4b03ba9caa09",
    "inpaint_data/texture_editing/group_2/leaf_10_5_xiaorong_1": "0baad2ab0df5463fa55a4b03ba9caa09",
    "inpaint_data/texture_editing/group_2/dot_pattern/ours": "0baad2ab0df5463fa55a4b03ba9caa09",
    "inpaint_data/texture_editing/group_2/line_pattern/ours": "0baad2ab0df5463fa55a4b03ba9caa09",
    "inpaint_data/texture_editing/group_2/26Mar2026-105110_flower": "0baad2ab0df5463fa55a4b03ba9caa09",    

    "inpaint_data/texture_editing/group_3/04Mar2026-060600_原图": "314d5728e6ac496abdf1fe7268ca9ae0",
    "inpaint_data/texture_editing/group_3/26Mar2026-120823_石灰岩": "314d5728e6ac496abdf1fe7268ca9ae0",
    "inpaint_data/texture_editing/group_3/26Mar2026-121329_砂岩": "314d5728e6ac496abdf1fe7268ca9ae0",
    "inpaint_data/texture_editing/group_3/26Mar2026-122334_大理石": "314d5728e6ac496abdf1fe7268ca9ae0", 
    "inpaint_data/texture_editing/group_3/26Mar2026-122724_大理石2": "314d5728e6ac496abdf1fe7268ca9ae0",


    # noadj
    "inpaint_data/rebuttal/adj/61ec1aa136b74edda418c3104fdb08ab/full_design": "61ec1aa136b74edda418c3104fdb08ab",
    "inpaint_data/rebuttal/adj/61ec1aa136b74edda418c3104fdb08ab/no_adj": "61ec1aa136b74edda418c3104fdb08ab",
    "inpaint_data/rebuttal/adj/90ef7d25938d4855b57794ba7e3420cd/full_design": "90ef7d25938d4855b57794ba7e3420cd", 
    "inpaint_data/rebuttal/adj/90ef7d25938d4855b57794ba7e3420cd/no_adj": "90ef7d25938d4855b57794ba7e3420cd",    
    "inpaint_data/rebuttal/adj/4078e7e485ab48ad86e121d771c1d07a/full_design": "4078e7e485ab48ad86e121d771c1d07a",
    "inpaint_data/rebuttal/adj/4078e7e485ab48ad86e121d771c1d07a/no_adj": "4078e7e485ab48ad86e121d771c1d07a",

    "inpaint_data/rebuttal/adj/27b5263cee8f451c98474758a6d779f2/seed1/full_design": "27b5263cee8f451c98474758a6d779f2",
    "inpaint_data/rebuttal/adj/27b5263cee8f451c98474758a6d779f2/seed1/no_adj": "27b5263cee8f451c98474758a6d779f2",
    "inpaint_data/rebuttal/adj/27b5263cee8f451c98474758a6d779f2/seed3/full_design": "27b5263cee8f451c98474758a6d779f2",
    "inpaint_data/rebuttal/adj/27b5263cee8f451c98474758a6d779f2/seed3/no_adj": "27b5263cee8f451c98474758a6d779f2",


    # no adain
    "inpaint_data/rebuttal/no_adain/1bb0cf7261174670ad1134093875e1d1/full_design": "1bb0cf7261174670ad1134093875e1d1",
    "inpaint_data/rebuttal/no_adain/1bb0cf7261174670ad1134093875e1d1/no_adain": "1bb0cf7261174670ad1134093875e1d1",
    "inpaint_data/rebuttal/no_adain/4c4cfe66151741ef91d85211d416753b/full_design": "4c4cfe66151741ef91d85211d416753b", 
    "inpaint_data/rebuttal/no_adain/4c4cfe66151741ef91d85211d416753b/no_adain": "4c4cfe66151741ef91d85211d416753b",    
    "inpaint_data/rebuttal/no_adain/43a3cc69312948dc8d72ac0eaeecfc0a/full_design": "43a3cc69312948dc8d72ac0eaeecfc0a",
    "inpaint_data/rebuttal/no_adain/43a3cc69312948dc8d72ac0eaeecfc0a/no_adain": "43a3cc69312948dc8d72ac0eaeecfc0a",
    "inpaint_data/rebuttal/no_adain/90789cdc08724e39b5bb3c70d6262c7a/full_design": "90789cdc08724e39b5bb3c70d6262c7a",
    "inpaint_data/rebuttal/no_adain/90789cdc08724e39b5bb3c70d6262c7a/no_adain": "90789cdc08724e39b5bb3c70d6262c7a",
    "inpaint_data/rebuttal/no_adain/cffd007512a74a47b48390e1777c2afa_house/full_design": "cffd007512a74a47b48390e1777c2afa",
    "inpaint_data/rebuttal/no_adain/cffd007512a74a47b48390e1777c2afa_house/no_adain": "cffd007512a74a47b48390e1777c2afa",

}

DEFAULT_LABELS_DIR = "../bishetest"
# ==========================================================

# 假设 pipeline 在 src 目录下，根据实际情况调整引用
try:
    from src.texture_inpainting_pipeline import TextureInpaintingPipeline
except ImportError:
    print("Warning: Could not import TextureInpaintingPipeline.")
    class TextureInpaintingPipeline:
        def __init__(self, config): pass
        def run(self): pass

def load_base_config(config_path):
    """【完全保留你的原始逻辑】加载基础YAML配置文件"""
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
    parser = argparse.ArgumentParser(description='SemantiSeed 精准指定 ID 工具')
    parser.add_argument('--gpu_ids', type=str, default="0")
    parser.add_argument('--obj_name', type=str, default="textured.obj")
    parser.add_argument('--png_name', type=str, default="textured.png")
    parser.add_argument('--log_file', type=str, default="processed_tasks.txt")
    return parser.parse_args()

def load_processed_set(log_file):
    """加载已处理的任务列表到内存集合中"""
    processed = set()
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                path = line.strip()
                if path:
                    processed.add(os.path.abspath(path))
    return processed

def gpu_worker(gpu_id, task_queue, base_config, labels_dir, lock, log_file, obj_name, png_name):
    print(f"[Worker Start] 进程启动，绑定 GPU ID: {gpu_id}")
    
    while True:
        try:
            task = task_queue.get(timeout=3) 
        except Empty:
            break

        result_dir = task["path"]
        model_id = task["id"]
        
        # ==================== 任务处理逻辑 ====================
        mesh_path = os.path.join(result_dir, obj_name)
        texture_path = os.path.join(result_dir, png_name)
        output_dir = os.path.join(result_dir, "after_component5_inpaint_0_25")
        
        if not (os.path.exists(mesh_path) and os.path.exists(texture_path)):
            print(f"[Worker GPU {gpu_id}] 跳过无效目录: {result_dir}")
            continue

        label_path = os.path.join(labels_dir, model_id, "labels.json")
        if not os.path.exists(label_path):
            print(f"[Worker GPU {gpu_id}] [Error] 无法找到对应的 label 文件: {label_path}")
            continue

        os.makedirs(output_dir, exist_ok=True)
        print(f"[Worker GPU {gpu_id}] 开始处理 Model: {model_id}")

        try:
            # 【完全保留你的原始逻辑】使用深拷贝，注入特定参数
            current_config = copy.deepcopy(base_config)
            current_config["gpu_id"] = gpu_id 
            current_config["mesh_path"] = mesh_path
            current_config["texture_path"] = texture_path
            current_config["output_dir"] = output_dir
            current_config["face2label_path"] = label_path

            pipeline = TextureInpaintingPipeline(current_config)
            pipeline.run()
            
            print(f"[Worker GPU {gpu_id}] 完成 Model: {model_id}")

            abs_path = os.path.abspath(result_dir)
            with lock:
                with open(log_file, "a", encoding='utf-8') as f:
                    f.write(abs_path + "\n")
                    f.flush()

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
    final_log_file = os.path.join(".", args.log_file)
    processed_set = load_processed_set(final_log_file)

    # 【核心修复】初始化带默认参数的 base_config
    base_config = load_base_config(None)

    task_queue = mp.Queue()
    task_count = 0
    skipped_count = 0
    
    print(f"--- 精准 ID 指定模式 ---")
    for path, mid in TASKS_CONFIG.items():
        abs_folder = os.path.abspath(path)
        # if abs_folder in processed_set:
        #     skipped_count += 1
        #     print(f"跳过已处理: {path}")
        #     continue
        task_queue.put({"path": path, "id": mid})
        task_count += 1

    print(f"  - 待处理任务: {task_count}")
    print(f"  - 跳过已完成: {skipped_count}")

    if task_count == 0:
        print("没有新任务需要处理。")
        return

    try:
        gpu_list = [int(x.strip()) for x in args.gpu_ids.split(',')]
    except ValueError:
        print("错误: gpu_ids 格式不正确")
        sys.exit(1)

    print(f"可用 GPU 列表: {gpu_list}")
    file_lock = mp.Lock()

    processes = []
    for gpu_id in gpu_list:
        p = mp.Process(target=gpu_worker, 
                       args=(gpu_id, task_queue, base_config, DEFAULT_LABELS_DIR, 
                             file_lock, final_log_file, args.obj_name, args.png_name))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n所有并行任务已完成。")

if __name__ == "__main__":
    main()