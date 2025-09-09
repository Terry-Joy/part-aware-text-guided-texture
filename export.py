import os
import trimesh

def convert_glb_to_obj(input_folder, output_folder):
    """
    将输入文件夹中的所有GLB文件转换为OBJ格式，并保存到输出文件夹
    :param input_folder: 包含GLB文件的源目录
    :param output_folder: 存储OBJ文件的目标目录
    """
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.glb'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.obj")
            
            try:
                # 加载GLB文件
                mesh = trimesh.load(input_path, file_type='glb')
                
                # 导出为OBJ格式
                mesh.export(output_path, file_type='obj')
                print(f"✅ 转换成功: {filename} -> {os.path.basename(output_path)}")
                
            except Exception as e:
                print(f"❌ 转换失败 {filename}: {str(e)}")

if __name__ == "__main__":
    # 配置路径（根据实际需求修改）
    input_dir = "./config/eval/eval_data_glb"  # GLB文件所在目录
    output_dir = "./config/eval/eval_data_obj"  # OBJ输出目录
    
    # 执行转换
    convert_glb_to_obj(input_dir, output_dir)