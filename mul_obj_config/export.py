import trimesh

# 1. 读取你手里那个唯一的 .glb 文件（force='mesh' 确保它变成一个完整的网格）
mesh = trimesh.load("table_and_chairs.glb", force="mesh")

# 2. 原地直接导出为 .obj
mesh.export("table_and_chairs_1.obj", file_type="obj")

print("✅ 哥们，搞定！纯净版 OBJ 已经生成，可以喂给 samesh 跑分割了！")