from pytorch3d.io import load_objs_as_meshes, load_obj, save_obj, IO
import torch
device = torch.device("cuda:0")
def load_mesh(mesh_path, scale_factor=2.0, auto_center=True, autouv=False):
	mesh = load_objs_as_meshes([mesh_path], device=device)
	return mesh
mesh = load_mesh(mesh_path= "./race_car.obj")

verts_list = mesh.verts_list()[0]
faces_list = mesh.faces_list()[0]
print(verts_list.shape)
print(faces_list.shape)
# print(len(verts_list))
# print(len(faces_list))
		