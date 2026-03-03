import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
from os.path import join
import shutil
from datetime import datetime

# 导入项目内部模块
from src.renderer.project import UVProjection as UVP
from .segment_sptial_inpaint import SpatialAware3DInpainting
from .commoponentAware3Dinpaint2 import ComponentAware3DInpainting
from .spital_commoponentAware3dinpaint3 import ComponentSparse3DInpainting
from .commoponentAware3Dinpaint3 import ComponentAware3DInpainting as ComponentAware3DInpainting_v2
from .commoponentAware3Dinpaint4 import ComponentAware3DInpaintingGPU as ComponentAware3DInpainting_v3
from .commoponentAware3Dinpaint5 import ComponentAware3DInpaintingGPU as ComponentAware3DInpainting_v4
from .commoponentAware3Dinpaint6 import ComponentAware3DInpaintingGPU2 as ComponentAware3DInpainting_v5
from diffusers.utils import numpy_to_pil

class TextureInpaintingPipeline:
    """纹理补全Pipeline，保持与原始项目相同的模块结构"""
    
    def __init__(self, config):
        self.config = config
        self._setup_device()
        self._setup_directories()
    
    def _setup_device(self):
        """设置计算设备"""
        gpu_id = self.config["gpu_id"]
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(self.device)
            print(f"使用GPU: {torch.cuda.get_device_name(gpu_id)}")
        else:
            self.device = torch.device("cpu")
            print("警告: 未检测到GPU，将在CPU上运行")
    
    def _setup_directories(self):
        """设置输出目录结构"""
        self.output_dir = self.config["output_dir"]
        self.result_dir = join(self.output_dir, "results")
        self.intermediate_dir = join(self.output_dir, "intermediate")
        
        for dir_path in [self.output_dir, self.result_dir, self.intermediate_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def _setup_renderers(self):
        """初始化渲染器"""
        texture_rgb_size = self.config["texture_rgb_size"]
        render_rgb_size = self.config["render_rgb_size"]
        
        # 初始化UV投影器 (RGB)
        self.uvp_rgb = UVP(
            texture_size=texture_rgb_size,
            render_size=render_rgb_size,
            sampling_mode="nearest",
            channels=3,
            device=self.device
        )
        
        # 加载网格
        mesh_path = self.config["mesh_path"]
        mesh_transform = self.config["mesh_transform"]
        mesh_autouv = self.config["mesh_autouv"]
        
        if mesh_path.lower().endswith(".obj"):
            self.uvp_rgb.load_mesh(
                mesh_path,
                scale_factor=mesh_transform.get("scale", 1.0),
                autouv=mesh_autouv
            )
        elif mesh_path.lower().endswith(".glb"):
            self.uvp_rgb.load_glb_mesh(
                mesh_path,
                scale_factor=mesh_transform.get("scale", 1.0),
                autouv=mesh_autouv
            )
        else:
            raise ValueError("不支持的网格格式，仅支持.obj和.glb文件")
        
        # 设置相机
        camera_azims = self.config["camera_azims"]
        camera_elev = self.config["camera_elev"]
        camera_centers = self.config["camera_centers"]
        
        self.camera_poses = [
    (elev, azim if azim >= 0 else azim + 360, camera_centers) 
    for azim, elev in zip(camera_azims, camera_elev)
]
        self.uvp_rgb.set_cameras_and_render_settings(
            self.camera_poses,
            centers=camera_centers,
            camera_distance=2.0
        )
        
        # 渲染几何并计算可见性权重
        _, _, _, cos_maps, _, _ = self.uvp_rgb.render_geometry()
        self.uvp_rgb.calculate_cos_angle_weights(cos_maps, fill=False)
        
        # 保存可见性信息
        self.max_cos_map = torch.max(torch.stack(self.uvp_rgb.cos_maps, dim=0), dim=0)[0]
        self.avg_cos_map = torch.mean(torch.stack(self.uvp_rgb.cos_maps, dim=0), dim=0)
        
        # 释放不必要的内存
        del _, cos_maps
    
    def _load_texture(self):
        """加载纹理图像"""
        texture_path = self.config["texture_path"]
        if not os.path.exists(texture_path):
            raise FileNotFoundError(f"纹理文件不存在: {texture_path}")
        
        print(f"加载纹理: {texture_path}")
        texture_img = Image.open(texture_path).convert("RGB")
        texture_tensor = T.ToTensor()(texture_img) 
        # texture_tensor = T.ToTensor()(texture_img) * 2 - 1  # 转为[-1,1]
        
        # 确保纹理尺寸匹配
        texture_size = self.config["texture_rgb_size"]
        if texture_tensor.shape[1:] != (texture_size, texture_size):
            print(f"调整纹理尺寸从 {texture_tensor.shape[1:]} 到 {(texture_size, texture_size)}")
            texture_tensor = T.Resize(
                (texture_size, texture_size),
                interpolation=T.InterpolationMode.BILINEAR
            )(texture_tensor)
        
        return texture_tensor.to(self.device)
    
    def _save_initial_texture(self, texture_tensor):
        """保存初始纹理贴图"""
        # 将张量转换为PIL图像 (确保值在[0,1]范围内)
        texture_tensor = texture_tensor.clamp(0, 1).cpu()
        
        # 转换为PIL图像
        texture_pil = T.ToPILImage()(texture_tensor)
        
        # 保存路径
        save_path = join(self.intermediate_dir, "initial_texture.png")
        
        # 保存图像
        texture_pil.save(save_path)
        print(f"初始纹理已保存: {save_path}")
        
        # 可选：也保存为OBJ格式的纹理映射
        initial_tex_dir = join(self.intermediate_dir, "initial_texture")
        os.makedirs(initial_tex_dir, exist_ok=True)
        
        # 保存带初始纹理的OBJ
        self.uvp_rgb.set_texture_map(texture_tensor)
        self.uvp_rgb.save_mesh(join(initial_tex_dir, "textured.obj"), texture_tensor.permute(1, 2, 0))
        
        # 渲染并保存初始纹理的视图
        camera_azims = [0, 60, 120, 180, 240, 300]
        camera_elevs = [20, 20, 20, 20, 20, 20]
        vis_camera_poses = [(elev, azim) for elev, azim in zip(camera_elevs, camera_azims)]
        vis_views = self.uvp_rgb.render_multi_views(vis_camera_poses)
        vis_rgb = torch.cat(vis_views, axis=-1)[:-1, ...]
        vis_rgb = vis_rgb.permute(1, 2, 0).cpu().numpy()[None, ...]
        vis_image = numpy_to_pil(vis_rgb)[0]    
        vis_image.save(join(initial_tex_dir, "origin_textured_views_rgb_vis.jpg"))

        textured_views = self.uvp_rgb.render_textured_views()
        views_rgb = torch.cat(textured_views, axis=-1)[:-1, ...]
        views_rgb = views_rgb.permute(1, 2, 0).cpu().numpy()[None, ...]
        view_image = numpy_to_pil(views_rgb)[0]
        view_image.save(join(initial_tex_dir, "initial_texture_views.jpg"))
        print(f"初始纹理模型和视图已保存至: {initial_tex_dir}")
    
    def _visualize_original(self, texture_tensor=None):
        """可视化原始纹理"""
        # 注意：这里我们不设置纹理，因为load_texture后纹理应该已经设置
        # 如果需要，可以取消下面的注释
        # if texture_tensor is not None:
        #     self.uvp_rgb.set_texture_map(texture_tensor)
        
        textured_views = self.uvp_rgb.render_textured_views()
        
        views_rgb = torch.cat(textured_views, axis=-1)[:-1, ...]
        views_rgb = views_rgb.permute(1, 2, 0).cpu().numpy()[None, ...]
        view_image = numpy_to_pil(views_rgb)[0]
        
        save_path = join(self.intermediate_dir, "original_texture_views.jpg")
        view_image.save(save_path)
        print(f"原始纹理视图已保存: {save_path}")
    
    def _perform_inpainting(self, texture_tensor=None):
        """执行纹理补全"""
        inpainting_method = self.config["inpainting_method"]
        face2label_path = self.config["face2label_path"]
        mesh = self.uvp_rgb.mesh
        
        # 调整维度 [C, H, W] -> [H, W, C]
        result_tex_rgb = texture_tensor
        red_mask = None
        
        print(f"使用 {inpainting_method} 方法进行纹理补全")
        
        try:
            if inpainting_method == "component" and face2label_path and os.path.exists(face2label_path):
                inpainter = ComponentAware3DInpainting(
                    mesh, self.device, self.max_cos_map, self.avg_cos_map, face2label_path
                )
                result_tex_rgb, _, red_mask = inpainter(result_tex_rgb)
            elif inpainting_method == "spatial":
                inpainter = SpatialAware3DInpainting(
                    mesh, self.device, self.max_cos_map
                )
                result_tex_rgb, _, red_mask = inpainter(result_tex_rgb)
            elif inpainting_method == "sparse" and face2label_path and os.path.exists(face2label_path):
                inpainter = ComponentSparse3DInpainting(
                    mesh, self.device, self.max_cos_map, face2label_path
                )
                result_tex_rgb, _, red_mask = inpainter(result_tex_rgb)
            elif inpainting_method == "component2" and face2label_path and os.path.exists(face2label_path):
                inpainter = ComponentAware3DInpainting_v2(
                    mesh, self.device, self.max_cos_map, self.avg_cos_map, face2label_path
                )
                result_tex_rgb, _, red_mask = inpainter(result_tex_rgb)
            elif inpainting_method == "component3" and face2label_path and os.path.exists(face2label_path):
                inpainter = ComponentAware3DInpainting_v3(
                    mesh, self.device, self.max_cos_map, self.avg_cos_map, face2label_path
                )
                result_tex_rgb, _, red_mask = inpainter(result_tex_rgb)
            elif inpainting_method == "component4" and face2label_path and os.path.exists(face2label_path):
                inpainter = ComponentAware3DInpainting_v4(
                    mesh, self.device, self.max_cos_map, self.avg_cos_map, face2label_path
                )
                result_tex_rgb, _, red_mask = inpainter(result_tex_rgb)
            elif inpainting_method == "component5" and face2label_path and os.path.exists(face2label_path):
                inpainter = ComponentAware3DInpainting_v5(
                    mesh, self.device, self.max_cos_map, self.avg_cos_map, face2label_path
                )
                result_tex_rgb, _, red_mask = inpainter(result_tex_rgb)
            else:
                print("警告: 未指定有效方法或缺少面标签，跳过高级补全")
                red_mask = torch.zeros_like(self.max_cos_map[..., 0])
        except Exception as e:
            print(f"补全过程出错: {e}")
            red_mask = torch.zeros_like(self.max_cos_map[..., 0])
        
        return result_tex_rgb, red_mask
    
    def _save_results(self, inpainted_texture, visibility_mask):
        """保存补全结果"""
        inpaint_dir = join(self.result_dir, "after_inpainting")
        os.makedirs(inpaint_dir, exist_ok=True)
        
        # 保存补全后的网格
        self.uvp_rgb.save_mesh(join(inpaint_dir, "textured.obj"), inpainted_texture)
        
        # 渲染并保存视图
        self.uvp_rgb.set_texture_map(inpainted_texture.permute(2, 0, 1))
        inpainted_views = self.uvp_rgb.render_textured_views()
        
        # 保存渲染视图
        views_rgb = torch.cat(inpainted_views, axis=-1)[:-1, ...]
        views_rgb = views_rgb.permute(1, 2, 0).cpu().numpy()[None, ...]
        inpaint_v = numpy_to_pil(views_rgb)[0]
        inpaint_v.save(join(inpaint_dir, "inpaint_textured_views_rgb.jpg"))

        camera_azims = [0, 60, 120, 180, 240, 300]
        camera_elevs = [20, 20, 20, 20, 20, 20]
        vis_camera_poses = [(elev, azim) for elev, azim in zip(camera_elevs, camera_azims)]
        vis_views = self.uvp_rgb.render_multi_views(vis_camera_poses)
        vis_rgb = torch.cat(vis_views, axis=-1)[:-1, ...]
        vis_rgb = vis_rgb.permute(1, 2, 0).cpu().numpy()[None, ...]
        vis_image = numpy_to_pil(vis_rgb)[0]    
        vis_image.save(join(inpaint_dir, "inpaint_textured_views_rgb_vis.jpg"))

        
        # 保存可视化掩码(如果有)
        # if visibility_mask is not None:
        #     red_mask_rgb = np.stack([visibility_mask.cpu().numpy()] * 3, axis=-1)
        #     red_vis_tex = torch.from_numpy(red_mask_rgb).float().to(self.device)
        #     self.uvp_rgb.save_mesh(join(inpaint_dir, "visibility_mask.obj"), red_vis_tex)
    
    def run(self):
        """执行完整的纹理补全流程"""
        print("===== 纹理补全Pipeline开始运行 =====")
        
        # 1. 设置渲染器
        self._setup_renderers()
        
        # 2. 加载纹理
        texture_tensor = self._load_texture()
        
        # 2.5 保存初始纹理贴图 - 新增步骤
        self._save_initial_texture(texture_tensor)
        
        # 3. 可视化原始纹理
        self._visualize_original(texture_tensor)
        
        # 4. 执行纹理补全
        inpainted_texture, visibility_mask = self._perform_inpainting(texture_tensor)
        
        # 5. 保存结果
        self._save_results(inpainted_texture, visibility_mask)
        
        print("===== 纹理补全Pipeline完成 =====")
        print(f"所有结果已保存至: {self.output_dir}")