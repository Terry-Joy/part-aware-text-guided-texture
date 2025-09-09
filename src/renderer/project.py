import torch
import pytorch3d


from pytorch3d.io import load_objs_as_meshes, load_obj, save_obj, IO

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    FoVOrthographicCameras,
    AmbientLights,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    TexturesUV
)

from .geometry import HardGeometryShader
from .shader import HardNChannelFlatShader
from .voronoi import voronoi_solve


# Pytorch3D based renderering functions, managed in a class
# Render size is recommended to be the same as your latent view size
# DO NOT USE "bilinear" sampling when you are handling latents.
# Stable Diffusion has 4 latent channels so use channels=4

class UVProjection():
    def __init__(self, texture_size=96, render_size=64, sampling_mode="nearest", channels=3, device=None):
        self.channels = channels
        self.device = device or torch.device("cpu")
        self.lights = AmbientLights(ambient_color=((1.0,)*channels,), device=self.device)
        self.target_size = (texture_size,texture_size)
        self.render_size = render_size
        self.sampling_mode = sampling_mode
        self.new_face_labels = None

    def load_face_labels(self, face2label_path):
        """加载面标签并映射到新网格"""
        import os
        import json
        assert os.path.exists(face2label_path), f"Face label file not found: {face2label_path}"
        
        # 加载原始面标签
        with open(face2label_path, "r", encoding="utf-8") as f:
            original_face2label = json.load(f)
        original_face2label = {int(face_id): int(label) for face_id, label in original_face2label.items()}
        
        # 创建新网格的面标签映射
        self.new_face_labels = {}
        for orig_face_id, label in original_face2label.items():
            # 假设UV展开后每个原始面对应一个新面
            self.new_face_labels[orig_face_id] = label
    # Load obj mesh, rescale the mesh to fit into the bounding box
    def load_mesh(self, mesh_path, scale_factor=2.0, auto_center=True, autouv=False):
        mesh = load_objs_as_meshes([mesh_path], device=self.device)
        # print('mesh_vertice list is: ', mesh.verts_list()[0].shape)
        if auto_center:
            # 如果要居中，先拿到顶点
            verts = mesh.verts_packed()
            max_bb = (verts - 0).max(0)[0]
            min_bb = (verts - 0).min(0)[0]
            # print('max_bb', max_bb)
            # print('min_bb', min_bb)
            max_extent = max(max_bb - min_bb)  # 最大维度尺寸
            scale = (max_bb - min_bb).max()
            # scale = (max_bb - min_bb).max()/2
            # print('scale', scale)
            center = (max_bb+min_bb) /2
            # 集体偏移-center
            mesh.offset_verts_(-center)
            # 缩放
            # mesh.scale_verts_(scale)
            mesh.scale_verts_((scale_factor / float(scale)))		
        else:
            mesh.scale_verts_((scale_factor))

        # print('mesh_vertice list 2 is: ', mesh.verts_list()[0].shape)
        # print('scale_factor', scale_factor)
        if autouv or (mesh.textures is None):
            # 生成uv
            mesh = self.uv_unwrap(mesh)
        self.mesh = mesh
        # print('self.verts_packed is: ', self.mesh.verts_packed().shape)
        # print('self.mesh.uv is: ', len(self.mesh.textures.verts_uvs_list()))
        # print('self.mesh.verts_uvs_padded shape is: ', self.mesh.textures.verts_uvs_padded().shape)
        # print('self.mesh.textures is: ', self.mesh.textures.verts_uvs_list()[0].shape)
        # print('self.mesh.uv face is: ', len(self.mesh.textures.faces_uvs_list()))
        # print('self.mesh.textures face is: ', self.mesh.textures.faces_uvs_list()[0].shape)


    def load_glb_mesh(self, mesh_path, scale_factor=2.0, auto_center=True, autouv=False):
        from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
        io = IO()
        io.register_meshes_format(MeshGlbFormat())
        with open(mesh_path, "rb") as f:
            mesh = io.load_mesh(f, include_textures=True, device=self.device)
        if auto_center:
            verts = mesh.verts_packed()
            max_bb = (verts - 0).max(0)[0]
            min_bb = (verts - 0).min(0)[0]
            scale = (max_bb - min_bb).max()/2 
            center = (max_bb+min_bb) /2
            mesh.offset_verts_(-center)
            mesh.scale_verts_((scale_factor / float(scale)))
        else:
            mesh.scale_verts_((scale_factor))
        if autouv or (mesh.textures is None):
            mesh = self.uv_unwrap(mesh)
        self.mesh = mesh


    # Save obj mesh
    def save_mesh(self, mesh_path, texture):
        save_obj(mesh_path, 
                self.mesh.verts_list()[0],
                self.mesh.faces_list()[0],
                verts_uvs= self.mesh.textures.verts_uvs_list()[0],
                faces_uvs= self.mesh.textures.faces_uvs_list()[0],
                texture_map=texture)

    # Code referred to TEXTure code (https://github.com/TEXTurePaper/TEXTurePaper.git)
    def uv_unwrap(self, mesh):
        verts_list = mesh.verts_list()[0]
        faces_list = mesh.faces_list()[0]
        print('verts_list shape is', verts_list.shape)
        print('faces_list shape is', faces_list.shape)

        import xatlas
        import numpy as np
        v_np = verts_list.cpu().numpy()
        f_np = faces_list.int().cpu().numpy()
        atlas = xatlas.Atlas()
        atlas.add_mesh(v_np, f_np)
        chart_options = xatlas.ChartOptions()
        chart_options.max_iterations = 4
        atlas.generate(chart_options=chart_options)
        vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

        vt = torch.from_numpy(vt_np.astype(np.float32)).type(verts_list.dtype).to(mesh.device)
        ft = torch.from_numpy(ft_np.astype(np.int64)).type(faces_list.dtype).to(mesh.device)
        # print('vt shape is', vt.shape)
        # print('ft shape is', ft.shape)
        # 新纹理图，((tex_size, tex_size), channels)
        new_map = torch.zeros(self.target_size+(self.channels,), device=mesh.device)
        new_tex = TexturesUV(
            [new_map],  # (H, W, C)
            [ft], 
            [vt], 
            sampling_mode=self.sampling_mode
            )

        mesh.textures = new_tex
        return mesh


    '''
        A functions that disconnect faces in the mesh according to
        its UV seams. The number of vertices are made equal to the
        number of unique vertices its UV layout, while the faces list
        is intact.
    '''
    def disconnect_faces(self):
        mesh = self.mesh
        verts_list = mesh.verts_list()
        faces_list = mesh.faces_list()
        verts_uvs_list = mesh.textures.verts_uvs_list() # list[(N, 2)]
        faces_uvs_list = mesh.textures.faces_uvs_list() # list[(M, 3))]
        # print('verts_list shape is: ', verts_list[0].shape)
        # print('faces_list shape is: ', faces_list[0].shape)
        # print('verts_uvs_list[0] shape is: ', verts_uvs_list[0].shape)
        # print('faces_uvs_list[0] shape is: ', faces_uvs_list[0].shape)
        # print('verts_uvs_list[0][5] is: ', verts_uvs_list[0][5])
        # print('faces_uvs_list[0][5] is: ', faces_uvs_list[0][5])
        # 打包每个面的顶点
        packed_list = [v[f] for v,f in zip(verts_list, faces_list)]
        # print(len(packed_list))
        # print('packed_list[0] shape is: ', packed_list[0].shape)
        # print('packed_list shape is: ', packed_list[0][5])

        # 创建新的不连通的顶点列表，大小等于verts_uvs_list的唯一顶点数量
        verts_disconnect_list = [
            torch.zeros(
                (verts_uvs_list[i].shape[0], 3), 
                dtype=verts_list[0].dtype, 
                device=verts_list[0].device
            ) 
            for i in range(len(verts_list))]
        # print('verts_disconnect_list shape is: ', verts_disconnect_list[0].shape)
        for i in range(len(verts_list)):
            verts_disconnect_list[i][faces_uvs_list] = packed_list[i]
        # print('verts_disconnect_list[0] shape is: ', verts_disconnect_list[0].shape)
        # verts_disconnect_list[0] shape is:  torch.Size([57607, 3])
        assert not mesh.has_verts_normals(), "Not implemented for vertex normals"
        self.mesh_d = Meshes(verts_disconnect_list, faces_uvs_list, mesh.textures)
        return self.mesh_d


    '''
        A function that construct a temp mesh for back-projection.
        Take a disconnected mesh and a rasterizer, the function calculates
        the projected faces as the UV, as use its original UV with pseudo
        z value as world space geometry.
    '''
    def construct_uv_mesh(self):
        mesh = self.mesh_d # (verts_disconnect_list, faces_uvs_list) = ([57607, 3], [98998, 3])
        verts_list = mesh.verts_list()
        verts_uvs_list = mesh.textures.verts_uvs_list()
        # faces_list = [torch.flip(faces, [-1]) for faces in mesh.faces_list()]
        new_verts_list = []
        for i, (verts, verts_uv) in enumerate(zip(verts_list, verts_uvs_list)):
            verts = verts.clone()
            verts_uv = verts_uv.clone()
            # (u, v, z)
            verts[...,0:2] = verts_uv[...,:]
            # [0, 1] -> [-1, 1]
            verts = (verts - 0.5) * 2
            verts[...,2] *= 1
            new_verts_list.append(verts)
        textures_uv = mesh.textures.clone()
        self.mesh_uv = Meshes(new_verts_list, mesh.faces_list(), textures_uv)
        return self.mesh_uv


    # Set texture for the current mesh.
    def set_texture_map(self, texture):
        new_map = texture.permute(1, 2, 0)
        new_map = new_map.to(self.device)
        new_tex = TexturesUV(
            [new_map], 
            self.mesh.textures.faces_uvs_padded(), 
            self.mesh.textures.verts_uvs_padded(), 
            sampling_mode=self.sampling_mode
            )
        self.mesh.textures = new_tex


    # Set the initial normal noise texture
    # No generator here for replication of the experiment result. Add one as you wish
    def set_noise_texture(self, channels=None):
        if not channels:
            channels = self.channels
        noise_texture = torch.normal(0, 1, (channels,) + self.target_size, device=self.device)
        self.set_texture_map(noise_texture)
        return noise_texture


    # Set the cameras given the camera poses and centers
    def set_cameras(self, camera_poses, centers=None, camera_distance=10.0, scale=None):
        elev = torch.FloatTensor([pose[0] for pose in camera_poses])
        azim = torch.FloatTensor([pose[1] for pose in camera_poses])
        # rotation matrix, transform matrix
        R, T = look_at_view_transform(dist=camera_distance, elev=elev, azim=azim, at=centers or ((0,0,0),))
        # 正交投影
        self.cameras = FoVOrthographicCameras(device=self.device, R=R, T=T, scale_xyz=scale or ((0.97,0.97,0.97),))


    # Set all necessary internal data for rendering and texture baking
    # Can be used to refresh after changing camera positions
    def set_cameras_and_render_settings(self, camera_poses, centers=None, camera_distance=2.7, render_size=None, scale=None):
        self.set_cameras(camera_poses, centers, camera_distance, scale=scale)
        # print('camera_poses is: ', camera_poses)
        # print('centers is: ', centers)
        # print('camera_distance is: ', camera_distance)
        if render_size is None:
            render_size = self.render_size
        if not hasattr(self, "renderer"):
            self.setup_renderer(size=render_size)
        if not hasattr(self, "mesh_d"):
            print('???')
            self.disconnect_faces()
        if not hasattr(self, "mesh_uv"):
            self.construct_uv_mesh() # (u, v, z)
        self.calculate_tex_gradient() # self.gradient_maps, all white
        self.calculate_visible_triangle_mask() # self.visible_triangles
        _,_,_,cos_maps,_, _ = self.render_geometry()
        self.calculate_cos_angle_weights(cos_maps)


    # Setup renderers for rendering
    # max faces per bin set to 30000 to avoid overflow in many test cases.
    # You can use default value to let pytorch3d handle that for you.
    def setup_renderer(self, size=64, blur=0.0, face_per_pix=1, perspective_correct=False, channels=None):
        # 渲染大小，模糊半径，每个像素的最大面数，透视校正，渲染通道
        if not channels:
            channels = self.channels
        self.raster_settings = RasterizationSettings(
            image_size=size,
            blur_radius=0.0,  # 无抗锯齿
            faces_per_pixel=1,  # 每个像素只考虑一个面
            bin_size=0,  # 使用朴素光栅化
            perspective_correct=False,  # 根据nvdiffrast的设置调整
            clip_barycentric_coords=False,  # 不裁剪重心坐标
            cull_backfaces=False,  # 不禁用背面剔除
            z_clip_value=None,  # 禁用深度裁剪
            cull_to_frustum=False,  # 不禁用视锥裁剪
        )
        # self.raster_settings = RasterizationSettings(
        # 	image_size=size, 
        # 	blur_radius=blur, 
        # 	faces_per_pixel=face_per_pix,
        # 	perspective_correct=perspective_correct,
        # 	cull_backfaces=True,
        # 	max_faces_per_bin=30000,
        # )

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(	
                cameras=self.cameras, 
                raster_settings=self.raster_settings,

            ),
            shader=HardNChannelFlatShader(
                device=self.device, 
                cameras=self.cameras,
                lights=self.lights,
                channels=channels
                # materials=materials
            )
        )


    # Bake screen-space cosine weights to UV space
    # May be able to reimplement using the generic "bake_texture" function, but it works so leave it here for now
    @torch.enable_grad()
    def calculate_cos_angle_weights(self, cos_angles, fill=True, channels=None):
        if not channels:
            channels = self.channels
        cos_maps = []
        tmp_mesh = self.mesh.clone()
        for i in range(len(self.cameras)):
            # 遍历所有摄像机
            zero_map = torch.zeros(self.target_size+(channels,), device=self.device, requires_grad=True)
            optimizer = torch.optim.SGD([zero_map], lr=1, momentum=0)
            optimizer.zero_grad()
            zero_tex = TexturesUV([zero_map], self.mesh.textures.faces_uvs_padded(), self.mesh.textures.verts_uvs_padded(), sampling_mode=self.sampling_mode)
            tmp_mesh.textures = zero_tex
            
            images_predicted = self.renderer(tmp_mesh, cameras=self.cameras[i], lights=self.lights)

            loss = torch.sum((cos_angles[i,:,:,0:1]**1 - images_predicted)**2)
            loss.backward()
            optimizer.step()

            if fill:
                zero_map = zero_map.detach() / (self.gradient_maps[i] + 1E-8)
                zero_map = voronoi_solve(zero_map, self.gradient_maps[i][...,0])
            else:
                # uvp_rgb最后是false
                zero_map = zero_map.detach() / (self.gradient_maps[i]+1E-8)
            cos_maps.append(zero_map)
        self.cos_maps = cos_maps

        
    # Get geometric info from fragment shader
    # Can be used for generating conditioning image and cosine weights
    # Returns some information you may not need, remember to release them for memory saving
    @torch.no_grad()
    def render_geometry(self, image_size=None):
        if image_size:
            size = self.renderer.rasterizer.raster_settings.image_size
            self.renderer.rasterizer.raster_settings.image_size = image_size
        shader = self.renderer.shader
        self.renderer.shader = HardGeometryShader(device=self.device, cameras=self.cameras, lights=self.lights)

        # self.renderer.shader = HardGeometryShader(device=self.device, cameras=self.cameras[0], lights=self.lights)
        tmp_mesh = self.mesh.clone()
        verts, normals, depths, cos_angles, texels, fragments = self.renderer(tmp_mesh.extend(len(self.cameras)), cameras=self.cameras, lights=self.lights)
        self.renderer.shader = shader
        # print('verts shape is: ', verts.shape)
        # print('normals shape is: ', normals.shape)
        # print('depths shape is: ', depths.shape)
        # print('cos_angles shape is: ', cos_angles.shape)
        # print('texels shape is: ', texels.shape)
        # print('fragments shape is: ', fragments.shape)
        if image_size:
            self.renderer.rasterizer.raster_settings.image_size = size

        return verts, normals, depths, cos_angles, texels, fragments


    # Project world normal to view space and normalize
    @torch.no_grad()
    def decode_view_normal(self, normals):
        w2v_mat = self.cameras.get_full_projection_transform()
        normals_view = torch.clone(normals)[:,:,:,0:3]
        normals_view = normals_view.reshape(normals_view.shape[0], -1, 3)
        normals_view = w2v_mat.transform_normals(normals_view)
        normals_view = normals_view.reshape(normals.shape[0:3]+(3,))
        normals_view[:,:,:,2] *= -1
        normals = (normals_view[...,0:3]+1) * normals[...,3:] / 2 + torch.FloatTensor(((((0.5,0.5,1))))).to(self.device) * (1 - normals[...,3:])
        # normals = torch.cat([normal for normal in normals], dim=1)
        normals = normals.clamp(0, 1)
        return normals


    # Normalize absolute depth to inverse depth
    @torch.no_grad()
    def decode_normalized_depth(self, depths, batched_norm=False):
        view_z, mask = depths.unbind(-1)
        view_z = view_z * mask + 100 * (1-mask)
        inv_z = 1 / view_z
        inv_z_min = inv_z * mask + 100 * (1-mask)
        if not batched_norm:
            max_ = torch.max(inv_z, 1, keepdim=True)
            max_ = torch.max(max_[0], 2, keepdim=True)[0]

            min_ = torch.min(inv_z_min, 1, keepdim=True)
            min_ = torch.min(min_[0], 2, keepdim=True)[0]
        else:
            max_ = torch.max(inv_z)
            min_ = torch.min(inv_z_min)
        inv_z = (inv_z - min_) / (max_ - min_)
        inv_z = inv_z.clamp(0,1)
        inv_z = inv_z[...,None].repeat(1,1,1,3)

        return inv_z


    # Multiple screen pixels could pass gradient to a same texel
    # We can precalculate this gradient strength and use it to normalize gradients when we bake textures
    @torch.enable_grad()
    def calculate_tex_gradient(self, channels=None):
        if not channels:
            channels = self.channels
        tmp_mesh = self.mesh.clone()
        gradient_maps = []
        # 存储每个相机视角下的梯度图
        for i in range(len(self.cameras)):
            # 初始化zero_map, 大小是(texture_size, texture_size, channels)
            zero_map = torch.zeros(self.target_size+(channels,), device=self.device, requires_grad=True)
            optimizer = torch.optim.SGD([zero_map], lr=1, momentum=0)
            optimizer.zero_grad()
            zero_tex = TexturesUV([zero_map], self.mesh.textures.faces_uvs_padded(), self.mesh.textures.verts_uvs_padded(), sampling_mode=self.sampling_mode)
            tmp_mesh.textures = zero_tex
            images_predicted = self.renderer(tmp_mesh, cameras=self.cameras[i], lights=self.lights)
            # hope images_predicted更接近全白
            loss = torch.sum((1 - images_predicted)**2)
            loss.backward()
            optimizer.step()

            gradient_maps.append(zero_map.detach())

        self.gradient_maps = gradient_maps


    # Get the UV space masks of triangles visible in each view
    # First get face ids from each view, then filter pixels on UV space to generate masks
    @torch.no_grad()
    def calculate_visible_triangle_mask(self, channels=None, image_size=(512,512)):
        if not channels:
            channels = self.channels

        pix2face_list = []
        for i in range(len(self.cameras)):
            # 遍历每个相机，设置渲染器的图像大小为image_size
            self.renderer.rasterizer.raster_settings.image_size=image_size
            # 获取每个像素对应的面的id
            pix2face = self.renderer.rasterizer(self.mesh_d, cameras=self.cameras[i]).pix_to_face
            # 恢复render_size
            self.renderer.rasterizer.raster_settings.image_size=self.render_size
            # 将每个视图的 pix_to_face 存储在 pix2face_list 中。
            pix2face_list.append(pix2face)

        if not hasattr(self, "mesh_uv"):
            self.construct_uv_mesh()

        # raster_settings = RasterizationSettings(
        # 	image_size=self.target_size,
        # 	blur_radius=0.0,
        # 	faces_per_pixel=10,  # 高采样率
        # 	bin_size=0,          # 禁用优化
        # 	max_faces_per_bin=None,
        # 	perspective_correct=False,
        # 	cull_backfaces=False,  # 禁用背面剔除
        # 	z_clip_value=0.001,    # 更小的裁剪值
        # 	cull_to_frustum=False, # 禁用视锥体裁剪
        # )

        raster_settings = RasterizationSettings(
            image_size=self.target_size,
            blur_radius=0.0,  # 无抗锯齿
            faces_per_pixel=1,  # 每个像素只考虑一个面
            bin_size=0,  # 使用朴素光栅化
            perspective_correct=False,  # 根据nvdiffrast的设置调整
            clip_barycentric_coords=False,  # 不裁剪重心坐标
            cull_backfaces=False,  # 不禁用背面剔除
            z_clip_value=None,  # 禁用深度裁剪
            cull_to_frustum=False,  # 不禁用视锥裁剪
        )

        R, T = look_at_view_transform(dist=2, elev=0, azim=0)
        cameras = FoVOrthographicCameras(device=self.device, R=R, T=T)

        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )
        uv_pix2face = rasterizer(self.mesh_uv).pix_to_face

        visible_triangles = []
        for i in range(len(pix2face_list)):
            # 获取当前视图中的可见面id, 并且将背景像素设置为-1
            valid_faceid = torch.unique(pix2face_list[i])
            valid_faceid = valid_faceid[1:] if valid_faceid[0]==-1 else valid_faceid
            # bool 掩码
            mask = torch.isin(uv_pix2face[0], valid_faceid, assume_unique=False)
            # uv_pix2face[0][~mask] = -1
            triangle_mask = torch.ones(self.target_size+(1,), device=self.device)
            triangle_mask[~mask] = 0
            
            # 拓展mask, 上下左右相邻的且原来为1的也标记为1
            triangle_mask[:,1:][triangle_mask[:,:-1] > 0] = 1
            triangle_mask[:,:-1][triangle_mask[:,1:] > 0] = 1
            triangle_mask[1:,:][triangle_mask[:-1,:] > 0] = 1
            triangle_mask[:-1,:][triangle_mask[1:,:] > 0] = 1
            visible_triangles.append(triangle_mask)

        self.visible_triangles = visible_triangles



    # Render the current mesh and texture from current cameras
    def render_textured_views(self):
        meshes = self.mesh.extend(len(self.cameras))
        images_predicted = self.renderer(meshes, cameras=self.cameras, lights=self.lights)

        return [image.permute(2, 0, 1) for image in images_predicted]


    # Bake views into a texture
    # First bake into individual textures then combine based on cosine weight
    @torch.enable_grad()
    def bake_texture(self, views=None, main_views=[], cos_weighted=True, channels=None, exp=None, noisy=False, generator=None):
        if not exp:
            exp=1
        if not channels:
            channels = self.channels
        views = [view.permute(1, 2, 0) for view in views] # 改成H, W, C
        tmp_mesh = self.mesh
        bake_maps = [torch.zeros(self.target_size+(views[0].shape[2],), device=self.device, requires_grad=True) for view in views]
        optimizer = torch.optim.SGD(bake_maps, lr=1, momentum=0)
        optimizer.zero_grad()
        loss = 0
        for i in range(len(self.cameras)):    
            bake_tex = TexturesUV([bake_maps[i]], tmp_mesh.textures.faces_uvs_padded(), tmp_mesh.textures.verts_uvs_padded(), sampling_mode=self.sampling_mode)
            tmp_mesh.textures = bake_tex
            images_predicted = self.renderer(tmp_mesh, cameras=self.cameras[i], lights=self.lights, device=self.device)
            predicted_rgb = images_predicted[..., :-1]
            loss += (((predicted_rgb[...] - views[i]))**2).sum()
            # 通过每个多视角图像获得对应的back_maps
        loss.backward(retain_graph=False)
        optimizer.step()

        total_weights = 0
        baked = 0
        for i in range(len(bake_maps)):
            # 计算每个归一化的baked_map
            normalized_baked_map = bake_maps[i].detach() / (self.gradient_maps[i] + 1E-8)
            # 每个都拓展
            bake_map = voronoi_solve(normalized_baked_map, self.gradient_maps[i][...,0])
            weight = self.visible_triangles[i] * (self.cos_maps[i]) ** exp
            if noisy:
                noise = torch.rand(weight.shape[:-1]+(1,), generator=generator).type(weight.dtype).to(weight.device)
                weight *= noise
            total_weights += weight
            baked += bake_map * weight
        baked /= total_weights + 1E-8
        baked = voronoi_solve(baked, total_weights[...,0])

        bake_tex = TexturesUV([baked], tmp_mesh.textures.faces_uvs_padded(), tmp_mesh.textures.verts_uvs_padded(), sampling_mode=self.sampling_mode)
        tmp_mesh.textures = bake_tex
        extended_mesh = tmp_mesh.extend(len(self.cameras))
        images_predicted = self.renderer(extended_mesh, cameras=self.cameras, lights=self.lights)
        learned_views = [image.permute(2, 0, 1) for image in images_predicted]

        return learned_views, baked.permute(2, 0, 1), total_weights.permute(2, 0, 1)


    # Move the internel data to a specific device
    def to(self, device):
        for mesh_name in ["mesh", "mesh_d", "mesh_uv"]:
            if hasattr(self, mesh_name):
                mesh = getattr(self, mesh_name)
                setattr(self, mesh_name, mesh.to(device))
        for list_name in ["visible_triangles", "visibility_maps", "cos_maps"]:
            if hasattr(self, list_name):
                map_list = getattr(self, list_name)
                for i in range(len(map_list)):
                    map_list[i] = map_list[i].to(device)
