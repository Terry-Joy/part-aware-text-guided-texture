import os
import torch
import torchvision.transforms as T
from PIL import Image
from typing import List, Tuple, Optional

from renderer.project import UVProjection as UVP


def save_image_tensor(img_tensor: torch.Tensor, out_path: str):
    """
    Accepts CHW (C,H,W) or HWC (H,W,C). Values expected in [0,1] float.
    """
    if img_tensor.dim() == 3 and img_tensor.shape[0] in (1, 3, 4):
        img = img_tensor.permute(1, 2, 0).cpu().clamp(0, 1).numpy()
    else:
        img = img_tensor.cpu().clamp(0, 1).numpy()

    pil = Image.fromarray((img * 255).astype("uint8"))
    pil.save(out_path)


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


class UVPPipeline:
    """
    Lightweight wrapper to load an .obj (with .mtl & textures) and render views using UVProjection.
    """
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.uvp = None

    def initialize_uvp_from_obj(
        self,
        mesh_path: str,
        texture_size: int = 1536,
        render_size: int = 512,
        channels: int = 3,
        autouv: bool = False,
        scale_factor: float = 1.0,
        auto_center: bool = True,
    ):
        assert os.path.exists(mesh_path), f"mesh not found: {mesh_path}"
        self.uvp = UVP(texture_size=texture_size, render_size=render_size, sampling_mode="nearest",
                       channels=channels, device=self.device)
        self.uvp.load_mesh(mesh_path, scale_factor=scale_factor, auto_center=auto_center, autouv=autouv)
        self.uvp.to(self.device)

    def set_cameras(
        self,
        azims: List[float],
        elevs: Optional[List[float]] = None,
        centers: Optional[Tuple[float, float, float]] = (0.0, 0.0, 0.0),
        camera_distance: float = 2.0,
        render_size: Optional[int] = None,
        scale: Optional[Tuple[float, float, float]] = None,
    ):
        if self.uvp is None:
            raise RuntimeError("UVP not initialized. Call initialize_uvp_from_obj first.")
        if elevs is None:
            elevs = [0.0] * len(azims)
        assert len(elevs) == len(azims), "elevs must match azims length"
        camera_poses = [(e, a) for e, a in zip(elevs, azims)]
        self.uvp.set_cameras_and_render_settings(camera_poses, centers=None,
                                                 camera_distance=camera_distance, render_size=None or None, scale=None)

    def render_and_save(
        self,
        out_dir: str,
        prefix: str = "view",
        save_rgb: bool = True,
        return_tensors: bool = True,
        concat_mode: str = "both",  # ✅ 新增选项: "single" / "concat" / "both"
    ):
        """
        Render current textured mesh from the cameras previously set and save views as png.
        Results will be saved under out_dir/prefix/*.png
        Returns list of CHW tensors (C,H,W) if return_tensors True.

        concat_mode:
            "single" - 仅保存单张图
            "concat" - 仅保存拼接图 (水平)
            "both"   - 两者都保存
        """
        if self.uvp is None:
            raise RuntimeError("UVP not initialized.")

        # ✅ 新逻辑：输出目录为 out_dir/prefix/
        output_dir = os.path.join(out_dir, prefix)
        ensure_dir(output_dir)

        if not hasattr(self.uvp, "renderer"):
            self.uvp.setup_renderer(size=self.uvp.render_size, channels=self.uvp.channels)

        views = self.uvp.render_textured_views()
        saved_paths = []

        # ========== 保存单张视图 ==========
        if concat_mode in ("single", "both"):
            for i, view in enumerate(views):
                out_path = os.path.join(output_dir, f"{prefix}_{i:02d}.png")
                save_image_tensor(view, out_path)
                saved_paths.append(out_path)

        # ========== 拼接保存 ==========
        if concat_mode in ("concat", "both"):
            pil_imgs = []
            for v in views:
                if v.dim() == 3 and v.shape[0] in (1, 3, 4):
                    img = v.permute(1, 2, 0).cpu().clamp(0, 1).numpy()
                else:
                    img = v.cpu().clamp(0, 1).numpy()
                pil_imgs.append(Image.fromarray((img * 255).astype("uint8")))

            widths, heights = zip(*(im.size for im in pil_imgs))
            total_width = sum(widths)
            max_height = max(heights)

            concat_img = Image.new("RGB", (total_width, max_height))
            x_offset = 0
            for im in pil_imgs:
                concat_img.paste(im, (x_offset, 0))
                x_offset += im.width

            concat_path = os.path.join(output_dir, f"{prefix}_concat.png")
            concat_img.save(concat_path)
            saved_paths.append(concat_path)

        if return_tensors:
            return views, saved_paths
        else:
            return saved_paths


if __name__ == "__main__":
    import glob
    # ours_best
    # obj_dir = "/home/zhangtianle/text-to-texture/SyncMVD/exp/buquan/49d0b382def54ff1b544cc6293ea1b22/ours_best/gs_15.5/results/after_inpainting" 

    # mvpaint
    # obj_dir = "/home/zhangtianle/text-to-texture/SyncMVD/exp/buquan/49d0b382def54ff1b544cc6293ea1b22/mvpaint_modified_byours/gs_15.5/results/after_inpainting" 

    # 2d_voronoi
    # obj_dir = "/home/zhangtianle/text-to-texture/SyncMVD/exp/buquan/49d0b382def54ff1b544cc6293ea1b22/voroni_solve_2d" 

    # TEXTURE exp
    # obj_dir = "/home/zhangtianle/text-to-texture/SyncMVD/duibi_experiment/texture_experiments/f0742330720b49e59740835a12380219/mesh"
    # out_dir = "/home/zhangtianle/text-to-texture/SyncMVD/exp/duibi/f0742330720b49e59740835a12380219/TEXTure_img"

    # Text2tex exp
    # obj_dir = "/home/zhangtianle/text-to-texture/SyncMVD/exp/duibi/text2tex_syncmvd_ablation_1/texture_outputs/f0742330720b49e59740835a12380219/20251012_165335/42-p36-h20-1.0-0.3-0.1/update/mesh"
    # out_dir = "/home/zhangtianle/text-to-texture/SyncMVD/exp/duibi/f0742330720b49e59740835a12380219/Text2tex_img"

    # ours exp
    # obj_dir = "/home/zhangtianle/text-to-texture/SyncMVD/inpaint_test_data/duibibuquan/origin" 
    # out_dir = "/home/zhangtianle/text-to-texture/SyncMVD/inpaint_test_data/duibibuquan/origin/rendered_views"
    # obj_dir = "/home/zhangtianle/text-to-texture/SyncMVD/inpaint_test_data/duibibuquan/ours_real" 
    # out_dir = "/home/zhangtianle/text-to-texture/SyncMVD/inpaint_test_data/duibibuquan/ours_real/rendered_views"
    # obj_dir = "/home/zhangtianle/text-to-texture/SyncMVD/inpaint_test_data/duibi_8a5c0fd0d4bc45128f3ee5c65a32e54f_prompt1_seed1" 
    # out_dir = "/home/zhangtianle/text-to-texture/SyncMVD/inpaint_test_data/duibi_8a5c0fd0d4bc45128f3ee5c65a32e54f_prompt1_seed1/rendered_views"
    # obj_dir = "/home/zhangtianle/text-to-texture/SyncMVD/inpaint_test_data/duibi_8a5c0fd0d4bc45128f3ee5c65a32e54f_prompt1_seed1/castex_res" 
    # out_dir = "/home/zhangtianle/text-to-texture/SyncMVD/inpaint_test_data/duibi_8a5c0fd0d4bc45128f3ee5c65a32e54f_prompt1_seed1/castex_res/rendered_views"
    # obj_dir = "/home/zhangtianle/text-to-texture/SyncMVD/inpaint_test_data/duibi_8a5c0fd0d4bc45128f3ee5c65a32e54f_prompt1_seed1/mvpaint_res" 
    # out_dir = "/home/zhangtianle/text-to-texture/SyncMVD/inpaint_test_data/duibi_8a5c0fd0d4bc45128f3ee5c65a32e54f_prompt1_seed1/mvpaint_res/rendered_views"   
    # obj_dir = "/home/zhangtianle/text-to-texture/SyncMVD/inpaint_test_data/duibi_8a5c0fd0d4bc45128f3ee5c65a32e54f_prompt1_seed1/paint3d_res" 
    # out_dir = "/home/zhangtianle/text-to-texture/SyncMVD/inpaint_test_data/duibi_8a5c0fd0d4bc45128f3ee5c65a32e54f_prompt1_seed1/paint3d_res/rendered_views"
    # obj_dir = "/home/zhangtianle/text-to-texture/SyncMVD/inpaint_test_data/duibi_8a5c0fd0d4bc45128f3ee5c65a32e54f_prompt1_seed1/texture" 
    # out_dir = "/home/zhangtianle/text-to-texture/SyncMVD/inpaint_test_data/duibi_8a5c0fd0d4bc45128f3ee5c65a32e54f_prompt1_seed1/texture/rendered_views"  
    obj_dir = "/root/autodl-tmp/part-aware-text-guided-texture/inpaint_data/07fe/" 
    out_dir = "/root/autodl-tmp/part-aware-text-guided-texture/inpaint_data/07fe/no_adj_rendered_views"  

    # obj_dir = "/home/zhangtianle/text-to-texture/SyncMVD/exp" 
    # out_dir = "/home/zhangtianle/text-to-texture/SyncMVD/exp/test_render"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    obj_files = glob.glob(f"{obj_dir}/textured.obj")
    if len(obj_files) == 0:
        raise FileNotFoundError(f"No OBJ found in {obj_dir}")
    elif len(obj_files) > 1:
        print(f"Multiple OBJ files found, using the first one: {obj_files[0]}")

    mesh_path = obj_files[0]

    pipeline = UVPPipeline(device=device)
    pipeline.initialize_uvp_from_obj(
        mesh_path,
        texture_size=1024,
        render_size=512,
        channels=3,
        autouv=False,
        scale_factor=2.0,
        auto_center=True
    )

    # azims = [0, 90, 180, 270, 180, 180, 0, 0]
    # elevs = [0, 0, 0, 0, 30, -30, 30, -60]
    # azims = [210, 240]
    # elevs = [0, 0]
    azims = [0, 60, 120, 180, 240, 300]
    elevs = [0, 0, 0, 0, 0, 0]

    pipeline.set_cameras(
        azims=azims,
        elevs=elevs,
        centers=(0, 0, 0),
        camera_distance=2.2
    )

    # ✅ 控制输出模式：single / concat / both
    # views, paths = pipeline.render_and_save(out_dir, prefix="part_aware_3dinpaint_ours_best", concat_mode="concat")
    # views, paths = pipeline.render_and_save(out_dir, prefix="mvpaint_modified_byours", concat_mode="concat")
    # views, paths = pipeline.render_and_save(out_dir, prefix="voroni_solve_2d", concat_mode="concat")
    views, paths = pipeline.render_and_save(out_dir, prefix="2fd7954d31dc47d7a48f9e0cb8faeeba", concat_mode="single")

    print("Saved rendered views:", paths)
