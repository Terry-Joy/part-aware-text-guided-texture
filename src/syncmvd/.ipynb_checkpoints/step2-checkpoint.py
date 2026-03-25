import torch
from diffusers.utils import randn_tensor

@torch.no_grad()
def step_tex(
        scheduler,
        uvp,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        texture: None,
        generator=None,
        return_dict: bool = True,
        guidance_scale = 1,
        main_views = [],
        hires_original_views = True,
        exp=None,
        cos_weighted=True,
        # 👇 新增的两个编辑参数
        painted_views=None, 
        painted_mask=None
):
    t = timestep
    prev_t = scheduler.previous_timestep(t)

    if model_output.shape[1] == sample.shape[1] * 2 and scheduler.variance_type in ["learned", "learned_range"]:
        model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
    else:
        predicted_variance = None

    # 1. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else scheduler.one
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    # 2. compute predicted original sample from predicted noise
    if scheduler.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    elif scheduler.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif scheduler.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {scheduler.config.prediction_type} must be one of `epsilon`, `sample` or `v_prediction`"
        )

    # 3. Clip or threshold "predicted x_0"
    if scheduler.config.thresholding:
        pred_original_sample = scheduler._threshold_sample(pred_original_sample)
    elif scheduler.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -scheduler.config.clip_sample_range, scheduler.config.clip_sample_range
        )

    # 4. Compute coefficients
    pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
    current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

    # 5. Bake Original and Texture
    if texture is None:
        sample_views = [view for view in sample]
        sample_views, texture, _ = uvp.bake_texture(views=sample_views, main_views=main_views, exp=exp)
        sample_views = torch.stack(sample_views, axis=0)[:,:-1,...]

    original_views = [view for view in pred_original_sample]
    original_views, original_tex, visibility_weights = uvp.bake_texture(views=original_views, main_views=main_views, exp=exp)
    uvp.set_texture_map(original_tex)
    original_views = uvp.render_textured_views()
    original_views = torch.stack(original_views, axis=0)[:,:-1,...]

    # 6. Compute predicted previous sample µ_t
    prev_tex = pred_original_sample_coeff * original_tex + current_sample_coeff * texture

    # 7. Add variance noise
    variance = 0
    if predicted_variance is not None:
        variance_views = [view for view in predicted_variance]
        variance_views, variance_tex, visibility_weights = uvp.bake_texture(views=variance_views, main_views=main_views, cos_weighted=cos_weighted, exp=exp)
        variance_views = torch.stack(variance_views, axis=0)[:,:-1,...]
    else:
        variance_tex = None

    if t > 0:
        device = texture.device
        variance_noise = randn_tensor(
            texture.shape, generator=generator, device=device, dtype=texture.dtype
        )
        if scheduler.variance_type == "fixed_small_log":
            variance = scheduler._get_variance(t, predicted_variance=variance_tex) * variance_noise
        elif scheduler.variance_type == "learned_range":
            variance = scheduler._get_variance(t, predicted_variance=variance_tex)
            variance = torch.exp(0.5 * variance) * variance_noise
        else:
            variance = (scheduler._get_variance(t, predicted_variance=variance_tex) ** 0.5) * variance_noise

    prev_tex = prev_tex + variance

    uvp.set_texture_map(prev_tex)
    prev_views = uvp.render_textured_views()
    
    # 拿到模型预测的 t-1 步的 views
    pred_prev_sample = torch.clone(sample)
    for i, view in enumerate(prev_views):
        pred_prev_sample[i] = view[:-1]

    # 👇 -------- 核心：简单版 RePaint 融合逻辑 --------
    if painted_views is not None and painted_mask is not None:
        # 1. 🚨 防御性编程：强制对齐设备和数据类型，防止 GPU/CPU 混用报错
        p_v = painted_views.to(device=sample.device, dtype=sample.dtype)
        p_m = painted_mask.to(device=sample.device, dtype=sample.dtype)
        
        if prev_t >= 0:
            # 拿到 t-1 步的 tensor
            prev_t_tensor = torch.tensor([prev_t], device=sample.device, dtype=torch.long)
            
            # 给干净原图加噪到 t-1
            noise = randn_tensor(p_v.shape, generator=generator, device=sample.device, dtype=p_v.dtype)
            noisy_known = scheduler.add_noise(p_v, noise, prev_t_tensor)
        else:
            # 2. 🚨 补漏：如果是最后一步出图 (prev_t < 0)，不需要任何噪声！
            # 直接拿纯净的特征图去拼贴，保证保留区域 100% 还原。
            noisy_known = p_v
            
        # 3. 强制拼贴：1 代表保留原图，0 代表交给模型重绘
        pred_prev_sample = noisy_known * p_m + pred_prev_sample * (1.0 - p_m)
        
        # 4. 把拼好的图再烤一次，确保 3D 视角不穿帮
        sample_views_list = [v for v in pred_prev_sample]
        _, merged_tex, _ = uvp.bake_texture(views=sample_views_list, main_views=main_views, exp=exp)
        prev_tex = merged_tex
    # 👆 ------------------------------------------

    if return_dict:
        return {"prev_sample": pred_prev_sample, "pred_original_sample": pred_original_sample, "prev_tex": prev_tex}
    return pred_prev_sample, pred_original_sample