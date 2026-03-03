import torch
from torch.nn import functional as F
from diffusers.models.attention_processor import Attention

def replace_attention_processors(
    module,
    processor,
    attention_mask=None,
    ref_attention_mask=None,
    use_adjacent_baseline=True,
    use_adjacent_segment=False,
    segment_mask_tokens=None,   # dict {(H,W): tensor[num_views, seq_len]}
    segment_weight=0.0,
    use_ref=True,
    ref_weight=0,
    yizhi=False,
    yizhi_weight=0.0,
    resolution=1024
):
    attn_processors = module.attn_processors

    # latent 基准尺寸（SD1.5@512→64，@768→96，@1024→128）
    base = resolution // 8
    # print('base is:', base)
    n_down = len(getattr(module, "down_blocks", [])) or 4  # 通常是 4

    def hw_from_key(k: str):
        if "down_blocks" in k:
            b = int(k.split(".")[1])          # 0,1,2,3
            scale = 2 ** b                    # 1,2,4,8
            # print('b, is, ', b, base, base // scale)
            # print('down_blocks k is: ', k)
            # print('down block name is: ', k, base // scale, base // scale)
            return base // scale, base // scale
        elif "mid_block" in k:
            # 最低分辨率
            # print('mid_block k is: ', k)
            # print('mid block name is: ', k, base // (2 ** (n_down - 1)), base // (2 ** (n_down - 1)))
            return base // (2 ** (n_down - 1)), base // (2 ** (n_down - 1))
        elif "up_blocks" in k:
            b = int(k.split(".")[1])          # 0,1,2,3
            # print('b is: ',b)
            scale = 2 ** (n_down - b - 1)                    # 和 down 一样
            # print('up block name is: ', k, base // scale, base // scale)
            # print('up_blocks k is: ', k, scale, base, base // scale)
            return base // scale, base // scale
        else:
            return None

    for k, v in attn_processors.items():
        if "attn1" not in k:
            continue
        # print('k is: ', k)
        hw = hw_from_key(k)
        if hw is None:
            continue
        H, W = hw

        seg_tokens = None
        if isinstance(segment_mask_tokens, dict) and len(segment_mask_tokens) > 0:
            if (H, W) in segment_mask_tokens:
                seg_tokens = segment_mask_tokens[(H, W)]
            # else:
            #     # 兜底：找最接近的 (H,W)
            #     avail = list(segment_mask_tokens.keys())
            #     nearest = min(avail, key=lambda s: abs(s[0]*s[1] - H*W))
            #     seg_tokens = segment_mask_tokens[nearest]

        attn_processors[k] = processor(
            # name=k,
            custom_attention_mask=attention_mask,
            ref_attention_mask=ref_attention_mask,
            ref_weight=ref_weight,
            use_adjacent_baseline=use_adjacent_baseline,
            use_adjacent_segment=use_adjacent_segment,
            segment_mask_tokens=seg_tokens,
            segment_weight=segment_weight,
            use_ref=use_ref,
            yizhi=yizhi,
            yizhi_weight=yizhi_weight,
        )

    module.set_attn_processor(attn_processors)

class SamplewiseAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        custom_attention_mask=None,         # 相邻视角索引 [[0,1], [1,2]...]
        ref_attention_mask=None,            # 全局参考视角索引
        use_adjacent_baseline=True,         # 主开关：相邻视角
        use_adjacent_segment=False,         # 废弃开关：已合并入 baseline，设为 False 即可
        segment_mask_tokens=None,           # token-level 部件ID
        segment_weight=1.0,                 # (废弃参数)
        use_ref=True,                       # Ref 开关
        ref_weight=0,                       # Ref 权重
        yizhi=False,                        # 抑制开关
        yizhi_weight=0.0,                   # 抑制强度 (软惩罚)
    ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        self.ref_weight = ref_weight
        self.custom_attention_mask = custom_attention_mask
        self.ref_attention_mask = ref_attention_mask
        self.use_adjacent_baseline = use_adjacent_baseline
        # use_adjacent_segment 逻辑上有冲突，我们在内部逻辑中将其视为 False，避免重复计算
        self.use_adjacent_segment = False 
        self.segment_mask_tokens = segment_mask_tokens
        self.segment_weight = segment_weight
        self.use_ref = use_ref
        self.yizhi = yizhi
        self.yizhi_weight = yizhi_weight

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, channels = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = torch.clone(hidden_states)
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states)

        encoder_hidden_states_f = encoder_hidden_states.reshape(
            1, -1, channels)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, 
                           head_dim).transpose(1, 2) # (B, num_head, S, head_dim)

        hidden_state_list = []
        
        # === 逐个 Sample 处理 ===
        for b_idx in range(batch_size):
            # 取当前视角的 query (1, heads, seq_len, dim)
            query_b = query[b_idx:b_idx+1]

            # ---------------------------------------------------
            # 1. Ref Attention (全局参考) - 必须保留
            # ---------------------------------------------------
            hidden_state_ref = 0
            if self.use_ref and self.ref_attention_mask is not None:
                print('use_ref')
                key_ref = key.clone()
                value_ref = value.clone()
                keys = [key_ref[view_idx] for view_idx in self.ref_attention_mask]
                values = [value_ref[view_idx] for view_idx in self.ref_attention_mask]

                key_ref = torch.stack(keys).view(len(keys), -1, attn.heads, head_dim).permute(
                    2, 0, 1, 3).contiguous().view(attn.heads, -1, head_dim)[None, ...]
                value_ref = torch.stack(values).view(len(values), -1, attn.heads, head_dim).permute(
                    2, 0, 1, 3).contiguous().view(attn.heads, -1, head_dim)[None, ...]

                # Ref 通常不需要 Mask，直接看全局
                hidden_state_ref = F.scaled_dot_product_attention(
                    query_b, key_ref, value_ref, attn_mask=None, dropout_p=0.0, is_causal=False
                )

            # ---------------------------------------------------
            # 2. Adjacent Attention (相邻视角 + 你的抑制机制)
            # ---------------------------------------------------
            hidden_adj_base = 0
            if self.use_adjacent_baseline:
                key_a = key.clone()
                value_a = value.clone()
                keys = [key_a[view_idx]
                        for view_idx in self.custom_attention_mask[b_idx]]
                values = [value_a[view_idx]
                          for view_idx in self.custom_attention_mask[b_idx]]
                key_a_group = torch.stack(keys) # (num_adj, seq_len, dim)
                value_a_group = torch.stack(values)

                key_a_group = key_a_group.view(key_a_group.shape[0], -1, attn.heads, head_dim).permute(
                    2, 0, 1, 3
                ).contiguous().view(attn.heads, -1, head_dim)[None, ...] # (1, heads, num_adj*seq_len, head_dim)
                value_a_group = value_a_group.view(value_a_group.shape[0], -1, attn.heads, head_dim).permute(
                    2, 0, 1, 3
                ).contiguous().view(attn.heads, -1, head_dim)[None, ...]

                # --- 制作 Attention Bias ---
                attn_bias = None
                
                # 只要有 mask tokens，我们就得处理背景（不管开不开抑制）
                if self.segment_mask_tokens is not None:
                    mask_tokens = self.segment_mask_tokens
                    seg_ids_q = mask_tokens[b_idx]      # (S_q)
                    seg_ids_k = torch.cat(              # (S_k)
                        [mask_tokens[v] for v in self.custom_attention_mask[b_idx]], dim=0
                    )
                    
                    # ======================================================
                    # 步骤 A: 确定基础 Bias (处理部件抑制)
                    # ======================================================
                    if self.yizhi:
                        # 开启抑制：计算相似度
                        # 【修正点】不要用 .float()，而是用 .to(query_b.dtype)
                        # 这样如果 query 是 FP16，sim 也会是 FP16
                        sim = (seg_ids_q[:, None] == seg_ids_k[None, :]).to(dtype=query_b.dtype)
                        
                        # 此时 attn_bias 也会自动继承 query_b.dtype
                        attn_bias = (1 - sim) * (-self.yizhi_weight)
                    else:
                        # 关闭抑制：创建一个全 0 矩阵
                        # 这里你原本写的对了，但为了保险，保持逻辑一致
                        attn_bias = torch.zeros(
                            (seg_ids_q.shape[0], seg_ids_k.shape[0]), 
                            device=query_b.device, 
                            dtype=query_b.dtype  # 确保类型一致
                        )
                    attn_bias = attn_bias[None, None, :, :]
                    # ======================================================
                    # 步骤 B: 背景硬屏蔽 (永久生效，独立于 yizhi)
                    # ======================================================
                    # 假设 ID=0 是背景，直接给 -inf
#                     bg_id = 0
#                     is_bg_k = (seg_ids_k == bg_id)
                    
#                     # 广播填入：不管 A 步骤算出来是多少，背景位置强行覆盖为 -inf
#                     attn_bias.masked_fill_(is_bg_k[None, None, None, :], -2.3)

                    # ======================================================
                    # 步骤 C: 形状调整
                    # ======================================================
                    # (S_q, S_k) -> (1, 1, S_q, S_k)
                    # attn_bias = attn_bias[None, None, :, :]
                    attn_bias = attn_bias.expand(1, attn.heads, -1, -1).to(query_b.dtype)

                # 计算 Attention
                hidden_adj_base = F.scaled_dot_product_attention(
                    query_b, key_a_group, value_a_group, attn_mask=attn_bias, dropout_p=0.0, is_causal=False
                )

            # ---------------------------------------------------
            # 3. 最终融合 (加权平均)
            # ---------------------------------------------------
            hidden_state = 0
            total_weight = 0

            # 加 Baseline (相邻)
            if self.use_adjacent_baseline:
                hidden_state += hidden_adj_base
                total_weight += 1
            
            # 加 Ref (全局参考)
            if self.use_ref and self.ref_attention_mask is not None:
                hidden_state += self.ref_weight * hidden_state_ref
                total_weight += self.ref_weight

            # 归一化
            if total_weight > 0:
                hidden_state = hidden_state / total_weight
            
            hidden_state_list.append(hidden_state)

        # === 整理输出 ===
        hidden_states = torch.cat(hidden_state_list)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(
                -1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

# 保持 replace_attention_processors 不变，
# 只要调用时传参 yizhi=True, yizhi_weight=2.0 即可