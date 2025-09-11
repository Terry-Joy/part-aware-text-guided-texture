# import argparse
# import json
# import os

# from PIL import Image
# from torchvision.transforms import Compose, Resize, GaussianBlur, InterpolationMode

import numpy as np
import torch
from torch.nn import functional as F

from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler, DDPMScheduler
from diffusers.models.attention_processor import Attention, AttentionProcessor


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
        )

    module.set_attn_processor(attn_processors)



class SamplewiseAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        # name="",
        custom_attention_mask=None,         # 相邻视角 mask
        ref_attention_mask=None,            # 全局参考视角 mask
        use_adjacent_baseline=True,         # 消融开关1: 原始相邻视角 attention
        use_adjacent_segment=False,         # 消融开关2: 部件级相邻 attention
        segment_mask_tokens=None,           # token-level 部件ID
        segment_weight=1.0,                 # 部件级 attention 强度
        use_ref=True,                        # 消融开关3: 全局 reference attention
        ref_weight=0
    ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        # self.name = name
        self.ref_weight = ref_weight
        self.custom_attention_mask = custom_attention_mask
        self.ref_attention_mask = ref_attention_mask
        self.use_adjacent_baseline = use_adjacent_baseline
        self.use_adjacent_segment = use_adjacent_segment
        self.segment_mask_tokens = segment_mask_tokens
        self.segment_weight = segment_weight
        self.use_ref = use_ref

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        # print(f"[Forward] {self.name} | hidden={hidden_states.shape}")
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, channels = ( # (B, S, C)
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

        query = attn.to_q(hidden_states) # (B, S, dim)

        if encoder_hidden_states is None:
            encoder_hidden_states = torch.clone(hidden_states)
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states)

        '''
        reshape encoder hidden state to a single batch
        '''
        encoder_hidden_states_f = encoder_hidden_states.reshape(
            1, -1, channels)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, 
                           head_dim).transpose(1, 2) # (B, num_head, S, head_dim)

        '''
        each time select 1 sample from q and compute with concated kv
        concat result hidden states afterwards
        '''
        hidden_state_list = []

        for b_idx in range(batch_size):
            # 取当前视角的query
            # shape (1, num_heads, seq_len, head_dim)
            query_b = query[b_idx:b_idx+1]
            # print('query_b shape:', query_b.shape)

            # --- 处理全局 reference attention ---
            if self.use_ref and self.ref_attention_mask is not None:
                key_ref = key.clone() # (num_views, seq_len, dim)
                value_ref = value.clone()
                keys = [key_ref[view_idx]
                        for view_idx in self.ref_attention_mask]
                values = [value_ref[view_idx]
                          for view_idx in self.ref_attention_mask]

                key_ref = torch.stack(keys) # (num_ref, seq_len, dim)
                key_ref = key_ref.view(key_ref.shape[0], -1, attn.heads, head_dim).permute(
                    2, 0, 1, 3 # (heads, num_ref, seq_len, head_dim)
                ).contiguous().view(attn.heads, -1, head_dim)[None, ...] # (1, heads, num_ref*seq_len, head_dim)
                value_ref = torch.stack(values)
                value_ref = value_ref.view(value_ref.shape[0], -1, attn.heads, head_dim).permute(
                    2, 0, 1, 3
                ).contiguous().view(attn.heads, -1, head_dim)[None, ...] # 

            # --- 处理相邻视角 attention ---
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

            # --- 相邻视角 baseline attention ---
            if self.use_adjacent_baseline:
                # print('query_b shape:', query_b.shape, 'key_a_group shape:', key_a_group.shape, 'value_a_group shape:', value_a_group.shape)
                hidden_adj_base = F.scaled_dot_product_attention(
                    query_b, key_a_group, value_a_group, attn_mask=None, dropout_p=0.0, is_causal=False
                )
            # --- 相邻视角 segment attention ---
            if self.use_adjacent_segment and self.segment_mask_tokens is not None:
                # print('use segment attention!')
                # print('query_b shape:', query_b.shape)
                mask_tokens = self.segment_mask_tokens  # 直接用对应层的 (H,W) mask

                seg_ids_q = mask_tokens[b_idx]  # (seq_len_q)
                seg_ids_k = torch.cat(
                    [mask_tokens[v] for v in self.custom_attention_mask[b_idx]],
                    dim=0
                )  # (seq_len_k)

                # seg_attn_mask = (
                #     (seg_ids_q[:, None] != seg_ids_k[None, :]).float()
                #     * -1e9 * self.segment_weight
                # ).to(query_b.dtype)

                # seg_attn_mask = seg_attn_mask[None, None, :, :]       # (1, 1, S_q, S_k)
                # seg_attn_mask = seg_attn_mask.expand(1, attn.heads, -1, -1)  # (1, H, S_q, S_k)


                sim = (seg_ids_q[:, None] == seg_ids_k[None, :]).float()
                # 相同部件 → 0 bias（无惩罚）
                # 不同部件 → 加一个小的负偏置，而不是 -1e9
                seg_attn_mask = (1 - sim) * (-self.segment_weight)   # 例如 -0.5 ~ -2
                seg_attn_mask = seg_attn_mask[None, None, :, :].expand(1, attn.heads, -1, -1).to(query_b.dtype)

                # print('seg_attn_mask shape:', seg_attn_mask.shape)
                # print('segment query_b shape:', query_b.shape, 'segment_key_a_group shape:', key_a_group.shape, 'segment_value_a_group shape:', value_a_group.shape)
                hidden_adj_seg = F.scaled_dot_product_attention(
                    query_b, key_a_group, value_a_group,
                    attn_mask=seg_attn_mask,
                    dropout_p=0.0,
                    is_causal=False
                )
            # --- 全局 reference attention ---
            if self.use_ref and self.ref_attention_mask is not None:
                hidden_state_ref = F.scaled_dot_product_attention(
                    query_b, key_ref, value_ref, attn_mask=None, dropout_p=0.0, is_causal=False
                )



            # --- 最终融合 ---
            hidden_state = 0
            total_weight = 0

            if self.use_adjacent_baseline:
                hidden_state += hidden_adj_base * 1
                total_weight += 1
            if self.use_adjacent_segment:
                hidden_state += hidden_adj_seg
                total_weight += 1
            if self.use_ref and self.ref_attention_mask is not None:
                hidden_state += self.ref_weight * hidden_state_ref
                total_weight += self.ref_weight

            hidden_state = hidden_state / total_weight

            hidden_state_list.append(hidden_state)

        hidden_states = torch.cat(hidden_state_list)

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(
                -1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

# [Forward] down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor | hidden=torch.Size([6, 9216, 320])
# [Forward] down_blocks.0.attentions.1.transformer_blocks.0.attn1.processor | hidden=torch.Size([6, 9216, 320])
# [Forward] down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor | hidden=torch.Size([6, 2304, 640])
# [Forward] down_blocks.1.attentions.1.transformer_blocks.0.attn1.processor | hidden=torch.Size([6, 2304, 640])
# [Forward] down_blocks.2.attentions.0.transformer_blocks.0.attn1.processor | hidden=torch.Size([6, 576, 1280])
# [Forward] down_blocks.2.attentions.1.transformer_blocks.0.attn1.processor | hidden=torch.Size([6, 576, 1280])
# [Forward] mid_block.attentions.0.transformer_blocks.0.attn1.processor | hidden=torch.Size([6, 144, 1280])
# [Forward] up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor | hidden=torch.Size([6, 576, 1280])
# [Forward] up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor | hidden=torch.Size([6, 576, 1280])
# [Forward] up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor | hidden=torch.Size([6, 576, 1280])
# [Forward] up_blocks.2.attentions.0.transformer_blocks.0.attn1.processor | hidden=torch.Size([6, 2304, 640])
# [Forward] up_blocks.2.attentions.1.transformer_blocks.0.attn1.processor | hidden=torch.Size([6, 2304, 640])
# [Forward] up_blocks.2.attentions.2.transformer_blocks.0.attn1.processor | hidden=torch.Size([6, 2304, 640])
# [Forward] up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor | hidden=torch.Size([6, 9216, 320])
# [Forward] up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor | hidden=torch.Size([6, 9216, 320])
# [Forward] up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor | hidden=torch.Size([6, 9216, 320])