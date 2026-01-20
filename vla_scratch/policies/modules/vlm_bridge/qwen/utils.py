import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from contextlib import contextmanager
import einops

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionModel,
    Qwen3VLVisionAttention,
    apply_rotary_pos_emb_vision,
    dynamic_rope_update,
)

from vla_scratch.policies.utils.transformers import apply_rotary_pos_emb
from vla_scratch.policies.utils.training import apply_checkpoint_when_training


def _qwen3vl_rot_pos_emb(
    self: "Qwen3VLVisionModel", grid_thw_list: List[Tuple[int, int, int]]
) -> torch.Tensor:
    """GPU-friendly rotary position builder without .item()/.tolist() syncs."""
    torch.cuda.nvtx.range_push("custom-rot_pos_emb")
    # freq_table = self.rotary_pos_emb(32)
    freq_table = self.prepared_freq_table

    merge_size = self.spatial_merge_size
    t, h, w = grid_thw_list[0]
    h_range = torch.arange(h, device=self.device)
    w_range = torch.arange(w, device=self.device)
    row_idx, col_idx = torch.meshgrid(h_range, w_range, indexing="ij")

    merged_h, merged_w = h // merge_size, w // merge_size
    row_idx_block = einops.rearrange(
        row_idx,
        "(m_h m_1) (m_w m_2) -> m_h m_w m_1 m_2",
        m_h=merged_h,
        m_w=merged_w,
        m_1=merge_size,
        m_2=merge_size,
    )
    col_idx_block = einops.rearrange(
        col_idx,
        "(m_h m_1) (m_w m_2) -> m_h m_w m_1 m_2",
        m_h=merged_h,
        m_w=merged_w,
        m_1=merge_size,
        m_2=merge_size,
    )

    coords = torch.stack(
        (row_idx_block.reshape(-1), col_idx_block.reshape(-1)), dim=-1
    )
    if t > 1:
        coords = coords.repeat(t, 1)
    pos_ids = einops.repeat(coords, "p d -> (b p) d", b=len(grid_thw_list))
    result = freq_table[pos_ids].flatten(1)
    torch.cuda.nvtx.range_pop()
    return result


def _qwen3vl_fast_pos_embed_interpolate(
    self: "Qwen3VLVisionModel", grid_thw_list: List[Tuple[int, int, int]]
) -> torch.Tensor:
    """Optimized bilinear pos embedding interpolation on CUDA (no host sync)."""
    torch.cuda.nvtx.range_push("custom-pos_embed_interp")
    num_grid_per_side = self.num_grid_per_side
    m_size = self.spatial_merge_size
    hidden_dim = self.pos_embed.embedding_dim
    device = self.pos_embed.weight.device

    t, h, w = grid_thw_list[0]

    def compute_indices_weights(h: int, w: int):
        h_idxs = torch.linspace(
            0, num_grid_per_side - 1, h, dtype=torch.float32, device=device
        )
        w_idxs = torch.linspace(
            0, num_grid_per_side - 1, w, dtype=torch.float32, device=device
        )

        h_floor = h_idxs.to(torch.long)
        w_floor = w_idxs.to(torch.long)
        h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
        w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

        dh = h_idxs - h_floor
        dw = w_idxs - w_floor

        dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
        h_floor_grid, w_floor_grid = torch.meshgrid(
            h_floor, w_floor, indexing="ij"
        )
        h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

        w11 = dh_grid * dw_grid
        w10 = dh_grid - w11
        w01 = dw_grid - w11
        w00 = 1 - dh_grid - w01

        h_grid = torch.stack(
            [h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid]
        )
        w_grid = torch.stack(
            [w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid]
        )
        h_grid_idx = h_grid * num_grid_per_side

        indices = (h_grid_idx + w_grid).reshape(4, -1)
        # shape: [4, num_patches]
        weights = (
            torch.stack([w00, w01, w10, w11], dim=0)
            .reshape(4, -1, 1)
            .to(dtype=self.pos_embed.weight.dtype)
        )
        # shape: [4, num_patches, 1]
        return indices, weights

    indices, weights = compute_indices_weights(h, w)
    indices = einops.repeat(indices, "four p -> four b p", b=len(grid_thw_list))
    weights = einops.repeat(
        weights, "four p 1 -> four b p 1", b=len(grid_thw_list)
    )

    embeds = self.pos_embed(indices) * weights
    combined = embeds.sum(dim=0)
    # shape: [b, num_patches, hidden_dim]
    combined = einops.rearrange(
        combined,
        "b (h m1 w m2) d -> b h w m1 m2 d",
        h=h // m_size,
        w=w // m_size,
        m1=m_size,
        m2=m_size,
    )
    repeated = einops.repeat(
        combined,
        "b h w m1 m2 d -> b t h w m1 m2 d",
        t=t,
    )
    torch.cuda.nvtx.range_pop()
    return repeated.reshape(-1, hidden_dim)


def _qwen3_vision_attn_fast_forward(
    self: "Qwen3VLVisionAttention",
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    grid_thw = cu_seqlens
    torch.cuda.nvtx.range_push("vision_attention")
    torch.cuda.nvtx.range_push("qkv_projection")
    qkv = einops.rearrange(
        self.qkv(hidden_states),
        "b_seq (three head dim) -> b_seq three head dim",
        three=3,
        head=self.num_heads,
    )
    # qkv = self.qkv(hidden_states).view(-1, 3, self.num_heads, self.head_dim)
    query_states, key_states, value_states = qkv.unbind(1)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(
        query_states, key_states, cos, sin
    )
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("attn")
    batch_size = (
        grid_thw.shape[0]
        if isinstance(grid_thw, torch.Tensor)
        else len(grid_thw)
    )
    query_states = einops.rearrange(
        query_states, "(b l) h d -> b h l d", b=batch_size
    )
    key_states = einops.rearrange(
        key_states, "(b l) h d -> b h l d", b=batch_size
    )
    value_states = einops.rearrange(
        value_states, "(b l) h d -> b h l d", b=batch_size
    )
    attn_out = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=None,
        is_causal=False,
        scale=self.scaling,
        enable_gqa=True,
    )
    attn_out = einops.rearrange(attn_out, "b h l d -> (b l) (h d)").contiguous()
    torch.cuda.nvtx.range_pop()

    attn_output = self.proj(attn_out)
    torch.cuda.nvtx.range_pop()
    return attn_output


def _qwen3_vision_model_fast_forward(
    self: "Qwen3VLVisionModel",
    pixel_values: torch.Tensor,
    grid_thw: List[Tuple[int, int, int]] | torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
            The final hidden states of the model.
        grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width of feature shape of each image in LLM.

    Returns:
        `torch.Tensor`: hidden_states.
    """
    torch.cuda.nvtx.range_push("vision_embed")
    hidden_states = self.patch_embed(pixel_values)
    if isinstance(grid_thw, torch.Tensor):
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
    else:
        pos_embeds = _qwen3vl_fast_pos_embed_interpolate(self, grid_thw)
    hidden_states = hidden_states + pos_embeds
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("rotary_pos_emb")
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)

    if isinstance(grid_thw, torch.Tensor):
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
    else:
        rotary_pos_emb = _qwen3vl_rot_pos_emb(self, grid_thw)

    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1).type(torch.float32)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())
    position_embeddings = tuple(
        pe.type(hidden_states.dtype) for pe in position_embeddings
    )
    torch.cuda.nvtx.range_pop()

    deepstack_feature_lists = []
    for layer_num, blk in enumerate(self.blocks):
        torch.cuda.nvtx.range_push(f"vision_block_{layer_num}")
        # hidden_states = blk(hidden_states, cu_seqlens=grid_thw_list, position_embeddings=position_embeddings)
        hidden_states = apply_checkpoint_when_training(
            self,
            blk,
            hidden_states,
            cu_seqlens=grid_thw,
            position_embeddings=position_embeddings,
        )
        torch.cuda.nvtx.range_pop()
        if layer_num in self.deepstack_visual_indexes:
            torch.cuda.nvtx.range_push(f"deepstack_merger_{layer_num}")
            deepstack_idx = self.deepstack_visual_indexes.index(layer_num)
            deepstack_feature = apply_checkpoint_when_training(
                self,
                self.deepstack_merger_list[deepstack_idx],
                hidden_states,
            )
            deepstack_feature_lists.append(deepstack_feature)
            torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("vision_merger")
    hidden_states = self.merger(hidden_states)
    torch.cuda.nvtx.range_pop()

    return hidden_states, deepstack_feature_lists


@torch.no_grad()
@dynamic_rope_update
def _qwen_text_rotary_forward_fp32(
    self: "Qwen3VLTextRotaryEmbedding", x, position_ids
):
    """Compute text rotary embeddings in fp32 then cast to input dtype."""
    if position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(
            3, position_ids.shape[0], -1
        )
    inv_freq_expanded = (
        self.inv_freq[None, None, :, None]
        .to(dtype=torch.float32)
        .expand(3, position_ids.shape[1], -1, 1)
    )
    position_ids_expanded = position_ids[:, :, None, :].to(dtype=torch.float32)

    device_type = "cpu" if x.device.type == "mps" else x.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _qwen_text_decoder_layer_custom_forward(
    self: "Qwen3VLTextDecoderLayer",
    hidden_states: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    past_key_values=None,
):
    self_attn = self.self_attn
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self_attn.head_dim)

    # Projections with QK norm
    q = self_attn.q_norm(self_attn.q_proj(hidden_states).view(hidden_shape))
    k = self_attn.k_norm(self_attn.k_proj(hidden_states).view(hidden_shape))
    v = self_attn.v_proj(hidden_states).view(hidden_shape)
    q = einops.rearrange(q, "b seq head dim -> b head seq dim")
    k = einops.rearrange(k, "b seq head dim -> b head seq dim")
    v = einops.rearrange(v, "b seq head dim -> b head seq dim")

    # RoPE
    cos, sin = position_embeddings
    q_rotate, k_rotate = apply_rotary_pos_emb(q, k, cos, sin)

    # from flash_attn import flash_attn_func
    # attn_out = flash_attn_func(
    #     einops.rearrange(q_rotate, "b h l d -> b l h d"),
    #     einops.rearrange(k_rotate, "b h l d -> b l h d"),
    #     einops.rearrange(v, "b h l d -> b l h d"),
    #     causal=False,
    #     softmax_scale=self_attn.scaling,
    # )
    # attn_out = einops.rearrange(
    #     attn_out, "b seq head dim -> b seq (head dim)"
    # ).contiguous()

    attn_out = F.scaled_dot_product_attention(
        q_rotate,
        k_rotate,
        v,
        attn_mask=attention_mask,
        dropout_p=0.0,
        enable_gqa=True,
        scale=self_attn.scaling,
    )
    attn_out = einops.rearrange(
        attn_out, "b head seq dim -> b seq (head dim)"
    ).contiguous()
    attn_out = self_attn.o_proj(attn_out)
    hidden_states = residual + attn_out

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states, (k_rotate, v)


orig_vision_forward = Qwen3VLVisionModel.forward
orig_attn_forward = Qwen3VLVisionAttention.forward
orig_layer_forward = Qwen3VLTextDecoderLayer.forward
orig_rotary_forward = Qwen3VLTextRotaryEmbedding.forward
REPLACED = False


def replace_qwen3vl_forward():
    global REPLACED
    if not REPLACED:
        Qwen3VLTextDecoderLayer.forward = (
            _qwen_text_decoder_layer_custom_forward
        )
        Qwen3VLVisionModel.forward = _qwen3_vision_model_fast_forward
        Qwen3VLVisionAttention.forward = _qwen3_vision_attn_fast_forward
        Qwen3VLTextRotaryEmbedding.forward = _qwen_text_rotary_forward_fp32
        REPLACED = True


def restore_qwen3vl_forward():
    global REPLACED
    if REPLACED:
        Qwen3VLTextDecoderLayer.forward = orig_layer_forward
        Qwen3VLVisionModel.forward = orig_vision_forward
        Qwen3VLVisionAttention.forward = orig_attn_forward
        Qwen3VLTextRotaryEmbedding.forward = orig_rotary_forward
        REPLACED = False


def is_qwen3vl_forward_replaced() -> bool:
    return REPLACED


def replace_context():
    """Context manager to replace Qwen3VL forward methods with optimized versions."""

    @contextmanager
    def _ctx():
        if REPLACED:
            # if already replaced, do nothing
            yield
        else:
            try:
                replace_qwen3vl_forward()
                yield
            finally:
                restore_qwen3vl_forward()

    return _ctx()
