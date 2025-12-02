import einops
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer

from vla_scratch.policies.utils.transformers import apply_rotary_pos_emb


def _gemma_decoder_layer_custom_forward(
    self: "GemmaDecoderLayer", hidden_states, prefix_att_mask, position_embeddings
):
    """Custom forward for a GemmaDecoderLayer used in prefix encoding.

    This mirrors the previous inline `compute_layer` function, but is defined
    as a bound method that attaches to `GemmaDecoderLayer` as `custom_forward`.
    """
    pre_att = self.input_layernorm(hidden_states)
    input_shape = hidden_states.shape[:-1]  # [batch_size, seq_len]
    head_shape = (*input_shape, -1, self.self_attn.head_dim)

    # attention
    torch.cuda.nvtx.range_push("project_qkv")
    q = self.self_attn.q_proj(pre_att).view(head_shape)
    k = self.self_attn.k_proj(pre_att).view(head_shape)
    v = self.self_attn.v_proj(pre_att).view(head_shape)
    q = einops.rearrange(q, "b seq head dim -> b head seq dim")
    k = einops.rearrange(k, "b seq head dim -> b head seq dim")
    v = einops.rearrange(v, "b seq head dim -> b head seq dim")
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("rotary_embedding")
    cos, sin = position_embeddings
    q_rotate, k_rotate = apply_rotary_pos_emb(q, k, cos, sin)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("attention")
    out_att = F.scaled_dot_product_attention(
        q_rotate,
        k_rotate,
        v,
        attn_mask=prefix_att_mask,
        scale=self.self_attn.scaling,
        enable_gqa=True,
    )
    out_att = einops.rearrange(
        out_att, "b head seq dim -> b seq (head dim)"
    ).contiguous()
    out_att = self.self_attn.o_proj(out_att)
    res_att = hidden_states + out_att
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("mlp")
    pre_mlp = self.post_attention_layernorm(res_att)
    out_mlp = self.mlp(pre_mlp)
    res_mlp = res_att + out_mlp
    torch.cuda.nvtx.range_pop()
    return res_mlp, (k_rotate, v)


orig_layer_forward = GemmaDecoderLayer.forward
REPLACED = False
def replace_paligemma_forward():
    """Context manager to replace GemmaDecoderLayer.forward with custom version."""

    @contextmanager
    def _ctx():
        try:
            GemmaDecoderLayer.forward = _gemma_decoder_layer_custom_forward
            yield
        finally:
            GemmaDecoderLayer.forward = orig_layer_forward

    return _ctx()
