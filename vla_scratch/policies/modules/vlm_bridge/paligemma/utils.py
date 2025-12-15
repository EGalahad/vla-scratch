import einops
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer
from transformers.models.siglip.modeling_siglip import (
    SiglipEncoder,
    BaseModelOutput as SiglipBaseModelOutput,
)

from vla_scratch.policies.utils.transformers import apply_rotary_pos_emb
from vla_scratch.policies.utils.training import apply_checkpoint_when_training


def _siglip_encoder_foward(
    self: "SiglipEncoder",
    inputs_embeds,
    attention_mask: torch.Tensor = None,
    **kwargs,
) -> SiglipBaseModelOutput:
    hidden_states = inputs_embeds
    for encoder_layer in self.layers:
        hidden_states = apply_checkpoint_when_training(
            self,
            encoder_layer,
            hidden_states,
            attention_mask,
            **kwargs,
        )
    return SiglipBaseModelOutput(last_hidden_state=hidden_states)


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

def _gemma2_decoder_layer_custom_forward(
    self: "Gemma2DecoderLayer", hidden_states, prefix_att_mask, position_embeddings
):
    """Custom forward for a GemmaDecoderLayer used in prefix encoding.
    """
    redisual = hidden_states

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
    out_att = self.post_attention_layernorm(out_att)
    res_att = redisual + out_att
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("mlp")
    pre_mlp = self.pre_feedforward_layernorm(res_att)
    out_mlp = self.mlp(pre_mlp)
    out_mlp = self.post_feedforward_layernorm(out_mlp)
    res_mlp = res_att + out_mlp
    torch.cuda.nvtx.range_pop()
    return res_mlp, (k_rotate, v)

orig_gemma_layer_forward = GemmaDecoderLayer.forward
orig_gemma2_layer_forward = Gemma2DecoderLayer.forward
orig_siglip_encoder_forward = SiglipEncoder.forward
REPLACED = False


def replace_paligemma_forward():
    global REPLACED
    if not REPLACED:
        GemmaDecoderLayer.forward = _gemma_decoder_layer_custom_forward
        Gemma2DecoderLayer.forward = _gemma2_decoder_layer_custom_forward
        SiglipEncoder.forward = _siglip_encoder_foward
        REPLACED = True


def restore_paligemma_forward():
    global REPLACED
    if REPLACED:
        GemmaDecoderLayer.forward = orig_gemma_layer_forward
        Gemma2DecoderLayer.forward = orig_gemma2_layer_forward
        SiglipEncoder.forward = orig_siglip_encoder_forward
        REPLACED = False


def is_paligemma_forward_replaced() -> bool:
    return REPLACED
