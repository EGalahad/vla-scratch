from dataclasses import dataclass
from typing import List, Tuple, Callable
import einops

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from torch.nn import RMSNorm

from transformers.activations import ACT2FN

from vla_scratch.policies.data_types import *


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# @torch.compile
def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@torch.compile
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaptiveModulation(nn.Module):
    def __init__(self, cond_dim: int, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.modulation = nn.Linear(cond_dim, dim * 3, bias=True)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shift, scale, gate = self.modulation(F.silu(x)).chunk(3, dim=-1)
        return shift, scale, gate


@torch.compile
def gated_activation(
    x: torch.Tensor, y: torch.Tensor, act: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    return act(x) * y


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, activation: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        gated = gated_activation(self.gate_proj(x), self.up_proj(x), self.act_fn)
        return self.down_proj(gated)


class RotaryEmbedding(nn.Module):
    def __init__(
        self, head_dim: int, max_position_embeddings: int, base: float
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(
        self, position_ids: torch.LongTensor, *, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_ids.dim() != 2:
            raise ValueError(
                f"Expected position_ids to have shape (batch, seq), got {position_ids.shape}"
            )
        freqs = torch.einsum(
            "bi,j->bij", position_ids.float(), self.inv_freq.to(position_ids.device)
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        return cos, sin


@dataclass
class DiTConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    mlp_activation: str = "silu"

    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    attention_bias: bool = False
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0


from typing import Literal
Variant = Literal["dummy", "300m", "2b"]

def get_config(variant: Variant) -> DiTConfig:
    """Returns config for specified gemma variant."""
    if variant == "dummy":
        return DiTConfig(
            hidden_size=64,
            num_hidden_layers=4,
            intermediate_size=128,
            num_attention_heads=8,
            num_key_value_heads=1,
            head_dim=16,
        )
    if variant == "300m":
        # 311M params
        return DiTConfig(
            hidden_size=1024,
            num_hidden_layers=18,
            intermediate_size=4096,
            num_attention_heads=8,
            num_key_value_heads=1,
            head_dim=256,
        )
    if variant == "2b":
        return DiTConfig(
            hidden_size=2048,
            num_hidden_layers=18,
            intermediate_size=16384,
            num_attention_heads=8,
            num_key_value_heads=1,
            head_dim=256,
        )
    raise ValueError(f"Unknown variant: {variant}")

class Attention(nn.Module):
    def __init__(self, config: DiTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        # Grouped Query Attention (GQA): num_key_value_heads < num_attention_heads
        # num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.num_att_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        self.scaling = self.head_dim**-0.5
        # TODO: not sure if we need dropout here
        # self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_att_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_att_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: HiddenState,
        position_embeddings: PositionEmbs,
        *,
        attention_mask: AttentionMask | None = None,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        q = einops.rearrange(
            q, "b n_q (h d) -> b h n_q d", h=self.num_att_heads, d=self.head_dim
        )
        k = self.k_proj(hidden_states)
        k = einops.rearrange(
            k,
            "b n_kv (h_kv d) -> b h_kv n_kv d",
            h_kv=self.num_kv_heads,
            d=self.head_dim,
        )
        v = self.v_proj(hidden_states)
        v = einops.rearrange(
            v,
            "b n_kv (h_kv d) -> b h_kv n_kv d",
            h_kv=self.num_kv_heads,
            d=self.head_dim,
        )
        # shape: q: (batch, num_heads, seq, head_dim)
        # shape: k, v: (batch, num_kv_heads, seq, head_dim)

        cos, sin = position_embeddings
        q_rotate, k_rotate = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k_rotate = torch.cat([past_k, k_rotate], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        attn_output = F.scaled_dot_product_attention(
            q_rotate,
            k_rotate,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.scaling,
            enable_gqa=True,
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, -1).contiguous()
        return self.o_proj(attn_output), k_rotate, v


class DecoderBlock(nn.Module):
    def __init__(self, config: DiTConfig, layer_idx: int):
        super().__init__()
        cond_dim = config.hidden_size
        self.ada_mod1 = AdaptiveModulation(cond_dim, config.hidden_size)
        self.input_layernorm = RMSNorm(
            normalized_shape=config.hidden_size,
            eps=config.rms_norm_eps,
            elementwise_affine=False,
        )
        self.attn = Attention(config, layer_idx)

        self.ada_mod2 = AdaptiveModulation(cond_dim, config.hidden_size)
        self.post_attention_layernorm = RMSNorm(
            normalized_shape=config.hidden_size,
            eps=config.rms_norm_eps,
            elementwise_affine=False,
        )
        self.mlp = MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation=config.mlp_activation,
        )

    def forward(
        self,
        hidden_states: HiddenState,
        position_embeddings: PositionEmbs,
        adarms_cond: AdarmsCond,
        *,
        attention_mask: AttentionMask | None = None,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        torch.cuda.nvtx.range_push(f"adarms")
        shift_msa, scale_msa, gate_msa = self.ada_mod1(adarms_cond)
        shift_mlp, scale_mlp, gate_mlp = self.ada_mod2(adarms_cond)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push(f"attention")
        pre_att = modulate(self.input_layernorm(hidden_states), shift_msa, scale_msa)
        out_att, k, v = self.attn.forward(
            pre_att,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        res_att = hidden_states + out_att * gate_msa.unsqueeze(1)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push(f"mlp")
        pre_mlp = modulate(self.post_attention_layernorm(res_att), shift_mlp, scale_mlp)
        out_mlp = self.mlp(pre_mlp)
        res_mlp = res_att + out_mlp * gate_mlp.unsqueeze(1)
        torch.cuda.nvtx.range_pop()
        return res_mlp, k, v


class DiTModel(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        self.blocks: List[DecoderBlock] = nn.ModuleList(
            [DecoderBlock(config, idx) for idx in range(config.num_hidden_layers)]
        )

        self.norm = RMSNorm(
            normalized_shape=config.hidden_size,
            eps=config.rms_norm_eps,
            elementwise_affine=False,
        )
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(basic_init)

        # Zero-out adaLN modulation layers in DiT blocks:
        def ada_mod_init(module):
            if isinstance(module, AdaptiveModulation):
                nn.init.constant_(module.modulation.weight, 0)
                nn.init.constant_(module.modulation.bias, 0)

        self.apply(ada_mod_init)

    def forward(
        self,
        inputs_embeds: HiddenState,
        position_ids: PositionIds,
        adarms_cond: AdarmsCond,
        *,
        attention_mask: AttentionMask | None = None,
        past_key_values: List[KVCache] | None = None,
        # num_kv = num_q + len(past_key_values)
    ) -> Tuple[HiddenState, List[KVCache]]:
        # We currently ignore past_key_values as caching is not implemented.
        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided.")

        cos, sin = self.rotary_emb(position_ids, dtype=inputs_embeds.dtype)
        if attention_mask is not None and attention_mask.dim() != 4:
            raise ValueError(
                "attention_mask is expected to have shape (batch, 1, seq, seq_kv)."
            )

        hidden_states = inputs_embeds
        kv_cache_list: List[KVCache] = []
        for i, layer in enumerate(self.blocks):
            if past_key_values is not None:
                kv_cache = past_key_values[i]
            else:
                kv_cache = None
            torch.cuda.nvtx.range_push(f"layer_{i}")
            if self.training:
                hidden_states, k, v = torch.utils.checkpoint.checkpoint(
                    layer.forward,
                    hidden_states,
                    (cos, sin),
                    adarms_cond,
                    attention_mask=attention_mask,
                    kv_cache=kv_cache,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                hidden_states, k, v = layer.forward(
                    hidden_states,
                    position_embeddings=(cos, sin),
                    adarms_cond=adarms_cond,
                    attention_mask=attention_mask,
                    kv_cache=kv_cache,
                )
            torch.cuda.nvtx.range_pop()
            
            kv_cache_list.append((k, v))

        hidden_states = self.norm(hidden_states)
        return hidden_states, kv_cache_list
