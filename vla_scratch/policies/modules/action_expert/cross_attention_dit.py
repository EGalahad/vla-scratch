from dataclasses import dataclass
from typing import List, Tuple, Callable
import einops

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import RMSNorm

from transformers.activations import ACT2FN

from vla_scratch.policies.utils.data_types import *
# from vla_scratch.policies.utils.transformers import apply_rotary_pos_emb
from vla_scratch.policies.utils.training import apply_checkpoint_when_training

from torch.nn.modules.lazy import LazyModuleMixin

class LazyRMSNorm(LazyModuleMixin, torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.normalized_shape = None

    def initialize_parameters(self, input):
        self.normalized_shape = tuple(input.shape[-1:])

    def forward(self, x):
        if self.normalized_shape is None:
            self.initialize_parameters(x)

        return torch.rms_norm(x, self.normalized_shape, eps=self.eps)
    
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
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    attn_dropout: float = 0.0
    mlp_dropout: float = 0.0
    mlp_activation: str = "silu"

    num_hidden_layers: int = 12
    layers_for_dispersive_loss = [6]
    # layers_for_dispersive_loss = [4, 6, 8]
    dispersive_loss_tau: float = 1.0

    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    attention_bias: bool = False
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0


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

        self.q_proj = nn.LazyLinear(
            self.num_att_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.LazyLinear(
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.LazyLinear(
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
        *,
        attention_mask: AttentionMask | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        q = einops.rearrange(
            q, "b n_q (h d) -> b h n_q d", h=self.num_att_heads, d=self.head_dim
        )
        kv_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        k = self.k_proj(kv_hidden_states)
        k = einops.rearrange(
            k,
            "b n_kv (h_kv d) -> b h_kv n_kv d",
            h_kv=self.num_kv_heads,
            d=self.head_dim,
        )
        v = self.v_proj(kv_hidden_states)
        v = einops.rearrange(
            v,
            "b n_kv (h_kv d) -> b h_kv n_kv d",
            h_kv=self.num_kv_heads,
            d=self.head_dim,
        )
        # shape: q: (batch, num_heads, seq, head_dim)
        # shape: k, v: (batch, num_kv_heads, seq, head_dim)

        q_rotate, k_rotate = q, k

        attn_output = F.scaled_dot_product_attention(
            q_rotate,
            k_rotate,
            v,
            # attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.scaling,
            enable_gqa=True,
        )
        attn_output = einops.rearrange(
            attn_output, "b h seq_q d -> b seq_q (h d)"
        ).contiguous()
        return self.o_proj(attn_output), (k_rotate, v)


class DecoderBlock(nn.Module):
    def __init__(self, config: DiTConfig, layer_idx: int):
        super().__init__()
        self.attn_dropout = config.attn_dropout
        self.mlp_dropout = config.mlp_dropout

        cond_dim = config.hidden_size
        self.ada_mod1 = AdaptiveModulation(cond_dim, config.hidden_size)
        self.input_layernorm = RMSNorm(
            normalized_shape=config.hidden_size,
            eps=config.rms_norm_eps,
            elementwise_affine=False,
        )
        self.encoder_layernorm = LazyRMSNorm(eps=config.rms_norm_eps)
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
        adarms_cond: AdarmsCond,
        *,
        attention_mask: AttentionMask | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        torch.cuda.nvtx.range_push(f"adarms")
        shift_msa, scale_msa, gate_msa = self.ada_mod1(adarms_cond)
        shift_mlp, scale_mlp, gate_mlp = self.ada_mod2(adarms_cond)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push(f"attention")
        pre_att = modulate(self.input_layernorm(hidden_states), shift_msa, scale_msa)
        if encoder_hidden_states is not None:
            encoder_hidden_states = self.encoder_layernorm(encoder_hidden_states)
        out_att, (k, v) = self.attn.forward(
            pre_att,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
        )
        if self.attn_dropout > 0.0:
            out_att = torch.dropout(out_att, p=self.attn_dropout, train=self.training)
        res_att = hidden_states + einops.einsum(
            out_att, gate_msa, "b s d, b d -> b s d"
        )
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push(f"mlp")
        pre_mlp = modulate(self.post_attention_layernorm(res_att), shift_mlp, scale_mlp)
        out_mlp = self.mlp(pre_mlp)
        if self.mlp_dropout > 0.0:
            out_mlp = torch.dropout(out_mlp, p=self.mlp_dropout, train=self.training)
        res_mlp = res_att + einops.einsum(out_mlp, gate_mlp, "b s d, b d -> b s d")
        torch.cuda.nvtx.range_pop()
        return res_mlp, (k, v)


class DiTModel(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        self.use_dropout = config.attn_dropout > 0.0 or config.mlp_dropout > 0.0
        blocks: List[DecoderBlock] = []
        # = nn.ModuleList(
        #     [DecoderBlock(config, idx) for idx in range(config.num_hidden_layers)]
        # )
        for idx in range(config.num_hidden_layers):
            blocks.append(DecoderBlock(config, idx * 2))
            blocks.append(DecoderBlock(config, idx * 2 + 1))
        self.blocks = nn.ModuleList(blocks)

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
        encoder_hidden_states: List[torch.Tensor] | None = None,
        # num_kv = num_q + len(past_key_values)
    ) -> (
        Tuple[HiddenState, List[KVCache]]
        | Tuple[HiddenState, List[KVCache], torch.Tensor | None]
    ):
        # We currently ignore past_key_values as caching is not implemented.
        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided.")

        if attention_mask is not None and attention_mask.dim() != 4:
            raise ValueError(
                "attention_mask is expected to have shape (batch, 1, seq, seq_kv)."
            )

        hidden_states = inputs_embeds
        kv_cache_list: List[KVCache] = []
        hidden_states_for_dispersive_loss = []
        for i, layer in enumerate(self.blocks):
            if i % 2 == 1:
                # cross-attention
                encoder_hidden_states_this_layer = encoder_hidden_states[i // 2]
            else:
                # self-attention
                encoder_hidden_states_this_layer = None
            torch.cuda.nvtx.range_push(f"layer_{i}")
            if i in self.config.layers_for_dispersive_loss:
                hidden_states_for_dispersive_loss.append(hidden_states)
            hidden_states, (k, v) = apply_checkpoint_when_training(
                self,
                layer,
                hidden_states,
                adarms_cond,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states_this_layer,
                preserve_rng_state=self.use_dropout,
            )
            torch.cuda.nvtx.range_pop()

            kv_cache_list.append((k, v))

        hidden_states = self.norm(hidden_states)

        dispersive_loss = 0
        if hidden_states_for_dispersive_loss:
            dispersive_loss = self._compute_dispersive_loss(
                hidden_states_for_dispersive_loss
            )

        return hidden_states, kv_cache_list, dispersive_loss

    def _compute_dispersive_loss(
        self, hidden_states_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """InfoNCE-based dispersive loss using squared L2 distance."""
        tau = self.config.dispersive_loss_tau
        hidden_states = torch.stack(hidden_states_list, dim=1)
        z = einops.rearrange(hidden_states, "b l s d -> (b l) (s d)")
        z_cast = z.type(torch.float32)
        diff = nn.functional.pdist(z_cast).pow(2) / z.shape[1]
        diff = diff.type(z.dtype)
        diff = torch.concat(
            (diff, diff, torch.zeros(z.shape[0], device=z.device, dtype=z.dtype))
        )
        return torch.log(torch.exp(-diff / tau).mean())
