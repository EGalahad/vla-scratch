from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from vla_scratch.policies.utils.training import apply_checkpoint_when_training

if TYPE_CHECKING:
    from transformers import PaliGemmaForConditionalGeneration


PI05_MODEL_ID = "lerobot/pi05_base"
PI05_MODEL_REVISION = "b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba"
PI05_MODEL_SHA256 = "0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f"
PI05_VLM_PREFIX = "paligemma_with_expert.paligemma."


def load_pi05_weights(
    model: "PaliGemmaForConditionalGeneration",
) -> dict[str, str]:
    """Download the pinned π0.5 base checkpoint and strictly load its VLM."""
    path = Path(
        hf_hub_download(
            repo_id=PI05_MODEL_ID,
            filename="model.safetensors",
            revision=PI05_MODEL_REVISION,
        )
    )
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        vlm_state = {
            key.removeprefix(PI05_VLM_PREFIX): checkpoint.get_tensor(key)
            for key in checkpoint.keys()
            if key.startswith(PI05_VLM_PREFIX)
        }
    if not vlm_state:
        raise ValueError(
            f"{path} has no keys starting with {PI05_VLM_PREFIX!r}"
        )

    embed_key = "model.language_model.embed_tokens.weight"
    head_key = "lm_head.weight"
    if embed_key not in vlm_state and head_key in vlm_state:
        vlm_state[embed_key] = vlm_state[head_key]
    vlm_state.pop(head_key, None)

    expected = set(model.state_dict()) - {"lm_head.weight"}
    actual = set(vlm_state)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    shape_mismatches = sorted(
        key
        for key in expected & actual
        if tuple(model.state_dict()[key].shape) != tuple(vlm_state[key].shape)
    )
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            "Invalid OpenPI PaliGemma weights: "
            f"missing={missing}, unexpected={unexpected}, "
            f"shape_mismatches={shape_mismatches}"
        )

    incompatible = model.load_state_dict(vlm_state, strict=False)
    if (
        set(incompatible.missing_keys) != {"lm_head.weight"}
        or incompatible.unexpected_keys
    ):
        raise RuntimeError(f"Unexpected load result: {incompatible}")
    model.tie_weights()

    return {
        "source": str(path.resolve()),
        "repo_id": PI05_MODEL_ID,
        "revision": PI05_MODEL_REVISION,
        "sha256": PI05_MODEL_SHA256,
    }


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    groups = query.shape[1] // key.shape[1]
    query = (query * scale).unflatten(1, (key.shape[1], groups))
    logits = torch.matmul(
        query.float(), key.float().unsqueeze(2).transpose(-1, -2)
    )
    if attention_mask is not None:
        attention_mask = attention_mask.unsqueeze(2)
        if attention_mask.dtype == torch.bool:
            logits = torch.where(
                attention_mask,
                logits,
                torch.tensor(-2.3819763e38, device=logits.device),
            )
        else:
            logits = logits + attention_mask.float()
    probabilities = torch.softmax(logits, dim=-1).to(value.dtype)
    attended = torch.matmul(probabilities, value.unsqueeze(2))
    return attended.flatten(1, 2)


def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = position_embeddings
    dtype = query.dtype
    query, key = query.float(), key.float()
    cos, sin = cos.float(), sin.float()
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    query = query * cos + rotate_half(query) * sin
    key = key * cos + rotate_half(key) * sin
    return query.to(dtype), key.to(dtype)


def gemma_decoder_layer_forward(
    layer,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """π0.5 Gemma layer forward returning the layer KV cache."""
    residual = hidden_states
    normalized = layer.input_layernorm(hidden_states)
    shape = (*normalized.shape[:-1], -1, layer.self_attn.head_dim)
    query = layer.self_attn.q_proj(normalized).view(shape).transpose(1, 2)
    key = layer.self_attn.k_proj(normalized).view(shape).transpose(1, 2)
    value = layer.self_attn.v_proj(normalized).view(shape).transpose(1, 2)
    query, key = _apply_rope(query, key, position_embeddings)
    attended = _attention(
        query,
        key,
        value,
        attention_mask,
        layer.self_attn.scaling,
    )
    attended = attended.transpose(1, 2).reshape(*normalized.shape[:-1], -1)
    attended = layer.self_attn.o_proj(attended)

    is_gemma2 = hasattr(layer, "pre_feedforward_layernorm")
    if is_gemma2:
        hidden_states = residual + layer.post_attention_layernorm(attended)
        residual = hidden_states
        normalized = layer.pre_feedforward_layernorm(hidden_states)
    else:
        hidden_states = residual + attended
        residual = hidden_states
        normalized = layer.post_attention_layernorm(hidden_states)

    mlp = layer.mlp
    projected = F.gelu(mlp.gate_proj(normalized), approximate="none")
    feed_forward = mlp.down_proj(projected * mlp.up_proj(normalized))
    if is_gemma2:
        feed_forward = layer.post_feedforward_layernorm(feed_forward)
    return residual + feed_forward, (key, value)


def _siglip_layer_forward(
    layer,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    residual = hidden_states
    attention = layer.self_attn
    norm = layer.layer_norm1
    normalized = F.layer_norm(
        hidden_states.float(),
        norm.normalized_shape,
        norm.weight.float(),
        None if norm.bias is None else norm.bias.float(),
        norm.eps,
    ).to(attention.q_proj.weight.dtype)
    batch, length, width = normalized.shape
    shape = (batch, length, attention.num_heads, attention.head_dim)
    query = attention.q_proj(normalized).view(shape).transpose(1, 2)
    key = attention.k_proj(normalized).view(shape).transpose(1, 2)
    value = attention.v_proj(normalized).view(shape).transpose(1, 2)
    attended = _attention(
        query,
        key,
        value,
        None,
        attention.scale,
    )
    hidden_states = residual + attention.out_proj(
        attended.transpose(1, 2).reshape(batch, length, width)
    )
    residual = hidden_states
    mlp = layer.mlp
    norm = layer.layer_norm2
    normalized = F.layer_norm(
        hidden_states.float(),
        norm.normalized_shape,
        norm.weight.float(),
        None if norm.bias is None else norm.bias.float(),
        norm.eps,
    ).to(mlp.fc1.weight.dtype)
    hidden_states = mlp.fc2(
        F.gelu(mlp.fc1(normalized), approximate="none")
    )
    return residual + hidden_states


def pi05_image_features(
    owner: torch.nn.Module,
    model: "PaliGemmaForConditionalGeneration",
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    """Run π0.5 SigLIP locally without mutating Hugging Face classes."""
    vision = model.model.vision_tower.vision_model
    embeddings = vision.embeddings
    patch = embeddings.patch_embedding
    hidden_states = F.conv2d(
        pixel_values.float(),
        patch.weight.float(),
        None if patch.bias is None else patch.bias.float(),
        stride=patch.stride,
        padding=patch.padding,
        dilation=patch.dilation,
        groups=patch.groups,
    ).flatten(2).transpose(1, 2)
    position = embeddings.position_embedding(embeddings.position_ids).float()
    hidden_states = (hidden_states + position).to(
        vision.encoder.layers[0].layer_norm1.weight.dtype
    )

    for layer in vision.encoder.layers:
        layer_forward = getattr(layer, "bridge_forward", None)
        if layer_forward is None:
            hidden_states = apply_checkpoint_when_training(
                owner,
                _siglip_layer_forward,
                layer,
                hidden_states,
            )
        else:
            hidden_states = apply_checkpoint_when_training(
                owner,
                layer_forward,
                hidden_states,
            )
    hidden_states = vision.post_layernorm(hidden_states)
    hidden_states = model.model.multi_modal_projector(hidden_states)
    return hidden_states / math.sqrt(model.config.text_config.hidden_size)
