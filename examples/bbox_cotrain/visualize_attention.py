#!/usr/bin/env python3
"""
Visualize attention scores between image and language instruction in VLA policy.

This script:
1. Loads a VLA policy from checkpoint
2. Loads a LeRobotDataset from root path
3. For each episode, processes the first frame image and language instruction
4. Extracts attention weights between image tokens and text tokens
5. Maps attention weights back to image space
6. Saves visualization as an image with instruction text overlaid

Usage:
    python scripts/visualize_attention.py \
        policy=pi-qwen \
        policy.state_history=0 \
        policy.action_horizon=10 \
        policy.transforms.0.max_length=500 \
        data=bbox_cotrain_train \
        checkpoint_path=/path/to/checkpoint_dir_or_file \
        dataset_root=/path/to/lerobot_dataset_root \
        output_dir=attention_visualizations \
        layer_idx=-1 \
        max_episodes=100
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple, cast

import einops
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm as cm
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, MISSING, OmegaConf

from vla_scratch.policies.config import PolicyConfig, create_policy
from vla_scratch.datasets.config import DataConfig
from vla_scratch.utils.checkpoint import (
    find_latest_checkpoint,
    load_model_from_checkpoint,
)
from vla_scratch.transforms.data_types import Observation, DataSample
from vla_scratch.transforms.common import ToTorch
from vla_scratch.policies.modules.vlm_bridge.qwen.bridge import Qwen3VLBridge
from vla_scratch.helpers.data import build_input_transforms, create_dataset
from vla_scratch.transforms.data_keys import (
    PROCESSED_IMAGE_KEY,
    PROCESSED_IMAGE_MASK_KEY,
    PROCESSED_STATE_KEY,
    TASK_KEY,
    GENERATION_PROMPT_KEY,
    GENERATION_ANSWER_KEY,
)

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def compute_attention_weights(
    q: torch.Tensor,
    k: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Compute attention weights manually to get the attention scores.

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_heads, seq_len, head_dim]
        attention_mask: Optional attention mask [batch, 1, seq_len, seq_len]
        scale: Scaling factor for attention

    Returns:
        Attention weights [batch, num_heads, seq_len, seq_len]
    """
    # Compute attention scores: Q @ K^T
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Apply attention mask if provided
    if attention_mask is not None:
        # Mask should be 0 for valid positions, -inf for masked positions
        attn_scores = attn_scores.masked_fill(
            attention_mask == 0, float("-inf")
        )

    # Apply softmax
    attn_weights = F.softmax(attn_scores, dim=-1)

    return attn_weights


@torch.inference_mode()
def extract_attention_from_bridge(
    bridge: Qwen3VLBridge,
    observation: Observation,
    layer_idx: int = -1,  # Last layer by default
    subtext: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract attention weights from the VLM bridge.

    Returns:
        attention_weights: [batch, num_heads, seq_len, seq_len]
        image_mask: [batch, seq_len] - boolean mask indicating image token positions
        text_mask: [batch, seq_len] - boolean mask indicating text token positions
    """
    from vla_scratch.policies.modules.vlm_bridge.qwen.processor import (
        QwenPolicyInput,
    )
    from vla_scratch.policies.modules.vlm_bridge.qwen.utils import (
        is_qwen3vl_forward_replaced,
        apply_rotary_pos_emb,
    )
    from vla_scratch.policies.utils.transformers import make_att_2d_masks

    assert isinstance(observation.policy_input, QwenPolicyInput)
    policy_td: QwenPolicyInput = observation.policy_input

    REPLACED = is_qwen3vl_forward_replaced()

    # Now observation should have proper batch dimension from unsqueeze(0)
    # So input_ids and attention_mask should already be [batch, seq_len]
    input_ids = policy_td.input_ids
    attention_mask = policy_td.attention_mask
    # import pdb; pdb.set_trace()
    # Embed text and images (same as bridge.encode)
    lm = bridge.causal_model.language_model
    inputs_embeds = lm.embed_tokens(input_ids)  # [batch, seq_len, hidden]

    # Handle pixel_values shape
    # After unsqueeze(0), pixel_values should be [batch, grid, patch] or [batch, (grid), patch]
    pixel_values = policy_td.pixel_values
    if pixel_values.ndim == 2:
        # Already in shape [(b grid), patch], no need to rearrange
        pass
    elif pixel_values.ndim == 3:
        # Shape is [batch, grid, patch] or [batch, (grid), patch]
        # Rearrange to [(b grid), patch] for vision model
        pixel_values = einops.rearrange(
            pixel_values, "b grid patch -> (b grid) patch"
        )
    else:
        raise ValueError(f"Unexpected pixel_values shape: {pixel_values.shape}")
    if REPLACED:
        grid_thw_list = policy_td.image_grid_thw_list
        grid_thw_list = sum(grid_thw_list, [])
        image_embeds, deepstack_image_embeds = bridge.causal_model.model.visual(
            pixel_values, grid_thw_list
        )
    else:
        grid_thw_tensor = policy_td.image_grid_thw.reshape(-1, 3)
        image_embeds, deepstack_image_embeds = bridge.causal_model.model.visual(
            pixel_values, grid_thw_tensor
        )

    image_mask = (
        input_ids == bridge.causal_model.model.config.image_token_id
    )  # [batch, seq_len]

    inputs_embeds.masked_scatter_(image_mask.unsqueeze(-1), image_embeds)

    input_pad_mask = attention_mask  # [batch, seq_len]

    # Handle position_ids shape
    # After unsqueeze(0), position_ids should be [batch, plane, 1, seq_len]
    # bridge.encode expects [plane, batch, seq_len]
    position_ids_tensor = policy_td.position_ids
    if position_ids_tensor.ndim == 4:
        # Shape is [batch, plane, 1, seq_len] - rearrange to [plane, batch, seq_len]
        if position_ids_tensor.shape[2] == 1:
            position_ids = einops.rearrange(
                position_ids_tensor, "b plane 1 s -> plane b s"
            )
        else:
            position_ids = einops.rearrange(
                position_ids_tensor, "b plane d s -> plane b s"
            )
    elif position_ids_tensor.ndim == 3:
        # Check if it's [plane, batch, seq_len] or [batch, plane, seq_len]
        if position_ids_tensor.shape[0] == 3:
            # Shape is [plane, batch, seq_len], already correct
            position_ids = position_ids_tensor
        elif position_ids_tensor.shape[1] == 3:
            # Shape is [batch, plane, seq_len], need to rearrange
            position_ids = einops.rearrange(
                position_ids_tensor, "b plane s -> plane b s"
            )
        else:
            raise ValueError(
                f"Cannot infer position_ids shape: {position_ids_tensor.shape}"
            )
    else:
        raise ValueError(
            f"Unexpected position_ids shape: {position_ids_tensor.shape}, expected 3D or 4D"
        )

    embs = inputs_embeds
    pad_masks = input_pad_mask

    prefix_att_2d = make_att_2d_masks(pad_masks, pad_masks)
    prefix_att_mask = einops.rearrange(prefix_att_2d, "b i j -> b 1 i j")

    position_emb = lm.rotary_emb.forward(embs, position_ids)
    # import pdb; pdb.set_trace()
    hidden_states = embs

    # Process through layers until target layer
    target_layer_idx = (
        layer_idx if layer_idx >= 0 else len(lm.layers) + layer_idx
    )
    attn_weights = None

    for idx, decoder_layer in enumerate(lm.layers):
        if idx == target_layer_idx:
            # Manually compute attention weights for this layer
            self_attn = decoder_layer.self_attn
            residual = hidden_states
            hidden_states_norm = decoder_layer.input_layernorm(hidden_states)

            input_shape = hidden_states_norm.shape[:-1]
            hidden_shape = (*input_shape, -1, self_attn.head_dim)

            # Projections with QK norm
            q = self_attn.q_norm(
                self_attn.q_proj(hidden_states_norm).view(hidden_shape)
            )
            k = self_attn.k_norm(
                self_attn.k_proj(hidden_states_norm).view(hidden_shape)
            )
            v = self_attn.v_proj(hidden_states_norm).view(hidden_shape)
            q = einops.rearrange(q, "b seq head dim -> b head seq dim")
            k = einops.rearrange(k, "b seq head dim -> b head seq dim")
            v = einops.rearrange(v, "b seq head dim -> b head seq dim")

            # Handle GQA (Grouped Query Attention) if num_attention_heads != num_key_value_heads
            # Get num_heads from config (same way as bridge.get_text_dims)
            cfg = bridge.causal_model.config.text_config
            num_att_heads = cfg.num_attention_heads
            num_kv_heads = cfg.num_key_value_heads

            if num_kv_heads != num_att_heads:
                # Need to repeat k and v to match q's number of heads
                num_key_value_groups = num_att_heads // num_kv_heads
                # k and v are [batch, num_kv_heads, seq, head_dim]
                # Expand to [batch, num_kv_heads, num_key_value_groups, seq, head_dim]
                k = k[:, :, None, :, :].expand(
                    -1, -1, num_key_value_groups, -1, -1
                )
                v = v[:, :, None, :, :].expand(
                    -1, -1, num_key_value_groups, -1, -1
                )
                # Reshape to [batch, num_att_heads, seq, head_dim]
                k = k.reshape(k.shape[0], num_att_heads, k.shape[3], k.shape[4])
                v = v.reshape(v.shape[0], num_att_heads, v.shape[3], v.shape[4])

            # RoPE
            cos, sin = position_emb
            q_rotate, k_rotate = apply_rotary_pos_emb(q, k, cos, sin)

            # Compute attention weights
            attn_weights = compute_attention_weights(
                q_rotate,
                k_rotate,
                attention_mask=prefix_att_mask,
                scale=self_attn.scaling,
            )

            # Compute attention output
            attn_out = torch.matmul(attn_weights, v)
            attn_out = einops.rearrange(
                attn_out, "b head seq dim -> b seq (head dim)"
            ).contiguous()
            attn_out = self_attn.o_proj(attn_out)
            hidden_states = residual + attn_out

            # Continue with MLP
            residual = hidden_states
            hidden_states = decoder_layer.post_attention_layernorm(
                hidden_states
            )
            hidden_states = decoder_layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
            break
        else:
            # Normal forward
            outputs = decoder_layer(
                hidden_states,
                attention_mask=prefix_att_mask,
                position_embeddings=position_emb,
                past_key_values=None,
            )
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs

    # Create text mask (opposite of image mask, but only for valid tokens)
    text_mask = ~image_mask & input_pad_mask.bool()  # [batch, seq_len]

    if subtext:
        tokenizer = getattr(bridge.processor, "tokenizer", None)
        if tokenizer is not None:

            def _find_subtext_mask(
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                valid_text_mask: torch.Tensor,
                subtext_value: str,
            ) -> Optional[torch.Tensor]:
                subtext_value = subtext_value.strip()
                if not subtext_value:
                    return None

                candidates = []
                base_ids = tokenizer.encode(
                    subtext_value, add_special_tokens=False
                )
                if base_ids:
                    candidates.append(base_ids)
                if not subtext_value.startswith(" "):
                    spaced_ids = tokenizer.encode(
                        f" {subtext_value}", add_special_tokens=False
                    )
                    if spaced_ids:
                        candidates.append(spaced_ids)

                if not candidates:
                    return None

                mask = torch.zeros_like(attention_mask, dtype=torch.bool)
                for b in range(input_ids.shape[0]):
                    valid_positions = torch.where(attention_mask[b].bool())[
                        0
                    ].tolist()
                    tokens = input_ids[b, valid_positions].tolist()
                    for cand in candidates:
                        if len(cand) > len(tokens):
                            continue
                        for start in range(len(tokens) - len(cand) + 1):
                            if tokens[start : start + len(cand)] == cand:
                                orig_indices = valid_positions[
                                    start : start + len(cand)
                                ]
                                if all(
                                    valid_text_mask[b, idx].item()
                                    for idx in orig_indices
                                ):
                                    mask[b, orig_indices] = True

                return mask if mask.any() else None

            subtext_mask = _find_subtext_mask(
                input_ids=input_ids,
                attention_mask=input_pad_mask,
                valid_text_mask=text_mask,
                subtext_value=subtext,
            )
            if subtext_mask is not None:
                text_mask = subtext_mask

    return attn_weights, image_mask, text_mask


def map_attention_to_image(
    attention_weights: torch.Tensor,
    image_mask: torch.Tensor,
    text_mask: torch.Tensor,
    image_shape: Tuple[int, int],
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int = 14,
) -> np.ndarray:
    """
    Map attention weights from token space to image space.

    Args:
        attention_weights: [batch, num_heads, seq_len, seq_len] - attention weights
        image_mask: [batch, seq_len] - boolean mask for image tokens
        text_mask: [batch, seq_len] - boolean mask for text tokens
        image_shape: (height, width) of original image
        image_grid_thw: [batch, 3] - (temporal, height, width) grid dimensions
        spatial_merge_size: Spatial merge size from vision model

    Returns:
        attention_map: [height, width] - attention scores mapped to image space
    """
    # Ensure we have batch dimension
    assert attention_weights.ndim == 4
    # Average over heads and batch (assuming batch_size=1)
    attn = attention_weights[0].mean(dim=0)  # [seq_len, seq_len]

    # Get image token indices (should already have batch dimension)
    # image_mask and text_mask should be [batch, seq_len] after unsqueeze(0)
    assert image_mask.ndim == 2
    assert text_mask.ndim == 2
    image_token_indices = torch.where(image_mask[0])[0]
    text_token_indices = torch.where(text_mask[0])[0]

    if len(image_token_indices) == 0 or len(text_token_indices) == 0:
        # No image or text tokens, return zero map
        return np.zeros(image_shape, dtype=np.float32)

    # Extract attention from text tokens to image tokens
    # attn[text_idx, image_idx] gives how much each text token attends to each image token
    text_to_image_attn = attn[text_token_indices][
        :, image_token_indices
    ]  # [num_text, num_image_tokens]
    # normalize over image tokens
    text_to_image_attn = text_to_image_attn / (
        text_to_image_attn.sum(dim=-1, keepdim=True) + 1e-8
    )

    # Average over all text tokens to get per-image-token attention
    image_token_attn = text_to_image_attn.mean(dim=0)  # [num_image_tokens]

    # Normalize attention values to [0, 1] before mapping to image space
    # This ensures the attention values are properly scaled for visualization
    if image_token_attn.max() > image_token_attn.min():
        image_token_attn = (image_token_attn - image_token_attn.min()) / (
            image_token_attn.max() - image_token_attn.min()
        )
    else:
        # If all values are the same, set to 0.5 for visibility
        image_token_attn = torch.ones_like(image_token_attn) * 0.5

    # Map image tokens back to spatial positions
    # Handle image_grid_thw shape - it might be [num_images, 3] or [batch, num_images, 3] or [batch, 1, 3]
    if image_grid_thw.ndim == 2:
        # Could be [num_images, 3] or [batch, 3]
        if image_grid_thw.shape[0] == 1:
            # [1, 3] - single image
            t, h, w = (
                image_grid_thw[0, 0].item(),
                image_grid_thw[0, 1].item(),
                image_grid_thw[0, 2].item(),
            )
            num_images = 1
        else:
            # [num_images, 3] - multiple images, use first image's dimensions
            t, h, w = (
                image_grid_thw[0, 0].item(),
                image_grid_thw[0, 1].item(),
                image_grid_thw[0, 2].item(),
            )
            num_images = image_grid_thw.shape[0]
    elif image_grid_thw.ndim == 3:
        # [batch, num_images, 3] or [batch, 1, 3]
        if image_grid_thw.shape[1] == 1:
            # [batch, 1, 3] - single image
            t, h, w = (
                image_grid_thw[0, 0, 0].item(),
                image_grid_thw[0, 0, 1].item(),
                image_grid_thw[0, 0, 2].item(),
            )
            num_images = 1
        else:
            # [batch, num_images, 3] - multiple images
            t, h, w = (
                image_grid_thw[0, 0, 0].item(),
                image_grid_thw[0, 0, 1].item(),
                image_grid_thw[0, 0, 2].item(),
            )
            num_images = image_grid_thw.shape[1]
    else:
        raise ValueError(
            f"Unexpected image_grid_thw shape: {image_grid_thw.shape}"
        )

    llm_grid_h = h // spatial_merge_size
    llm_grid_w = w // spatial_merge_size
    llm_grid_t = t  # Temporal dimension

    # Reshape attention to spatial grid
    num_image_tokens = len(image_token_attn)
    expected_tokens_per_image = llm_grid_t * llm_grid_h * llm_grid_w
    expected_total_tokens = num_images * expected_tokens_per_image

    # Handle multiple frames or multiple images
    if num_image_tokens == expected_total_tokens:
        # Multiple images: reshape to [num_images, t, h, w] and average over images and temporal dimension
        if num_images > 1:
            print(
                f"Info: Found {num_images} images, averaging attention over all images"
            )
            attn_map_4d = (
                image_token_attn.reshape(
                    num_images, llm_grid_t, llm_grid_h, llm_grid_w
                )
                .cpu()
                .numpy()
            )
            # Average over images and temporal dimension
            attn_map = attn_map_4d.mean(axis=(0, 1))  # [llm_grid_h, llm_grid_w]
        elif llm_grid_t > 1:
            # Single image, multiple frames: reshape to [t, h, w] and average over temporal dimension
            attn_map_3d = (
                image_token_attn.reshape(llm_grid_t, llm_grid_h, llm_grid_w)
                .cpu()
                .numpy()
            )
            attn_map = attn_map_3d.mean(axis=0)  # [llm_grid_h, llm_grid_w]
        else:
            # Single image, single frame: direct reshape
            attn_map = (
                image_token_attn.reshape(llm_grid_h, llm_grid_w).cpu().numpy()
            )
    elif num_image_tokens == expected_tokens_per_image:
        # Single image: handle temporal dimension
        if llm_grid_t > 1:
            # Multiple frames: reshape to [t, h, w] and average
            attn_map_3d = (
                image_token_attn.reshape(llm_grid_t, llm_grid_h, llm_grid_w)
                .cpu()
                .numpy()
            )
            attn_map = attn_map_3d.mean(axis=0)  # [llm_grid_h, llm_grid_w]
        else:
            # Single frame: direct reshape
            attn_map = (
                image_token_attn.reshape(llm_grid_h, llm_grid_w).cpu().numpy()
            )
    elif num_image_tokens % expected_tokens_per_image == 0:
        # Multiple images: reshape and average
        inferred_num_images = num_image_tokens // expected_tokens_per_image
        print(
            f"Info: Inferred {inferred_num_images} images from token count, averaging attention over all images"
        )
        attn_map_4d = (
            image_token_attn.reshape(
                inferred_num_images, llm_grid_t, llm_grid_h, llm_grid_w
            )
            .cpu()
            .numpy()
        )
        attn_map = attn_map_4d.mean(axis=(0, 1))  # [llm_grid_h, llm_grid_w]
    else:
        # Fallback: try to reshape to square grid if possible
        print(
            f"Warning: num_image_tokens={num_image_tokens}, expected={expected_total_tokens} (num_images={num_images}, t={llm_grid_t}, h={llm_grid_h}, w={llm_grid_w}, tokens_per_image={expected_tokens_per_image})"
        )
        # Try to infer grid dimensions from number of tokens
        import math

        grid_size = int(math.sqrt(num_image_tokens))
        if grid_size * grid_size == num_image_tokens:
            print(f"Using inferred grid size: {grid_size}x{grid_size}")
            attn_map = (
                image_token_attn.reshape(grid_size, grid_size).cpu().numpy()
            )
        else:
            # Last resort: create uniform map with correct dimensions
            attn_map = np.zeros((llm_grid_h, llm_grid_w), dtype=np.float32)
            attn_map.fill(image_token_attn.mean().item())

    # Upsample to original image size using PIL
    # Attention is already normalized to [0, 1] above
    attn_img = Image.fromarray((attn_map * 255).astype(np.uint8), mode="L")
    attn_img = attn_img.resize(
        (image_shape[1], image_shape[0]), Image.Resampling.BILINEAR
    )
    attn_map = np.array(attn_img).astype(np.float32) / 255.0

    # Ensure values are in [0, 1] after resize (may have slight variations)
    attn_map = np.clip(attn_map, 0.0, 1.0)

    return attn_map


def visualize_attention(
    img_chw: np.ndarray,
    attention_map: np.ndarray,
    output_path: str,
    instruction: Optional[str] = None,
    alpha: float = 0.5,
):
    """
    Visualize attention map overlaid on the original image.

    Args:
        img_chw: Original image [3, H, W] in uint8
        attention_map: Attention map [H, W] in float32
        output_path: Path to save visualization
        instruction: Optional text instruction to display on the image
        alpha: Transparency for attention overlay
    """
    # Normalize attention map to [0, 1] (should already be normalized, but ensure it)
    attn_min, attn_max = attention_map.min(), attention_map.max()
    if attn_max > attn_min:
        attn_norm = (attention_map - attn_min) / (attn_max - attn_min)
    else:
        # If all values are the same, create a uniform map
        attn_norm = np.ones_like(attention_map) * 0.5

    # Create colormap
    cmap = cm.get_cmap("jet")
    attn_colored = cmap(attn_norm)[:, :, :3]  # [H, W, 3] in [0, 1]
    attn_colored = (attn_colored * 255).astype(np.uint8)

    # Overlay on original image
    img_hwc = img_chw.transpose(1, 2, 0)  # [H, W, 3]
    # Ensure same shape
    if attn_colored.shape[:2] != img_hwc.shape[:2]:
        attn_colored = np.array(
            Image.fromarray(attn_colored).resize(
                (img_hwc.shape[1], img_hwc.shape[0]), Image.Resampling.BILINEAR
            )
        )

    overlay = (alpha * attn_colored + (1 - alpha) * img_hwc).astype(np.uint8)

    # Convert to PIL Image for text drawing
    img_pil = Image.fromarray(overlay)

    # Add instruction text if provided
    if instruction:
        draw = ImageDraw.Draw(img_pil)
        # Try to load a font, fallback to default if not available
        try:
            # Try to use a larger font size
            font_size = max(20, min(img_pil.height // 20, 40))
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                font_size,
            )
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None

        # Prepare text with word wrapping
        max_width = img_pil.width - 40  # Leave margins
        words = instruction.split()
        lines = []
        current_line = []
        current_width = 0

        for word in words:
            if font:
                bbox = draw.textbbox((0, 0), word, font=font)
                word_width = bbox[2] - bbox[0]
            else:
                word_width = len(word) * 10  # Approximate

            if current_width + word_width > max_width and current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_width
            else:
                current_line.append(word)
                current_width += word_width + (
                    10 if font else 5
                )  # Add space width

        if current_line:
            lines.append(" ".join(current_line))

        # Draw text with background for better visibility
        y_offset = 10
        for line in lines:
            if font:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_height = bbox[3] - bbox[1]
            else:
                text_height = 15

            # Draw semi-transparent background
            padding = 5
            bg_coords = [
                10,
                y_offset - padding,
                10 + max_width + 20,
                y_offset + text_height + padding,
            ]
            bg_img = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
            bg_draw = ImageDraw.Draw(bg_img)
            bg_draw.rectangle(
                bg_coords, fill=(0, 0, 0, 180)
            )  # Semi-transparent black
            img_pil = Image.alpha_composite(
                img_pil.convert("RGBA"), bg_img
            ).convert("RGB")
            draw = ImageDraw.Draw(img_pil)

            # Draw text
            draw.text((15, y_offset), line, fill=(255, 255, 255), font=font)
            y_offset += text_height + 5

    # Save
    img_pil.save(output_path)
    print(f"Saved attention visualization to {output_path}")


def load_image(image_path: str) -> np.ndarray:
    """Load and preprocess image."""
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def extract_object_name(instruction: str) -> str:
    """
    Extract object name from instruction text.

    Supports formats like:
    - "put {object} on {location}" -> returns "{object}"
    - "put {object} in {location}" -> returns "{object}"
    - "place {object} on {location}" -> returns "{object}"
    - "move {object} to {location}" -> returns "{object}"

    Args:
        instruction: Full instruction text

    Returns:
        Extracted object name, or original instruction if pattern doesn't match
    """
    instruction = instruction.strip().lower()

    # Common patterns for "put X on Y" type instructions
    patterns = [
        (r"^put\s+(.+?)\s+on\s+", "put X on Y"),
        (r"^put\s+(.+?)\s+in\s+", "put X in Y"),
        (r"^put\s+(.+?)\s+into\s+", "put X into Y"),
        (r"^place\s+(.+?)\s+on\s+", "place X on Y"),
        (r"^place\s+(.+?)\s+in\s+", "place X in Y"),
        (r"^move\s+(.+?)\s+to\s+", "move X to Y"),
        (r"^move\s+(.+?)\s+on\s+", "move X on Y"),
        (
            r"^pick\s+up\s+(.+?)\s+and\s+put\s+it\s+on\s+",
            "pick up X and put it on Y",
        ),
        (
            r"^pick\s+(.+?)\s+up\s+and\s+put\s+it\s+on\s+",
            "pick X up and put it on Y",
        ),
    ]

    for pattern, _ in patterns:
        match = re.match(pattern, instruction)
        if match:
            object_name = match.group(1).strip()
            # Remove any trailing words that might be part of location
            # For example, "banana on plate" -> "banana"
            # But we want to keep compound names like "red apple"
            return object_name

    # If no pattern matches, try to extract first noun phrase
    # This is a fallback for other formats
    words = instruction.split()
    if len(words) >= 2:
        # Common verbs to skip
        verbs = {"put", "place", "move", "pick", "grab", "take", "set"}
        if words[0] in verbs:
            # Return the next word(s) as object name
            # Usually object is the second or third word
            if len(words) >= 3 and words[1] in {"up", "the", "a", "an"}:
                return " ".join(words[2:4]) if len(words) >= 4 else words[2]
            else:
                return words[1] if len(words) >= 2 else words[0]

    # Fallback: return original instruction
    return instruction


def convert_lerobot_image_to_numpy(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert LeRobotDataset image tensor to numpy array.

    Args:
        img_tensor: Image tensor from LeRobotDataset, shape (C, H, W) or (H, W, C)
                   dtype can be float32 [0,1] or uint8 [0,255]

    Returns:
        numpy array in (H, W, C) format, uint8
    """
    # Convert to numpy
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.cpu().numpy()
    else:
        img = img_tensor

    # Handle different shapes
    if img.ndim == 4:
        # Take first frame if multiple frames
        img = img[0]

    # Handle channel-first vs channel-last
    if (
        img.shape[0] in [1, 3]
        and img.shape[0] < img.shape[1]
        and img.shape[0] < img.shape[2]
    ):
        # Channel-first: (C, H, W) -> (H, W, C)
        img = np.transpose(img, (1, 2, 0))

    # Handle different dtypes
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            # Normalized float [0, 1]
            img = (img * 255).astype(np.uint8)
        else:
            # Float with values > 1, normalize first
            img = img.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).astype(np.uint8)

    # Handle grayscale images
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    return img


@dataclass
class VisualizeAttentionConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"policy": "pi"},
            {"data": "libero-ipec"},
        ]
    )

    # configs
    data: DataConfig = MISSING
    policy: PolicyConfig = MISSING
    checkpoint_path: Optional[str] = None

    # visualization parameters
    output_dir: str = (
        "attention_visualizations"  # Output directory for all visualizations
    )
    layer_idx: int = 6  # Last layer by default
    max_episodes: Optional[int] = (
        None  # Limit number of episodes to process (None = all)
    )


cs = ConfigStore.instance()
cs.store(name="visualize_attention", node=VisualizeAttentionConfig)


@hydra.main(config_name="visualize_attention", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    args = cast(VisualizeAttentionConfig, OmegaConf.to_object(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model from policy config
    # Disable augmentation if present
    for i, spec in enumerate(list(args.data.input_transforms or [])):
        if isinstance(spec, dict) and "enable_aug" in spec:
            spec.update({"enable_aug": False})
            args.data.input_transforms[i] = spec

    # Initialize policy dimensions from dataset
    args.data.action_horizon = args.policy.action_horizon
    args.data.state_history = args.policy.state_history
    dataset = create_dataset(
        args.data,
        args.policy,
        skip_norm_stats=False,
        skip_policy_transforms=True,
    )
    if len(dataset) > 0:
        data_sample, _ = dataset[0]
        if data_sample.action_chunk is not None:
            action_dim = int(data_sample.action_chunk.actions.shape[-1])
            if args.policy.action_dim is None:
                args.policy.action_dim = action_dim
        if data_sample.observation.state is not None:
            state_dim = int(data_sample.observation.state.shape[-1])
            if args.policy.state_dim is None:
                args.policy.state_dim = state_dim
    print("Initializing model...")
    with torch.device(device):
        model = create_policy(args.policy)
    model.eval()
    print("Model initialized.")

    # Resolve checkpoint path (supports file or directory)
    if args.checkpoint_path is not None:
        ckpt = find_latest_checkpoint(args.checkpoint_path)
        if ckpt is None:
            raise FileNotFoundError(
                f"No checkpoint found under {args.checkpoint_path}"
            )
        print(f"Loading checkpoint: {ckpt}")
        missing, unexpected = load_model_from_checkpoint(
            model, ckpt, device, strict=False
        )
        print("Checkpoint loaded.")
        if missing:
            print(f"Warning: Missing keys when loading checkpoint: {missing}")
        if unexpected:
            print(
                f"Warning: Unexpected keys when loading checkpoint: {unexpected}"
            )
    else:
        print(
            "Warning: No checkpoint_path provided, using untrained model weights"
        )

    model.eval()

    # Get processor from model
    if not hasattr(model, "vlm_bridge"):
        raise ValueError("Model does not have vlm_bridge attribute")

    bridge = model.vlm_bridge
    if not isinstance(bridge, Qwen3VLBridge):
        raise ValueError(f"Expected Qwen3VLBridge, got {type(bridge)}")

    # Build input transforms to get the processor
    input_transforms = [ToTorch()] + build_input_transforms(
        args.data, args.policy
    )

    # Load LeRobotDataset
    dataset: "LeRobotDataset" = dataset.base_dataset.dataset
    print(
        f"Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames"
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Determine number of episodes to process
    num_episodes = dataset.num_episodes
    if args.max_episodes is not None:
        num_episodes = min(num_episodes, args.max_episodes)

    print(f"Processing {num_episodes} episodes (all frames)...")

    # Get image key from dataset
    image_key = "observation.images.image"
    if image_key not in dataset.features:
        # Try alternative keys
        camera_keys = (
            dataset.meta.camera_keys
            if hasattr(dataset.meta, "camera_keys")
            else []
        )
        if camera_keys:
            image_key = camera_keys[0]
        else:
            raise ValueError(
                f"Could not find image key in dataset. Available keys: {list(dataset.features.keys())}"
            )

    print(f"Using image key: {image_key}")

    # Process each episode
    for episode_idx in range(num_episodes):
        print(f"\nProcessing episode {episode_idx}...")

        start_idx = dataset.meta.episodes["dataset_from_index"][episode_idx]
        end_idx = dataset.meta.episodes["dataset_to_index"][episode_idx]

        for frame_idx in range(start_idx, end_idx + 1):
            frame_data = dataset[frame_idx]

            # Extract image and instruction
            img_tensor = frame_data[image_key]

            # Get instruction/task - LeRobotDataset adds "task" field in __getitem__
            instruction = frame_data.get("task", "")

            # Extract object name from instruction
            object_name = extract_object_name(instruction)
            print(f"  Full instruction: {instruction}")
            print(f"  Extracted object name: {object_name}")

            # Use object name as the input for the model
            model_input_instruction = instruction

            img_chw = (img_tensor * 255).type(torch.uint8).cpu().numpy()
            original_shape = img_chw.shape[1:]  # (H, W)

            # Create payload dict (same format as build_payload and serve_policy input)
            payload = {
                PROCESSED_IMAGE_KEY: img_chw,  # (3, H, W) - will be processed by transforms
                PROCESSED_IMAGE_MASK_KEY: np.ones((1,), dtype=bool),
                PROCESSED_STATE_KEY: np.zeros(
                    (args.policy.state_history + 1, args.policy.state_dim or 1),
                    dtype=np.float32,
                ),
                TASK_KEY: model_input_instruction,
                GENERATION_PROMPT_KEY: "",
                GENERATION_ANSWER_KEY: "",
            }

            # Process through transforms (same as serve_policy.py)
            data_sample = payload
            for transform in input_transforms:
                data_sample: "DataSample" = transform.compute(data_sample)

            # Move to device and add batch dimension (same as serve_policy.py line 147)
            data_sample = data_sample.to(device).unsqueeze(0)

            attn_weights, image_mask, text_mask = extract_attention_from_bridge(
                bridge,
                data_sample.observation,
                layer_idx=args.layer_idx,
                subtext=object_name,
            )

            # Get image grid info
            policy_input = data_sample.observation.policy_input
            image_grid_thw = policy_input.image_grid_thw

            # Get spatial merge size from vision model
            spatial_merge_size = getattr(
                bridge.processor.image_processor, "merge_size", 14
            )

            # Map attention to image space
            attention_map = map_attention_to_image(
                attn_weights,
                image_mask,
                text_mask,
                original_shape,
                image_grid_thw,
                spatial_merge_size=spatial_merge_size,
            )

            # Visualize and save
            # Display both full instruction and extracted object name for clarity
            display_text = f"{instruction}\n(Object: {object_name})"
            ep_idx = (
                int(frame_data["episode_index"].item())
                if "episode_index" in frame_data
                else episode_idx
            )
            frame_idx_in_ep = (
                int(frame_data["frame_index"].item())
                if "frame_index" in frame_data
                else (frame_idx - start_idx)
            )
            episode_dir = output_dir / f"episode_{ep_idx}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            output_path = episode_dir / f"frame_{frame_idx_in_ep}.png"
            visualize_attention(
                img_chw,
                attention_map,
                str(output_path),
                instruction=display_text,
            )

    print(
        f"\nDone! Processed {num_episodes} episodes. Results saved to {output_dir}"
    )


if __name__ == "__main__":
    main()
