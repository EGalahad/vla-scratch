import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, MISSING, OmegaConf
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from vla_scratch.policies.config import PolicyConfig, create_policy
from vla_scratch.datasets.config import DataConfig
from vla_scratch.helpers.data import create_dataset
from vla_scratch.utils.checkpoint import (
    find_latest_checkpoint,
    load_model_from_checkpoint,
)
from vla_scratch.policies.modules.vlm_bridge.qwen.utils import (
    restore_qwen3vl_forward,
)


@dataclass
class PredictBboxConfig:
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
        "bbox_visualizations"  # Output directory for all visualizations
    )
    max_episodes: Optional[int] = None


cs = ConfigStore.instance()
cs.store(name="predict_bbox", node=PredictBboxConfig)


@hydra.main(config_name="predict_bbox", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    args = cast(PredictBboxConfig, OmegaConf.to_object(cfg))

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
        skip_policy_transforms=False,
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
    # Restore original forward methods before using model for generation
    # This is needed because the bridge may have replaced forward methods
    restore_qwen3vl_forward()
    model_for_generation = model.vlm_bridge.causal_model
    processor = model.vlm_bridge.processor

    # Load LeRobotDataset
    lerobot_dataset: "LeRobotDataset" = dataset.base_dataset.dataset
    print(
        f"Dataset loaded: {lerobot_dataset.num_episodes} episodes, {lerobot_dataset.num_frames} frames"
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Get image key from dataset
    image_key = "observation.images.image"
    if image_key not in lerobot_dataset.features:
        # Try alternative keys
        camera_keys = (
            lerobot_dataset.meta.camera_keys
            if hasattr(lerobot_dataset.meta, "camera_keys")
            else []
        )
        if camera_keys:
            image_key = camera_keys[0]
        else:
            raise ValueError(
                f"Could not find image key in dataset. Available keys: {list(lerobot_dataset.features.keys())}"
            )

    print(f"Using image key: {image_key}")

    # Determine number of episodes to process
    num_episodes = lerobot_dataset.num_episodes
    if args.max_episodes is not None:
        num_episodes = min(num_episodes, args.max_episodes)

    print(f"Processing {num_episodes} episodes (all frames)...")

    # Bbox prompt (same as in dataset.py)
    bbox_prompt = (
        "Please return bounding boxes for all task-relevant objects in JSON format as"
        '[{"bbox_2d": [x1, y1, x2, y2], "label": "<object_name>"}]'
    )
    # import pdb;pdb.set_trace()
    # Process each episode (all frames)
    for episode_idx in range(num_episodes):
        print(f"\nProcessing episode {episode_idx}...")

        start_idx = lerobot_dataset.meta.episodes["dataset_from_index"][
            episode_idx
        ]
        end_idx = lerobot_dataset.meta.episodes["dataset_to_index"][episode_idx]

        for frame_idx in range(start_idx, end_idx + 1):
            frame_data = lerobot_dataset[frame_idx]

            # Extract image
            img_tensor = frame_data[image_key]

            # Get episode and frame indices
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

            # Convert image tensor to PIL Image
            image_pil = convert_tensor_to_pil(img_tensor)
            # Get instruction/task from frame data
            instruction = frame_data.get("task", "")
            # Generate bbox prediction
            with torch.no_grad():
                bbox_json = generate_bbox(
                    image=image_pil,
                    task=instruction,
                    prompt=bbox_prompt,
                    model=model_for_generation,
                    processor=processor,
                    device=device,
                )

            # Parse bbox JSON
            # The output format is typically: [{"bbox_2d": [...], "label": "..."}, ...]<|im_end|>
            print(f"Bbox JSON: {bbox_json}")
            bboxes = parse_bbox_json(bbox_json)

            # Visualize and save
            episode_dir = output_dir / f"episode_{ep_idx}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            output_path = episode_dir / f"frame_{frame_idx_in_ep}.png"
            visualize_bbox(
                image_pil,
                bboxes,
                str(output_path),
                ep_idx,
                frame_idx_in_ep,
                instruction=instruction,
            )

    print(
        f"\nDone! Processed {num_episodes} episodes. Results saved to {output_dir}"
    )


def convert_tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """
    Convert image tensor from LeRobotDataset to PIL Image.

    Args:
        img_tensor: Image tensor, shape (C, H, W) or (H, W, C), dtype float32 [0,1] or uint8 [0,255]

    Returns:
        PIL Image in RGB format
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

    # Convert to PIL Image
    return Image.fromarray(img)


def prefill(
    *,
    image: Image.Image,
    task: str,
    prompt: str,
    model,
    processor,
    device: torch.device,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Prefill pass that returns cache, cache_position, attention_mask, and next token."""
    prompt_sep_text = "<<<PROMPT_SEP>>>"
    content: List[Dict] = [
        {"type": "image", "image": image},
        {"type": "text", "text": task},
        {"type": "text", "text": prompt_sep_text},
        {"type": "text", "text": prompt},
    ]
    messages = [{"role": "user", "content": content}]

    encoded = processor.apply_chat_template(
        [messages],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        text_kwargs={
            "max_length": 500,  # Match training config: policy.transforms.0.max_length=500
            "truncation": False,  # Match training: truncation=False in QwenProcessor.compute
            "padding": "max_length",  # Match training config
            "return_tensors": "pt",
        },
    )
    model_inputs = {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in encoded.items()
    }

    with torch.inference_mode():
        outputs = model(**model_inputs, use_cache=True)

    hidden_states = outputs.hidden_states
    past_key_values = outputs.past_key_values
    cache_position = torch.tensor(
        [model_inputs["input_ids"].shape[1]], device=device
    )
    attention_mask = model_inputs["attention_mask"]
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return (
        hidden_states,
        past_key_values,
        cache_position,
        attention_mask,
        next_token,
    )


def decode(
    *,
    model,
    processor,
    past_key_values,
    cache_position: torch.Tensor,
    attention_mask: torch.Tensor,
    next_token: torch.Tensor,
    max_new_tokens: int = 256,
) -> str:
    """Manual autoregressive decoding loop with explicit KV cache updates."""
    generated_ids: list[torch.Tensor] = []
    eos_token_id = processor.tokenizer.encode("<|im_end|>")[0]
    device = attention_mask.device

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            generated_ids.append(next_token)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        device=device,
                        dtype=attention_mask.dtype,
                    ),
                ],
                dim=1,
            )

            decode_inputs = model.prepare_inputs_for_generation(
                input_ids=next_token,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                cache_position=cache_position,
            )
            decode_outputs = model(**decode_inputs)
            past_key_values = decode_outputs.past_key_values
            cache_position = cache_position + 1

            next_token = decode_outputs.logits[:, -1, :].argmax(
                dim=-1, keepdim=True
            )
            if next_token.item() == eos_token_id:
                break

    return processor.decode(torch.cat(generated_ids, dim=-1)[0])


def generate_bbox(
    image: Image.Image,
    task: str,
    prompt: str,
    model,
    processor,
    device: torch.device,
    max_new_tokens: int = 256,
) -> str:
    """Generate bbox prediction using prefill and decode."""
    hidden_states, kv_cache, cache_position, attention_mask, next_token = (
        prefill(
            image=image,
            task=task,
            prompt=prompt,
            model=model,
            processor=processor,
            device=device,
        )
    )
    decoded = decode(
        model=model,
        processor=processor,
        past_key_values=kv_cache,
        cache_position=cache_position,
        attention_mask=attention_mask,
        next_token=next_token,
        max_new_tokens=max_new_tokens,
    )
    return decoded


def parse_bbox_json(bbox_json: str) -> List[Dict]:
    """
    Parse bbox JSON from model output.
    The output format is typically: [{"bbox_2d": [...], "label": "..."}, ...]<|im_end|>

    Args:
        bbox_json: Raw JSON string from model output

    Returns:
        List of bbox dicts with keys "bbox_2d" and "label"
    """
    bbox_json_clean = bbox_json.strip()

    # Remove special tokens like <|im_end|>, <|im_start|>, etc.
    bbox_json_clean = re.sub(r"<\|[^|]+\|>", "", bbox_json_clean)

    # The JSON array should be at the start, before any special tokens
    # Find the first '[' and the matching ']' before any special tokens
    start_idx = bbox_json_clean.find("[")
    if start_idx == -1:
        return []

    # Find the matching closing bracket
    bracket_count = 0
    end_idx = start_idx
    for i in range(start_idx, len(bbox_json_clean)):
        if bbox_json_clean[i] == "[":
            bracket_count += 1
        elif bbox_json_clean[i] == "]":
            bracket_count -= 1
            if bracket_count == 0:
                end_idx = i + 1
                break

    if end_idx <= start_idx:
        return []

    bbox_json_clean = bbox_json_clean[start_idx:end_idx]

    try:
        bboxes = json.loads(bbox_json_clean)
        if not isinstance(bboxes, list):
            return []
        return bboxes
    except json.JSONDecodeError:
        print(f"Warning: Failed to parse bbox JSON")
        print(f"  Raw output: {bbox_json[:200]}...")
        print(f"  Cleaned: {bbox_json_clean[:200]}...")
        return []


def visualize_bbox(
    image: Image.Image,
    bboxes: List[Dict],
    output_path: str,
    ep_idx: int,
    frame_idx: int,
    instruction: Optional[str] = None,
):
    """
    Visualize bounding boxes on image and save.

    Args:
        image: PIL Image
        bboxes: List of dicts with keys "bbox_2d" and "label"
                bbox_2d format: [x1, y1, x2, y2] where coordinates are in [0, 1000] range
        output_path: Path to save visualization
        ep_idx: Episode index
        frame_idx: Frame index
    """
    # Create a copy for drawing
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    # Try to load a font
    try:
        font_size = max(16, min(img_draw.height // 30, 24))
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None

    # Get image dimensions
    img_width, img_height = img_draw.size

    # Draw each bbox
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    for idx, bbox in enumerate(bboxes):
        if "bbox_2d" not in bbox or "label" not in bbox:
            continue

        coords = bbox["bbox_2d"]
        label = bbox["label"]

        # Convert from [0, 1000] range to pixel coordinates
        x1 = int(coords[0] * img_width / 1000)
        y1 = int(coords[1] * img_height / 1000)
        x2 = int(coords[2] * img_width / 1000)
        y2 = int(coords[3] * img_height / 1000)

        # Clamp to image bounds
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))

        # Skip invalid bboxes
        if x1 >= x2 or y1 >= y2:
            continue

        # Choose color
        color = colors[idx % len(colors)]

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label background and text
        if font:
            bbox_text = draw.textbbox((0, 0), label, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
        else:
            text_width = len(label) * 8
            text_height = 12

        # Position label above bbox
        label_y = max(0, y1 - text_height - 5)
        if label_y < 0:
            label_y = y2 + 5

        # Draw background rectangle for text
        bg_coords = [
            x1,
            label_y,
            x1 + text_width + 10,
            label_y + text_height + 5,
        ]
        draw.rectangle(bg_coords, fill=color)

        # Draw text
        draw.text((x1 + 5, label_y + 2), label, fill=(255, 255, 255), font=font)

    # Add instruction text if provided (similar to visualize_attention.py)
    if instruction:
        # Prepare text with word wrapping
        max_width = img_draw.width - 40  # Leave margins
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
            bg_img = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
            bg_draw = ImageDraw.Draw(bg_img)
            bg_draw.rectangle(
                bg_coords, fill=(0, 0, 0, 180)
            )  # Semi-transparent black
            img_draw = Image.alpha_composite(
                img_draw.convert("RGBA"), bg_img
            ).convert("RGB")
            draw = ImageDraw.Draw(img_draw)

            # Draw text
            draw.text((15, y_offset), line, fill=(255, 255, 255), font=font)
            y_offset += text_height + 5

    # Add episode and frame info at bottom
    info_text = f"Episode {ep_idx}, Frame {frame_idx}"
    if font:
        bbox_info = draw.textbbox((0, 0), info_text, font=font)
        info_height = bbox_info[3] - bbox_info[1]
    else:
        info_height = 12

    # Draw info background at bottom right
    info_y = img_draw.height - info_height - 15
    info_bg_coords = [
        img_draw.width - len(info_text) * 8 - 30,
        info_y - 5,
        img_draw.width - 10,
        info_y + info_height + 5,
    ]
    draw.rectangle(info_bg_coords, fill=(128, 128, 128))
    draw.text(
        (img_draw.width - len(info_text) * 8 - 25, info_y),
        info_text,
        fill=(255, 255, 255),
        font=font,
    )

    # Save
    img_draw.save(output_path)


if __name__ == "__main__":
    main()
