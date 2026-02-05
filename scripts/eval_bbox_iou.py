#!/usr/bin/env python3

"""
Offline bbox evaluation on LeRobot-style datasets with `meta/bboxes.jsonl`.

Computes:
- mean IoU (reported as IoU AUC; since ∫_0^1 1[iou>=t] dt = iou)
- acc@0.25 / acc@0.5 / acc@0.75: fraction of GT boxes with max IoU above threshold

Intended usage for LIBERO:
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/eval_bbox_iou.py \
    policy=pi-paligemma data=libero-90-bbox \
    checkpoint_path=/path/to/checkpoint_3 merge_policy_cfg=true \
    num_samples=512 batch_size=8 max_new_tokens=256 \
    output_json=outputs/bbox_eval_libero_90.json output_dir=outputs/bbox_eval_libero_90_episodes
"""

from __future__ import annotations

import json
import inspect
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast, TYPE_CHECKING, Iterable

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf, MISSING

from vla_scratch.datasets.config import DataConfig
from vla_scratch.helpers.data import create_dataset
from vla_scratch.policies.config import PolicyConfig
from vla_scratch.utils.checkpoint import (
    find_latest_checkpoint,
    load_model_from_checkpoint,
    merge_policy_cfg_from_checkpoint,
)

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from vla_scratch.policies.base import BasePolicy


def _clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _xyxy_to_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ax1, ay1, ax2, ay2 = min(ax1, ax2), min(ay1, ay2), max(ax1, ax2), max(ay1, ay2)
    bx1, by1, bx2, by2 = min(bx1, bx2), min(by1, by2), max(bx1, bx2), max(by1, by2)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    img = img_tensor.detach().cpu().numpy()
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[0] < img.shape[1]:
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return Image.fromarray(img)


def _parse_pred_bboxes(text: str) -> List[Tuple[float, float, float, float]]:
    # Expect JSON list of {"bbox_2d":[x1,y1,x2,y2], "label":...}
    s = text.strip()
    # Strip special tokens like <|im_end|>
    s = s.replace("<|im_end|>", "")
    start = s.find("[")
    if start < 0:
        return []
    bracket = 0
    end = -1
    for i in range(start, len(s)):
        if s[i] == "[":
            bracket += 1
        elif s[i] == "]":
            bracket -= 1
            if bracket == 0:
                end = i + 1
                break
    if end <= start:
        return []
    s = s[start:end]
    try:
        parsed = json.loads(s)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    out: List[Tuple[float, float, float, float]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        coords = item.get("bbox_2d")
        if not (isinstance(coords, list) and len(coords) == 4):
            continue
        try:
            x1, y1, x2, y2 = [float(c) for c in coords]
        except Exception:
            continue
        # Most datasets use [0,1000] ints; normalize to [0,1]
        x1, y1, x2, y2 = x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0
        out.append((_clamp01(x1), _clamp01(y1), _clamp01(x2), _clamp01(y2)))
    return out


def _get_policy_transform_spec(
    policy_cfg: PolicyConfig, *, target_suffix: str
) -> Optional[dict]:
    tfs = getattr(policy_cfg, "transforms", None) or []
    for spec in tfs:
        if not isinstance(spec, dict):
            continue
        target = str(spec.get("_target_", ""))
        if target.endswith(target_suffix):
            return dict(spec)
    return None


def _resolve_prompt_sep_text(policy_cfg: PolicyConfig) -> str:
    spec = _get_policy_transform_spec(
        policy_cfg,
        target_suffix="vla_scratch.policies.modules.vlm_bridge.paligemma.processor.PaligemmaProcessor",
    )
    if spec is None:
        return "<<<PROMPT_SEP>>>"
    # Training transform hardcodes this, but keep a safe fallback.
    return str(spec.get("prompt_sep_text", "<<<PROMPT_SEP>>>"))


def _resolve_paligemma_encode_kwargs(policy_cfg: PolicyConfig) -> dict[str, Any]:
    spec = _get_policy_transform_spec(
        policy_cfg,
        target_suffix="vla_scratch.policies.modules.vlm_bridge.paligemma.processor.PaligemmaProcessor",
    )
    if spec is None:
        return {}
    out: dict[str, Any] = {}
    for k in ("max_length", "truncation", "padding"):
        if k in spec:
            out[k] = spec[k]
    return out


def _call_with_supported_kwargs(fn, **kwargs):
    sig = inspect.signature(fn)
    # If the callable accepts **kwargs, do not filter (nn.Module.__call__ often does).
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        supported = dict(kwargs)
    else:
        supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**supported)


def _prefill_paligemma(
    *,
    model,
    processor,
    images: List[Image.Image],
    task: str,
    prompt_sep_text: str,
    prompt: str,
    device: torch.device,
    encode_kwargs: dict[str, Any],
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Training-style text: "<image>"*N + task + sep + generation_prompt
    text = "".join(["<image>"] * len(images) + [f"{task}{prompt_sep_text}{prompt}"])
    encoded = processor(
        text=text,
        images=[images],
        return_tensors="pt",
        **encode_kwargs,
    )
    model_inputs = {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in encoded.items()
    }
    # Some processor/model versions may surface `inputs_embeds` in addition to `input_ids`.
    # PaliGemma requires exactly one of them.
    if "input_ids" in model_inputs and "inputs_embeds" in model_inputs:
        model_inputs.pop("inputs_embeds", None)
    with torch.inference_mode():
        outputs = _call_with_supported_kwargs(model, use_cache=True, **model_inputs)
    past_key_values = getattr(outputs, "past_key_values", None)
    if past_key_values is None:
        raise RuntimeError("Model did not return past_key_values; cannot do manual decode.")
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise RuntimeError("Model did not return logits; cannot decode next token.")
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(model_inputs["input_ids"], dtype=torch.long)
    cache_position = torch.tensor([int(model_inputs["input_ids"].shape[1])], device=device)
    return past_key_values, cache_position, attention_mask, next_token


def _decode_paligemma(
    *,
    model,
    tokenizer,
    past_key_values,
    cache_position: torch.Tensor,
    attention_mask: torch.Tensor,
    next_token: torch.Tensor,
    max_new_tokens: int,
) -> str:
    generated_ids: List[int] = []
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = getattr(getattr(model, "config", None), "eos_token_id", None)
    if eos_token_id is None:
        raise RuntimeError("Could not resolve eos_token_id for decoding.")
    device = attention_mask.device

    with torch.inference_mode():
        for _ in range(int(max_new_tokens)):
            generated_ids.append(int(next_token.item()))
            if int(next_token.item()) == int(eos_token_id):
                break
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
            if "input_ids" in decode_inputs and "inputs_embeds" in decode_inputs:
                decode_inputs.pop("inputs_embeds", None)
            decode_outputs = _call_with_supported_kwargs(model, **decode_inputs)
            past_key_values = getattr(decode_outputs, "past_key_values", past_key_values)
            cache_position = cache_position + 1
            logits = getattr(decode_outputs, "logits", None)
            if logits is None:
                raise RuntimeError("Decode step did not return logits.")
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

    return tokenizer.decode(generated_ids, skip_special_tokens=False)


class _BBoxEvalFrameDataset(Dataset):
    def __init__(
        self,
        *,
        transformed_ds: Dataset,
        lerobot_ds: "LeRobotDataset",
        indices: List[int],
        gt_map: Dict[Tuple[int, int], List[Tuple[float, float, float, float]]],
    ) -> None:
        super().__init__()
        self._transformed_ds = transformed_ds
        self._lerobot_ds = lerobot_ds
        self._indices = indices
        self._gt_map = gt_map

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> Optional[Dict[str, Any]]:
        idx = int(self._indices[i])
        # Use raw frame only for episode/frame indices (GT alignment).
        frame_raw = self._lerobot_ds[idx]
        # Use transformed dataset to match training-time task/images/prompt formatting.
        sample, _perf = self._transformed_ds[idx]

        ep_idx = (
            int(frame_raw["episode_index"].item())
            if "episode_index" in frame_raw
            else 0
        )
        fr_idx = (
            int(frame_raw["frame_index"].item())
            if "frame_index" in frame_raw
            else idx
        )
        gt_boxes = self._gt_map.get((ep_idx, fr_idx))
        if not gt_boxes:
            return None

        obs = sample.observation
        if obs is None or obs.images is None:
            return None
        # obs.images: (num_cam, 3, H, W) uint8
        images: List[Image.Image] = [
            _tensor_to_pil(obs.images[j]) for j in range(int(obs.images.shape[0]))
        ]

        task = str(obs.task)
        prompt = str(getattr(obs, "generation_prompt", ""))
        if not prompt:
            return None

        return {
            "dataset_index": idx,
            "episode_index": ep_idx,
            "frame_index": fr_idx,
            "task": task,
            "prompt": prompt,
            "images": images,
            "gt_boxes": gt_boxes,
        }


def _collate_bbox_eval(items: List[Optional[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return [x for x in items if x is not None]


@dataclass
class EvalBboxConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"policy": "pi-paligemma"},
            {"data": "libero-90-bbox"},
        ]
    )

    data: DataConfig = MISSING
    policy: PolicyConfig = MISSING
    checkpoint_path: Optional[str] = None
    merge_policy_cfg: bool = False

    # generation
    max_new_tokens: int = 256
    use_bf16: bool = True

    # eval controls
    num_samples: int = 512  # number of frames to evaluate (with GT bboxes)
    batch_size: int = 1  # NOTE: manual decode path currently supports only batch_size=1
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True

    thresholds: Tuple[float, float, float] = (0.25, 0.5, 0.75)
    output_json: str = "outputs/bbox_eval.json"
    output_dir: Optional[str] = None


cs = ConfigStore.instance()
cs.store(name="eval_bbox_iou", node=EvalBboxConfig())


def _load_gt_bboxes(meta_root: Path) -> Dict[Tuple[int, int], List[Tuple[float, float, float, float]]]:
    bbox_path = meta_root / "meta" / "bboxes.jsonl"
    if not bbox_path.exists():
        raise FileNotFoundError(f"Missing bbox annotations: {bbox_path}")
    gt: Dict[Tuple[int, int], List[Tuple[float, float, float, float]]] = {}
    with bbox_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ep = int(rec["episode_index"])
            fr = int(rec["frame_index"])
            bboxes = rec.get("bbox") or []
            boxes_xyxy: List[Tuple[float, float, float, float]] = []
            for b in bboxes:
                coords = b.get("bbox_normalized")
                if not (isinstance(coords, list) and len(coords) == 4):
                    continue
                x1, y1, x2, y2 = [float(c) for c in coords]
                boxes_xyxy.append((_clamp01(x1), _clamp01(y1), _clamp01(x2), _clamp01(y2)))
            if boxes_xyxy:
                gt[(ep, fr)] = boxes_xyxy
    return gt


@hydra.main(config_name="eval_bbox_iou", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    args = cast(EvalBboxConfig, OmegaConf.to_object(cfg))
    if (checkpoint_path := cfg.get("checkpoint_path")) is not None:
        cfg.checkpoint_path = find_latest_checkpoint(checkpoint_path)
    if cfg.get("merge_policy_cfg", False):
        cfg = merge_policy_cfg_from_checkpoint(cfg, cfg.get("checkpoint_path"))
        OmegaConf.resolve(cfg)
        args = cast(EvalBboxConfig, OmegaConf.to_object(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(args.use_bf16) and device.type == "cuda"

    # Disable augmentation if present
    for i, spec in enumerate(list(args.data.input_transforms or [])):
        if isinstance(spec, dict) and "enable_aug" in spec:
            spec.update({"enable_aug": False})
            args.data.input_transforms[i] = spec

    # Align temporal params for policy instantiation (not used for bbox gen itself)
    args.data.action_horizon = args.policy.action_horizon
    args.data.state_history = args.policy.state_history

    # Build transformed dataset, but skip policy transforms so we can:
    # - use the dataset-provided training prompt (observation.generation_prompt)
    # - do our own prompt-only encoding for generation (no teacher-forced suffix)
    dataset = create_dataset(
        args.data,
        args.policy,
        skip_norm_stats=False,
        skip_policy_transforms=True,
    )
    # Infer dims for policy config
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")
    sample0, _ = dataset[0]
    sample0_batch = sample0.unsqueeze(0)
    if sample0_batch.action_chunk is not None:
        args.policy.action_dim = int(sample0_batch.action_chunk.actions.shape[-1])
    if sample0_batch.observation.state is not None:
        args.policy.state_dim = int(sample0_batch.observation.state.shape[-1])

    with torch.device(device):
        model: "BasePolicy" = args.policy.instantiate()

    if (ckpt := args.checkpoint_path) is not None:
        missing, unexpected = load_model_from_checkpoint(
            model, ckpt, device, strict=False
        )
        if missing:
            print(f"Warning: Missing keys when loading checkpoint: {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys when loading checkpoint: {unexpected}")

    model.eval()
    if use_bf16:
        model.bfloat16()

    if int(args.batch_size) != 1:
        raise ValueError(
            f"batch_size={args.batch_size} is not supported yet; set batch_size=1 for manual decode."
        )

    # Use the underlying LeRobot dataset so we can access episode/frame indices + meta root.
    base = dataset.base_dataset
    if not hasattr(base, "dataset"):
        raise TypeError(
            "Expected LIBERO base dataset to expose a `.dataset` (LeRobotDataset)."
        )
    lerobot_ds: "LeRobotDataset" = base.dataset  # type: ignore[assignment]

    meta_root = Path(str(lerobot_ds.meta.root))
    gt_map = _load_gt_bboxes(meta_root)

    prompt_sep_text = _resolve_prompt_sep_text(args.policy)
    encode_kwargs = _resolve_paligemma_encode_kwargs(args.policy)
    model_for_generation = model.vlm_bridge.causal_model
    processor = model.vlm_bridge.processor
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise TypeError("Could not resolve tokenizer from PaliGemma processor.")

    # Build a list of dataset indices that have GT bboxes, up to `num_samples`.
    eligible_indices: List[int] = []
    for idx in tqdm(range(int(lerobot_ds.num_frames)), desc="Scanning GT frames"):
        frame = lerobot_ds[idx]
        ep_idx = int(frame["episode_index"].item()) if "episode_index" in frame else 0
        fr_idx = int(frame["frame_index"].item()) if "frame_index" in frame else idx
        if (ep_idx, fr_idx) in gt_map:
            eligible_indices.append(int(idx))
            if len(eligible_indices) >= int(args.num_samples):
                break

    if not eligible_indices:
        raise ValueError(
            "Found 0 frames with GT bboxes; check `meta/bboxes.jsonl` and dataset indices."
        )

    dl_workers = int(args.num_workers)
    dl_prefetch = int(args.prefetch_factor)
    dataset_dl = _BBoxEvalFrameDataset(
        transformed_ds=dataset,
        lerobot_ds=lerobot_ds,
        indices=eligible_indices,
        gt_map=gt_map,
    )
    dl_kwargs: Dict[str, Any] = {}
    if dl_workers > 0:
        dl_kwargs["prefetch_factor"] = dl_prefetch
        dl_kwargs["persistent_workers"] = bool(args.persistent_workers)
    loader = DataLoader(
        dataset_dl,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=dl_workers,
        pin_memory=bool(args.pin_memory) and device.type == "cuda",
        collate_fn=_collate_bbox_eval,
        **dl_kwargs,
    )

    ious: List[float] = []
    per_frame: List[Dict[str, Any]] = []
    per_episode: Dict[int, List[Dict[str, Any]]] = {}
    thresholds = tuple(float(t) for t in args.thresholds)
    acc_counts = {str(t): 0 for t in thresholds}

    num_frames = 0
    num_gt = 0
    num_pred_total = 0

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else torch.cpu.amp.autocast(enabled=False)
    )

    start = time.monotonic()
    with autocast_ctx:
        for batch in tqdm(loader, desc="Evaluating bbox IoU"):
            if not batch:
                continue
            if len(batch) != 1:
                raise RuntimeError(
                    f"Expected batch_size=1, got batch of size {len(batch)}."
                )
            it = batch[0]
            ep_idx = int(it["episode_index"])
            fr_idx = int(it["frame_index"])
            gt_boxes = cast(List[Tuple[float, float, float, float]], it["gt_boxes"])
            images = cast(List[Image.Image], it["images"])
            task = cast(str, it["task"])
            prompt = cast(str, it["prompt"])

            past_kv, cache_pos, attn_mask, next_tok = _prefill_paligemma(
                model=model_for_generation,
                processor=processor,
                images=images,
                task=task,
                prompt_sep_text=prompt_sep_text,
                prompt=prompt,
                device=device,
                encode_kwargs=encode_kwargs,
            )
            pred_text = _decode_paligemma(
                model=model_for_generation,
                tokenizer=tokenizer,
                past_key_values=past_kv,
                cache_position=cache_pos,
                attention_mask=attn_mask,
                next_token=next_tok,
                max_new_tokens=int(args.max_new_tokens),
            )

            pred_boxes = _parse_pred_bboxes(pred_text)
            num_pred_total += len(pred_boxes)

            frame_ious: List[float] = []
            for gt in gt_boxes:
                best = 0.0
                for pb in pred_boxes:
                    best = max(best, _xyxy_to_iou(gt, pb))
                frame_ious.append(best)
                ious.append(best)
            for t in thresholds:
                acc_counts[str(t)] += sum(1 for v in frame_ious if v >= t)

            num_frames += 1
            num_gt += len(gt_boxes)
            rec = {
                "dataset_index": int(it["dataset_index"]),
                "episode_index": ep_idx,
                "frame_index": fr_idx,
                "num_gt": len(gt_boxes),
                "num_pred": len(pred_boxes),
                "ious": frame_ious,
            }
            per_frame.append(rec)
            per_episode.setdefault(ep_idx, []).append(rec)

    elapsed = time.monotonic() - start

    mean_iou = float(np.mean(ious)) if ious else 0.0
    # IoU AUC: integral of acc(t) over t∈[0,1] equals mean IoU across GT boxes.
    iou_auc = mean_iou
    acc = {f"acc@{t}": (float(acc_counts[str(t)]) / float(max(1, num_gt))) for t in thresholds}

    report: Dict[str, Any] = {
        "checkpoint": str(args.checkpoint_path) if args.checkpoint_path else None,
        "data": getattr(args.data, "_target_", None),
        "meta_root": str(meta_root),
        "prompt_sep_text": str(prompt_sep_text),
        "encode_kwargs": dict(encode_kwargs),
        "num_frames_evaluated": int(num_frames),
        "num_gt_boxes": int(num_gt),
        "num_pred_boxes_total": int(num_pred_total),
        "mean_iou": mean_iou,
        "iou_auc": iou_auc,
        "thresholds": list(thresholds),
        **acc,
        "elapsed_s": float(elapsed),
        "per_frame": per_frame,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote bbox IoU report: {out_path}")

    if args.output_dir:
        out_dir = Path(str(args.output_dir))
    else:
        stem = out_path.with_suffix("").name
        out_dir = out_path.parent / f"{stem}_episodes"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep_idx, frames in per_episode.items():
        ep_ious: List[float] = [
            v for rec in frames for v in cast(List[float], rec["ious"])
        ]
        ep_num_gt = sum(int(rec["num_gt"]) for rec in frames)
        ep_num_pred_total = sum(int(rec["num_pred"]) for rec in frames)
        ep_mean_iou = float(np.mean(ep_ious)) if ep_ious else 0.0

        ep_acc_counts = {str(t): 0 for t in thresholds}
        for rec in frames:
            for t in thresholds:
                ep_acc_counts[str(t)] += sum(
                    1 for v in cast(List[float], rec["ious"]) if v >= t
                )
        ep_acc = {
            f"acc@{t}": (float(ep_acc_counts[str(t)]) / float(max(1, ep_num_gt)))
            for t in thresholds
        }

        ep_report: Dict[str, Any] = {
            "checkpoint": report["checkpoint"],
            "data": report["data"],
            "meta_root": report["meta_root"],
            "episode_index": int(ep_idx),
            "num_frames_evaluated": int(len(frames)),
            "num_gt_boxes": int(ep_num_gt),
            "num_pred_boxes_total": int(ep_num_pred_total),
            "mean_iou": ep_mean_iou,
            "iou_auc": ep_mean_iou,
            "thresholds": list(thresholds),
            **ep_acc,
            "elapsed_s": None,
            "per_frame": frames,
        }

        ep_path = out_dir / f"episode_{int(ep_idx)}.json"
        ep_path.write_text(json.dumps(ep_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
