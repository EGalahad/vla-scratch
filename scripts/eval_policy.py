#!/usr/bin/env python3


"""
Evaluate a policy on a dataset using Hydra configs (mirrors train_policy grammar).

- Expects data and policy groups (e.g., data=libero-spatial, policy=pi-qwen)
- Optionally loads a checkpoint and computes sample MSE or generation loss over a subset
"""

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Optional, cast, TYPE_CHECKING
import os
import art
from setproctitle import setproctitle

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf, MISSING

import torch
from torch.utils.data import DataLoader, Subset

from vla_scratch.datasets.config import DataConfig
from vla_scratch.helpers.data import create_dataset
from vla_scratch.policies.config import PolicyConfig
from vla_scratch.utils.checkpoint import (
    find_latest_checkpoint,
    load_model_from_checkpoint,
    merge_policy_cfg_from_checkpoint,
)
from vla_scratch.helpers.training import eval_sample_mse
from vla_scratch.helpers.training import eval_generation

if TYPE_CHECKING:
    from vla_scratch.transforms.data_types import DataSample
    from vla_scratch.policies.base import BasePolicy

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class EvalConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"policy": "pi-qwen"},
            {"data": "libero-spatial"},
        ]
    )
    # Hydra configs
    data: DataConfig = MISSING
    policy: PolicyConfig = MISSING

    # Eval controls
    eval_type: str = "sample_mse"  # "sample_mse" or "generation"
    batch_size: int = 32
    num_workers: int = 16
    num_samples: int = 512
    num_steps: int = 10
    # Runtime
    checkpoint_path: Optional[str] = None
    merge_policy_cfg: bool = False
    use_bf16: bool = True  # Enable bf16 autocast for inference


cs = ConfigStore.instance()
cs.store(name="eval", node=EvalConfig())


@hydra.main(config_name="eval", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    art.tprint("VLA-SCRATCH", font="big")
    setproctitle("vla-eval")
    if (checkpoint_path := cfg.get("checkpoint_path")) is not None:
        cfg.checkpoint_path = find_latest_checkpoint(checkpoint_path)
    if cfg.get("merge_policy_cfg", False):
        cfg = merge_policy_cfg_from_checkpoint(cfg, cfg.get("checkpoint_path"))
        OmegaConf.resolve(cfg)

    # Convert to typed objects after merge
    eval_cfg = cast(EvalConfig, OmegaConf.to_object(cfg))

    # Data + policy configs
    data_cfg: DataConfig = eval_cfg.data
    policy_cfg: PolicyConfig = eval_cfg.policy

    data_cfg.action_horizon = policy_cfg.action_horizon
    data_cfg.state_history = policy_cfg.state_history

    for i, spec in enumerate(list(data_cfg.input_transforms or [])):
        if isinstance(spec, dict) and "enable_aug" in spec:
            spec.update({"enable_aug": False})
            data_cfg.input_transforms[i] = spec

    # Create transformed dataset (includes normalization + policy transforms + ToTensorClass)
    dataset = create_dataset(data_cfg, policy_cfg)

    # Infer dims from one sample
    sample0: "DataSample" = dataset[0][0].unsqueeze(0)
    policy_cfg.state_dim = int(sample0.observation.state.shape[-1])
    policy_cfg.action_dim = int(sample0.action_chunk.actions.shape[-1])

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.device(device):
        model: "BasePolicy" = policy_cfg.instantiate()
    model.compute_loss(sample0.to(device))

    if (ckpt := eval_cfg.checkpoint_path) is not None:
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
    model.eval()
    if eval_cfg.use_bf16:
        model.bfloat16()

    # Dataloader — subset for speed
    total = len(dataset)
    num = min(int(eval_cfg.num_samples), total)
    indices = list(range(num))
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=int(eval_cfg.batch_size),
        shuffle=False,
        num_workers=int(eval_cfg.num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=dataset.collate_fn,
    )

    # Evaluate MSE
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if eval_cfg.use_bf16
        else nullcontext()
    )
    with autocast_ctx:
        if eval_cfg.eval_type == "sample_mse":
            metrics = eval_sample_mse(
                model,
                loader,
                device,
                num_sample_steps=int(eval_cfg.num_steps),
                local_rank=0,
            )
            mse = float(metrics["sample_mse"])
            print(
                "Eval sample MSE over "
                f"{num} samples (batch={eval_cfg.batch_size}, steps={eval_cfg.num_steps}): "
                f"{mse:.6f}"
            )
        elif eval_cfg.eval_type == "generation":
            metrics = eval_generation(
                model=model,
                dataloader=loader,
                device=device,
                local_rank=0,
            )
            metrics_flat = metrics.flatten_keys(separator="/").to_dict(
                convert_tensors=True
            )
            metrics_str = ", ".join(
                f"{k}={float(v):.6f}" for k, v in sorted(metrics_flat.items())
            )
            print(
                f"Eval generation loss over {num} samples (batch={eval_cfg.batch_size}): "
                f"{metrics_str}"
            )
        else:
            raise ValueError(
                f"Unsupported eval_type='{eval_cfg.eval_type}', expected 'sample_mse' or 'generation'."
            )


if __name__ == "__main__":
    main()
