#!/usr/bin/env python3


"""
Compute and save normalization statistics for any configured dataset/policy.

Hydra usage mirrors train_policy: pass data=... and policy=... groups.

Examples:
  uv run python scripts/compute_norm_stats.py data=moz policy=pi \
      data.action_horizon=30 data.state_history=1 \
      num_samples=4096 batch_size=64 num_workers=8

  uv run python scripts/compute_norm_stats.py data=libero-ipec policy=pi \
      policy.action_horizon=30 policy.state_history=10
"""

from dataclasses import dataclass, field
from typing import Any, cast
from tqdm import tqdm


import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf, MISSING

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

from vla_scratch.datasets.config import DataConfig
from vla_scratch.helpers.data import create_dataset
from vla_scratch.policies.config import PolicyConfig

from vla_scratch.transforms.data_keys import PROCESSED_ACTION_KEY, PROCESSED_STATE_KEY
from vla_scratch.transforms.data_types import DataSample
from vla_scratch.transforms.normalization import (
    save_norm_stats,
    NormStats,
    FieldNormStats,
)

@dataclass
class NormStatsConfig:
    defaults: list[Any] = field(
        default_factory=lambda: ["_self_", {"policy": "pi"}, {"data": "moz"}]
    )
    data: DataConfig = MISSING
    policy: PolicyConfig = MISSING

    # Compute controls
    num_samples: int = 4096
    batch_size: int = 32
    num_workers: int = 8
    pin_memory: bool = False


cs = ConfigStore.instance()
cs.store(name="norm_stats", node=NormStatsConfig())


def compute_and_save_norm_stats(
    data_config: DataConfig,
    policy_config: PolicyConfig,
    num_samples: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> NormStats:
    dataset = create_dataset(data_config, policy_config, skip_norm_stats=True, skip_policy_transforms=True)
    dataset_size = len(dataset)

    if data_config.norm_stats_path is None:
        raise ValueError("DataConfig.norm_stats_path must be set to save stats.")

    num_samples = min(num_samples, dataset_size)
    batch_size = min(batch_size, num_samples)

    rng = np.random.default_rng()
    indices = rng.choice(dataset_size, size=num_samples, replace=False).tolist()
    subset = Subset(dataset, indices)

    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )

    batches = []
    for batch, _ in tqdm(dataloader, desc="Computing norm stats"):
        batches.append(batch)
    stacked: DataSample = torch.cat(batches)
    state_tensor = stacked.observation.state
    action_tensor = stacked.action_chunk.actions

    def _compute_norm_stats_for_tensor(tensor: torch.Tensor) -> FieldNormStats:
        mean = tensor.mean(dim=0)
        std = tensor.std(dim=0, unbiased=False)
        q01 = torch.quantile(tensor, 0.01, dim=0)
        q99 = torch.quantile(tensor, 0.99, dim=0)
        return FieldNormStats(
            mean_=mean, std_=std, q01=q01, q99=q99, batch_size=tensor.shape[1:]
        )

    stats = {
        PROCESSED_STATE_KEY: _compute_norm_stats_for_tensor(state_tensor),
        PROCESSED_ACTION_KEY: _compute_norm_stats_for_tensor(action_tensor),
    }

    stats_path = save_norm_stats(data_config, policy_config, stats)
    print(f"Saved normalization stats to: {stats_path}")
    return stacked, stats


@hydra.main(config_name="norm_stats", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    run_cfg = cast(NormStatsConfig, OmegaConf.to_object(cfg))
    data_cfg: DataConfig = run_cfg.data
    policy_cfg: PolicyConfig = run_cfg.policy

    # Keep temporal params aligned if one is overridden
    data_cfg.action_horizon = policy_cfg.action_horizon
    data_cfg.state_history = policy_cfg.state_history

    print(
        f"Computing norm stats for data={data_cfg._target_} policy={policy_cfg._target_} "
        f"(horizon={data_cfg.action_horizon}, history={data_cfg.state_history})"
    )

    _, _ = compute_and_save_norm_stats(
        data_cfg,
        policy_cfg,
        num_samples=int(cfg.num_samples),
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
    )


if __name__ == "__main__":
    main()
