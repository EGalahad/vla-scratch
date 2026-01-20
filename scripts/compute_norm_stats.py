#!/usr/bin/env python3


"""
Compute and save normalization statistics for any configured dataset/policy.

Hydra usage mirrors train_policy: pass data=... and policy=... groups.

Examples:
  uv run python scripts/compute_norm_stats.py data=libero-spatial policy=pi-qwen \
      data.action_horizon=30 data.state_history=1 \
      num_samples=4096 batch_size=64 num_workers=8
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast, TYPE_CHECKING
import shutil
import tempfile
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

from vla_scratch.transforms.data_keys import (
    PROCESSED_ACTION_KEY,
    PROCESSED_STATE_KEY,
)
from vla_scratch.transforms.normalization import (
    save_norm_stats,
    NormStats,
    FieldNormStats,
)
from vla_scratch.utils.config import resolve_config_placeholders
from vla_scratch.utils.paths import REPO_ROOT

if TYPE_CHECKING:
    from vla_scratch.transforms.data_types import DataSample


@dataclass
class NormStatsConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"policy": "pi-qwen"},
            {"data": "libero-spatial"},
        ]
    )
    data: DataConfig = MISSING
    policy: PolicyConfig = MISSING

    # Compute controls
    num_samples: int = 16384
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
    output_dir: Path,
) -> tuple["DataSample", NormStats, Path]:
    dataset = create_dataset(
        data_config,
        policy_config,
        skip_norm_stats=True,
        skip_policy_transforms=True,
    )
    dataset_size = len(dataset)

    if data_config.norm_stats_path is None:
        raise ValueError(
            "DataConfig.norm_stats_path must be set to save stats."
        )

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
    stacked: "DataSample" = torch.cat(batches)
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

    stats_path = save_norm_stats(output_dir, data_config, policy_config, stats)
    print(f"Saved normalization stats to: {stats_path}")
    return stacked, stats, stats_path


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

    temp_dir = Path(tempfile.mkdtemp(prefix="norm_stats-", dir="/tmp"))
    OmegaConf.save(OmegaConf.structured(run_cfg), temp_dir / "cfg.yaml")

    _, _, _ = compute_and_save_norm_stats(
        data_cfg,
        policy_cfg,
        num_samples=int(cfg.num_samples),
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        output_dir=temp_dir,
    )

    resolved_path = resolve_config_placeholders(
        data_cfg.norm_stats_path, data_cfg=data_cfg, policy_cfg=policy_cfg
    )
    if resolved_path is None:
        raise ValueError(
            "DataConfig.norm_stats_path must be set to save stats."
        )

    if str(resolved_path).startswith("hf:"):
        from huggingface_hub import HfApi, get_token

        raw = str(resolved_path)[len("hf:") :]
        parts = raw.split("/", 2)
        if len(parts) >= 2:
            repo_id = "/".join(parts[:2])
            subpath = parts[2] if len(parts) == 3 else ""
        else:
            repo_id = raw
            subpath = ""

        revision = None
        if "@" in repo_id:
            repo_id, revision = repo_id.split("@", 1)

        api = HfApi(token=get_token())
        last_exc: Exception | None = None
        for repo_type in ("dataset", "model"):
            try:
                api.create_repo(
                    repo_id=repo_id, repo_type=repo_type, exist_ok=True
                )
                api.upload_folder(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    folder_path=str(temp_dir),
                    path_in_repo=subpath or "",
                    revision=revision,
                )
                print(f"Uploaded normalization stats to hf:{repo_id}")
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        assert last_exc is not None
        raise last_exc

    target_dir = Path(str(resolved_path)).expanduser()
    if not target_dir.is_absolute():
        target_dir = REPO_ROOT / target_dir
    target_dir = target_dir.resolve()
    if target_dir.exists():
        raise FileExistsError(
            f"Target norm stats path already exists: {target_dir}"
        )
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(temp_dir), str(target_dir))
    print(f"Moved normalization stats to: {target_dir}")


if __name__ == "__main__":
    main()
