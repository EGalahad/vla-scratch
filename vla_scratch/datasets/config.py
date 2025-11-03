from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Dict, Optional
import importlib

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

from vla_scratch.datasets.common import PROCESSED_ACTION_KEY, PROCESSED_STATE_KEY
from vla_scratch.datasets.transforms import TransformedDataset, TransformFn
from vla_scratch.datasets.transforms import (
    load_norm_stats,
    save_norm_stats,
    NormStats,
    FieldNormStats,
)
from vla_scratch.datasets.transforms import Normalize, ToTensorClass
from vla_scratch.policies.config import PolicyConfig


@dataclass
class DataConfig:
    _target_: str
    action_horizon: Optional[int] = None
    state_history: Optional[int] = None
    transforms: List[Any] = field(default_factory=list)
    norm_stats_path: Path | None = None


def _locate_class(target: str) -> type:
    module_name, _, attr_name = target.rpartition(".")
    if not module_name:
        raise ValueError(f"Target '{target}' must be a fully-qualified path.")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ImportError(f"Cannot import '{attr_name}' from '{module_name}'.") from exc


def _instantiate_transform(spec: Any) -> TransformFn:
    if isinstance(spec, TransformFn):
        return spec
    if isinstance(spec, Dict):
        target = spec.get("_target_")
        if target is None:
            raise ValueError("Transform configuration must define '_target_'.")
        kwargs = {k: v for k, v in spec.items() if k != "_target_"}
        cls = _locate_class(target)
        obj = cls(**kwargs)
        if isinstance(obj, TransformFn):
            return obj
        if callable(getattr(obj, "compute", None)):
            return obj
        raise TypeError(f"Instance of '{target}' is not a TransformFn.")
    raise TypeError(f"Unsupported transform specification: {spec!r}")


def create_dataset(
    data_config: DataConfig,
    policy_config: PolicyConfig,
    *,
    skip_norm_stats: bool = False,
) -> TransformedDataset:
    dataset_cls = _locate_class(data_config._target_)
    base_dataset = dataset_cls(data_config)

    transforms: List[TransformFn] = []

    dataset_transforms = [
        _instantiate_transform(spec) for spec in data_config.transforms
    ]
    transforms.extend(dataset_transforms)

    if not skip_norm_stats and data_config.norm_stats_path is not None:
        import vla_scratch

        repo_root = Path(vla_scratch.__file__).parents[1]
        stats_path = repo_root / data_config.norm_stats_path
        stats = load_norm_stats(stats_path)
        transforms.append(Normalize(norm_stats=stats))

    policy_transforms = [
        _instantiate_transform(spec) for spec in policy_config.transforms
    ]
    transforms.extend(policy_transforms)
    transforms.append(ToTensorClass())

    return TransformedDataset(base_dataset, transforms)


def compute_and_save_norm_stats(
    data_config: DataConfig,
    policy_config: PolicyConfig,
    num_samples: int = 4096,
    batch_size: int = 64,
    num_workers: int = 16,
    pin_memory: bool = False,
) -> NormStats:
    dataset = create_dataset(data_config, policy_config, skip_norm_stats=True)
    dataset_size = len(dataset)

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

    from tqdm import tqdm

    for batch, _ in tqdm(dataloader, desc="Computing norm stats"):
        batches.append(batch)

    stacked = torch.cat(batches)
    state_tensor = stacked.observation.state
    action_tensor = stacked.action_chunk.actions

    def _compute_norm_stats_for_tensor(tensor: torch.Tensor) -> FieldNormStats:
        mean = tensor.mean(dim=0)
        std = tensor.std(dim=0, unbiased=False)
        q01 = torch.quantile(tensor, 0.01, dim=0)
        q99 = torch.quantile(tensor, 0.99, dim=0)
        return FieldNormStats(mean_=mean, std_=std, q01=q01, q99=q99)

    stats = {
        PROCESSED_STATE_KEY: _compute_norm_stats_for_tensor(state_tensor),
        PROCESSED_ACTION_KEY: _compute_norm_stats_for_tensor(action_tensor),
    }

    if data_config.norm_stats_path is None:
        raise ValueError("DataConfig.norm_stats_path must be set to save stats.")

    import vla_scratch

    repo_root = Path(vla_scratch.__file__).parents[1]
    stats_path = repo_root / data_config.norm_stats_path
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    save_norm_stats(stats_path, stats)
    print(f"Saved normalization stats to: {stats_path}")
    return stats


if __name__ == "__main__":
    # Example usage
    from tensordict import TensorDict
    from vla_scratch.datasets.libero.config import LiberoIPECConfig
    from vla_scratch.policies.pi.config import PiConfig

    data_config = LiberoIPECConfig()
    policy_config = PiConfig()
    data_config.action_horizon = policy_config.action_horizon
    data_config.state_history = policy_config.state_history
    batches, stats = compute_and_save_norm_stats(data_config, policy_config)

    # visualize the state and action distributions as histograms
    # and the normalization bounds as vertical lines
    def visualize_distribution(batches: TensorDict, stats: NormStats):
        import matplotlib.pyplot as plt

        state_tensor = batches.observation.state  # shape: [num_samples, 10, 11]
        action_tensor = batches.action_chunk.actions  # shape: [num_samples, 30, 10]

        import math

        def visualize_distribution(
            tensor: torch.Tensor,
            stats_obj: FieldNormStats,
            title_prefix: str,
            stride: int,
        ):
            feature_dim = tensor.shape[-1]
            flattened = tensor.detach().cpu().reshape(-1, feature_dim)
            dims = list(range(0, feature_dim, max(1, stride)))
            if not dims:
                return None

            mean = stats_obj.mean_.detach().cpu()
            q01 = stats_obj.q01.detach().cpu()
            q99 = stats_obj.q99.detach().cpu()

            cols = min(4, len(dims))
            rows = math.ceil(len(dims) / cols)
            fig, axes = plt.subplots(
                rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False
            )
            axes = axes.ravel()

            for idx, dim in enumerate(dims):
                ax = axes[idx]
                values = flattened[:, dim].numpy()
                ax.hist(
                    values, bins=50, color="steelblue", alpha=0.8, edgecolor="white"
                )
                ax.axvline(
                    float(q01[dim]), color="tab:red", linestyle="--", label="q01"
                )
                ax.axvline(
                    float(q99[dim]), color="tab:green", linestyle="--", label="q99"
                )
                ax.axvline(float(mean[dim]), color="black", linestyle="-", label="mean")
                ax.set_title(f"{title_prefix} dim {dim}")
                ax.set_xlabel("value")
                ax.set_ylabel("count")

            for ax in axes[len(dims) :]:
                ax.axis("off")

            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper right")

            fig.suptitle(f"{title_prefix} histogram (stride={stride})")
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            return fig

        print("Visualizing state distributions...")
        state_stride = 2
        state_tensor = state_tensor[:, ::state_stride, :].reshape(
            state_tensor.shape[0], -1
        )
        state_stats: FieldNormStats = stats[PROCESSED_STATE_KEY][
            ::state_stride
        ].reshape(-1)
        state_stride = 1
        visualize_distribution(state_tensor, state_stats, "State", state_stride)
        plt.savefig("state_distribution.png")

        print("Visualizing action distributions...")
        action_stride = 6
        action_tensor = action_tensor[:, ::action_stride, :].reshape(
            action_tensor.shape[0], -1
        )
        action_stats: FieldNormStats = stats[PROCESSED_ACTION_KEY][
            ::action_stride
        ].reshape(-1)
        action_stride = 1
        visualize_distribution(action_tensor, action_stats, "Action", action_stride)
        plt.savefig("action_distribution.png")
