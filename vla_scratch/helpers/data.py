from __future__ import annotations

from typing import Any, List, Sequence, TYPE_CHECKING
from omegaconf import DictConfig

from vla_scratch.transforms.base import TransformedDataset
from vla_scratch.transforms.common import ToDataSample
from vla_scratch.transforms.normalization import DeNormalize, Normalize, load_norm_stats
from vla_scratch.utils.config import locate_class

if TYPE_CHECKING:
    from vla_scratch.datasets.config import DataConfig
    from vla_scratch.policies.config import PolicyConfig
    from vla_scratch.transforms.base import TransformFn


def instantiate_transform(spec: Any) -> Any:
    """Instantiate a transform object from a spec.

    Accepts either an existing object with a `.compute()` method, or a dict with
    a `_target_` key and constructor kwargs. Returns an object exposing
    `.compute(sample) -> Any`.
    """
    # Already a transform-like object
    if hasattr(spec, "compute") and callable(getattr(spec, "compute")):
        return spec

    # Config mapping
    if isinstance(spec, dict) or isinstance(spec, DictConfig):
        target = spec.get("_target_")
        if target is None:
            raise ValueError("Transform configuration must define '_target_'.")
        kwargs = {k: v for k, v in spec.items() if k != "_target_"}
        cls = locate_class(target)
        obj = cls(**kwargs)
        if hasattr(obj, "compute") and callable(getattr(obj, "compute")):
            return obj
        raise TypeError(f"Instance of '{target}' does not expose a 'compute' method.")

    raise TypeError(f"Unsupported transform specification: {spec!r}")


def make_transforms(specs: Sequence[Any]) -> List["TransformFn"]:
    """Instantiate transform specs into concrete transform objects."""
    return [instantiate_transform(spec) for spec in specs]


def build_input_transforms(
    data_cfg: "DataConfig",
    policy_cfg: "PolicyConfig",
    *,
    add_noise: bool = False,
) -> Sequence["TransformFn"]:
    dataset_tfs = make_transforms(data_cfg.input_transforms)

    norm_tf: List["TransformFn"] = []
    if data_cfg.norm_stats_path is not None:
        stats = load_norm_stats(data_cfg, policy_cfg)
        norm_tf = [
            Normalize(
                norm_stats=stats,
                strict=False,
                noise_cfg=data_cfg.noise_cfg,
                enable_aug=add_noise,
            )
        ]

    policy_tfs = make_transforms(policy_cfg.transforms)
    return dataset_tfs + norm_tf + [ToDataSample()] + policy_tfs


def build_output_transforms(
    data_cfg: "DataConfig",
    policy_cfg: "PolicyConfig",
) -> Sequence["TransformFn"]:
    denorm_tf: List["TransformFn"] = []
    if data_cfg.norm_stats_path is not None:
        stats = load_norm_stats(data_cfg, policy_cfg)
        denorm_tf = [DeNormalize(norm_stats=stats, strict=False)]

    inv_tfs = make_transforms(data_cfg.output_inv_transforms)

    from vla_scratch.transforms.common import ToNumpy

    return denorm_tf + inv_tfs + [ToNumpy()]


def create_dataset(
    data_cfg: "DataConfig",
    policy_cfg: "PolicyConfig",
    *,
    skip_norm_stats: bool = False,
    skip_policy_transforms: bool = False,
    add_noise: bool = False,
) -> TransformedDataset:
    """Create a dataset pipeline applying configured transforms."""
    # Eval-specific fields on EvalDatasetCfg are ignored here.
    input_tfs = data_cfg.input_transforms
    output_tfs = data_cfg.output_transforms
    dataset_tfs = make_transforms(list(input_tfs) + list(output_tfs))

    norm_tf: List["TransformFn"] = []
    if (not skip_norm_stats) and data_cfg.norm_stats_path is not None:
        stats = load_norm_stats(data_cfg, policy_cfg)
        norm_tf = [
            Normalize(
                norm_stats=stats,
                noise_cfg=data_cfg.noise_cfg,
                enable_aug=add_noise,
            )
        ]

    policy_tfs = (
        make_transforms(policy_cfg.transforms) if not skip_policy_transforms else []
    )

    pipeline = dataset_tfs + norm_tf + [ToDataSample()] + policy_tfs

    base_dataset = data_cfg.instantiate()
    return TransformedDataset(base_dataset, pipeline)

