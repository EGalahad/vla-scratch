import torch
import numpy as np
from typing import Dict, Mapping, TYPE_CHECKING, List, Tuple, Optional
from tensordict import TensorClass

from vla_scratch.utils.math import scale_transform, unscale_transform
from vla_scratch.utils.config import resolve_config_placeholders
from vla_scratch.utils.paths import REPO_ROOT

if TYPE_CHECKING:
    from vla_scratch.datasets.config import DataConfig
    from vla_scratch.policies.config import PolicyConfig


class FieldNormStats(TensorClass):
    mean_: torch.Tensor
    std_: torch.Tensor
    q01: torch.Tensor
    q99: torch.Tensor

NormStats = Dict[str, FieldNormStats]


def load_norm_stats(data_cfg: "DataConfig", policy_cfg: "PolicyConfig") -> NormStats:
    stats_path_str = resolve_config_placeholders(
        data_cfg.norm_stats_path, data_cfg=data_cfg, policy_cfg=policy_cfg
    )
    stats_path = REPO_ROOT / str(stats_path_str)
    loaded = np.load(stats_path, allow_pickle=True)
    try:
        if hasattr(loaded, "files"):
            raw = {key: loaded[key] for key in loaded.files}
        else:
            raw = loaded.item()
    finally:
        if hasattr(loaded, "close"):
            loaded.close()

    stats: NormStats = {}
    for key, components in raw.items():
        if isinstance(components, np.ndarray) and components.dtype == object:
            components = components.item()
        if not isinstance(components, Mapping):
            raise TypeError(f"Normalization entry '{key}' must be a mapping")
        stats[key] = FieldNormStats(
            mean_=components["mean_"],
            std_=components["std_"],
            q01=components["q01"],
            q99=components["q99"],
        )
    return stats


def save_norm_stats(data_config: "DataConfig", policy_config: "PolicyConfig", stats: NormStats) -> None:
    stats_path_str = resolve_config_placeholders(
        data_config.norm_stats_path, data_cfg=data_config, policy_cfg=policy_config
    )
    stats_path = REPO_ROOT / str(stats_path_str)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    flat = {
        key: {
            "mean_": value.mean_.detach().cpu().numpy(),
            "std_": value.std_.detach().cpu().numpy(),
            "q01": value.q01.detach().cpu().numpy(),
            "q99": value.q99.detach().cpu().numpy(),
        }
        for key, value in stats.items()
    }
    np.savez_compressed(stats_path, **flat)
    return stats_path


class Normalize(torch.nn.Module):
    def __init__(
        self,
        norm_stats: NormStats,
        *,
        use_quantiles: bool = True,
        strict: bool = False,
        noise_cfg: Optional[Mapping[str, Mapping[str, Dict[str, float]]]] = None,
        enable_aug: bool = False,
    ) -> None:
        super().__init__()
        self.norm_stats = norm_stats
        self.use_quantiles = use_quantiles
        self.strict = strict
        self._fn = self._normalize_quantile if use_quantiles else self._normalize
        self._noise_cfg = self._prepare_noise_cfg(noise_cfg)
        self.enable_aug = enable_aug
        if self._noise_cfg:
            missing_norm_keys = [k for k in self._noise_cfg if k not in self.norm_stats]
            if missing_norm_keys:
                raise KeyError(
                    f"Noise requested for keys without normalization stats: {missing_norm_keys}"
                )

    def compute(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for key, stats in self.norm_stats.items():
            if key not in sample:
                if self.strict:
                    raise KeyError(f"Missing key '{key}' for normalization")
                continue
            normalized = self._fn(sample[key], stats)
            if self.enable_aug:
                sample[key] = self._apply_noise(key, normalized)
            else:
                sample[key] = normalized
        return sample

    @staticmethod
    def _normalize(tensor: torch.Tensor, stats: FieldNormStats) -> torch.Tensor:
        return ((tensor - stats.mean_) / (stats.std_ + 1e-6)).clamp(-1.5, 1.5)

    @staticmethod
    def _normalize_quantile(
        tensor: torch.Tensor, stats: FieldNormStats
    ) -> torch.Tensor:
        return scale_transform(tensor, stats.q01, stats.q99).clamp(-1.5, 1.5)

    @staticmethod
    def _parse_range(range_key: str) -> Tuple[int, int]:
        try:
            start_str, end_str = range_key.split("-")
            start, end = int(start_str), int(end_str)
        except Exception as exc:  # noqa: PERF203
            raise ValueError(
                f"Noise range '{range_key}' must be formatted as 'start-end'"
            ) from exc
        if end <= start:
            raise ValueError(f"Noise range '{range_key}' must satisfy end > start")
        return start, end

    @staticmethod
    def _prepare_noise_cfg(
        noise_cfg: Optional[Mapping[str, Mapping[str, Dict[str, float]]]]
    ) -> Dict[str, List[Tuple[slice, Dict[str, float]]]]:
        if noise_cfg is None:
            return {}

        prepared: Dict[str, List[Tuple[slice, Dict[str, float]]]] = {}
        for target_key, ranges in noise_cfg.items():
            if not isinstance(ranges, Mapping):
                raise TypeError("Noise config entries must be mappings of ranges to cfgs")

            parsed_ranges: List[Tuple[slice, Dict[str, float]]] = []
            for range_key, cfg in ranges.items():
                start, end = Normalize._parse_range(range_key)
                parsed_ranges.append((slice(start, end), cfg))

            if parsed_ranges:
                prepared[target_key] = parsed_ranges
        return prepared

    def _apply_noise(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        noise_specs = self._noise_cfg.get(key)
        if not noise_specs:
            return tensor

        noisy = tensor.clone()
        for span, cfg in noise_specs:
            noise_type = cfg.get("type")
            if noise_type == "gaussian":
                std = cfg.get("std")
                noise = torch.randn_like(noisy[..., span]).clamp_(-3, 3) * float(std)
                noisy[..., span] = noisy[..., span] + noise
            else:
                raise ValueError(f"Unsupported noise type '{noise_type}' for key '{key}'")
        return noisy

class DeNormalize(torch.nn.Module):
    def __init__(
        self,
        norm_stats: NormStats,
        *,
        use_quantiles: bool = True,
        strict: bool = False,
    ) -> None:
        super().__init__()
        self.norm_stats = norm_stats
        self.use_quantiles = use_quantiles
        self.strict = strict
        self._fn = self._denormalize_quantile if use_quantiles else self._denormalize

    def compute(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for key, stats in self.norm_stats.items():
            if key not in sample:
                if self.strict:
                    raise KeyError(f"Missing key '{key}' for denormalization")
                continue
            sample[key] = self._fn(sample[key], stats)
        return sample

    @staticmethod
    def _denormalize(tensor: torch.Tensor, stats: FieldNormStats) -> torch.Tensor:
        return tensor.clamp(-1.5, 1.5) * (stats.std_ + 1e-6) + stats.mean_

    @staticmethod
    def _denormalize_quantile(
        tensor: torch.Tensor, stats: FieldNormStats
    ) -> torch.Tensor:
        return unscale_transform(tensor.clamp(-1.5, 1.5), stats.q01, stats.q99)
