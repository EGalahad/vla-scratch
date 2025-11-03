import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, SupportsIndex, Tuple

import jaxtyping as at
import numpy as np
import torch
from tensordict import TensorClass, TensorDict
from torch.utils.data import Dataset

from vla_scratch.datasets.common import (
    PROCESSED_ACTION_KEY,
    PROCESSED_IMAGE_KEY,
    PROCESSED_IMAGE_MASK_KEY,
    PROCESSED_STATE_KEY,
    TOKENIZED_KEY,
    TOKENIZED_MASK_KEY,
)
from vla_scratch.datasets.data_types import ActionChunk, DataSample, Observation


class TransformFn:
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def compute(self, sample: Dict) -> Dict:
        raise NotImplementedError


class TransformedDataset(Dataset):
    @staticmethod
    def collate_fn(batch):
        return tuple(torch.stack(items) for items in zip(*batch))

    def __init__(self, dataset: Dataset, transforms: Sequence[TransformFn]):
        self.base_dataset = dataset
        self.transforms = list(transforms)
        self._log_names = [tr.__repr__() for tr in self.transforms]

    def __getitem__(self, index: SupportsIndex) -> Tuple[Any, TensorDict]:
        perf: Dict[str, float] = {}

        start = time.perf_counter()
        sample = self.base_dataset[index]
        perf["base_dataset_load"] = time.perf_counter() - start

        for transform, name in zip(self.transforms, self._log_names):
            start = time.perf_counter()
            sample = transform.compute(sample)
            perf[name] = time.perf_counter() - start

        return sample, TensorDict(perf)

    def __len__(self) -> int:
        return len(self.base_dataset)


class FieldNormStats(TensorClass):
    mean_: at.Float[torch.Tensor, "*feature_dim"]
    std_: at.Float[torch.Tensor, "*feature_dim"]
    q01: at.Float[torch.Tensor, "*feature_dim"]
    q99: at.Float[torch.Tensor, "*feature_dim"]


NormStats = Dict[str, FieldNormStats]


class Normalize(TransformFn):
    def __init__(
        self,
        norm_stats: Dict[str, FieldNormStats],
        *,
        use_quantiles: bool = True,
        strict: bool = False,
    ) -> None:
        self.norm_stats = norm_stats
        self.use_quantiles = use_quantiles
        self.strict = strict

    def __repr__(self) -> str:
        keys = ", ".join(sorted(self.norm_stats.keys()))
        return (
            f"{self.__class__.__name__}(keys=[{keys}], "
            f"use_quantiles={self.use_quantiles}, strict={self.strict})"
        )

    def compute(self, sample: Dict) -> Dict:
        normalizer = self._normalize_quantile if self.use_quantiles else self._normalize
        for key, stats in self.norm_stats.items():
            if key not in sample:
                if self.strict:
                    raise KeyError(
                        f"Normalization stats provided for '{key}' "
                        "but the key is missing in the sample."
                    )
                continue
            sample[key] = normalizer(sample[key], stats)
        return sample

    @staticmethod
    def _normalize(tensor: torch.Tensor, stats: FieldNormStats) -> torch.Tensor:
        mean = stats.mean_
        std = stats.std_
        return ((tensor - mean) / (std + 1e-6)).clamp(-1.5, 1.5)

    @staticmethod
    def _normalize_quantile(
        tensor: torch.Tensor, stats: FieldNormStats
    ) -> torch.Tensor:
        q01 = stats.q01
        q99 = stats.q99
        return ((tensor - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0).clamp(-1.5, 1.5)


class ToTensorClass(TransformFn):
    """Convert the standardized dict into structured TensorClasses."""

    def compute(self, sample: Dict) -> DataSample:
        observation = Observation(
            images=sample[PROCESSED_IMAGE_KEY],
            image_masks=sample.get(PROCESSED_IMAGE_MASK_KEY),
            state=sample[PROCESSED_STATE_KEY],
            tokenized_prompt=sample[TOKENIZED_KEY],
            tokenized_prompt_mask=sample[TOKENIZED_MASK_KEY],
        )
        action = ActionChunk(actions=sample[PROCESSED_ACTION_KEY])
        return DataSample(observation=observation, action_chunk=action)


def load_norm_stats(path: Path) -> Dict[str, FieldNormStats]:
    path = Path(path)
    loaded = np.load(path, allow_pickle=True)
    try:
        if hasattr(loaded, "files"):
            raw = {key: loaded[key] for key in loaded.files}
        else:
            raw = loaded.item()
    finally:
        if hasattr(loaded, "close"):
            loaded.close()

    result: Dict[str, FieldNormStats] = {}
    for key, components in raw.items():
        if isinstance(components, np.ndarray) and components.dtype == object:
            components = components.item()
        if not isinstance(components, Mapping):
            raise TypeError(
                f"Expected normalization entry for '{key}' to be a mapping, "
                f"got {type(components).__name__}."
            )
        result[key] = FieldNormStats(
            mean_=torch.as_tensor(components["mean_"], dtype=torch.float32),
            std_=torch.as_tensor(components["std_"], dtype=torch.float32),
            q01=torch.as_tensor(components["q01"], dtype=torch.float32),
            q99=torch.as_tensor(components["q99"], dtype=torch.float32),
        )
    return result


def save_norm_stats(path: Path, stats: NormStats) -> None:
    path = Path(path)
    flat: Dict[str, Dict[str, np.ndarray]] = {}
    for key, value in stats.items():
        flat[key] = {
            "mean_": value.mean_.detach().cpu().numpy(),
            "std_": value.std_.detach().cpu().numpy(),
            "q01": value.q01.detach().cpu().numpy(),
            "q99": value.q99.detach().cpu().numpy(),
        }
    np.savez_compressed(path, **flat)
