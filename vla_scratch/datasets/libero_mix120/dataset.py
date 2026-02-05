from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, List, Tuple

import bisect
import torch

from vla_scratch.datasets.libero.dataset import LIBERODataset

if TYPE_CHECKING:
    from vla_scratch.datasets.libero_mix120.config import LiberoMix120Config


class LiberoMix120Dataset(torch.utils.data.Dataset):
    """
    A mixed dataset wrapper for LIBERO suites.

    This behaves like a concatenation of the component datasets, so sampling
    indices uniformly from `range(len(dataset))` yields a distribution where
    each component contributes proportionally to its own length.
    """

    def __init__(self, config: "LiberoMix120Config"):
        self._datasets: List[LIBERODataset] = []

        for base_cfg in (
            config.libero_spatial,
            config.libero_goal,
            config.libero_object,
            config.libero_90,
        ):
            ds_cfg = replace(
                base_cfg,
                action_horizon=config.action_horizon,
                state_history=config.state_history,
                video_backend=config.video_backend,
                bbox_only=config.bbox_only,
                remove_bbox=config.remove_bbox,
            )
            self._datasets.append(LIBERODataset(ds_cfg))

        if not self._datasets:
            raise ValueError(
                "LiberoMix120Dataset requires at least one dataset."
            )

        lengths = [len(ds) for ds in self._datasets]
        if any(l <= 0 for l in lengths):
            raise ValueError(
                f"LiberoMix120Dataset has an empty component: lengths={lengths}"
            )

        self._offsets: List[int] = []
        total = 0
        for l in lengths:
            total += int(l)
            self._offsets.append(total)
        self._total_len = total

    def __len__(self) -> int:
        return self._total_len

    def _locate(self, idx: int) -> Tuple[int, int]:
        if idx < 0:
            idx = self._total_len + idx
        if idx < 0 or idx >= self._total_len:
            raise IndexError(idx)
        dataset_idx = bisect.bisect_right(self._offsets, idx)
        prev = 0 if dataset_idx == 0 else self._offsets[dataset_idx - 1]
        local_idx = idx - prev
        return dataset_idx, local_idx

    def __getitem__(self, idx: int):
        dataset_idx, local_idx = self._locate(int(idx))
        return self._datasets[dataset_idx][local_idx]
