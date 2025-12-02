from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from vla_scratch.policies.utils.data_types import (
        HiddenState,
        PrefixPadMask,
        KVCache,
    )
    from vla_scratch.transforms.data_types import Observation


class BasePolicy(nn.Module, ABC):
    """Minimal policy interface required by training/eval/serving scripts."""

    @abstractmethod
    def encode_prefix(
        self,
        observation: "Observation",
    ) -> Tuple["HiddenState", "PrefixPadMask", List["KVCache"]]:
        """Encode the observation prefix and return KV cache artifacts."""

    @abstractmethod
    def predict_suffix(
        self,
        state: torch.Tensor,
        prefix_pad_masks: "PrefixPadMask",
        prefix_key_values: List["KVCache"],
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict action flow / displacement loss for the diffusion objective."""

    @abstractmethod
    def sample_actions(
        self,
        observation: "Observation",
        num_steps: int,
    ) -> torch.Tensor:
        """Sample actions in evaluation/serving mode."""

    def apply_fsdp(self, *args, **kwargs) -> None:  # pragma: no cover - optional hook
        """Optional shard hook for policies that support FSDP."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement FSDP sharding."
        )
