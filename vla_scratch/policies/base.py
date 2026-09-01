from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Dict, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from vla_scratch.transforms.data_types import Observation, DataSample
    from vla_scratch.policies.modules.vlm_bridge.base import VLMOutputs


class BasePolicy(nn.Module, ABC):
    """Minimal policy interface required by training/eval/serving scripts."""

    def forward(
        self,
        data_sample: "DataSample",
    ) -> Tuple[torch.Tensor, Dict]:
        return self.compute_loss(data_sample)

    def initialize_weights(self):
        pass

    @abstractmethod
    def encode_prefix(
        self,
        observation: "Observation",
    ) -> Tuple[torch.Tensor, "VLMOutputs", Dict]:
        """Encode the observation prefix and return KV cache artifacts."""

    @abstractmethod
    def predict_suffix(
        self,
        state: torch.Tensor,
        suffix_input,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Predict action flow / displacement loss for the diffusion objective."""

    @abstractmethod
    def sample_actions(
        self,
        observation: "Observation",
        num_steps: int,
    ) -> torch.Tensor:
        """Sample actions in evaluation/serving mode."""

    @abstractmethod
    def compute_loss(
        self,
        data_sample: "DataSample",
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute training loss between predicted and target actions."""

    def apply_fsdp(self, *args, **kwargs):
        """Optional shard hook for policies that support FSDP."""
        return self
