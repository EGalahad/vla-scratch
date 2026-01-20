from __future__ import annotations

from typing import Dict, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
from tensordict import TensorClass
import jaxtyping as at

if TYPE_CHECKING:
    from vla_scratch.transforms.data_types import Observation

TARGET_IGNORE_ID = -100


class VLMOutputs(TensorClass):
    last_hidden_state: at.Float[torch.Tensor, "*b seq_len hidden"]
    prefix_pad_masks: at.Bool[torch.Tensor, "*b seq_len"]
    key_states: at.Float[torch.Tensor, "*b n_layer n_head seq_len head_dim"]
    value_states: at.Float[torch.Tensor, "*b n_layer n_head seq_len head_dim"]
    hidden_state_list: at.Float[torch.Tensor, "*b n_layer seq_len hidden"]


class VLMBridge(nn.Module):
    causal_model: nn.Module

    def get_text_dims(self) -> Tuple[int, int, int]:
        raise NotImplementedError

    @property
    def hidden_size(self) -> int:
        raise NotImplementedError

    def encode(
        self,
        observation: "Observation",
        *,
        extra_embs: Optional[torch.Tensor] = None,
        extra_pad_masks: Optional[torch.Tensor] = None,
        extra_att_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[VLMOutputs, Dict]:
        raise NotImplementedError

    def apply_fsdp(self, mp_policy, mesh):
        raise NotImplementedError
