from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from vla_scratch.transforms.data_types import Observation


class VLMBridge(nn.Module):
    """Abstract base class for VLM bridges.

    Responsibilities:
      - Handle model-specific preprocessing (tokenization/vision).
      - Run the VLM transformer forward with optional checkpointing.
      - Return (hidden_states, prefix_pad_masks, kv_cache_list).
    """

    causal_model: nn.Module

    def get_text_dims(self) -> Tuple[int, int, int]:
        raise NotImplementedError

    @property
    def hidden_size(self) -> int:
        raise NotImplementedError

    def encode_prefix(
        self,
        *,
        observation: "Observation",
        extra_embs: Optional[torch.Tensor] = None,
        extra_pad_masks: Optional[torch.Tensor] = None,
        extra_att_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        List[Tuple[torch.Tensor, torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
        raise NotImplementedError

    def apply_fsdp(self, mp_policy, mesh):
        raise NotImplementedError
