import torch
import jaxtyping as at
from typing import Tuple

HiddenState = at.Float[torch.Tensor, "*batch_size n_q hidden_dim"]
PositionIds = at.Int64[torch.Tensor, "*batch_size n_q"]
PositionEmbs = Tuple[
    at.Float[torch.Tensor, "*batch_size n_q head_dim"], at.Float[torch.Tensor, "*batch_size n_q head_dim"]
]


AttentionMask = at.Bool[torch.Tensor, "*batch_size 1 n_q n_kv"]
AdarmsCond = at.Float[torch.Tensor, "*batch_size cond_dim"]
KVCache = Tuple[
    at.Float[torch.Tensor, "*batch_size depth n_past_kv n_kv_heads head_dim"],
    at.Float[torch.Tensor, "*batch_size depth n_past_kv n_kv_heads head_dim"],
]

PrefixPadMask = at.Bool[torch.Tensor, "*batch_size n_past_kv"]

__all__ = [
    "HiddenState",
    "PositionIds",
    "PositionEmbs",
    "PrefixPadMask",
    "AttentionMask",
    "AdarmsCond",
    "KVCache",
]