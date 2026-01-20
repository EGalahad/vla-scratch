from __future__ import annotations

from typing import Iterable, List, TYPE_CHECKING

import torch
import torch.utils.checkpoint as checkpoint
from torch.distributed.fsdp._fully_shard import fully_shard

if TYPE_CHECKING:
    from torch.distributed.fsdp._fully_shard import FSDPModule


def set_forward_backward_prefetch(
    layers: List["FSDPModule"],
    num_to_forward_prefetch: int,
    num_to_backward_prefetch: int,
) -> None:
    for i, layer in enumerate(layers):
        if i >= len(layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [
            layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)
    for i, layer in enumerate(layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [
            layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)


def fully_shard_layers(
    layers: Iterable["torch.nn.Module"],
    mesh,
    mp_policy,
    num_to_prefetch: int = 2,
) -> None:
    # Skip FSDP if mesh is None (non-distributed mode)
    if mesh is None:
        return
    
    layers_list = list(layers)
    for layer in layers_list:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
    set_forward_backward_prefetch(
        layers_list,
        num_to_backward_prefetch=num_to_prefetch,
        num_to_forward_prefetch=num_to_prefetch,
    )


def apply_checkpoint_when_training(
    module: torch.nn.Module,
    func,
    *args,
    preserve_rng_state: bool = False,
    disable: bool = False,
    **kwargs,
):
    if module.training and not disable:
        return checkpoint.checkpoint(
            func,
            *args,
            use_reentrant=False,
            preserve_rng_state=preserve_rng_state,
            **kwargs,
        )
    return func(*args, **kwargs)
