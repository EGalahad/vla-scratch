from __future__ import annotations

import datetime
import logging
import os
from typing import TYPE_CHECKING, Any, Mapping, Set

import einops
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from vla_scratch.helpers.data import create_dataset
from vla_scratch.policies.utils.transformers import sample_noise

if TYPE_CHECKING:
    from tensordict import TensorDict
    from scripts.train_policy import TrainConfig
    from vla_scratch.policies.base import BasePolicy
    from vla_scratch.transforms.data_types import DataSample

logger = logging.getLogger(__name__)

local_rank = 0
global_rank = 0
world_size = 1


def setup_dist():
    """Initialize DDP process group using env:// init and optionally build a device mesh."""
    global local_rank, global_rank, world_size
    mesh = None
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        timeout_sec = int(os.environ.get("TORCH_DDP_TIMEOUT_SEC", 600))
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=timeout_sec),
        )
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size > 1:
            nproc_per_node = int(os.environ["LOCAL_WORLD_SIZE"])
            nnodes = world_size // nproc_per_node
            assert world_size == nproc_per_node * nnodes
            if nnodes > 1:
                mesh = dist.device_mesh.init_device_mesh(
                    "cuda",
                    (nnodes, nproc_per_node),
                    mesh_dim_names=("node", "process"),
                )
            else:
                mesh = dist.device_mesh.init_device_mesh(
                    "cuda",
                    (world_size,),
                    mesh_dim_names=("process",),
                )
    except ValueError:
        local_rank = 0
        global_rank = 0
        world_size = 1
        mesh = None
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size, mesh


def print_with_rank(string: str) -> None:
    print(f"[Rank {global_rank}] {string}")


def get_beta_dist(
    alpha: float,
    beta: float,
    device: torch.device | str,
) -> torch.distributions.Distribution:
    """Construct a Beta distribution on the training device."""
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    return torch.distributions.Beta(beta_t, alpha_t)


def sample_time(
    time_dist: torch.distributions.Distribution,
    shape: torch.Size,
) -> torch.Tensor:
    """Sample diffusion timesteps with a small clamp to avoid numerical issues."""
    return time_dist.sample(shape) * 0.999 + 0.001


def sample_and_select_noise(
    actions: torch.Tensor,
    train_cfg: "TrainConfig",
    *,
    device: torch.device,
) -> torch.Tensor:
    """Draw noisy action candidates and choose the closest ones to the target actions."""
    batch_size, action_horizon, action_dim = actions.shape
    candidate_shape = (
        batch_size,
        train_cfg.num_noise_before_topk,
        action_horizon,
        action_dim,
    )
    noise_candidates = sample_noise(
        candidate_shape,
        device,
        dtype=actions.dtype,
    )
    action_flat = einops.rearrange(
        actions, "b h d -> b 1 (h d)", h=action_horizon, d=action_dim
    )
    noise_flat = einops.rearrange(
        noise_candidates,
        "b k h d -> b k (h d)",
        h=action_horizon,
        d=action_dim,
    )
    if train_cfg.num_noise_before_topk == train_cfg.num_noise_per_sample:
        selected_noise_flat = noise_flat
    else:
        distances = torch.sum((noise_flat - action_flat) ** 2, dim=-1)
        topk_indices = torch.topk(
            distances,
            k=train_cfg.num_noise_per_sample,
            dim=1,
            largest=False,
        ).indices
        gather_idx = topk_indices.unsqueeze(-1).expand(
            -1,
            -1,
            noise_flat.shape[-1],
        )
        selected_noise_flat = torch.gather(
            noise_flat,
            dim=1,
            index=gather_idx,
        )

    selected_noise = einops.rearrange(
        selected_noise_flat,
        "b k (h d) -> (k b) h d",
        h=action_horizon,
        d=action_dim,
    )
    return selected_noise


def _create_dataloader(
    *,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool,
    train_cfg: "TrainConfig",
    world_size: int,
    global_rank: int,
) -> DataLoader:
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=shuffle,
            drop_last=shuffle,
        )
    else:
        sampler = None

    def collate_fn(batch):
        return tuple(torch.stack(items) for items in zip(*batch))

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=train_cfg.num_workers,
        persistent_workers=train_cfg.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    if train_cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = train_cfg.prefetch_factor
    if sampler is not None:
        loader_kwargs["sampler"] = sampler
    else:
        loader_kwargs["shuffle"] = shuffle

    return DataLoader(dataset, **loader_kwargs)


def create_dataloaders(
    train_cfg: "TrainConfig",
    world_size: int,
    global_rank: int,
    *,
    add_noise: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_cfg.data.action_horizon = train_cfg.policy.action_horizon
    train_cfg.data.state_history = train_cfg.policy.state_history

    full_dataset = create_dataset(
        train_cfg.data,
        train_cfg.policy,
        add_noise=add_noise,
    )

    if not (0.0 < train_cfg.eval_fraction < 1.0):
        raise ValueError("eval_fraction must be within (0, 1).")

    total_samples = len(full_dataset)
    eval_size = max(1, int(total_samples * train_cfg.eval_fraction))
    train_size = total_samples - eval_size

    split_generator = torch.Generator().manual_seed(train_cfg.split_seed)
    train_dataset, eval_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, eval_size],
        generator=split_generator,
    )
    train_size = len(train_dataset)

    subtrain_size = max(1, int(train_size * train_cfg.eval_fraction))
    subtrain_generator = torch.Generator().manual_seed(train_cfg.split_seed + 1)
    subtrain_indices = torch.randperm(train_size, generator=subtrain_generator)[
        :subtrain_size
    ].tolist()
    subtrain_dataset = torch.utils.data.Subset(train_dataset, subtrain_indices)

    dataloader = _create_dataloader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=train_cfg.batch_size,
        train_cfg=train_cfg,
        world_size=world_size,
        global_rank=global_rank,
    )
    eval_dataloader = _create_dataloader(
        dataset=eval_dataset,
        shuffle=False,
        batch_size=32,
        train_cfg=train_cfg,
        world_size=world_size,
        global_rank=global_rank,
    )
    subtrain_dataloader = _create_dataloader(
        dataset=subtrain_dataset,
        shuffle=False,
        batch_size=32,
        train_cfg=train_cfg,
        world_size=world_size,
        global_rank=global_rank,
    )

    return (
        dataloader,
        eval_dataloader,
        subtrain_dataloader,
    )


def build_param_lr_groups(
    model: torch.nn.Module,
    lr_cfg: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Create optimizer parameter groups from a learning-rate mapping."""
    if not lr_cfg:
        return [{"params": list(model.parameters()), "name": "base"}]

    base_lr = lr_cfg.get("base")
    used_params: Set[int] = set()
    param_groups: list[dict[str, Any]] = []

    for module_path, lr in lr_cfg.items():
        if module_path == "base":
            continue
        try:
            module = model
            for attr in module_path.split("."):
                module = getattr(module, attr)
        except AttributeError:
            logger.warning(
                "Learning rate config references missing module path '%s'; skipping.",
                module_path,
            )
            continue

        params = [p for p in module.parameters() if p.requires_grad]
        if not params:
            continue

        param_groups.append({"params": params, "lr": float(lr), "name": module_path})
        used_params.update(id(p) for p in params)

    remaining_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in used_params
    ]
    if remaining_params:
        base_group: dict[str, Any] = {"params": remaining_params, "name": "base"}
        if base_lr is not None:
            base_group["lr"] = float(base_lr)
        param_groups.append(base_group)

    return param_groups


@torch.inference_mode()
def compute_sample_mse(
    model: "BasePolicy",
    dataloader: DataLoader,
    device: torch.device,
    num_sample_steps: int,
    local_rank: int,
) -> torch.Tensor:
    squared_errors = []

    pbar = tqdm(
        range(len(dataloader)),
        desc="Evaluating sample MSE",
        disable=local_rank != 0,
    )
    dataloader_iter = iter(dataloader)
    for _ in pbar:
        batch, _ = next(dataloader_iter)
        batch: "DataSample" = batch.to(device)
        predicted_actions = model.sample_actions(
            observation=batch.observation,
            num_steps=num_sample_steps,
        )
        target_actions = batch.action_chunk.actions

        squared_error = F.mse_loss(
            predicted_actions,
            target_actions,
            reduction="none",
        )
        squared_errors.append(squared_error.mean())

    return torch.stack(squared_errors).mean()


def expand_tensor(t: torch.Tensor, repeat_times: int) -> torch.Tensor:
    return t.expand(repeat_times, *t.shape).reshape(-1, *t.shape[1:])


def aggregate_tensordict(td: "TensorDict", world_size: int) -> dict[str, float]:
    flat_td = td.flatten_keys(separator="/")
    if world_size <= 1:
        return flat_td.to_dict(convert_tensors=True)
    flat_dict = flat_td.to_dict()
    keys_sorted = sorted(flat_dict.keys())

    vec = torch.stack(
        [flat_dict[k].detach().reshape(1) for k in keys_sorted],
        dim=0,
    ).squeeze(-1)

    dist.all_reduce(vec, op=dist.ReduceOp.AVG)

    agg_values = vec.detach().cpu().tolist()
    return {k: float(agg_values[i]) for i, k in enumerate(keys_sorted)}
