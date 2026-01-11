from __future__ import annotations

import copy
import itertools
import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from collections.abc import Mapping as ABCMapping
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader, DistributedSampler

import datetime
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Mapping, Set, Tuple, Iterator, Optional
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.distributed.tensor import DTensor

from vla_scratch.helpers.data import create_dataset
from vla_scratch.utils.dataloader import DistributedRankAwareBatchSampler

from tensordict import TensorDict
if TYPE_CHECKING:
    from scripts.train_policy import TrainConfig
    from vla_scratch.policies.base import BasePolicy
    from vla_scratch.transforms.data_types import DataSample

logger = logging.getLogger(__name__)

local_rank = 0
global_rank = 0
world_size = 1


def setup_dist():
    """Initialize dist process group using env:// init and optionally build a device mesh."""
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


def _create_dataloader(
    *,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool,
    train_cfg: "TrainConfig",
    world_size: int,
    global_rank: int,
) -> DataLoader:
    base_sampler = getattr(dataset, "sampler", None)
    distributed_batch_sampler = None

    if world_size > 1:
        if base_sampler is not None:
            distributed_batch_sampler = DistributedRankAwareBatchSampler(
                base_sampler,
                batch_size=batch_size,
                drop_last=shuffle,
                num_replicas=world_size,
                rank=global_rank,
            )
            sampler = None
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=global_rank,
                shuffle=shuffle,
                drop_last=shuffle,
            )
    else:
        sampler = base_sampler

    def collate_fn(batch):
        return tuple(torch.stack(items) for items in zip(*batch))

    loader_kwargs = dict(
        num_workers=train_cfg.num_workers,
        persistent_workers=train_cfg.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    if train_cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = train_cfg.prefetch_factor
    if distributed_batch_sampler is not None:
        loader_kwargs["batch_sampler"] = distributed_batch_sampler
    elif sampler is not None:
        loader_kwargs["batch_size"] = batch_size
        loader_kwargs["sampler"] = sampler
    else:
        loader_kwargs["batch_size"] = batch_size
        loader_kwargs["shuffle"] = shuffle

    return DataLoader(dataset, **loader_kwargs)


def _subset_dataset(
    dataset: torch.utils.data.Dataset,
    fraction: float,
    *,
    seed: int,
) -> torch.utils.data.Dataset:
    if fraction >= 1.0:
        return dataset
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction for eval subset must be within (0, 1].")
    total_samples = len(dataset)
    subset_size = max(1, int(total_samples * fraction))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_samples, generator=generator)[:subset_size].tolist()
    return torch.utils.data.Subset(dataset, indices)


def _normalize_cfg_for_hash(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize_cfg_for_hash(asdict(value))
    if isinstance(value, DictConfig):
        return _normalize_cfg_for_hash(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, ABCMapping):
        return {
            str(key): _normalize_cfg_for_hash(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_cfg_for_hash(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize_cfg_for_hash(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _dataset_cache_key(data_cfg: Any, *, add_noise: bool) -> str:
    payload = {
        "data_cfg": _normalize_cfg_for_hash(data_cfg),
        "add_noise": add_noise,
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def create_dataloaders(
    train_cfg: "TrainConfig",
    world_size: int,
    global_rank: int,
    *,
    add_noise: bool = False,
) -> tuple[dict[str, DataLoader], dict[str, tuple[DataLoader, str]]]:
    train_loaders: Dict[str, DataLoader] = {}
    dataset_cache: Dict[str, torch.utils.data.Dataset] = {}
    if len(train_cfg.train_data.keys()) > 0:
        train_items = [
            (key, data.data, data.batch_size) for key, data in train_cfg.train_data.items()
        ]
    else:
        train_items = [("train", train_cfg.data, train_cfg.batch_size)]

    def get_or_create_dataset(
        data_cfg: Any,
        *,
        add_noise: bool,
    ) -> torch.utils.data.Dataset:
        key = _dataset_cache_key(data_cfg, add_noise=add_noise)
        dataset = dataset_cache.get(key)
        if dataset is None:
            dataset = create_dataset(
                data_cfg,
                train_cfg.policy,
                add_noise=add_noise,
            )
            dataset_cache[key] = dataset
        return dataset

    for name, data_cfg, batch_size in train_items:
        data_cfg.action_horizon = train_cfg.policy.action_horizon
        data_cfg.state_history = train_cfg.policy.state_history

        dataset = get_or_create_dataset(data_cfg, add_noise=add_noise)
        train_loaders[name] = _create_dataloader(
            dataset=dataset,
            shuffle=True,
            batch_size=batch_size,
            train_cfg=train_cfg,
            world_size=world_size,
            global_rank=global_rank,
        )

    eval_loaders: Dict[str, Tuple[DataLoader, str]] = {}

    extra_eval_seed = train_cfg.split_seed
    for idx, (name, eval_cfg) in enumerate(train_cfg.eval_data.items()):
        data_cfg = eval_cfg.data
        data_cfg.action_horizon = train_cfg.policy.action_horizon
        data_cfg.state_history = train_cfg.policy.state_history

        dataset = get_or_create_dataset(data_cfg, add_noise=False)
        dataset = _subset_dataset(dataset, eval_cfg.eval_fraction, seed=extra_eval_seed + idx)
        eval_loader = _create_dataloader(
            dataset=dataset,
            shuffle=False,
            batch_size=train_cfg.eval_batch_size,
            train_cfg=train_cfg,
            world_size=world_size,
            global_rank=global_rank,
        )
        eval_loaders[name] = (eval_loader, eval_cfg.eval_type)

    return train_loaders, eval_loaders


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

        if hasattr(module, "parameters"):
            params = [p for p in module.parameters() if p.requires_grad]
        elif isinstance(module, torch.nn.Parameter):
            params = [module] if module.requires_grad else []
        elif torch.is_tensor(module):
            params = [module] if module.requires_grad else []
        else:
            logger.warning(
                "Learning rate config references unsupported module path '%s'; skipping.",
                module_path,
            )
            continue
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
def eval_sample_mse(
    model: "BasePolicy",
    dataloader: DataLoader,
    device: torch.device,
    local_rank: int,
    *,
    num_sample_steps: int,
) -> TensorDict:
    eval_loss_tds = []

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
            reduction="mean",
        )
        eval_loss_td = TensorDict({"sample_mse": squared_error})
        eval_loss_tds.append(eval_loss_td)

    return torch.stack(eval_loss_tds).mean()


@torch.inference_mode()
def eval_generation(
    model: "BasePolicy",
    dataloader: DataLoader,
    device: torch.device,
    local_rank: int,
) -> TensorDict:
    eval_loss_tds = []

    pbar = tqdm(
        range(len(dataloader)),
        desc="Evaluating generation",
        disable=local_rank != 0,
    )
    dataloader_iter = iter(dataloader)
    for _ in pbar:
        batch, _ = next(dataloader_iter)
        batch: "DataSample" = batch.to(device)
        _, _, log_dict = model.encode_prefix(
            observation=batch.observation,
        )
        eval_loss_dict = {
            key.replace("loss/", ""): value
            for key, value in log_dict.items()
            if key.startswith("loss/")
        }
        eval_loss_tds.append(TensorDict(eval_loss_dict))

    return torch.stack(eval_loss_tds).mean()


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


def _add_dtype_bytes(
    dtype_bytes: dict[torch.dtype, int], dtype: torch.dtype, num_bytes: int
) -> None:
    dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + num_bytes


def _accumulate_tensor_stats(
    tensors: list[torch.Tensor | DTensor],
) -> tuple[int, dict[torch.dtype, int]]:
    """Return total bytes and float-dtype bytes for a list of tensors."""
    total_bytes_local = 0
    total_bytes_global = 0
    float_dtype_bytes: dict[torch.dtype, int] = {}
    for tensor in tensors:
        if isinstance(tensor, DTensor):
            tensor = tensor.to_local()
            num_bytes = tensor.numel() * tensor.element_size()
            total_bytes_local += num_bytes
            total_bytes_global += num_bytes * dist.get_world_size()
        else:
            num_bytes = tensor.numel() * tensor.element_size()
            total_bytes_local += num_bytes
            total_bytes_global += num_bytes
        if tensor.is_floating_point():
            _add_dtype_bytes(float_dtype_bytes, tensor.dtype, num_bytes)
    return total_bytes_local, total_bytes_global, float_dtype_bytes


def _accumulate_state_stats(state: Any) -> tuple[int, dict[torch.dtype, int]]:
    """Recursively return total bytes and float-dtype bytes for optimizer state."""
    if isinstance(state, (torch.Tensor, DTensor)):
        local_tensor = state.to_local() if isinstance(state, DTensor) else state
        num_bytes = local_tensor.numel() * local_tensor.element_size()
        float_dtype_bytes: dict[torch.dtype, int] = {}
        if local_tensor.is_floating_point():
            _add_dtype_bytes(float_dtype_bytes, local_tensor.dtype, num_bytes)
        return num_bytes, float_dtype_bytes

    if isinstance(state, Mapping):
        total_bytes = 0
        float_dtype_bytes: dict[torch.dtype, int] = {}
        for value in state.values():
            child_bytes, child_dtype_bytes = _accumulate_state_stats(value)
            total_bytes += child_bytes
            for dtype, bytes_ in child_dtype_bytes.items():
                _add_dtype_bytes(float_dtype_bytes, dtype, bytes_)
        return total_bytes, float_dtype_bytes

    if isinstance(state, (list, tuple)):
        total_bytes = 0
        float_dtype_bytes: dict[torch.dtype, int] = {}
        for value in state:
            child_bytes, child_dtype_bytes = _accumulate_state_stats(value)
            total_bytes += child_bytes
            for dtype, bytes_ in child_dtype_bytes.items():
                _add_dtype_bytes(float_dtype_bytes, dtype, bytes_)
        return total_bytes, float_dtype_bytes

    return 0, {}


def _format_dtype_bytes(dtype_bytes: dict[torch.dtype, int]) -> str:
    if not dtype_bytes:
        return "none"
    parts = [
        f"{dtype.__str__().replace('torch.', '')}={bytes_ / (1024**2):.2f}MB"
        for dtype, bytes_ in sorted(dtype_bytes.items(), key=lambda x: x[0].__str__())
    ]
    return ", ".join(parts)


def log_model_state_sizes(
    policy: "BasePolicy", optimizer: torch.optim.Optimizer
) -> None:
    """Print local-shard sizes (MB) for params, buffers, grads, and optimizer state."""
    params = list(policy.parameters())
    buffers = list(policy.buffers())
    grads = [p.grad for p in params if p.grad is not None]

    param_bytes_local, param_bytes_global, param_dtype_bytes = _accumulate_tensor_stats(
        params
    )
    buffer_bytes_local, buffer_bytes_global, buffer_dtype_bytes = (
        _accumulate_tensor_stats(buffers)
    )
    grad_bytes_local, grad_bytes_global, grad_dtype_bytes = _accumulate_tensor_stats(
        grads
    )
    optim_bytes_local = 0
    optim_dtype_bytes: dict[torch.dtype, int] = {}
    for state in optimizer.state.values():
        state_bytes, state_dtype_bytes = _accumulate_state_stats(state)
        optim_bytes_local += state_bytes
        for dtype, bytes_ in state_dtype_bytes.items():
            _add_dtype_bytes(optim_dtype_bytes, dtype, bytes_)

    def to_mb(num_bytes: int) -> float:
        return num_bytes / (1024**2)

    total_bytes_local = (
        param_bytes_local + buffer_bytes_local + grad_bytes_local + optim_bytes_local
    )
    total_bytes_global = param_bytes_global + buffer_bytes_global + grad_bytes_global
    # global means full model, not sum over all ranks
    msg = (
        "Tensor sizes (MB): "
        f"params[local={to_mb(param_bytes_local):.2f}, "
        f"global={to_mb(param_bytes_global):.2f}], "
        f"buffers[local={to_mb(buffer_bytes_local):.2f}, "
        f"global={to_mb(buffer_bytes_global):.2f}], "
        f"grads[local={to_mb(grad_bytes_local):.2f}, "
        f"global={to_mb(grad_bytes_global):.2f}], "
        f"optim_state[local={to_mb(optim_bytes_local):.2f}] | "
        f"Total[local={to_mb(total_bytes_local):.2f}, "
        f"global={to_mb(total_bytes_global):.2f}]; "
        "float dtypes: "
        f"params[{_format_dtype_bytes(param_dtype_bytes)}], "
        f"buffers[{_format_dtype_bytes(buffer_dtype_bytes)}], "
        f"grads[{_format_dtype_bytes(grad_dtype_bytes)}], "
        f"optim_state[{_format_dtype_bytes(optim_dtype_bytes)}]"
    )
    print_with_rank(msg)


class PrefetchingEpochIterator(Iterator[Iterator]):
    """Prefetch the next epoch's first batch while the current epoch runs."""

    def __init__(
        self,
        # iterator_fn: Callable[[int], Iterator],
        dataloader: DataLoader,
        num_epochs: int,
        *,
        max_workers: int = 1,
    ):
        # self.iterator_fn = iterator_fn
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.epoch_idx = 0
        self.current_iter: Optional[Iterator] = None

        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.prefetched_iter: Optional[Iterator] = None

        self._submit_prefetch(0)

    def __iter__(self) -> "PrefetchingEpochIterator":
        return self

    def __next__(self) -> Iterator:
        if self.epoch_idx >= self.num_epochs:
            raise StopIteration

        self._cleanup_prev_iter()

        prefetched_first_batch = self.prefetch_batch_future.result()
        self.current_iter = self.prefetched_iter

        next_epoch = self.epoch_idx + 1
        self._submit_prefetch(next_epoch)
        self.epoch_idx += 1

        return itertools.chain((prefetched_first_batch,), self.current_iter)

    def _submit_prefetch(self, epoch_idx: int):
        if epoch_idx >= self.num_epochs:
            return
        dataloader = copy.copy(self.dataloader)
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch_idx)
        self.prefetched_iter = iter(dataloader)
        # return self.executor.submit(
        #     lambda: next(self.prefetched_iter),
        # )
        self.prefetch_batch_future = self.executor.submit(
            lambda iter: next(iter),
            self.prefetched_iter,
        )

    def _cleanup_prev_iter(self, final=False) -> None:
        if self.dataloader.persistent_workers and not final:
            # do not shutdown workers if persistent
            return
        if self.current_iter is not None and hasattr(self.current_iter, "_shutdown_workers"):
            self.current_iter._shutdown_workers()  # type: ignore[attr-defined]
        self.current_iter = None

    def finalize(self) -> None:
        self._cleanup_prev_iter()


class EagerEpochIterator(Iterator[Iterator]):
    """Create a fresh iterator each epoch without prefetching."""

    def __init__(
        self,
        dataloader: DataLoader,
        num_epochs: int,
    ):
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.epoch_idx = 0
        self.current_iter: Optional[Iterator] = None

    def __iter__(self) -> "EagerEpochIterator":
        return self

    def __next__(self) -> Iterator:
        if self.epoch_idx >= self.num_epochs:
            raise StopIteration

        self._cleanup_prev_iter()
        dataloader = self.dataloader
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(self.epoch_idx)
        iterator = iter(dataloader)
        self.current_iter = iterator
        self.epoch_idx += 1
        return iterator

    def _cleanup_prev_iter(self, final=False) -> None:
        if self.dataloader.persistent_workers and not final:
            # do not shutdown workers if persistent
            return
        if self.current_iter is not None and hasattr(self.current_iter, "_shutdown_workers"):
            self.current_iter._shutdown_workers()  # type: ignore[attr-defined]
        self.current_iter = None

    def finalize(self) -> None:
        self._cleanup_prev_iter()
