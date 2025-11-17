from typing import Any, Sequence, List, TYPE_CHECKING

from vla_scratch.datasets.config import DataConfig
from vla_scratch.policies.config import PolicyConfig
from vla_scratch.transforms.base import (
    TransformFn,
    TransformedDataset,
)
from vla_scratch.transforms.common import ToObservation
from vla_scratch.transforms.normalization import (
    Normalize,
    DeNormalize,
    load_norm_stats,
)
from vla_scratch.utils.config import instantiate_transform, locate_class


def make_transforms(specs: Sequence[Any]) -> List[TransformFn]:
    """Instantiate a sequence of transform specs into TransformFn objects."""
    return [instantiate_transform(spec) for spec in specs]


def build_input_transforms(
    data_cfg: DataConfig, policy_cfg: "PolicyConfig"
) -> Sequence[TransformFn]:
    transforms: list[TransformFn] = []

    from vla_scratch.datasets.libero.transforms import LiberoAction  # type: ignore

    # Use explicit input_transforms only
    input_specs = data_cfg.input_transforms
    dataset_transforms = make_transforms(input_specs)
    transforms.extend(
        [tf for tf in dataset_transforms if not isinstance(tf, LiberoAction)]
    )

    if data_cfg.norm_stats_path is not None:
        stats = load_norm_stats(data_cfg, policy_cfg)
        transforms.append(Normalize(norm_stats=stats, strict=False))

    policy_transforms = make_transforms(policy_cfg.transforms)
    transforms.extend(policy_transforms)

    transforms.append(ToObservation())
    return transforms


def build_output_transforms(
    data_cfg: DataConfig, policy_cfg: "PolicyConfig"
) -> Sequence[TransformFn]:
    transforms: list[TransformFn] = []
    if data_cfg.norm_stats_path is not None:
        stats = load_norm_stats(data_cfg, policy_cfg)
        transforms.append(DeNormalize(norm_stats=stats, strict=False))

    # Apply dataset-provided inverse output transforms (map to original dataset
    # action format), then convert to numpy for network/payloads.
    inv_specs = data_cfg.output_inv_transforms
    transforms.extend(make_transforms(inv_specs))

    from vla_scratch.transforms.common import ToNumpy

    transforms.append(ToNumpy())
    return transforms


def create_dataset(
    data_cfg: DataConfig,
    policy_cfg: "PolicyConfig",
    *,
    skip_norm_stats: bool = False,
) -> TransformedDataset:
    """Create a training dataset applying input -> output -> normalize -> policy -> ToTensorClass.

    Uses DataConfig.input_transforms and output_transforms exclusively.
    """
    # Assemble dataset transforms
    input_specs = data_cfg.input_transforms
    output_specs = data_cfg.output_transforms
    ds_transforms = make_transforms(list(input_specs) + list(output_specs))

    # Optionally add normalization of processed state/action
    norm_tf: List[TransformFn] = []
    if (not skip_norm_stats) and data_cfg.norm_stats_path is not None:
        stats = load_norm_stats(data_cfg, policy_cfg)
        norm_tf = [Normalize(norm_stats=stats)]

    policy_tfs = make_transforms(policy_cfg.transforms)

    from vla_scratch.transforms.common import ToDataSample

    pipeline = ds_transforms + norm_tf + policy_tfs + [ToDataSample()]

    # Build underlying dataset via DataConfig target
    base_cls = locate_class(data_cfg._target_)
    base = base_cls(data_cfg)
    return TransformedDataset(base, pipeline)


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from tensordict import TensorDict
from tqdm import tqdm

from vla_scratch.policies.pi.policy import PiPolicy
from vla_scratch.transforms.data_types import DataSample

if TYPE_CHECKING:
    from scripts.train_policy import TrainConfig


def create_dataloaders(train_cfg: "TrainConfig", world_size: int, global_rank: int):
    train_cfg.data.action_horizon = train_cfg.policy.action_horizon
    train_cfg.data.state_history = train_cfg.policy.state_history

    full_dataset = create_dataset(
        train_cfg.data,
        train_cfg.policy,
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

    def _create_dataloader(
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool,
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

    dataloader = _create_dataloader(
        train_dataset, shuffle=True, batch_size=train_cfg.batch_size
    )
    eval_dataloader = _create_dataloader(eval_dataset, shuffle=False, batch_size=32)
    subtrain_dataloader = _create_dataloader(
        subtrain_dataset, shuffle=False, batch_size=32
    )
    return (
        dataloader,
        eval_dataloader,
        subtrain_dataloader,
    )


@torch.inference_mode()
def compute_sample_mse(
    model: PiPolicy,
    dataloader: DataLoader,
    device: torch.device,
    num_sample_steps: int,
    local_rank: int,
) -> torch.Tensor:
    squared_errors = []

    pbar = range(len(dataloader))
    if local_rank == 0:
        pbar = tqdm(pbar, desc=f"Evaluating sample MSE")
    dataloader_iter = iter(dataloader)
    for i in pbar:
        batch, _ = next(dataloader_iter)
        batch: DataSample = batch.to(device)
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


def aggregate_tensordict(td: TensorDict, world_size: int) -> TensorDict:
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
