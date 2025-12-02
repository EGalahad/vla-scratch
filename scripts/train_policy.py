from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import os
import time
from typing import cast, Any, Optional, List, Tuple, TYPE_CHECKING
from tqdm import tqdm
import wandb
import datetime
from setproctitle import setproctitle

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.utils.data import DistributedSampler
from torch.distributed.fsdp._fully_shard import FSDPModule
from torch.optim.lr_scheduler import CosineAnnealingLR

from tensordict import TensorDict

import hydra
from hydra.core.config_store import ConfigStore
from hydra.conf import HydraConf, JobConf, RunDir
from omegaconf import DictConfig, OmegaConf, MISSING

if TYPE_CHECKING:
    from vla_scratch.transforms.data_types import DataSample

from vla_scratch.policies.base import BasePolicy
from vla_scratch.policies.config import PolicyConfig
from vla_scratch.datasets.config import DataConfig


from vla_scratch.helpers.training import (
    build_param_lr_groups,
    create_dataloaders,
    get_beta_dist,
    aggregate_tensordict,
    compute_sample_mse,
    expand_tensor,
    print_with_rank,
    sample_and_select_noise,
    sample_time,
    setup_dist,
)

from vla_scratch.utils.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from vla_scratch.utils.config import save_train_config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_float32_matmul_precision("high")


@dataclass
class WandbCfg:
    project: str = "vla-scratch"
    mode: str = "disabled"
    tags: List[str] = field(default_factory=lambda: [])


@dataclass
class TrainConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"policy": "pi"},
            {"data": "libero-ipec"},
        ]
    )

    # data loader
    num_workers: int = 8
    prefetch_factor: int = 6
    split_seed: int = 42
    # optimization
    epochs: int = 20
    batch_size: int = 16
    grad_accum_steps: int = 1

    # Learning rates keyed by module path; "base" applies to remaining params
    lr: dict[str, float] = field(default_factory=lambda: {"base": 3e-6})
    # Linearly ramp LR from 0 to base LR over this many optimizer steps (0 disables)
    warmup_steps: int = 0
    # LR scheduling: start cosine anneal from the last N epochs
    # Set to 0 to disable cosine annealing
    cosine_anneal_epoch: int = 0

    betas: Tuple[float] = (0.99, 0.9999)
    eps: float = 1e-8
    weight_decay: float = 1e-4

    clip_grad_norm: float = 1.0
    num_noise_per_sample: int = 8
    num_noise_before_topk: int = 8
    detach_kv_cache: bool = False
    disp_loss_weight: float = 0.25

    # logging and evaluation
    exp_name: str = "pi-training"
    log_interval: int = 32
    eval_interval: int = 512
    save_interval: int = 1  # in epochs
    eval_fraction: float = 0.003
    eval_num_sample_steps: int = 10

    # data
    data: DataConfig = MISSING
    # model
    policy: PolicyConfig = MISSING
    checkpoint_path: Optional[str] = None
    load_optimizer: bool = True
    # wandb
    wandb: WandbCfg = field(default_factory=WandbCfg)

    # Hydra behavior overrides
    # - Do not change cwd automatically (job.chdir=False)
    # - Do not create .hydra subdir (output_subdir=null)
    # - Keep Hydra run dir as current directory (run.dir='.')
    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            job=JobConf(chdir=False),
            output_subdir=None,
            run=RunDir(dir="."),
        )
    )


cs = ConfigStore.instance()
cs.store(name="train", node=TrainConfig())


@hydra.main(config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    train_cfg = cast(TrainConfig, OmegaConf.to_object(cfg))

    if train_cfg.num_noise_before_topk < train_cfg.num_noise_per_sample:
        raise ValueError(
            "num_noise_before_topk must be >= num_noise_per_sample for top-k selection"
        )

    # Resolve checkpoint path (supports file or directory)
    if train_cfg.checkpoint_path is not None:
        cp = Path(train_cfg.checkpoint_path).resolve()
        # If a directory is provided, pick latest matching checkpoint
        if cp.is_dir():
            latest = find_latest_checkpoint(cp)
            train_cfg.checkpoint_path = latest if latest is not None else None
        else:
            train_cfg.checkpoint_path = cp

    # create timestamped output directory with exp_name
    now = datetime.datetime.now()
    date_stamp = now.strftime("%Y-%m-%d")
    time_stamp = now.strftime("%H-%M-%S")
    run_dir = Path("./outputs") / date_stamp / f"{time_stamp}-{train_cfg.exp_name}"
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)
    setproctitle(f"{time_stamp}-{train_cfg.exp_name}")

    assert (
        train_cfg.eval_interval % train_cfg.log_interval == 0
    ), "eval-interval must be multiple of log-interval"

    local_rank, global_rank, world_size, mesh = setup_dist()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    print_with_rank("create dataloaders...")
    (
        dataloader,
        eval_dataloader,
        subtrain_dataloader,
    ) = create_dataloaders(train_cfg, world_size, global_rank, add_noise=True)

    dummy_data: "DataSample" = next(iter(dataloader))[0]
    train_cfg.policy.action_dim = dummy_data.action_chunk.actions.shape[-1]
    train_cfg.policy.state_dim = dummy_data.observation.state.shape[-1]

    print_with_rank("create model...")
    with torch.device(device):
        model: BasePolicy = train_cfg.policy.instantiate()

    if world_size > 1:
        model.apply_fsdp(
            param_type=torch.bfloat16,
            reduce_type=torch.bfloat16,
            output_dtype=torch.float32,
            mesh=mesh,
        )
        model: FSDPModule | BasePolicy

    global_batch_size = train_cfg.batch_size * train_cfg.grad_accum_steps * world_size
    lr_cfg = dict(train_cfg.lr)
    param_groups = build_param_lr_groups(model, lr_cfg)

    # Scale learning rates by sqrt(batch) while preserving per-group ratios
    lr_scale = np.sqrt(global_batch_size)
    base_lr_dict = {name: float(value) * lr_scale for name, value in lr_cfg.items()}

    for group in param_groups:
        target_lr = base_lr_dict[group["name"]]
        group["lr"] = target_lr
        group["initial_lr"] = target_lr

    base_lr = base_lr_dict["base"]
    betas = tuple(np.pow(beta, global_batch_size) for beta in train_cfg.betas)
    eps = train_cfg.eps / np.sqrt(global_batch_size)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        betas=betas,
        eps=eps,
        weight_decay=train_cfg.weight_decay,
        foreach=False,
        fused=True,
    )

    if train_cfg.checkpoint_path is not None:
        load_checkpoint(
            model=model,
            checkpoint=train_cfg.checkpoint_path,
            global_rank=global_rank,
            optimizer=optimizer if train_cfg.load_optimizer else None,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if global_rank == 0:
        run = wandb.init(
            project=train_cfg.wandb.project,
            mode=train_cfg.wandb.mode,
            tags=train_cfg.wandb.tags,
        )
        run.config.update(OmegaConf.to_container(cfg))

        default_run_name = (
            f"{train_cfg.exp_name}-{datetime.datetime.now().strftime('%m-%d-%H-%M')}"
        )
        run_idx = run.name.split("-")[-1]
        run.name = f"{run_idx}-{default_run_name}"

        save_train_config(cfg, run_dir)

    time_dist = get_beta_dist(1.0, 1.5, device=device)

    global_step = 0
    scheduler = None
    steps_per_epoch = max(1, len(dataloader) // train_cfg.grad_accum_steps)
    last_time = time.perf_counter()
    log_tds = []

    for epoch in range(train_cfg.epochs):
        # Initialize cosine annealing at the start of the first scheduled epoch
        if (
            scheduler is None
            and train_cfg.cosine_anneal_epoch > 0
            and epoch >= train_cfg.epochs - train_cfg.cosine_anneal_epoch
        ):
            # Number of remaining optimizer steps including this epoch
            remaining_epochs = train_cfg.epochs - epoch
            cosine_steps = max(1, steps_per_epoch * remaining_epochs)
            # Start cosine anneal from current LR down to 1e-7 by training end

            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
                eta_min=1e-7,
            )
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)

        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"Epoch {epoch+1}/{train_cfg.epochs}",
            disable=local_rank != 0,
        )

        model.train()
        data_loader_iter = iter(dataloader)
        for i in pbar:
            torch.cuda.nvtx.range_push("Zero Grad")
            if isinstance(model, FSDPModule):
                model.unshard()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.nvtx.range_pop()

            for _ in range(train_cfg.grad_accum_steps):
                torch.cuda.nvtx.range_push("DataLoader")
                data_sample, perf_dict = next(data_loader_iter)
                data_sample: "DataSample" = data_sample.to(device, non_blocking=True)
                perf_dict = perf_dict.to(device, non_blocking=True)
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("Model Encode Prefix")
                _, prefix_pad_masks, prefix_key_values = model.encode_prefix(
                    observation=data_sample.observation,
                )
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("Noise Sampling")
                actions = data_sample.action_chunk.actions
                selected_noise = sample_and_select_noise(
                    actions,
                    train_cfg,
                    device=device,
                )
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("Expand Data Sample")
                data_sample = expand_tensor(data_sample, train_cfg.num_noise_per_sample)
                prefix_pad_masks = expand_tensor(
                    prefix_pad_masks, train_cfg.num_noise_per_sample
                )
                prefix_key_values = [
                    (
                        expand_tensor(k, train_cfg.num_noise_per_sample),
                        expand_tensor(v, train_cfg.num_noise_per_sample),
                    )
                    for k, v in prefix_key_values
                ]
                torch.cuda.nvtx.range_pop()

                if train_cfg.detach_kv_cache:
                    torch.cuda.nvtx.range_push("Detach KV Cache")
                    prefix_key_values = [
                        (k.detach(), v.detach()) for k, v in prefix_key_values
                    ]
                    torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("Apply Noise")
                actions = data_sample.action_chunk.actions
                u_t = selected_noise - actions
                timestep = sample_time(time_dist, data_sample.shape)
                noisy_actions = actions + timestep[:, None, None] * u_t
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("Model Predict Suffix")
                v_t, disp_loss = model.predict_suffix(
                    state=data_sample.observation.state,
                    prefix_pad_masks=prefix_pad_masks,
                    prefix_key_values=prefix_key_values,
                    noisy_actions=noisy_actions,
                    time=timestep,
                )
                flow_mse = F.mse_loss(u_t, v_t, reduction="none").mean()
                loss = flow_mse + train_cfg.disp_loss_weight * disp_loss
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("Loss Backward")
                (loss / train_cfg.grad_accum_steps / world_size).backward()
                torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("Optimizer Step")
            norm_before_clip = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=train_cfg.clip_grad_norm
            )
            # Linear warmup: ramp LR from 0 to base_lr over warmup_steps
            if train_cfg.warmup_steps and global_step < train_cfg.warmup_steps:
                warmup_factor = float(global_step + 1) / float(train_cfg.warmup_steps)
                for group in optimizer.param_groups:
                    target_lr = group["initial_lr"]
                    group["lr"] = target_lr * warmup_factor
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            torch.cuda.nvtx.range_pop()

            log_td = {}
            log_td["loss/flow_mse"] = flow_mse.detach()
            log_td["loss/disp_loss"] = disp_loss.detach()
            if isinstance(norm_before_clip, DTensor):
                norm_before_clip = norm_before_clip.full_tensor()
            log_td["loss/grad_norm"] = norm_before_clip
            log_td = TensorDict(log_td, [])
            log_td["loading"] = perf_dict.mean(dim=0)

            log_tds.append(log_td)

            global_step += 1

            if global_step % train_cfg.log_interval == 0:
                # log metrics
                log_dict = {
                    "epoch": global_step / steps_per_epoch,
                    "step": global_step,
                    "samples": global_step * global_batch_size,
                }
                for group in optimizer.param_groups:
                    log_dict[f"lr/{group['name']}"] = group["lr"]

                # log fps
                this_time = time.perf_counter()
                elapsed_time = this_time - last_time
                last_time = this_time
                fps = global_batch_size * train_cfg.log_interval / elapsed_time
                log_dict["perf/fps"] = fps / world_size
                log_dict["perf/fps.total"] = fps

                # log train stats (aggregate deterministically with a single all_reduce)
                log_td_mean: TensorDict = (
                    torch.stack(log_tds).mean(dim=0).type(torch.float32)
                )
                log_tds.clear()

                log_dict.update(aggregate_tensordict(log_td_mean, world_size))

                if global_step % train_cfg.eval_interval == 0:
                    # No pre-barrier; evaluation collectives below will synchronize
                    model.eval()
                    if isinstance(model, FSDPModule):
                        model.unshard()
                    for key, eval_dataloader in [
                        ("eval", eval_dataloader),
                        ("train", subtrain_dataloader),
                    ]:
                        eval_mse = compute_sample_mse(
                            model=model,
                            dataloader=eval_dataloader,
                            device=device,
                            num_sample_steps=train_cfg.eval_num_sample_steps,
                            local_rank=local_rank,
                        )
                        if world_size > 1:
                            dist.all_reduce(eval_mse, op=dist.ReduceOp.AVG)
                        log_dict[f"loss/sample_mse-{key}"] = eval_mse.item()
                    if isinstance(model, FSDPModule):
                        model.reshard()
                    model.train()

                log_string = "\n".join(
                    [
                        (
                            f"{key}={value:.8f}"
                            if isinstance(value, float)
                            else f"{key}={value}"
                        )
                        for key, value in log_dict.items()
                    ]
                )
                if local_rank == 0:
                    print(log_string)
                if global_rank == 0:
                    run.log(log_dict)

        if (epoch + 1) % train_cfg.save_interval == 0:
            save_checkpoint(model, optimizer, global_rank, f"checkpoint_{epoch+1}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
