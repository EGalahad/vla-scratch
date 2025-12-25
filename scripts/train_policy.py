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
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
torch._dynamo.config.recompile_limit = 64

from tensordict import TensorDict

import hydra
from hydra.core.config_store import ConfigStore
from hydra.conf import HydraConf, JobConf, RunDir
from omegaconf import DictConfig, OmegaConf, MISSING

from vla_scratch.policies.config import PolicyConfig
from vla_scratch.datasets.config import DataConfig, EvalDataCfg, TrainDataCfg

from vla_scratch.helpers.training import (
    aggregate_tensordict,
    build_param_lr_groups,
    create_dataloaders,
    eval_generation,
    eval_sample_mse,
    log_model_state_sizes,
    print_with_rank,
    setup_dist,
    EagerEpochIterator,
    PrefetchingEpochIterator,
)

from vla_scratch.utils.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from vla_scratch.utils.config import save_train_config


if TYPE_CHECKING:
    from vla_scratch.transforms.data_types import DataSample
    from torch.distributed.tensor import DTensor
    from vla_scratch.policies.base import BasePolicy
    from torch.distributed.fsdp._fully_shard import FSDPModule

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
            {"policy": "pi-qwen"},
            {"data": "libero-ipec"},
            {"train_data": "none"},
            {"eval_data": "none"},
        ]
    )

    # data loader
    num_workers: int = 4
    prefetch_factor: int = 2
    split_seed: int = 42
    epoch_iterator: str = "eager"  # "prefetch" or "eager"
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

    # logging and evaluation
    exp_name: str = "pi-training"
    log_interval: int = 32
    eval_interval: int = 512
    save_interval: int = 1  # in epochs

    # data
    data: DataConfig = MISSING
    train_data: TrainDataCfg = field(default_factory=TrainDataCfg)
    eval_data: EvalDataCfg = field(default_factory=EvalDataCfg)
    # eval_data datasets are DataConfig entries with eval_fraction/eval_type overrides.
    eval_num_sample_steps: int = 10
    eval_batch_size: int = 32

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

import vla_scratch.configs
@hydra.main(config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    train_cfg = cast(TrainConfig, OmegaConf.to_object(cfg))

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
    cfg.run_dir = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)
    setproctitle(f"{train_cfg.exp_name}")

    assert (
        train_cfg.eval_interval % train_cfg.log_interval == 0
    ), "eval-interval must be multiple of log-interval"

    local_rank, global_rank, world_size, mesh = setup_dist()
    device = torch.device(type="cuda", index=local_rank)
    cfg.world_size = world_size

    print_with_rank("create dataloaders...")
    train_loaders, eval_loaders = create_dataloaders(
        train_cfg,
        world_size,
        global_rank,
        add_noise=True,
    )

    try:
        first_loader = next(iter(train_loaders.values()))
        dummy_data, _ = next(iter(first_loader))
    except RuntimeError as e:
        print("If you see input ids shape incompatible errors here, please increase the max_length in processor config in vla_scratch/policies/pi/config.py!")
        raise e
    dummy_data: "DataSample" = dummy_data[0:1].to(device)
    train_cfg.policy.action_dim = dummy_data.action_chunk.actions.shape[-1]
    train_cfg.policy.state_dim = dummy_data.observation.state.shape[-1]

    print_with_rank("create model...")
    with torch.device(device):
        model: "BasePolicy" = train_cfg.policy.instantiate()

    # Warmup pass
    print_with_rank("warmup pass...")
    loss, _ = model.compute_loss(dummy_data)
    loss.backward()

    # for eval_key, (eval_dataloader, eval_type) in eval_loaders.items():
    #     if eval_type == "sample_mse":
    #         eval_metrics = eval_sample_mse(
    #             model=model,
    #             dataloader=eval_dataloader,
    #             device=device,
    #             num_sample_steps=train_cfg.eval_num_sample_steps,
    #             local_rank=local_rank,
    #         )
    #     elif eval_type == "generation":
    #         eval_metrics = eval_generation(
    #             model=model,
    #             dataloader=eval_dataloader,
    #             device=device,
    #             local_rank=local_rank,
    #         )
    #     else:
    #         raise ValueError(
    #             f"Unsupported eval_type '{eval_type}' for dataset '{eval_key}'."
    #         )

    model.initialize_weights()

    model.apply_fsdp(
        param_type=torch.bfloat16,
        reduce_type=torch.float32,
        output_dtype=torch.float32,
        mesh=mesh,
    )
    model: "FSDPModule" | "BasePolicy"

    if train_cfg.train_data:
        train_batch_sizes_by_name = {
            name: cfg.batch_size for name, cfg in train_cfg.train_data.items()
        }
    else:
        train_batch_sizes_by_name = {"train": train_cfg.batch_size}
    total_batch_size = sum(train_batch_sizes_by_name.values())
    global_batch_size = total_batch_size * train_cfg.grad_accum_steps * world_size
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
    log_model_state_sizes(model, optimizer)

    if train_cfg.checkpoint_path is not None:
        missing, unexpected = load_checkpoint(
            model=model,
            checkpoint=train_cfg.checkpoint_path,
            global_rank=global_rank,
            optimizer=optimizer if train_cfg.load_optimizer else None,
        )
        if local_rank == 0:
            print(f"Loaded checkpoint '{train_cfg.checkpoint_path}'")
            if len(missing) > 0:
                print(f"  Missing keys: {missing}")
            if len(unexpected) > 0:
                print(f"  Unexpected keys: {unexpected}")

    if global_rank == 0:
        run = wandb.init(
            project=train_cfg.wandb.project,
            mode=train_cfg.wandb.mode,
            tags=train_cfg.wandb.tags,
        )
        # update cfg
        run.config.update(OmegaConf.to_container(cfg))
        # update train_cfg
        # run.config.update(train_cfg.asdict())

        default_run_name = (
            f"{train_cfg.exp_name}-{datetime.datetime.now().strftime('%m-%d-%H-%M')}"
        )
        run_idx = run.name.split("-")[-1]
        run.name = f"{run_idx}-{default_run_name}"

        train_cfg_path = save_train_config(train_cfg, run_dir)
        run.save(str(train_cfg_path), base_path=str(run_dir))
        # save cfg
        cfg_path = run_dir / "cfg.yaml"
        with open(cfg_path, "w") as f:
            OmegaConf.save(cfg, f)
        run.save(str(cfg_path), base_path=str(run_dir))
        
    global_step = 0
    scheduler = None
    steps_per_epoch_by_name = {
        name: max(1, len(loader) // train_cfg.grad_accum_steps)
        for name, loader in train_loaders.items()
    }
    steps_per_epoch = min(steps_per_epoch_by_name.values())
    last_time = time.perf_counter()
    log_tds = []

    if train_cfg.epoch_iterator == "prefetch":
        epoch_iterator_cls = PrefetchingEpochIterator
    elif train_cfg.epoch_iterator == "eager":
        epoch_iterator_cls = EagerEpochIterator
    else:
        raise ValueError(
            f"Unsupported epoch_iterator '{train_cfg.epoch_iterator}', expected 'eager' or 'prefetch'."
        )
    epoch_iterators = {
        name: epoch_iterator_cls(dataloader=loader, num_epochs=train_cfg.epochs)
        for name, loader in train_loaders.items()
    }
    for epoch in range(train_cfg.epochs):
        data_loader_iters = {
            name: next(epoch_iterator)
            for name, epoch_iterator in epoch_iterators.items()
        }
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

        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"Epoch {epoch+1}/{train_cfg.epochs}",
            disable=local_rank != 0,
        )

        model.train()
        for i in pbar:
            torch.cuda.nvtx.range_push("Zero Grad")
            model.unshard()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.nvtx.range_pop()

            log_td = {}
            for _ in range(train_cfg.grad_accum_steps):
                for train_key, data_loader_iter in data_loader_iters.items():
                    torch.cuda.nvtx.range_push("DataLoader")
                    data_sample, perf_dict = next(data_loader_iter)
                    data_sample: "DataSample" = data_sample.to(device, non_blocking=True)
                    perf_dict: TensorDict = perf_dict.to(device, non_blocking=True)
                    torch.cuda.nvtx.range_pop()

                    loss, log_dict = model.compute_loss(data_sample)
                    torch.cuda.nvtx.range_push("Loss Backward")
                    (loss / train_cfg.grad_accum_steps / world_size).backward()
                    torch.cuda.nvtx.range_pop()

                    log_dict = {f"{key.split('/')[0]}/{train_key}.{key.split('/')[1]}": val for key, val in log_dict.items()}
                    perf_dict = {f"loading/{train_key}.{key}": val for key, val in perf_dict.mean(dim=0).items()}
                    log_td.update(log_dict)
                    log_td.update(perf_dict)

            torch.cuda.nvtx.range_push("Optimizer Step")
            norm_before_clip: "DTensor" = torch.nn.utils.clip_grad_norm_(
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

            log_td["loss/grad_norm"] = norm_before_clip.full_tensor()
            log_td = TensorDict(log_td, [])

            log_tds.append(log_td)

            global_step += 1

            if global_step % train_cfg.log_interval == 0:
                # log metrics
                log_dict = {
                    "step": global_step,
                }
                for train_key, batch_size in train_batch_sizes_by_name.items():
                    train_global_batch = (
                        batch_size * train_cfg.grad_accum_steps * world_size
                    )
                    log_dict[f"{train_key}.epoch"] = (
                        global_step / steps_per_epoch_by_name[train_key]
                    )
                    log_dict[f"{train_key}.samples"] = global_step * train_global_batch
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
                log_td_mean: TensorDict = torch.stack(log_tds).mean(dim=0)
                log_tds.clear()

                if global_step % train_cfg.eval_interval == 0 and len(eval_loaders) > 0:
                    model.eval()
                    model.unshard()
                    for eval_key, (eval_dataloader, eval_type) in eval_loaders.items():
                        if eval_type == "sample_mse":
                            eval_metrics = eval_sample_mse(
                                model=model,
                                dataloader=eval_dataloader,
                                device=device,
                                num_sample_steps=train_cfg.eval_num_sample_steps,
                                local_rank=local_rank,
                            )
                        elif eval_type == "generation":
                            eval_metrics = eval_generation(
                                model=model,
                                dataloader=eval_dataloader,
                                device=device,
                                local_rank=local_rank,
                            )
                        else:
                            raise ValueError(
                                f"Unsupported eval_type '{eval_type}' for dataset '{eval_key}'."
                            )

                        for metric_name in eval_metrics.keys():
                            log_td_mean[f"eval/{eval_key}.{metric_name}"] = eval_metrics[metric_name]
                    model.reshard()
                    model.train()

                log_dict.update(aggregate_tensordict(log_td_mean, world_size))

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
                dist.barrier()

        if (epoch + 1) % train_cfg.save_interval == 0:
            save_checkpoint(model, optimizer, global_rank, f"checkpoint_{epoch+1}")

    for epoch_iterator in epoch_iterators.values():
        if hasattr(epoch_iterator, "finalize"):
            epoch_iterator.finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
