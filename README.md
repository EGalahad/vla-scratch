<div align="left">
  <h2>
  VLA-Scratch: 
  <!-- <br/> -->
  a Fully Modular, High-Performance, Transparent Stack<br/>
  For Vision-Language-Action Models
  </h2>
</div>
<!-- add a huggingface badge, a twitter badge and a github star badge -->

## ✨ Key-Features

- **Explicit Data Model for Composable Modules** 
    - `TensorClass` is used to define explicit data boundaries for [dataset](vla_scratch/transforms/data_types.py) and [policy](vla_scratch/policies/utils/data_types.py), making our codebase fully typed and modular.
    - This unified sample schema allows heterogeneous dataset co-training, fast `.to(device)` calls, and clearer contracts between datasets, policies, and transforms.
    ![alt text](assets/data_model.png)
- **Dedicated Tuning for a First-Class Performance Stack** 
    - The [Qwen3-VL bridge](vla_scratch/policies/modules/vlm_bridge/qwen/) rewrites the forward path to minimize host-device syncs, making throughput 2x higher.
    - Layer-wise FSDP sharding and gradient checkpointing saves memory up to 2x, making it easier to scale model parameters.
    - TODO: figure here.
- **Clarity-Focused Hydra Workflow for Seamless Experimentation** 
    - Every [policy](vla_scratch/policies/pi/config.py) and [data](vla_scratch/datasets/config.py) config is registered with Hydra's `ConfigStore`, so experiments are overrideable with minimal boilerplate.
    - Training, eval, and serving scripts share a common config grammar, so switching between workflows is seamless.
    - TODO: example script snippet here.


## 🗂️ Codebase Structure

VLA-Scratch is a modular, high-performance VLA stack built around Hydra configs, TensorClass data models, and reusable training helpers.

- `vla_scratch/transforms/` — dataset/policy transforms plus TensorClass definitions for observations/actions: [Documentation →](vla_scratch/transforms/README.md).
- `vla_scratch/datasets/` — dataset configs, loaders, and transforms registered under `data=`: [Documentation →](vla_scratch/datasets/README.md).
- `vla_scratch/policies/` — policy interfaces, bridges, and action heads registered under `policy=`: [Documentation →](vla_scratch/policies/README.md).
- `scripts/` — training/eval/serving entry points with shared Hydra grammar: [Documentation →](scripts/README.md).
<!-- - `vla_scratch/helpers/` — repo-aware data + training helpers (prefetch dataloaders, FSDP setup, normalization I/O).
- `vla_scratch/utils/` — standalone utilities (config discovery, checkpointing, filesystem paths) that avoid repo-level imports.
- `examples/` — ready-to-run experiment folders demonstrating LIBERO usage and viz workflows.
- `tests/` — smoke tests and unit tests for policies, transforms, and helpers. -->

## 🚀 Quickstart

```bash
# Setup virtual environment and install dependencies:
uv sync
source .venv/bin/activate

# Apply lerobot dataset patch https://github.com/huggingface/lerobot/issues/959
patch -N -d .venv/lib/python3.10/site-packages/lerobot/datasets -p0 < ./patches/lerobot_dataset.patch
```

### Training
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    train_policy.py \
    policy=pi-qwen \
    policy.state_history=0 \
    policy.action_horizon=10 \
    policy.use_state=False \
    policy.transforms.0.max_length=500 \
    data=dont_blind \
    eval_data=dont_blind_8_8_eval \
    batch_size=32 \
    lr.base=3e-6 \
    +lr.vlm_bridge=3e-7 \
    +lr.action_expert=3e-6 \
    wandb.mode=online
```

<!-- 

```bash
hf download IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot --repo-type dataset

export HF_CACHE_TEMP=/home/elijah/.cache/huggingface/
ln -s ${HF_CACHE_TEMP}/hub/datasets--IPEC-COMMUNITY--libero_spatial_no_noops_1.0.0_lerobot/snapshots/cb7508999d4a8caa65b6448d6d700e1f347b809e ${HF_CACHE_TEMP}/lerobot/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot
```
 -->
