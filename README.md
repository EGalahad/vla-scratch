<!-- <div align="left">
  <h1>
  VLA-Scratch: 
  a Modular, Performant, Efficient Stack<br/>
  For Vision-Language-Action Models
  </h1>
</div> -->
# VLA-Scratch: a Modular, Performant, Efficient Stack For Vision-Language-Action Models
[![PyTorch FSDP2](https://img.shields.io/badge/PyTorch-FSDP2-ee4c2c?logo=pytorch&logoColor=white)](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)
[![PyTorch TensorDict](https://img.shields.io/badge/PyTorch-TensorDict-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/tensordict/)
[![Qwen3-VL](https://img.shields.io/badge/Qwen3-VL-1d9bf0?logo=github&logoColor=white)](https://github.com/QwenLM/Qwen3-VL)
[![Checkpoints](https://img.shields.io/badge/Checkpoints-1d9bf0?logo=huggingface&logoColor=white)](https://huggingface.co/collections/elijahgalahad/vla-scratch)

## 🚀 Quickstart

We use [uv](https://docs.astral.sh/uv/) to manage dependencies. Run `GIT_LFS_SKIP_SMUDGE=1 uv sync` to set up the environment.

Verify your installation with the following commands:

```bash
# Training
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_policy.py \
    policy=pi-qwen \
    data=libero-spatial \
    lr.base=5e-5 \
    +lr.vlm_bridge=1e-5 \
    +lr.action_expert=5e-5 \
    wandb.mode=online

# Evaluation
uv run scripts/eval_policy.py \
    checkpoint_path=hf:elijahgalahad/libero-spatial-qwen \
    data=libero-spatial \
    data.video_backend=pyav \
    merge_policy_cfg=true

# Serving
uv run scripts/serve_policy.py \
    checkpoint_path=hf:elijahgalahad/libero-spatial-qwen \
    data=libero-spatial \
    merge_policy_cfg=true
```

See [scripts/README.md](scripts/README.md) for more training commands. See [examples/libero](examples/libero/README.md) for LIBERO evaluation instructions.

## ✨ Key-Features

- **Explicit Data Model for Composable Modules** 
    - `TensorClass` is used to define explicit data boundaries between modules, making our codebase fully typed and modular. This allows heterogeneous dataset co-training, and [clearer data flow](vla_scratch/transforms/README.md) between datasets, policies, and transforms.
    ![data model](assets/data_model.png)
- **Dedicated Tuning for a First-Class Performance Stack** 
    - The [Qwen3-VL bridge](vla_scratch/policies/modules/vlm_bridge/qwen/bridge.py) rewrites the forward pass to [eliminate all host-device syncs](vla_scratch/policies/README.md).
    - Layer-wise FSDP sharding and gradient checkpointing reduces memory usage, making it easier to scale model parameters.
    ![performance](assets/performance-result.png)
- **Rich Feature Set Out-of-the-Box**
    - Multi-source dataset co-training: VQA and robotic datasets co-training.
    - Multi VLM backbone support: Qwen3-VL, PaliGemma 1/2, SmolVLM.
    - Simulation-Ready Serving Scripts.
- **Clarity-Focused Hydra Workflow for Seamless Experimentation** 
    - Every [policy](vla_scratch/policies/pi/config.py) and [data](vla_scratch/datasets/config.py) config is registered with Hydra's `ConfigStore`, so experiments are overrideable with minimal boilerplate.
    - Training, eval, and serving scripts share a common config grammar, so switching between workflows is seamless.

## 🗂️ Codebase Structure

VLA-Scratch is a fully modular, high-performance VLA stack built around TensorClass data models, hierarchical config system, and reusable training helpers.

| Path                                                 | Description                                  |
|------------------------------------------------------|----------------------------------------------|
| [`vla_scratch/transforms/`](vla_scratch/transforms/) | Data transforms and TensorClass models       |
| [`vla_scratch/datasets/`](vla_scratch/datasets/)     | Dataset loaders and transforms               |
| [`vla_scratch/policies/`](vla_scratch/policies/)     | Policy interfaces, bridges, and action heads |
| [`scripts/`](scripts/)                               | Training/eval/serving scripts                |
