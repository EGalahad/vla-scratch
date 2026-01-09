<div align="left">
  <h1>
  VLA-Scratch: 
  <!-- <br/> -->
  a Modular, Performant, Efficient Stack<br/>
  For Vision-Language-Action Models
  </h1>
</div>
<!-- add a huggingface badge, a twitter badge and a github star badge -->

## 🚀 Quickstart

Setup with `uv sync`. Verify your installation with the following commands:

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
uv run \
    policy=pi-qwen \
    policy.state_history=1 \
    policy.action_horizon=30 \
    policy.transforms.0.max_length=500 \
    data=libero-spatial \
    data.video_backend=pyav \
    batch_size=16 \
    checkpoint_path=hf:elijahgalahad/libero_policy
```


See [instructions](INSTRUCTIONS.md) for more training and evaluation commands.
## ✨ Key-Features

- **Explicit Data Model for Composable Modules** 
    - `TensorClass` is used to define explicit data boundaries between modules, making our codebase fully typed and modular. This allows heterogeneous dataset co-training, and [clearer data flow](vla_scratch/transforms/README.md) between datasets, policies, and transforms.
    ![data model](assets/data_model.png)
- **Dedicated Tuning for a First-Class Performance Stack** 
    - The [Qwen3-VL bridge](vla_scratch/policies/modules/vlm_bridge/qwen/bridge.py) rewrites the forward path to [minimize host-device syncs](vla_scratch/policies/README.md), making throughput 2x higher.
    - Layer-wise FSDP sharding and gradient checkpointing saves memory up to 2x, making it easier to scale model parameters.
    - TODO: figure here.
- **Clarity-Focused Hydra Workflow for Seamless Experimentation** 
    - Every [policy](vla_scratch/policies/pi/config.py) and [data](vla_scratch/datasets/config.py) config is registered with Hydra's `ConfigStore`, so experiments are overrideable with minimal boilerplate.
    - Training, eval, and serving scripts share a common config grammar, so switching between workflows is seamless.
    - TODO: example script snippet here.


## 🎯 Capabilities

TODO: Heterogeneous dataset co-training (VQA, Action), Multi VLM backbone, Simulation-ready Serving scripts, feature rich visualizations, etc.

## 🗂️ Codebase Structure

VLA-Scratch is a fully modular, high-performance VLA stack built around Hydra configs, TensorClass data models, and reusable training helpers.

| Path | Description |
| --- | --- |
| [`vla_scratch/transforms/`](vla_scratch/transforms/) | Data transforms and TensorClass models |
| [`vla_scratch/datasets/`](vla_scratch/datasets/) | Dataset loaders and transforms |
| [`vla_scratch/policies/`](vla_scratch/policies/) | Policy interfaces, bridges, and action heads |
| [`scripts/`](scripts/) | Training/eval/serving scripts |