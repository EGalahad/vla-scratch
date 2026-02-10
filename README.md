# VLA-Scratch

[![PyTorch FSDP2](https://img.shields.io/badge/PyTorch-FSDP2-ee4c2c?logo=pytorch&logoColor=white)](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)
[![PyTorch TensorDict](https://img.shields.io/badge/PyTorch-TensorDict-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/tensordict/)
[![Qwen3-VL](https://img.shields.io/badge/Qwen3-VL-1d9bf0?logo=github&logoColor=white)](https://github.com/QwenLM/Qwen3-VL)
[![Checkpoints](https://img.shields.io/badge/Checkpoints-1d9bf0?logo=huggingface&logoColor=white)](https://huggingface.co/collections/elijahgalahad/vla-scratch)

VLA-Scratch is a modular, performant and efficient stack with minimal dependencies that makes training, evaluating, and serving Vision-Language-Action (VLA) models fast and approachable.

---

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

---

## ✨ Key-Features

- **Explicit Data Model for Composable Modules** 
    - `TensorClass` is used to define explicit data boundaries between modules, making our codebase fully typed and modular. 
    - This allows heterogeneous dataset co-training, and [clearer data flow](vla_scratch/transforms/README.md) between datasets, policies, and transforms.
    ![data model](assets/data_model.png)
- **Dedicated Tuning for a First-Class Performance Stack** 
    - We rewrite the forward pass of VLMs to [eliminate all host-device syncs](vla_scratch/policies/README.md).
    - Instead of relying on generic libaries like `accelerate`, `vla-scratch` leverage native `torch` operations like `FSDP2` and gradient checkpointing for dedicated performance tuning.
    ![performance](assets/performance-result.png)
- **Rich Feature Set Out-of-the-Box**
    - Multi-source dataset co-training: VQA and robotic datasets [co-training](examples/bbox_cotrain).
    - Multi VLM backbone support: [Qwen3-VL](vla_scratch/policies/modules/vlm_bridge/qwen/bridge.py), [PaliGemma 1/2](vla_scratch/policies/modules/vlm_bridge/paligemma/bridge.py), [SmolVLM](vla_scratch/policies/modules/vlm_bridge/smolvlm/bridge.py).
    - Simulation-ready checkpoints and [serving scripts](examples/).
- **Clarity-Focused Hydra Workflow for Seamless Experimentation** 
    - Every [policy](vla_scratch/policies/pi/config.py) and [data](vla_scratch/datasets/config.py) config is registered with Hydra, so experiments are overrideable with minimal boilerplate.
    - Training, eval, and serving scripts share a common config grammar, so switching between workflows is seamless.

---

## 🗂️ Codebase Structure

| Path                                                 | Description                                  |
|------------------------------------------------------|----------------------------------------------|
| [`vla_scratch/transforms/`](vla_scratch/transforms/) | Data transforms and TensorClass models       |
| [`vla_scratch/datasets/`](vla_scratch/datasets/)     | Dataset loaders and transforms               |
| [`vla_scratch/policies/`](vla_scratch/policies/)     | Policy interfaces, bridges, and action heads |
| [`scripts/`](scripts/)                               | Training/eval/serving scripts                |

---

## 🧭 Developer Guide

| Use Case               | Where to go                                                                                                                                                                                                                                                                                                                                                                                         |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Create a new dataset   | 1) Create a new folder, e.g. [`vla_scratch/datasets/<dataset_name>`](vla_scratch/datasets/).<br>2) Add implementation and configs (e.g. `dataset.py`, `config.py`).<br>3) Register the dataset config with Hydra `ConfigStore` in `config.py` under the `data` group.                                                                                                                               |
| Add a new VLM backbone | 1) Create a new folder under [`vla_scratch/policies/modules/vlm_bridge/`](vla_scratch/policies/modules/vlm_bridge/).<br>2) Implement preprocessing in `processor.py`, define the `policy_input` contract between preprocessing and encoding, and implement the main encoding in `bridge.py`.<br>3) Wire the backbone into [`vla_scratch/policies/pi/policy.py`](vla_scratch/policies/pi/policy.py). |
| Add a new loss term    | 1) Add the loss inside `compute_loss` in [`vla_scratch/policies/pi/policy.py`](vla_scratch/policies/pi/policy.py)<br>2) Return it in `log_dict` so it is logged to Weights & Biases.                                                                                                                                                                                                                |
| Format code            | Run `uvx ruff format`.                                                                                                                                                                                                                                                                                                                                                                              |


## 🛠️ Troubleshooting (To be Continued)

<details>
<summary><strong>RTX 5090 Compatibility</strong></summary>

> **Issue**  
> PyTorch stable has not yet fully supported **CUDA 12.8** for **RTX 5090**  
> (Blackwell-generation GPU, compute capability **sm_120**).  
> This may lead to **kernel launch failures** or **CUDA unavailable** errors.

> **Solution**  
> Use [PyTorch-Nightly](https://forums.developer.nvidia.com/t/software-migration-guide-for-nvidia-blackwell-rtx-gpus-a-guide-to-cuda-12-8-pytorch-tensorrt-and-llama-cpp/321330) instead of stable PyTorch.  

Add the following to your `<pyproject.toml>`:

    ```
    [[tool.uv.index]]
    name = "pytorch-nightly-cu128"
    url = "https://download.pytorch.org/whl/nightly/cu128"
    default = true

    [[tool.uv.index]]
    url = "https://pypi.org/simple"

    [tool.uv.sources]
    torch = { index = "pytorch-nightly-cu128" }
    torchvision = { index = "pytorch-nightly-cu128" }
    triton = { index = "pytorch-nightly-cu128" }

    ```