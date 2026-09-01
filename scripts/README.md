## Training
```bash
# LIBERO with Qwen3-VL
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_policy.py \
    policy=pi-qwen \
    policy.state_history=0 \
    policy.action_horizon=30 \
    policy.use_state=False \
    policy.transforms.0.max_length=180 \
    data=libero-spatial \
    eval_data=libero-spatial \
    lr.base=5e-5 \
    +lr.vlm_bridge=1e-5 \
    +lr.action_expert=5e-5 \
    wandb.mode=online

# LIBERO with PaliGemma
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_policy.py \
    policy=pi-paligemma \
    policy.state_history=0 \
    policy.action_horizon=30 \
    policy.use_state=False \
    policy.transforms.0.max_length=550 \
    data=libero-spatial \
    eval_data=libero-spatial \
    lr.base=5e-5 \
    +lr.vlm_bridge=1e-5 \
    +lr.action_expert=5e-5 \
    wandb.mode=online
```

### π0.5 VLM + DiT

`policy=pi05` uses a dedicated π0.5 bridge, processor, and numerical forward.
It automatically downloads the pinned `lerobot/pi05_base` safetensors through
the standard Hugging Face cache and strictly loads only its pretrained VLM
tensors. The action expert remains this repository's randomly initialized
cross-attention DiT; it is not the π0.5 joint-attention action expert.

LIBERO camera order is front then wrist. For another dataset, set the
processor transform's `front_camera_indices` to every external/front camera;
all remaining camera indices use wrist-camera augmentation.

Set `HF_HOME` if the default cache location is unsuitable. No weight path,
OpenPI checkout, JAX runtime, or conversion command is required at training
time. Missing, unexpected, or shape-mismatched VLM keys are rejected; action
expert tensors in the base checkpoint are ignored.

```bash
# global batch = 32 samples/rank * 8 ranks = 256
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_policy.py \
    policy=pi05 \
    data=libero-spatial-ipec \
    data.root=/path/to/libero_spatial_ipec_v3 \
    batch_size=32 \
    epochs=1000 \
    max_steps=10000 \
    exp_name=pi05-vlm-dit-libero-spatial-10k \
    warmup_steps=1000 \
    lr.base=5e-5 \
    +lr.vlm_bridge=1e-5 \
    +lr.action_expert=5e-5 \
    betas='[0.9,0.95]' \
    weight_decay=1e-10 \
    clip_grad_norm=1.0 \
    num_workers=8 \
    wandb.mode=online
```

`max_steps=10000` is an optimizer-step limit and writes
`checkpoint_step_10000`, even when the limit falls in the middle of an epoch.
The IPEC v2.1 dataset is converted to LeRobot v3 for this repository; OpenPI
and RLinf consume the same episodes in their original v2.1 format.

Implementation checks cover strict VLM-only loading, local OpenPI-compatible
SigLIP/Gemma forward shapes, warmup, sampler epochs, gradient accumulation,
and full-precision checkpoint resume. Hardware acceptance is still required:
the official checkpoint/JAX numerical comparison, 8-GPU 200-step throughput
run, and 30k-step LIBERO Spatial evaluation (current reference: 91/100).

### Upload Checkpoints

The training output structure is:

```
outputs/<YYYY-MM-DD>/<HH-MM-SS>-<exp_name>/
├── cfg.yaml                    # training config snapshot
└── checkpoint_<epoch>/         # checkpoint directory per save interval
    ├── model.pt                # model weights
    └── optimizer.pt            # optimizer state (excluded from upload)
```

Use the command below to upload the latest checkpoint (automatically picks the highest-numbered `checkpoint_*`) and `cfg.yaml` to Hugging Face:

```bash
uv run bash scripts/helpers/upload_checkpoint.sh <user_name/repo_name> <checkpoint_path>
```

## Evaluation
See [examples](../examples/README.md) for details about evaluation in LIBERO and other simulation environments.
