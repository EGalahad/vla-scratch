# Scripts

Hydra entry points in this directory orchestrate training, evaluation, and tooling around VLA policies. All scripts accept the same `data=` and `policy=` overrides used in `scripts/train_policy.py`, so you can mix and match datasets and policies without editing Python.

## Core entry points
- `train_policy.py` — distributed trainer that wires `vla_scratch.helpers.training` utilities, Layerwise FSDP, and gradient checkpointing. It builds dataloaders via `create_dataloaders`, resolves Hydra configs, and owns logging/checkpointing policy.
- `eval.py` — lightweight evaluation loop that mirrors the training config grammar, instantiates datasets through `create_dataset`, and reports MSE over configurable subsets/rollouts.
- `serve_policy.py` — runs a policy in inference-only mode, constructing the same observation TensorClasses used during training so you can export checkpoints straight into serving.
- `compute_norm_stats.py` — hydrates datasets, streams samples through the transform stack, and records normalization statistics expected by `Normalize`/`DeNormalize`.

<!-- ## Tooling and diagnostics
- `inspect_checkpoint_diff.py` compares checkpoints produced by different runs and prints tensor-level diffs (useful for smoke-testing FSDP checkpointing).
- `add_detect_label_with_vlm.py` and `merge_bbox_jsonl.py` provide data curation helpers for grounding/vision-language annotation workflows.
- `ablations/` hosts experimental entry points (e.g., `libero_interleave_pos_emb`) for rapid prototyping without perturbing the main training script. -->

### Usage pattern
All scripts are invoked via `uv run python scripts/<name>.py <overrides>`. For example:

```bash
uv run python scripts/train_policy.py policy=pi data=libero-ipec \
    policy.state_history=10 policy.action_horizon=30 \
    wandb.mode=online
```

To add a new script, follow these practices:
1. Register typed Hydra configs next to the script (see `train_policy.py`/`eval.py` for templates).
2. Import shared helpers from `vla_scratch.helpers`/`vla_scratch.utils` rather than re-implementing infrastructure.
3. Accept `policy`/`data` overrides so downstream automation stays uniform.
