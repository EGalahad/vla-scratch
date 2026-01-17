# LIBERO Eval Example

## 🚀 Policy Serving
```bash
uv run scripts/serve_policy.py \
    checkpoint_path=hf:<checkpoint_id> \
    data=libero-spatial \
    merge_policy_cfg=true
```
More training/serving commands live in `examples/libero/scripts/`.

Pretrained checkpoints: [wandb runs](https://wandb.ai/elijahgalahad/vla-scratch/workspace?nw=iztnlef3txj)

| Huggingface Id                                                                                            | Gradient Steps | Run Time | Success Rate |
|-----------------------------------------------------------------------------------------------------------|----------------|----------|--------------|
| [`elijahgalahad/libero-spatial-qwen`](https://huggingface.co/elijahgalahad/libero-spatial-qwen)           | 10k            | 2h 58m   | 94%          |
| [`elijahgalahad/libero-spatial-paligemma`](https://huggingface.co/elijahgalahad/libero-spatial-paligemma) | 10k            | 4h 14m   | 88%          |
| [`elijahgalahad/libero-spatial-smolvlm`](https://huggingface.co/elijahgalahad/libero-spatial-smolvlm)     | 10k            | 2h 18m   | 76%          |

## 🤖 Simulation Environment

Set up simulation virtual environment (`examples/libero/.venv`):
```bash
# 1) Clone LIBERO repository
git clone git@github.com:Lifelong-Robot-Learning/LIBERO.git ../LIBERO
export LIBERO_ROOT=$(pwd)/../LIBERO

# 2) Install dependencies with uv
uv sync --project examples/libero
source examples/libero/.venv/bin/activate
uv pip install -r $LIBERO_ROOT/requirements.txt
uv pip install -e $LIBERO_ROOT
export PYTHONPATH=$PYTHONPATH:$LIBERO_ROOT
```

Run the simulation with policy client:
```bash
source examples/libero/.venv/bin/activate
export LIBERO_ROOT=$(pwd)/../LIBERO
export PYTHONPATH=$PYTHONPATH:$LIBERO_ROOT

python examples/libero/simulation.py \
    host=127.0.0.1 port=8000 \
    libero_task_suite=libero_spatial \
    headless=false \
    action_chunk_size=5 \
    episodes_per_task=10
```
