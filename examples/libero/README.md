# LIBERO policy rollouts

Simulator env (`examples/libero/.venv`):
```bash
export LIBERO_ROOT=$(pwd)/../LIBERO
# uv venv --python 3.10 examples/libero/.venv
uv sync --project examples/libero 
source examples/libero/.venv/bin/activate
uv pip install -r $LIBERO_ROOT/requirements.txt
uv pip install -e $LIBERO_ROOT
export PYTHONPATH=$PYTHONPATH:$LIBERO_ROOT
```

Policy server (root venv):
```bash
source .venv/bin/activate
python scripts/serve_policy.py \
  policy=pi-qwen \
  policy.state_history=1 \
  policy.action_horizon=20 \
  data=libero-ipec-spatial \
  checkpoint_path=hf:elijahgalahad/libero_policy
```

Sim client:
```bash
source examples/libero/.venv/bin/activate
export LIBERO_ROOT=$(pwd)/../LIBERO
export PYTHONPATH=$PYTHONPATH:$LIBERO_ROOT
python examples/libero/eval_libero_policy.py host=127.0.0.1 port=8000 libero_task_suite=libero_spatial headless=false action_chunk_size=5 episodes_per_task=10
# Show windowed rendering: headless=false (optionally render_every=N)
# Use absolute / global OSC commands: control_delta=false
# If EGL issues: MUJOCO_GL=glx python examples/libero/eval_libero_policy.py ...
```

The eval script sends training-aligned obs keys (`lerobot_ipec`), receives action chunks, and applies the first step in the LIBERO env.
