# VL-Think policy rollouts

Simulator env (`examples/bbox_cotrain/.venv`):
```bash
uv sync --project examples/bbox_cotrain  # installs pyzmq/msgpack/gym etc.
source examples/bbox_cotrain/.venv/bin/activate

export BLINDVLA_ROOT=$(pwd)/../BlindVLA
uv pip install -e $BLINDVLA_ROOT/ManiSkill
uv pip install -e $BLINDVLA_ROOT/SimplerEnv

# (Optional) quick sanity check
python $BLINDVLA_ROOT/replay_lerobot.py --repo-id horipse01/lerobot_merged_restricted_val --episodes 0 1 2 --max-episodes 10  --env-id PutOnPlateInScene25MultiCarrot2-v1 --record_dir examples/bbox_cotrain/videos
```

Policy server (root venv):
```bash
source .venv/bin/activate
python scripts/serve_policy.py \
  policy=pi-qwen \
  policy.state_history=0 \
  policy.action_horizon=10 \
  policy.transforms.0.max_length=500 \
  data=bbox_cotrain_train \
  inference_steps=10 \
  checkpoint_path=hf:elijahgalahad/checkpoint-action_a_bbox_ab
```

Sim client (adapt BlindVLA batched eval to call the ZMQ server):
```bash
source examples/bbox_cotrain/.venv/bin/activate

# Single-env eval
python examples/bbox_cotrain/eval_bbox_cotrain.py render=true sim_backend=cpu port=8000 obj_set=train
```
