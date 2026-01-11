# SimplerEnv Eval Example

Simulator env setup (`examples/bbox_cotrain/.venv`):
```bash
git clone git@github.com:EGalahad/BlindVLA.git ../BlindVLA
export BLINDVLA_ROOT=$(pwd)/../BlindVLA

uv sync --project examples/bbox_cotrain  # installs pyzmq/msgpack/gym etc.
source examples/bbox_cotrain/.venv/bin/activate
uv pip install -e $BLINDVLA_ROOT/ManiSkill
uv pip install -e $BLINDVLA_ROOT/SimplerEnv
```

Policy server (root venv):
```bash
uv run scripts/serve_policy.py \
    checkpoint_path=hf:elijahgalahad/checkpoint-action_a_bbox_ab
    data=bbox_cotrain_test \
    merge_policy_cfg=true
```

Sim client (adapt BlindVLA batched eval to call the ZMQ server):
```bash
source examples/bbox_cotrain/.venv/bin/activate
python examples/bbox_cotrain/simulation.py render=true sim_backend=cpu port=8000 obj_set=train
```
