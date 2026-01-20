export HYDRA_FULL_ERROR=1

for arg in "$@"; do
    if [[ "$arg" == *=* ]]; then
        export "$arg"
    fi
done

DL_WORKERS=${DL_WORKERS:-8}
PREFETCH=${PREFETCH:-2}
NPROCS=${NPROCS:-8}

echo "  DL_WORKERS=$DL_WORKERS"
echo "  PREFETCH=$PREFETCH"
echo "  NPROCS=$NPROCS"
echo

uv run torchrun --standalone --nnodes=1 --nproc_per_node=$NPROCS \
    scripts/train_policy.py \
    policy=pi-paligemma \
    policy.state_history=0 \
    policy.action_horizon=30 \
    policy.transforms.0.max_length=560 \
    data=libero-spatial \
    batch_size=32 \
    eval_data=libero-spatial \
    num_workers=$DL_WORKERS \
    prefetch_factor=$PREFETCH \
    lr.base=5e-5 \
    +lr.vlm_bridge=1e-5 \
    +lr.action_expert=5e-5 \
    epochs=150 \
    save_interval=50 \
    wandb.mode=online \
