# Simulation Benchmark Examples

## Files at a Glance

| Path            | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `bbox_cotrain/` | BBox co-training rollouts using BlindVLA + ManiSkill with a ZMQ policy RPC. |
| `libero/`       | LIBERO policy rollouts with windowed rendering and action chunking.         |

## Workflow

1. Start the policy server on the remote GPU machine (see the subfolder README for flags).
2. Use VS Code port forwarding to map the remote policy port to local port `8000`.
3. Run the simulator client on your local desktop/laptop and connect to `localhost:8000` so the simulation renders locally.
