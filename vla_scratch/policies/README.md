# Policies Folder Layout

## Files at a Glance

| Path                           | Description                                                 |
|--------------------------------|-------------------------------------------------------------|
| `base.py`                      | `BasePolicy` interface.                                     |
| `config.py`                    | Hydra `PolicyConfig` definitions and registrations.         |
| `utils/transformers.py`        | Shared math helpers (noise, rotary, sinusoidal embeddings). |
| `utils/training.py`            | FSDP + gradient checkpointing helpers for training.         |
| `modules/action_expert/dit.py` | Diffusion action head shared across policies.               |
| `modules/vlm_bridge/*`         | VLM utilities for Paligemma, Qwen, etc.   |
| `pi/`                          | Pi policy implementation (config + model).        |


## Core Execution Flow

1. Policies expose `encode_prefix` and `predict_suffix`; `compute_loss` and `sample_actions` simply compose those primitives during training/serving.
2. `encode_prefix` calls the chosen VLM bridge, which runs the vision-language encoder and caches its activations as [`VLMOutputs`](../../vla_scratch/policies/modules/vlm_bridge/data_types.py).
3. `predict_suffix` consumes `VLMOutputs`, denoises a Gaussian sample via the action expert head, and produces articulated actions.
4. Layer-wise FSDP sharding and gradient checkpointing are attached through helpers in `utils/training.py`.
5. Each VLM bridge expects `Observation.policy_input` to be prepared by its matching processor (e.g., `QwenBridge` with `QwenProcessor`), ensuring modality-specific tensors are present before the forward pass.

## Optimizations

Most performance gains come from spotting CUDA↔CPU syncs and eliminating them via three strategies:

1. **Move “bookkeeping” into the processor.** If the computation is sequential / token-by-token, but doesn’t depend on any model activations, it belongs in the data pipeline. In Qwen, the mRoPE indices fall into this bucket: we compute `position_ids` and `mrope_position_deltas` during preprocessing (see `vla_scratch/policies/modules/vlm_bridge/qwen/processor.py:QwenProcessor.get_rope_index`) so batches arrive at the policy already fully annotated.

2. **Keep dynamic shape metadata reside on CPU.** The syncs that remain are usually caused by allocating tensors whose size depends on values of CUDA tensors. With Qwen3-VL, operations like `fast_pos_embed_interpolate` depend on the values in `image_thw` CUDA tensor. The fix is to pass CPU-native shapes (`image_grid_thw_list`) as Python tuples inside the policy input (`modules/vlm_bridge/qwen/processor.py:QwenPolicyInput`), and then route the forward through the optimized path that consumes that list (`modules/vlm_bridge/qwen/utils.py:_qwen3vl_fast_pos_embed_interpolate`).

3. **Make the forward pass “shape-stable”.** One more trick is avoiding variable-sized intermediates when injecting deepstack visual features. The classic pattern is:
    ```python
    local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
    hidden_states[visual_pos_masks, :] = local_this
    ```
            
    it can be optimized to
    ```python
    delta = torch.zeros_like(hidden_states)
    delta.masked_scatter_(visual_pos_masks.unsqueeze(-1), visual_embeds)
    hidden_states.add_(delta)
    ```

    Because `local_this` has a shape that depends on `visual_pos_masks.sum()`, and producing it can trigger a sync when the mask lives on the GPU. In our bridge, we follow the “scatter-then-add” approach for deepstack features (see `vla_scratch/policies/modules/vlm_bridge/qwen/bridge.py:Qwen3VLBridge.encode`).


## When adding a new Policy
1. Define its config under `policies/<name>/config.py` and register it with Hydra.
2. Implement the policy model (subclass `BasePolicy`). Reuse modules/bridges/utilities where possible.
3. Add any policy-specific processors under `modules/...` if they will be shared, or locally under the policy if truly bespoke.
