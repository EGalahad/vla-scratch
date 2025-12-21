# Transform Pipeline & Data Keys

This repo keeps explicit data boundary by enforcing a common transform pipeline.

The core data structure is the `DataSample` defined in [vla_scratch/transforms/data_types.py](data_types.py).
```
DataSample
  ├── Observation
  │     ├── images
  │     ├── image_masks
  │     ├── state
  │     ├── task
  │     └── policy_input  <--- policy-specific inputs go here
  └── ActionChunk
        └── actions
```

## Files at a Glance

| File               | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `data_keys.py`     | Canonical keys for processed data used across datasets and policies.        |
| `data_types.py`    | TensorClass definitions for `DataSample`, `Observation`, and `ActionChunk`. |
| `common.py`        | Shared transform utilities (e.g. `ToDataSample`).                           |
| `normalization.py` | Normalization and denormalization transforms for state/action tensors.      |

## Core Execution Flow
The full transform pipeline is composed of four stages:

```
pipeline = dataset_transforms + [Normalize()] + [ToDataSample()] + policy_transforms
```

1. **Dataset transforms** live beside the dataset (e.g. [vla_scratch/datasets/libero/transforms.py](../../vla_scratch/datasets/libero/transforms.py)), to produce [canonical processed keys](../../vla_scratch/transforms/data_keys.py).
2. **Normalization transforms** normalizes the state/action tensors.
3. **`ToDataSample()`** converts the key-value dict into the structured `DataSample`.
4. **Policy transforms** live beside each policy (e.g. [vla_scratch/policies/modules/vlm_bridge/qwen/processor.py](../../vla_scratch/policies/modules/vlm_bridge/qwen/processor.py)) to emit whatever policy-specific processing and writes into `Observation.policy_input`.

The keys in and the type of the data is progressively transforms along the pipeline:
- **Dataset keys**: each dataset declares its own keys (e.g. `vla_scratch/datasets/libero/data_keys.py`). These names describe the raw modalities coming from the source data (`CAM_FRONT_KEY`, `ARM_CMD_CART_POS_KEY`, ...). Dataset transforms expect these keys.
- **Processed keys**: once the dataset transforms is done,, they store them under the [canonical processed keys](../../vla_scratch/transforms/data_keys.py).
- **Policy-input**: after `ToDataSample()` the policy-specific transforms can attach any additional structures (e.g. Paligemma/Qwen tensor classes). Those keys are internal to the policy and never reused outside.

Data evolution diagram

| Stage                    | Output representation                                                       |
|--------------------------|-----------------------------------------------------------------------------|
| Raw dataset              | Dict keyed per dataset (e.g., `CAM_FRONT_KEY`, `ARM_CMD_CART_POS_KEY`).     |
| Dataset transforms       | Canonical processed keys (`PROCESSED_STATE_KEY`, `PROCESSED_ACTION_KEY`, …) |
| `Normalize()` | Canonical processed keys (normalized)                                       |
| `ToDataSample()`         | `DataSample` TensorClass (`Observation`, `ActionChunk`)                     |
| Policy transforms        | `DataSample` with `Observation.policy_input` populated                      |
