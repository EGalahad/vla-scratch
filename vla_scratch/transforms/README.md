# Transform Pipeline & Data Keys

This repo keeps data processing predictable by enforcing a common transform pipeline and a consistent key taxonomy.

data types
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

## Transform pipeline
Every dataset config builds transforms in `vla_scratch/helpers/data.py` using the following recipe:

```
pipeline = dataset_transforms + normalization_transforms + [ToDataSample()] + policy_transforms
```

1. **Dataset transforms** live beside the dataset (e.g. `vla_scratch/datasets/libero/transforms.py`). They consume the raw values emitted by `__getitem__` and reshape / convert them into standardized tensors using dataset-specific keys.
2. **Normalization transforms** (`Normalize` / `DeNormalize`) act on the processed tensors defined in `vla_scratch/transforms/data_keys.py` (such as `PROCESSED_STATE_KEY`, `PROCESSED_ACTION_KEY`, etc.). Stats come from the dataset config.
3. **`ToDataSample()`** converts the key-value dict into the structured `DataSample` tensor class shared across policies.
4. **Policy transforms** are declared in each policy config (e.g. VLM processors) to emit whatever extra tensors the policy expects.

## Key conventions
- **Dataset keys**: each dataset declares its own keys (e.g. `vla_scratch/datasets/libero/data_keys.py`). These names describe the raw modalities coming from the source data (`CAM_FRONT_KEY`, `ARM_CMD_CART_POS_KEY`, ...). Dataset transforms expect and emit these keys.
- **Processed keys**: once the dataset transforms have assembled tensors ready for normalization, they store them under the canonical keys defined in `vla_scratch/transforms/data_keys.py` (`PROCESSED_STATE_KEY`, `PROCESSED_ACTION_KEY`, `PROCESSED_IMAGE_KEY`, ...). Only these keys are normalized / denormalized.
- **Policy-input keys**: after `ToDataSample()` the policy-specific transforms can attach any additional structures (e.g. Paligemma/Qwen tensor classes). Those keys are internal to the policy and never reused outside.

By keeping dataset-specific keys separate from the processed keys, we can freely combine datasets and policies while reusing normalization/statistics code.

Key evolution diagram

```
raw dataset (dataset-specific keys)
      │   e.g. libero: CAM_FRONT_KEY, ARM_CMD_CART_POS_KEY, ...
      ▼
Dataset transforms (`vla_scratch/datasets/<name>/transforms.py`)
      │   populate canonical processed keys:
      │   PROCESSED_STATE_KEY, PROCESSED_ACTION_KEY, PROCESSED_IMAGE_KEY, ...
      ▼
Normalization transforms (`Normalize` / `DeNormalize`)
      │   operate only on PROCESSED_* keys
      ▼
ToDataSample()
      │   builds Observation + ActionChunk tensor classes:
      │   Observation.images, Observation.state, ActionChunk.actions, ...
      ▼
Policy transforms (declared in policy config)
      │   attach policy-specific inputs (Observation.policy_input)
      ▼
Policy forward
```
