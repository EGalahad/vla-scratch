# Datasets

Datasets define how raw demonstrations are materialized, normalized, and converted into the `DataSample` observations consumed by policies. 

## Files at a Glance

| Path | Description |
| --- | --- |
| `config.py` | Dataset config dataclass and `ConfigStore` registrations. |
| `<dataset_name>/` | Dataset-specific loaders, transforms, config, and data keys. |

## Core Execution Flow

1. Hydra instantiates a `DataConfig` via `create_dataset` in `vla_scratch/helpers/data.py`.
2. The factory resolves the dataset class (`_target_`) and wiring (transforms, normalization) defined in the config.

For the detailed explanation of dataset transforms and data flow, please refer to the [Transforms README](../../vla_scratch/transforms/README.md).

## Adding a dataset
1. Create `vla_scratch/datasets/<name>/config.py` with a dataclass inheriting `DataConfig`. 
2. Implement the dataset and any private transforms in the same package.
3. Register the config with `ConfigStore` under `group="data"`. Point `_target_` to your custom dataset class.
4. Define the file path you would like to save the normalization stats (e.g. `normalization_stats/libero/IPEC-COMMUNITY/libero_spatial-horizon_{data.action_horizon}-history_{data.state_history}.npz`).
5. Run training with `data=<name>` to use your dataset.
