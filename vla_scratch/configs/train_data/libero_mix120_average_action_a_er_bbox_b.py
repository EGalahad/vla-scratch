from dataclasses import replace

from hydra.core.config_store import ConfigStore

from vla_scratch.datasets.config import TrainDataCfg, TrainDatasetCfg
from vla_scratch.datasets.libero.config import libero_er_bbox_config
from vla_scratch.datasets.libero_mix120.config import libero_mix_120_average_bbox_config


cs = ConfigStore.instance()

libero_mix120_average_action_a_er_bbox_b_train_cfg = TrainDataCfg(
    datasets={
        # Dataset A: action supervision + CE (bbox prompts) supervision.
        "action_a": TrainDatasetCfg(
            data=libero_mix_120_average_bbox_config,
            batch_size=16,
        ),
        # Dataset B: CE-only supervision (no actions returned when bbox_only=True).
        "bbox_b": TrainDatasetCfg(
            data=replace(libero_er_bbox_config, bbox_only=True),
            batch_size=16,
        ),
    }
)

cs.store(
    name="libero-mix120-average-action_a-er-bbox_b",
    node=libero_mix120_average_action_a_er_bbox_b_train_cfg,
    group="train_data",
)

