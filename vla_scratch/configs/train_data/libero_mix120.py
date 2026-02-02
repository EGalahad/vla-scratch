from hydra.core.config_store import ConfigStore

from vla_scratch.datasets.config import TrainDataCfg, TrainDatasetCfg
from vla_scratch.datasets.libero.config import (
    libero_90_config,
    libero_goal_config,
    libero_object_config,
    libero_spatial_config,
)


cs = ConfigStore.instance()

libero_mix120_train_cfg = TrainDataCfg(
    datasets={
        "libero90": TrainDatasetCfg(data=libero_90_config, batch_size=4),
        "libero_goal": TrainDatasetCfg(
            data=libero_goal_config, batch_size=4
        ),
        "libero_object": TrainDatasetCfg(
            data=libero_object_config, batch_size=4
        ),
        "libero_spatial": TrainDatasetCfg(
            data=libero_spatial_config, batch_size=4
        ),
    }
)

cs.store(name="libero-mix120", node=libero_mix120_train_cfg, group="train_data")
