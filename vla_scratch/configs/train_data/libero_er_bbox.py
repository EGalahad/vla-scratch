from hydra.core.config_store import ConfigStore

from vla_scratch.datasets.config import TrainDataCfg, TrainDatasetCfg
from vla_scratch.datasets.libero.config import libero_er_bbox_config


cs = ConfigStore.instance()

libero_er_bbox_train_cfg = TrainDataCfg(
    datasets={
        "libero_er": TrainDatasetCfg(data=libero_er_bbox_config, batch_size=4),
    }
)

cs.store(
    name="libero-er-bbox",
    node=libero_er_bbox_train_cfg,
    group="train_data",
)

