from hydra.core.config_store import ConfigStore

from vla_scratch.datasets.config import EvalDataCfg, EvalDatasetCfg
from vla_scratch.datasets.libero.config import (
    libero_90_bbox_config,
    libero_goal_bbox_config,
    libero_object_bbox_config,
    libero_spatial_bbox_config,
)


cs = ConfigStore.instance()

libero_mix120_bbox_eval_cfg = EvalDataCfg(
    datasets={
        "libero90": EvalDatasetCfg(
            data=libero_90_bbox_config,
            eval_fraction=0.05,
            eval_type="sample_mse",
        ),
        "libero_goal": EvalDatasetCfg(
            data=libero_goal_bbox_config,
            eval_fraction=0.05,
            eval_type="sample_mse",
        ),
        "libero_object": EvalDatasetCfg(
            data=libero_object_bbox_config,
            eval_fraction=0.05,
            eval_type="sample_mse",
        ),
        "libero_spatial": EvalDatasetCfg(
            data=libero_spatial_bbox_config,
            eval_fraction=0.05,
            eval_type="sample_mse",
        ),
    }
)

cs.store(
    name="libero-mix120-bbox",
    node=libero_mix120_bbox_eval_cfg,
    group="eval_data",
)

