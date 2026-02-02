from hydra.core.config_store import ConfigStore

from vla_scratch.datasets.config import EvalDataCfg, EvalDatasetCfg
from vla_scratch.datasets.libero.config import (
    libero_90_config,
    libero_goal_config,
    libero_object_config,
    libero_spatial_config,
)


cs = ConfigStore.instance()

libero_mix120_eval_cfg = EvalDataCfg(
    datasets={
        "libero90": EvalDatasetCfg(
            data=libero_90_config, eval_fraction=0.05, eval_type="sample_mse"
        ),
        "libero_goal": EvalDatasetCfg(
            data=libero_goal_config,
            eval_fraction=0.05,
            eval_type="sample_mse",
        ),
        "libero_object": EvalDatasetCfg(
            data=libero_object_config,
            eval_fraction=0.05,
            eval_type="sample_mse",
        ),
        "libero_spatial": EvalDatasetCfg(
            data=libero_spatial_config,
            eval_fraction=0.05,
            eval_type="sample_mse",
        ),
    }
)

cs.store(name="libero-mix120", node=libero_mix120_eval_cfg, group="eval_data")
