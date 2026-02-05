from hydra.core.config_store import ConfigStore

from vla_scratch.datasets.config import EvalDataCfg, EvalDatasetCfg
from vla_scratch.datasets.libero_mix120.config import (
    libero_mix_120_average_bbox_config,
    libero_mix_120_average_config,
)


cs = ConfigStore.instance()

libero_mix_120_average_eval_cfg = EvalDataCfg(
    datasets={
        "libero_mix_120_average": EvalDatasetCfg(
            data=libero_mix_120_average_config,
            eval_fraction=0.05,
            eval_type="sample_mse",
        ),
    }
)

libero_mix_120_average_bbox_eval_cfg = EvalDataCfg(
    datasets={
        "libero_mix_120_average": EvalDatasetCfg(
            data=libero_mix_120_average_bbox_config,
            eval_fraction=0.05,
            eval_type="sample_mse",
        ),
    }
)

cs.store(name="libero-mix-120-average", node=libero_mix_120_average_eval_cfg, group="eval_data")
cs.store(
    name="libero-mix-120-average-bbox",
    node=libero_mix_120_average_bbox_eval_cfg,
    group="eval_data",
)

