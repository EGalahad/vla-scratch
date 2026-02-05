from hydra.core.config_store import ConfigStore

from vla_scratch.datasets.config import EvalDataCfg, EvalDatasetCfg
from vla_scratch.datasets.libero.config import libero_er_bbox_config


cs = ConfigStore.instance()

libero_er_bbox_eval_cfg = EvalDataCfg(
    datasets={
        "libero_er": EvalDatasetCfg(
            data=libero_er_bbox_config,
            eval_fraction=0.05,
            eval_type="sample_mse",
        ),
    }
)

cs.store(
    name="libero-er-bbox",
    node=libero_er_bbox_eval_cfg,
    group="eval_data",
)

