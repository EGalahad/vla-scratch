from hydra.core.config_store import ConfigStore

from vla_scratch.datasets.config import EvalDataCfg, EvalDatasetCfg
from vla_scratch.datasets.droid.config import droid_config


droid_eval_cfg = EvalDataCfg(
    datasets={
        "droid": EvalDatasetCfg(
            data=droid_config,
            eval_fraction=0.05,
            eval_type="sample_mse",
        )
    }
)

cs = ConfigStore.instance()
cs.store(name="droid", node=droid_eval_cfg, group="eval_data")
