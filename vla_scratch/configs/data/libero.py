from hydra.core.config_store import ConfigStore

from vla_scratch.datasets.config import EvalDataCfg, EvalDatasetCfg
from vla_scratch.datasets.libero.config import libero_ipec_spatial_config, libero_spatial_config

cs = ConfigStore.instance()

libero_ipec_spatial_eval_cfg = EvalDataCfg(
    datasets={
        "spatial": EvalDatasetCfg(
            data=libero_ipec_spatial_config,
            eval_fraction=0.05,
            eval_type="sample_mse",
        )
    }
)

libero_spatial_eval_cfg = EvalDataCfg(
    datasets={
        "spatial": EvalDatasetCfg(
            data=libero_spatial_config,
            eval_fraction=0.05,
            eval_type="sample_mse",
        )
    }
)
cs.store(name="libero-ipec-spatial", node=libero_ipec_spatial_eval_cfg, group="eval_data")
cs.store(name="libero-spatial", node=libero_spatial_eval_cfg, group="eval_data")
