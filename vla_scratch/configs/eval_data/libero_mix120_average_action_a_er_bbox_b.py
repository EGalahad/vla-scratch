from dataclasses import replace

from hydra.core.config_store import ConfigStore

from vla_scratch.datasets.config import EvalDataCfg, EvalDatasetCfg
from vla_scratch.datasets.libero.config import libero_er_bbox_config
from vla_scratch.datasets.libero_mix120.config import libero_mix_120_average_bbox_config


cs = ConfigStore.instance()

libero_mix120_average_action_a_er_bbox_b_eval_cfg = EvalDataCfg(
    datasets={
        # Evaluate action quality on dataset A.
        "action_a": EvalDatasetCfg(
            data=libero_mix_120_average_bbox_config,
            eval_fraction=0.05,
            eval_type="sample_mse",
        ),
        # Evaluate generation quality (CE/prompted decoding) on dataset B.
        "bbox_b": EvalDatasetCfg(
            data=replace(libero_er_bbox_config, bbox_only=True),
            eval_fraction=0.05,
            eval_type="generation",
        ),
    }
)

cs.store(
    name="libero-mix120-average-action_a-er-bbox_b",
    node=libero_mix120_average_action_a_er_bbox_b_eval_cfg,
    group="eval_data",
)

