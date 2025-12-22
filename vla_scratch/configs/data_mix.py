from dataclasses import replace

from hydra.core.config_store import ConfigStore

from vla_scratch.datasets.config import EvalDataCfg, EvalDatasetCfg, TrainDataCfg, TrainDatasetCfg
from vla_scratch.datasets.dont_blind.config import (
    default_dont_blind_config,
    dont_blind_8_8_objects_config_train,
    dont_blind_8_8_objects_config_test,
)

cs = ConfigStore.instance()

dont_blind_8_8_eval_cfg = EvalDataCfg(
    datasets={
        "action_train": EvalDatasetCfg(
            data=dont_blind_8_8_objects_config_train,
            eval_fraction=0.05,
            eval_type="sample_mse",
        ),
        "action_test": EvalDatasetCfg(
            data=dont_blind_8_8_objects_config_test,
            eval_fraction=0.05,
            eval_type="sample_mse",
        ),
        "generation_train": EvalDatasetCfg(
            data=replace(dont_blind_8_8_objects_config_train, bbox_only=True),
            eval_fraction=0.1,
            eval_type="generation",
        ),
        "generation_test": EvalDatasetCfg(
            data=replace(dont_blind_8_8_objects_config_test, bbox_only=True),
            eval_fraction=0.1,
            eval_type="generation",
        ),
    }
)
cs.store(name="dont_blind_8_8_eval", node=dont_blind_8_8_eval_cfg, group="eval_data")

dont_blind_8_8_train_mix_cfg = TrainDataCfg(
    datasets={
        "action_a": TrainDatasetCfg(
            data=dont_blind_8_8_objects_config_train,
            batch_size=16,
        ),
        "bbox_ab": TrainDatasetCfg(
            data=replace(default_dont_blind_config, bbox_only=True),
            batch_size=16,
        ),
    }
)
cs.store(
    name="dont_blind_8_8_train_mix",
    node=dont_blind_8_8_train_mix_cfg,
    group="train_data",
)
