from dataclasses import replace
from hydra.core.config_store import ConfigStore

from vla_scratch.datasets.config import (
    EvalDataCfg,
    EvalDatasetCfg,
    TrainDataCfg,
    TrainDatasetCfg,
)
from vla_scratch.datasets.bbox_cotrain.config import (
    train_cotrain_config,
    test_cotrain_config,
)

cs = ConfigStore.instance()


cotrain_eval_cfg = EvalDataCfg(
    datasets={
        "action_train": EvalDatasetCfg(
            data=train_cotrain_config,
            eval_fraction=0.02,
            eval_type="sample_mse",
        ),
        "action_test": EvalDatasetCfg(
            data=test_cotrain_config,
            eval_fraction=0.1,
            eval_type="sample_mse",
        ),
        "generation_train": EvalDatasetCfg(
            data=replace(train_cotrain_config, bbox_only=True),
            eval_fraction=0.02,
            eval_type="generation",
        ),
        "generation_test": EvalDatasetCfg(
            data=replace(test_cotrain_config, bbox_only=True),
            eval_fraction=0.1,
            eval_type="generation",
        ),
    }
)
test_eval_cfg = EvalDataCfg(
    datasets={
        "action_test": EvalDatasetCfg(
            data=test_cotrain_config,
            eval_fraction=0.1,
            eval_type="sample_mse",
        ),
        "generation_test": EvalDatasetCfg(
            data=replace(test_cotrain_config, bbox_only=True),
            eval_fraction=0.1,
            eval_type="generation",
        ),
    }
)
cs.store(name="bbox_cotrain_eval", node=cotrain_eval_cfg, group="eval_data")
cs.store(name="bbox_test_eval", node=test_eval_cfg, group="eval_data")

cotrain_mix_cfg = TrainDataCfg(
    datasets={
        "action_a": TrainDatasetCfg(
            data=train_cotrain_config,
            batch_size=16,
        ),
        "bbox_b": TrainDatasetCfg(
            data=replace(test_cotrain_config, bbox_only=True),
            batch_size=16,
        ),
    }
)
cotrain_baseline_cfg = TrainDataCfg(
    datasets={
        "action_a": TrainDatasetCfg(
            data=train_cotrain_config,
            batch_size=16,
        ),
    }
)

cs.store(name="bbox_cotrain_mix", node=cotrain_mix_cfg, group="train_data")
cs.store(
    name="bbox_cotrain_baseline", node=cotrain_baseline_cfg, group="train_data"
)
