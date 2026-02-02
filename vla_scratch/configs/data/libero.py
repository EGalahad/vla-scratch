from dataclasses import replace
from hydra.core.config_store import ConfigStore

from vla_scratch.datasets.config import (
    EvalDataCfg,
    EvalDatasetCfg,
    TrainDataCfg,
    TrainDatasetCfg,
)
from vla_scratch.datasets.libero.config import (
    libero_spatial_config,
    libero_90_config,
    libero_goal_config,
    libero_object_config,
    libero_object_bbox_config,
    libero_90_bbox_config,
    libero_goal_bbox_config,
    libero_spatial_bbox_config,
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

cs = ConfigStore.instance()
cs.store(name="libero-spatial", node=libero_spatial_eval_cfg, group="eval_data")
libero_spatial_bbox_eval_cfg = EvalDataCfg(
    datasets={
        "action": EvalDatasetCfg(
            data=libero_spatial_bbox_config,
            eval_fraction=0.05,
            eval_type="sample_mse",
        ),
        "generation": EvalDatasetCfg(
            data=replace(libero_spatial_bbox_config, bbox_only=True),
            eval_fraction=0.1,
            eval_type="generation",
        ),
    }
)
cs.store(
    name="libero-spatial-bbox",
    node=libero_spatial_bbox_eval_cfg,
    group="eval_data",
)

libero_90_eval_cfg = EvalDataCfg(
    datasets={
        "libero_90": EvalDatasetCfg(
            data=libero_90_config,
            eval_fraction=0.05,
            eval_type="sample_mse",
        )
    }
)
cs.store(name="libero-90", node=libero_90_eval_cfg, group="eval_data")
libero_90_bbox_eval_cfg = EvalDataCfg(
    datasets={
        "action": EvalDatasetCfg(
            data=libero_90_bbox_config,
            eval_fraction=0.05,
            eval_type="sample_mse",
        ),
        "generation": EvalDatasetCfg(
            data=replace(libero_90_bbox_config, bbox_only=True),
            eval_fraction=0.1,
            eval_type="generation",
        ),
    }
)
cs.store(name="libero-90-bbox", node=libero_90_bbox_eval_cfg, group="eval_data")

libero_goal_eval_cfg = EvalDataCfg(
    datasets={
        "goal": EvalDatasetCfg(
            data=libero_goal_config,
            eval_fraction=0.05,
            eval_type="sample_mse",
        )
    }
)
cs.store(name="libero-goal", node=libero_goal_eval_cfg, group="eval_data")
libero_goal_bbox_eval_cfg = EvalDataCfg(
    datasets={
        "action": EvalDatasetCfg(
            data=libero_goal_bbox_config,
            eval_fraction=0.05,
            eval_type="sample_mse",
        ),
        "generation": EvalDatasetCfg(
            data=replace(libero_goal_bbox_config, bbox_only=True),
            eval_fraction=0.1,
            eval_type="generation",
        ),
    }
)
cs.store(
    name="libero-goal-bbox",
    node=libero_goal_bbox_eval_cfg,
    group="eval_data",
)

libero_object_eval_cfg = EvalDataCfg(
    datasets={
        "object": EvalDatasetCfg(
            data=libero_object_config,
            eval_fraction=0.05,
            eval_type="sample_mse",
        )
    }
)
cs.store(name="libero-object", node=libero_object_eval_cfg, group="eval_data")

libero_object_train_cfg = TrainDataCfg(
    datasets={"object": TrainDatasetCfg(data=libero_object_config, batch_size=16)}
)
cs.store(name="libero-object", node=libero_object_train_cfg, group="train_data")

libero_spatial_train_cfg = TrainDataCfg(
    datasets={"spatial": TrainDatasetCfg(data=libero_spatial_config, batch_size=16)}
)
cs.store(name="libero-spatial", node=libero_spatial_train_cfg, group="train_data")

libero_90_train_cfg = TrainDataCfg(
    datasets={"libero_90": TrainDatasetCfg(data=libero_90_config, batch_size=16)}
)
cs.store(name="libero-90", node=libero_90_train_cfg, group="train_data")

libero_goal_train_cfg = TrainDataCfg(
    datasets={"goal": TrainDatasetCfg(data=libero_goal_config, batch_size=16)}
)
cs.store(name="libero-goal", node=libero_goal_train_cfg, group="train_data")

libero_object_bbox_eval_cfg = EvalDataCfg(
    datasets={
        "action": EvalDatasetCfg(
            data=libero_object_bbox_config,
            eval_fraction=0.05,
            eval_type="sample_mse",
        ),
        "generation": EvalDatasetCfg(
            data=replace(libero_object_bbox_config, bbox_only=True),
            eval_fraction=0.1,
            eval_type="generation",
        ),
    }
)
cs.store(
    name="libero-object-bbox",
    node=libero_object_bbox_eval_cfg,
    group="eval_data",
)

libero_object_bbox_train_cfg = TrainDataCfg(
    datasets={
        "object": TrainDatasetCfg(data=libero_object_bbox_config, batch_size=16)
    }
)
cs.store(
    name="libero-object-bbox",
    node=libero_object_bbox_train_cfg,
    group="train_data",
)

libero_spatial_bbox_train_cfg = TrainDataCfg(
    datasets={
        "spatial": TrainDatasetCfg(
            data=libero_spatial_bbox_config, batch_size=16
        )
    }
)
cs.store(
    name="libero-spatial-bbox",
    node=libero_spatial_bbox_train_cfg,
    group="train_data",
)

libero_90_bbox_train_cfg = TrainDataCfg(
    datasets={"libero_90": TrainDatasetCfg(data=libero_90_bbox_config, batch_size=16)}
)
cs.store(
    name="libero-90-bbox",
    node=libero_90_bbox_train_cfg,
    group="train_data",
)

libero_goal_bbox_train_cfg = TrainDataCfg(
    datasets={"goal": TrainDatasetCfg(data=libero_goal_bbox_config, batch_size=16)}
)
cs.store(
    name="libero-goal-bbox",
    node=libero_goal_bbox_train_cfg,
    group="train_data",
)
