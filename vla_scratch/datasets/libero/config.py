from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from vla_scratch.datasets.config import DataConfig
from hydra.core.config_store import ConfigStore


@dataclass
class LiberoConfig(DataConfig):
    _target_: str = "vla_scratch.datasets.libero.dataset.LIBERODataset"
    repo_id: str = "libero_spatial_no_noops_lerobot"
    # Optional local dataset root. If set, expects the dataset to live at
    # `<root_path>/<repo_id>/...` (LeRobot on-disk layout).
    root_path: Optional[Path] = "data/libero_spatial_lerobot_v3.0/"
    norm_stats_path: str = "norm_stats/pi05_libero/libero_spatial_lerobot/libero_spatial_no_noops_lerobot"
    # If true, only return frames that have bbox annotations, and do not return actions.
    # Intended for generation-only evaluation.
    bbox_only: bool = False
    # If true, ignore bbox annotations even if present on disk.
    # Useful for ablations (action-only training/eval).
    remove_bbox: bool = True


libero_spatial_config = LiberoConfig()
libero_spatial_bbox_config = LiberoConfig(remove_bbox=False)

libero_90_config = LiberoConfig(
    repo_id="libero_90_no_noops_lerobot",
    root_path="data/libero_90_lerobot_v3.0",
    norm_stats_path="norm_stats/pi05_libero/libero_90_lerobot/libero_90_no_noops_lerobot",
)
libero_90_bbox_config = LiberoConfig(
    repo_id="libero_90_no_noops_lerobot",
    root_path="data/libero_90_lerobot_v3.0",
    norm_stats_path="norm_stats/pi05_libero/libero_90_lerobot/libero_90_no_noops_lerobot",
    remove_bbox=False,
)

libero_goal_config = LiberoConfig(
    repo_id="libero_goal_no_noops_lerobot",
    root_path="data/libero_goal_lerobot_v3.0",
    norm_stats_path="norm_stats/pi05_libero/libero_goal_lerobot/libero_goal_no_noops_lerobot",
)
libero_goal_bbox_config = LiberoConfig(
    repo_id="libero_goal_no_noops_lerobot",
    root_path="data/libero_goal_lerobot_v3.0",
    norm_stats_path="norm_stats/pi05_libero/libero_goal_lerobot/libero_goal_no_noops_lerobot",
    remove_bbox=False,
)

libero_object_config = LiberoConfig(
    repo_id="libero_object_no_noops_lerobot",
    root_path="data/libero_object_lerobot_v3.0",
    norm_stats_path="norm_stats/pi05_libero/libero_object_lerobot/libero_object_no_noops_lerobot",
)

libero_object_bbox_config = LiberoConfig(
    repo_id="libero_object_no_noops_lerobot",
    root_path="data/libero_object_lerobot_v3.0",
    norm_stats_path="norm_stats/pi05_libero/libero_object_lerobot/libero_object_no_noops_lerobot",
    remove_bbox=False,
)

cs = ConfigStore.instance()
cs.store(name="libero-spatial", node=libero_spatial_config, group="data")
cs.store(
    name="libero-spatial-bbox", node=libero_spatial_bbox_config, group="data"
)
cs.store(name="libero-90", node=libero_90_config, group="data")
cs.store(name="libero-90-bbox", node=libero_90_bbox_config, group="data")
cs.store(name="libero-goal", node=libero_goal_config, group="data")
cs.store(name="libero-goal-bbox", node=libero_goal_bbox_config, group="data")
cs.store(name="libero-object", node=libero_object_config, group="data")
cs.store(
    name="libero-object-bbox", node=libero_object_bbox_config, group="data"
)
