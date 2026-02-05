from __future__ import annotations

from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from vla_scratch.datasets.config import DataConfig
from vla_scratch.datasets.libero.config import (
    libero_90_config,
    libero_goal_config,
    libero_object_config,
    libero_spatial_config,
)
from vla_scratch.datasets.libero.config import LiberoConfig


@dataclass
class LiberoMix120Config(DataConfig):
    _target_: str = "vla_scratch.datasets.libero_mix120.dataset.LiberoMix120Dataset"

    # Component datasets (defaults match train_pi05_libero120.sh mix120 recipe)
    libero_spatial: LiberoConfig = libero_spatial_config
    libero_goal: LiberoConfig = libero_goal_config
    libero_object: LiberoConfig = libero_object_config
    libero_90: LiberoConfig = libero_90_config

    # Single stats path for the mixed distribution
    norm_stats_path: str = "norm_stats/pi05_libero/libero_mix_120_average_lerobot/libero_mix_120_average_no_noops_lerobot"

    # Match LiberoConfig flags (applied to all components)
    bbox_only: bool = False
    remove_bbox: bool = True


cs = ConfigStore.instance()
libero_mix_120_average_config = LiberoMix120Config()
libero_mix_120_average_bbox_config = LiberoMix120Config(remove_bbox=False)

cs.store(
    name="libero-mix-120-average",
    node=libero_mix_120_average_config,
    group="data",
)
cs.store(
    name="libero-mix-120-average-bbox",
    node=libero_mix_120_average_bbox_config,
    group="data",
)
