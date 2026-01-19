from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore

from vla_scratch.datasets.config import DataConfig
from vla_scratch.utils.paths import REPO_ROOT


@dataclass
class DontBlindConfig(DataConfig):
    """
    Config for the BlindVLA LeRobot dataset.
    """

    _target_: str = (
        "vla_scratch.datasets.dont_blind.lerobot_dataset.DontBlindDataset"
    )
    repo_id: str = "elijahgalahad/blindvla_1k_lerobot"
    root_path: Optional[Path] = None

    bbox_only: bool = False

    episodes: Optional[Any] = None
    # Regex filters applied to info.json splits keys (e.g., \".*banana.*\")
    splits: List[str] = field(default_factory=lambda: ["train"])

    norm_stats_path: Optional[Path] = (
        "normalization_stats/dont_blind/lerobot_norm_stats-horizon_{data.action_horizon}-history_{data.state_history}.npz"
    )


default_dont_blind_config = DontBlindConfig()

dont_blind_8_8_objects_config_train = DontBlindConfig(
    splits=[
        ".*banana.*",
        ".*fast_food_cup.*",
        ".*toy_bear.*",
        ".*pipe.*",
        ".*7up_can.*",
        ".*bread.*",
        ".*kitchen_shovel.*",
        ".*plant.*",
    ]
)
dont_blind_8_8_objects_config_test = DontBlindConfig(
    splits=[
        ".*plastic_bottle.*",
        ".*zuchinni.*",
        ".*golf_ball.*",
        ".*ketchup_bottle.*",
        ".*watering_can.*",
        ".*bbq_sauce.*",
        ".*carrot.*",
        ".*hamburger.*",
    ]
)

ours = DontBlindConfig(
    repo_id="ours",
    root_path=REPO_ROOT,
    splits=[".*"],
    norm_stats_path="normalization_stats/ours/lerobot_norm_stats-horizon_{data.action_horizon}-history_{data.state_history}.npz",
)

cs = ConfigStore.instance()
cs.store(name="dont_blind", node=default_dont_blind_config, group="data")
cs.store(
    name="dont_blind_8_8_train",
    node=dont_blind_8_8_objects_config_train,
    group="data",
)
cs.store(
    name="dont_blind_8_8_test",
    node=dont_blind_8_8_objects_config_test,
    group="data",
)
cs.store(name="ours", node=ours, group="data")
