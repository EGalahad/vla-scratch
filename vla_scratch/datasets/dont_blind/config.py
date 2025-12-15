from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from hydra.core.config_store import ConfigStore

from vla_scratch.datasets.config import DataConfig


@dataclass
class DontBlindConfig(DataConfig):
    """
    Config for the BlindVLA LeRobot dataset.
    """

    _target_: str = "vla_scratch.datasets.dont_blind.lerobot_dataset.DontBlindDataset"
    repo_id: str = "elijahgalahad/blindvla_1k_lerobot"
    root_path: Optional[Path] = None
    episodes: Optional[List[int]] = None
    # Regex filters applied to info.json splits keys (e.g., \".*banana.*\")
    splits: List[str] = field(default_factory=lambda: ["train"])
    # No action transforms are applied for this dataset; keep raw actions.

    norm_stats_path: Optional[Path] = "normalization_stats/dont_blind/lerobot_norm_stats-horizon_{data.action_horizon}-history_{data.state_history}.npz"


default_dont_blind_config = DontBlindConfig()

dont_blind_8_8_objects_config_train = DontBlindConfig(
    splits=[".*banana.*", ".*fast_food_cup.*", ".*toy_bear.*", ".*pipe.*", ".*7up_can.*", ".*bread.*", ".*kitchen_shovel.*", ".*plant.*"]
)
dont_blind_8_8_objects_config_test = DontBlindConfig(
    splits=[".*plastic_bottle.*", ".*zuchinni.*", ".*golf_ball.*", ".*ketchup_bottle.*", ".*watering_can.*", ".*bbq_sauce.*", ".*carrot.*", ".*hamburger.*"]
)
dont_blind_12_4_objects_config_train = DontBlindConfig(
    splits=[
        ".*banana.*",
        ".*fast_food_cup.*",
        ".*toy_bear.*",
        ".*pipe.*",
        ".*7up_can.*",
        ".*bread.*",
        ".*kitchen_shovel.*",
        ".*plant.*",
        ".*plastic_bottle.*",
        ".*zuchinni.*",
        ".*golf_ball.*",
        ".*ketchup_bottle.*",
    ]
)
dont_blind_12_4_objects_config_test = DontBlindConfig(
    splits=[
        ".*watering_can.*",
        ".*bbq_sauce.*",
        ".*carrot.*",
        ".*hamburger.*",
    ]
)

cs = ConfigStore.instance()
cs.store(name="dont_blind", node=default_dont_blind_config, group="data")
cs.store(name="dont_blind_8_8_train", node=dont_blind_8_8_objects_config_train, group="data")
cs.store(name="dont_blind_8_8_test", node=dont_blind_8_8_objects_config_test, group="data")
cs.store(name="dont_blind_12_4_train", node=dont_blind_12_4_objects_config_train, group="data")
cs.store(name="dont_blind_12_4_test", node=dont_blind_12_4_objects_config_test, group="data")
