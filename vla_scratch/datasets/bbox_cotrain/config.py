from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from vla_scratch.datasets.config import DataConfig


@dataclass
class CoTrainConfig(DataConfig):
    """
    Config for the BlindVLA LeRobot dataset.
    """

    _target_: str = "vla_scratch.datasets.bbox_cotrain.dataset.CoTrainDataset"
    repo_id: str = "horipse01/lerobot_merged"
    root_path: Optional[Path] = None

    bbox_only: bool = False
    remove_bbox: bool = False

    episodes: Optional[Any] = None
    # Regex filters applied to info.json splits keys (e.g., \".*banana.*\")
    splits: List[str] = field(default_factory=lambda: [".*"])

    norm_stats_path: Optional[Path] = (
        "normalization_stats/bbox_cotrain/lerobot_norm_stats-horizon_{data.action_horizon}-history_{data.state_history}.npz"
    )


train_cotrain_config = CoTrainConfig(repo_id="horipse01/lerobot_merged_restricted")
test_cotrain_config = CoTrainConfig(repo_id="horipse01/lerobot_merged_restricted_val")

cs = ConfigStore.instance()
cs.store(name="bbox_cotrain_train", node=train_cotrain_config, group="data")
cs.store(name="bbox_cotrain_test", node=test_cotrain_config, group="data")
