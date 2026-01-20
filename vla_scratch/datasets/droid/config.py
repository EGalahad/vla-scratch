from dataclasses import dataclass

from vla_scratch.datasets.config import DataConfig
from hydra.core.config_store import ConfigStore


@dataclass
class DROIDConfig(DataConfig):
    _target_: str = "vla_scratch.datasets.droid.dataset.DROIDDataset"
    repo_id: str = "lerobot/droid_100"
    norm_stats_path: str | None = None  # DROID doesn't have norm stats yet


droid_config = DROIDConfig()

cs = ConfigStore.instance()
cs.store(name="droid", node=droid_config, group="data")
