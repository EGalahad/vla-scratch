from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List

from vla_scratch.datasets.config import DataConfig


@dataclass
class LiberoIPECConfig(DataConfig):
    @staticmethod
    def _default_transform_configs() -> list[Dict[str, Any]]:
        return [
            {"_target_": "vla_scratch.datasets.libero.transforms.LiberoState"},
            {"_target_": "vla_scratch.datasets.libero.transforms.LiberoAction"},
            {"_target_": "vla_scratch.datasets.libero.transforms.StructurePrompt"},
            {"_target_": "vla_scratch.datasets.libero.transforms.LiberoImages"},
        ]

    _target_: str = "vla_scratch.datasets.libero.lerobot_ipec.IPECDataset"
    repo_id: str = "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot"

    transforms: List[Any] = field(default_factory=_default_transform_configs)

    norm_stats_path: Path = Path(
        "normalization_stats/libero/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot.npz"
    )

from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="libero-ipec", node=LiberoIPECConfig, group="data")