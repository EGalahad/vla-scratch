from dataclasses import dataclass, field
from typing import List
from copy import deepcopy
from hydra.core.config_store import ConfigStore

from vla_scratch.datasets.config import DataConfig
from vla_scratch.transforms.data_keys import PROCESSED_STATE_KEY


@dataclass
class LiberoIPECConfig(DataConfig):
    _target_: str = (
        "vla_scratch.datasets.libero_global.lerobot_ipec.IPECDataset"
    )
    repo_id: List[str] = field(
        default_factory=lambda: [
            "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot",
            "IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot",
            "IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot",
            # "IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot",
            # "IPEC-COMMUNITY/libero_90_no_noops_1.0.0_lerobot",
        ]
    )
    norm_stats_path: str = "normalization_stats/libero/IPEC-COMMUNITY/libero-horizon_{data.action_horizon}-history_{data.state_history}.npz"

    input_transforms: List[dict] = field(
        default_factory=lambda: [
            {
                "_target_": "vla_scratch.datasets.libero.transforms.LiberoGlobalState"
            },
            {"_target_": "vla_scratch.datasets.libero.transforms.LiberoImages"},
        ]
    )
    output_transforms: List[dict] = field(
        default_factory=lambda: [
            {
                "_target_": "vla_scratch.datasets.libero.transforms.LiberoActionToLocal"
            },
        ]
    )
    output_inv_transforms: List[dict] = field(
        default_factory=lambda: [
            {
                "_target_": "vla_scratch.datasets.libero.transforms.LiberoActionToGlobal"
            },
        ]
    )


default_libero_config = LiberoIPECConfig(
    repo_id=[
        "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot",
        "IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot",
        "IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot",
        # "IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot",
        # "IPEC-COMMUNITY/libero_90_no_noops_1.0.0_lerobot",
    ],
)
libero_ipec_spatial_config = deepcopy(default_libero_config)
libero_ipec_spatial_config.repo_id = [
    "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot"
]
libero_ipec_spatial_config.norm_stats_path = "normalization_stats/libero/IPEC-COMMUNITY/libero_spatial-horizon_{data.action_horizon}-history_{data.state_history}.npz"

libero_ipec_spatial_noised_config = deepcopy(libero_ipec_spatial_config)
libero_ipec_spatial_noised_config.noise_cfg = {
    PROCESSED_STATE_KEY: {
        # pos
        "0-3": {
            "type": "gaussian",
            "std": 0.2,
        },
        # rot
        "3-9": {
            "type": "gaussian",
            "std": 0.1,
        },
        # gripper
        "9-11": {
            "type": "gaussian",
            "std": 0.2,
        },
    }
}


cs = ConfigStore.instance()
cs.store(
    name="libero-ipec-spatial", node=libero_ipec_spatial_config, group="data"
)
cs.store(
    name="libero-ipec-spatial-noised",
    node=libero_ipec_spatial_noised_config,
    group="data",
)


@dataclass
class LiberoConfig(DataConfig):
    _target_: str = (
        "vla_scratch.datasets.libero_global.lerobot_dataset.LIBERODataset"
    )
    repo_id: List[str] = field(
        default_factory=lambda: ["elijahgalahad/libero_spatial_noops_v30"]
    )
    norm_stats_path: str = "normalization_stats/libero/lerobot-global-horizon_{data.action_horizon}-history_{data.state_history}.npz"
    input_transforms: List[dict] = field(
        default_factory=lambda: [
            {
                "_target_": "vla_scratch.datasets.libero.transforms.LiberoGlobalState"
            },
            {"_target_": "vla_scratch.datasets.libero.transforms.LiberoImages"},
        ]
    )
    output_transforms: List[dict] = field(
        default_factory=lambda: [
            {
                "_target_": "vla_scratch.datasets.libero.transforms.LiberoActionToLocal"
            },
        ]
    )
    output_inv_transforms: List[dict] = field(
        default_factory=lambda: [
            {
                "_target_": "vla_scratch.datasets.libero.transforms.LiberoActionToGlobal"
            },
        ]
    )


libero_spatial_config = LiberoConfig()
cs.store(name="libero-global-spatial", node=libero_spatial_config, group="data")
