from dataclasses import dataclass, field
from typing import List
from copy import deepcopy
from hydra.core.config_store import ConfigStore

from vla_scratch.datasets.config import DataConfig
from vla_scratch.transforms.data_keys import PROCESSED_STATE_KEY


@dataclass
class LiberoIPECConfig(DataConfig):
    _target_: str = "vla_scratch.datasets.libero.lerobot_ipec.IPECDataset"
    repo_id: List[str] = field(
        default_factory=lambda: [
            "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot",
            "IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot",
            "IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot",
            # "IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot",
            # "IPEC-COMMUNITY/libero_90_no_noops_1.0.0_lerobot",
        ]
    )
    norm_stats_path: str = (
        "normalization_stats/libero/IPEC-COMMUNITY/libero-horizon_{data.action_horizon}-history_{data.state_history}.npz"
    )


default_libero_config = LiberoIPECConfig(
    repo_id=[
        "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot",
        "IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot",
        "IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot",
        # "IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot",
        # "IPEC-COMMUNITY/libero_90_no_noops_1.0.0_lerobot",
    ],
    input_transforms=[
        {"_target_": "vla_scratch.datasets.libero.transforms.LiberoGlobalState"},
        {"_target_": "vla_scratch.datasets.libero.transforms.LiberoImages"},
    ],
    output_transforms=[
        {"_target_": "vla_scratch.datasets.libero.transforms.LiberoActionToLocal"},
    ],
    output_inv_transforms=[
        {"_target_": "vla_scratch.datasets.libero.transforms.LiberoActionToGlobal"},
    ],
)
libero_spatial_config = deepcopy(default_libero_config)
libero_spatial_config.repo_id = ["IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot"]
libero_spatial_config.norm_stats_path = "normalization_stats/libero/IPEC-COMMUNITY/libero_spatial-horizon_{data.action_horizon}-history_{data.state_history}.npz"

libero_spatial_noised_config = deepcopy(libero_spatial_config)
libero_spatial_noised_config.noise_cfg = {
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

# libero_spatial_delta_action_config = deepcopy(libero_spatial_config)
# libero_spatial_delta_action_config.output_transforms = []
# libero_spatial_delta_action_config.output_inv_transforms = []


# @dataclass
# class LiberoElijahConfig(DataConfig):
#     _target_: str = "vla_scratch.datasets.libero.lerobot_elijah.ElijahDataset"
#     repo_id: List[str] = field(
#         default_factory=lambda: [
#             "elijahgalahad/libero_spatial_noops_v21",
#         ]
#     )
#     norm_stats_path: str = (
#         "normalization_stats/libero/elijahgalahad/libero-horizon_{data.action_horizon}-history_{data.state_history}.npz"
#     )


# default_libero_elijah_config = LiberoElijahConfig(
#     input_transforms=[
#         {"_target_": "vla_scratch.datasets.libero.transforms.LiberoGlobalState"},
#         {"_target_": "vla_scratch.datasets.libero.transforms.LiberoImages"},
#     ],
#     output_transforms=[
#         {"_target_": "vla_scratch.datasets.libero.transforms.LiberoActionToLocal"},
#     ],
#     output_inv_transforms=[
#         {"_target_": "vla_scratch.datasets.libero.transforms.LiberoActionToGlobal"},
#     ],
# )

# libero_elijah_noised_config = deepcopy(default_libero_elijah_config)
# libero_elijah_noised_config.noise_cfg = {
#     PROCESSED_STATE_KEY: {
#         # left pos
#         "0-3": {
#             "type": "gaussian",
#             "std": 0.2,
#         },
#         # left rot
#         "3-9": {
#             "type": "gaussian",
#             "std": 0.1,
#         },
#     }
# }


cs = ConfigStore.instance()
cs.store(name="libero-ipec-global", node=default_libero_config, group="data")
cs.store(name="libero-ipec-spatial", node=libero_spatial_config, group="data")
cs.store(name="libero-ipec-spatial-noised", node=libero_spatial_noised_config, group="data")

# cs.store(name="libero-elijah", node=default_libero_elijah_config, group="data")
# cs.store(name="libero-elijah-noised", node=libero_elijah_noised_config, group="data")
