# this file hosts the LeRobotDataset class for loading libero dataset from IPEC-COMMUNITY
from typing import TYPE_CHECKING

import torch
import numpy as np
from lerobot.datasets.lerobot_dataset import (
    LeRobotDatasetMetadata,
    LeRobotDataset,
)

from vla_scratch.transforms.data_keys import (
    PROCESSED_STATE_KEY,
    PROCESSED_ACTION_KEY,
    PROCESSED_IMAGE_KEY,
    PROCESSED_IMAGE_MASK_KEY,
    TASK_KEY,
)

if TYPE_CHECKING:
    from vla_scratch.datasets.libero.config import LiberoConfig


class LIBERODataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config: "LiberoConfig",
    ):
        self.action_horizon = action_horizon = config.action_horizon
        self.state_history = state_history = config.state_history

        meta_data = LeRobotDatasetMetadata(
            config.repo_id, root=config.root
        )
        fps = meta_data.fps

        self.cmd_keys: list[str] = [
            key for key in meta_data.features.keys() if "cmd" in key
        ]
        self.cmd_keys.append("actions")
        self.cmd_keys.append("actions_orig")
        self.state_keys: list[str] = [
            key for key in meta_data.features.keys() if "state" in key
        ]
        delta_timestamps = {}
        for key in self.cmd_keys:
            delta_timestamps[key] = (
                np.linspace(0, action_horizon - 1, action_horizon, dtype=int)
                / fps
            ).tolist()

        for key in self.state_keys:
            delta_timestamps[key] = (
                np.linspace(-state_history, 0, state_history + 1, dtype=int)
                / fps
            ).tolist()

        self.dataset = LeRobotDataset(
            repo_id=config.repo_id,
            root=config.root,
            delta_timestamps=delta_timestamps,
            video_backend=config.video_backend,
        )
        assert fps == self.dataset.fps

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = torch.stack(
            [item["images.cam_front"], item["images.cam_wrist"]], dim=0
        )
        img = (img * 255).to(torch.uint8)
        img_mask = torch.ones((img.shape[0], 1), dtype=torch.bool)

        state = torch.cat(
            [
                item["arm_state_cart_pos"],
                item["arm_state_cart_rot"],
                item["gripper_state_qpos"],
            ],
            dim=-1,
        )
        state = state[1:]
        actions = item["actions_orig"]

        processed = {
            PROCESSED_IMAGE_KEY: img,
            PROCESSED_IMAGE_MASK_KEY: img_mask,
            PROCESSED_STATE_KEY: state,
            PROCESSED_ACTION_KEY: actions,
            TASK_KEY: item.get("task"),
        }
        return processed


class LIBEROIpecDataset(torch.utils.data.Dataset):
    """LIBERO Spatial v2.1 data in the simulator action space."""

    def __init__(self, config: "LiberoConfig"):
        self.action_horizon = config.action_horizon
        self.state_history = config.state_history
        metadata = LeRobotDatasetMetadata(
            config.repo_id, root=config.root
        )
        fps = metadata.fps
        self.dataset = LeRobotDataset(
            repo_id=config.repo_id,
            root=config.root,
            delta_timestamps={
                "action": (
                    np.arange(self.action_horizon, dtype=int) / fps
                ).tolist(),
                "observation.state": (
                    np.arange(-self.state_history, 1, dtype=int) / fps
                ).tolist(),
            },
            video_backend=config.video_backend,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        images = torch.stack(
            [
                item["observation.images.image"],
                item["observation.images.wrist_image"],
            ],
            dim=0,
        )
        actions = item["action"].clone()
        actions[..., -1] = 1 - 2 * actions[..., -1]
        return {
            PROCESSED_IMAGE_KEY: (images * 255).to(torch.uint8),
            PROCESSED_IMAGE_MASK_KEY: torch.ones(
                (images.shape[0], 1), dtype=torch.bool
            ),
            PROCESSED_STATE_KEY: item["observation.state"][1:],
            PROCESSED_ACTION_KEY: actions,
            TASK_KEY: item.get("task"),
        }
