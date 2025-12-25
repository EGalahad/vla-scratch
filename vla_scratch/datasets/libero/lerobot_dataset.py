# this file hosts the LeRobotDataset class for loading libero dataset from IPEC-COMMUNITY
from typing import TYPE_CHECKING

import torch
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset

from .data_keys import (
    CAM_FRONT_KEY,
    CAM_WRIST_KEY,
    TASK_NAME_KEY,
    ARM_STATE_CART_POS_KEY,
    ARM_STATE_CART_ROT_KEY,
    GRIPPER_STATE_QPOS_KEY,
    ARM_CMD_CART_POS_KEY,
    ARM_CMD_CART_ROT_KEY,
    GRIPPER_CMD_ACTION_KEY,
)

if TYPE_CHECKING:
    from vla_scratch.datasets.libero.config import LiberoIPECConfig


class LIBERODataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config: "LiberoIPECConfig",
    ):
        self.action_horizon = action_horizon = config.action_horizon
        self.state_history = state_history = config.state_history

        meta_data = LeRobotDatasetMetadata(config.repo_id[0])
        fps = meta_data.fps

        self.cmd_keys: list[str] = [
            key for key in meta_data.features.keys() if "cmd" in key
        ]
        self.state_keys: list[str] = [
            key for key in meta_data.features.keys() if "state" in key
        ]
        delta_timestamps = {}
        for key in self.cmd_keys:
            delta_timestamps[key] = (
                np.linspace(0, action_horizon - 1, action_horizon, dtype=int) / fps
            ).tolist()

        for key in self.state_keys:
            delta_timestamps[key] = (
                np.linspace(-state_history, 0, state_history + 1, dtype=int) / fps
            ).tolist()

        self.lerobot_datasets = [
            LeRobotDataset(repo_id=repo_id, delta_timestamps=delta_timestamps)
            for repo_id in config.repo_id
        ]
        assert fps == self.lerobot_datasets[0].fps

        self.idx_map = []
        for dataset_idx, dataset in enumerate(self.lerobot_datasets):
            for frame_in_dataset in range(dataset.num_frames):
                self.idx_map.append((dataset_idx, frame_in_dataset))

    def __len__(self):
        return len(self.idx_map)

    def __getitem__(self, idx):
        dataset_idx, frame_in_dataset = self.idx_map[idx]
        item = self.lerobot_datasets[dataset_idx][frame_in_dataset]
        item[CAM_FRONT_KEY] = item.pop("images.cam_front")
        item[CAM_WRIST_KEY] = item.pop("images.cam_wrist")
        item[TASK_NAME_KEY] = item.pop("task")
        item[GRIPPER_CMD_ACTION_KEY] = item[GRIPPER_CMD_ACTION_KEY].unsqueeze(-1)
        return item
