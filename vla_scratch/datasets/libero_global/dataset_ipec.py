# this file hosts the LeRobotDataset class for loading libero dataset from IPEC-COMMUNITY
from typing import TYPE_CHECKING

import torch
import numpy as np
from lerobot.datasets.lerobot_dataset import (
    LeRobotDatasetMetadata,
    LeRobotDataset,
)

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
from vla_scratch.utils.math import (
    unscale_transform,
    quat_from_angle_axis,
    quat_mul,
    axis_angle_from_quat,
)

if TYPE_CHECKING:
    from vla_scratch.datasets.libero.config import LiberoIPECConfig


class IPECDataset(torch.utils.data.Dataset):
    # norm_stats_path = "normalization_stats/libero_proprio_stats.npz"
    actions_low = torch.tensor([-0.05, -0.05, -0.05, -0.5, -0.5, -0.5, -1.0])
    actions_high = torch.tensor([0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 1.0])

    def __init__(
        self,
        config: "LiberoIPECConfig",
    ):
        self.action_horizon = action_horizon = config.action_horizon
        self.state_history = state_history = config.state_history

        meta_data = LeRobotDatasetMetadata(config.repo_id[0])
        fps = meta_data.fps

        delta_timestamps = {
            "action": (
                np.linspace(0, action_horizon - 1, action_horizon, dtype=int)
                / fps
            ).tolist(),
            "observation.state": (
                np.linspace(
                    -state_history,
                    action_horizon - 1,
                    state_history + action_horizon,
                    dtype=int,
                )
                / fps
            ).tolist(),
        }
        self.lerobot_datasets = [
            LeRobotDataset(
                repo_id=repo_id,
                delta_timestamps=delta_timestamps,
                video_backend=config.video_backend,
            )
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

        full_state = item.pop("observation.state")
        history_state = full_state[: self.state_history + 1]
        future_state = full_state[self.state_history :]

        actions = item.pop("action")
        actions[:, -1] = 1 - 2 * actions[:, -1]  # convert [0, 1] to [1, -1]

        # images and task
        item[CAM_FRONT_KEY] = item.pop("observation.images.image")
        item[CAM_WRIST_KEY] = item.pop("observation.images.wrist_image")
        item[TASK_NAME_KEY] = item.pop("task")

        # state
        item[ARM_STATE_CART_POS_KEY] = history_state[:, 0:3]
        item[ARM_STATE_CART_ROT_KEY] = history_state[:, 3:6]
        item[GRIPPER_STATE_QPOS_KEY] = history_state[:, 6:]

        # cmd
        actions = unscale_transform(
            actions,
            self.actions_low,
            self.actions_high,
        )

        future_pos_w = future_state[:, 0:3]
        future_rotvec_w = future_state[:, 3:6]
        angle = torch.linalg.norm(future_rotvec_w, dim=-1)
        future_quat_w = quat_from_angle_axis(angle, future_rotvec_w)

        cmd_pos_delta = actions[:, 0:3]
        cmd_rotvec_delta = actions[:, 3:6]
        cmd_angle_delta = torch.linalg.norm(cmd_rotvec_delta, dim=-1)
        cmd_quat_delta = quat_from_angle_axis(cmd_angle_delta, cmd_rotvec_delta)

        cmd_pos_w = future_pos_w + cmd_pos_delta
        cmd_quat_w = quat_mul(cmd_quat_delta, future_quat_w)
        cmd_rot_w = axis_angle_from_quat(cmd_quat_w)

        item[ARM_CMD_CART_POS_KEY] = cmd_pos_w
        item[ARM_CMD_CART_ROT_KEY] = cmd_rot_w
        item[GRIPPER_CMD_ACTION_KEY] = actions[:, 6:7]
        return item
