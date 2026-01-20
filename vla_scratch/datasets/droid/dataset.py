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
    from vla_scratch.datasets.droid.config import DROIDConfig


class DROIDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config: "DROIDConfig",
    ):
        self.action_horizon = action_horizon = config.action_horizon
        self.state_history = state_history = config.state_history

        meta_data = LeRobotDatasetMetadata(config.repo_id)
        fps = meta_data.fps

        self.cmd_keys: list[str] = [
            key for key in meta_data.features.keys() if "cmd" in key
        ]
        # DROID uses "action" (singular), not "actions" (plural)
        self.cmd_keys.append("action")
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
            delta_timestamps=delta_timestamps,
            video_backend=config.video_backend,
        )
        assert fps == self.dataset.fps

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # DROID uses different image keys than LIBERO
        # DROID: observation.images.wrist_image_left, observation.images.exterior_image_1_left, observation.images.exterior_image_2_left
        img = torch.stack(
            [
                item["observation.images.wrist_image_left"],
                item["observation.images.exterior_image_1_left"], 
                item["observation.images.exterior_image_2_left"]
            ], 
            dim=0
        )
        img = (img * 255).to(torch.uint8)
        img_mask = torch.ones((img.shape[0], 1), dtype=torch.bool)

        # DROID uses "observation.state" directly (already concatenated)
        state = item["observation.state"]
        if len(state.shape) > 1:
            state = state[1:]  # Remove first timestep if needed
        
        # DROID uses "action" (singular)
        actions = item["action"]

        processed = {
            PROCESSED_IMAGE_KEY: img,
            PROCESSED_IMAGE_MASK_KEY: img_mask,
            PROCESSED_STATE_KEY: state,
            PROCESSED_ACTION_KEY: actions,
            TASK_KEY: item.get("language_instruction", ""),  # DROID uses language_instruction
        }
        return processed
