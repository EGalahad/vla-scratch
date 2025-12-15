import re
from typing import TYPE_CHECKING, Iterable, List, Set

import torch
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from vla_scratch.transforms.data_keys import (
    PROCESSED_ACTION_KEY,
    PROCESSED_IMAGE_KEY,
    PROCESSED_IMAGE_MASK_KEY,
    PROCESSED_STATE_KEY,
    TASK_KEY,
)

if TYPE_CHECKING:
    from vla_scratch.datasets.dont_blind.config import DontBlindConfig


def _expand_split(value) -> List[int]:
    """
    Split values can be either lists of episode ids or range strings like \"0:1376\".
    """
    if isinstance(value, str):
        if ":" in value:
            start, end = value.split(":")
            return list(range(int(start), int(end)))
        return []
    if isinstance(value, Iterable):
        return list(value)
    return []


def _select_episodes(meta: LeRobotDatasetMetadata, split_patterns: List[str]) -> List[int] | None:
    """
    Build an episode list by matching split names against provided regex patterns.
    """
    if not split_patterns:
        return None

    compiled = [re.compile(pat) for pat in split_patterns]
    selected: Set[int] = set()
    for split_name, value in meta.info.get("splits", {}).items():
        if any(p.search(split_name) for p in compiled):
            selected.update(_expand_split(value))
    return sorted(selected) if selected else []


class DontBlindDataset(torch.utils.data.Dataset):
    """
    Minimal LeRobot dataset wrapper for BlindVLA.

    - Keeps actions as-is (no extra transforms).
    - Emits processed keys defined in `vla_scratch.transforms.data_keys`.
    """

    def __init__(self, config: "DontBlindConfig"):
        root = config.root_path
        repo_id = config.repo_id

        meta_root = root / repo_id if root else None
        meta = LeRobotDatasetMetadata(repo_id=repo_id, root=meta_root)

        # If explicit episodes are provided, they take precedence.
        if config.episodes is not None:
            episodes = config.episodes
        else:
            episodes = _select_episodes(meta, config.splits)

        fps = meta.fps
        self.action_horizon = config.action_horizon
        self.state_history = config.state_history
        delta_timestamps = {
            "actions": (np.linspace(0, self.action_horizon - 1, self.action_horizon, dtype=int) / fps).tolist(),
        }

        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            root=meta_root,
            delta_timestamps=delta_timestamps,
            episodes=episodes,
        )
        # Expose selection metadata for downstream inspection utilities.
        self.episodes = episodes
        self.metadata = meta

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Images: single camera, add camera dimension and convert to uint8.
        img = item["observation.images.image"]
        actions = item["actions"]  # shape: (action_horizon, action_dim)

        state_len = self.state_history + 1

        processed = {
            PROCESSED_IMAGE_KEY: (img * 255).to(torch.uint8).unsqueeze(0),
            PROCESSED_IMAGE_MASK_KEY: torch.ones((1, 1), dtype=torch.bool),
            PROCESSED_ACTION_KEY: actions,
            PROCESSED_STATE_KEY: torch.randn((state_len, 1), dtype=torch.float32),
            TASK_KEY: item.get("task"),
        }
        return processed
