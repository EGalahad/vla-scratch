from __future__ import annotations

import json
from typing import TYPE_CHECKING, List, Tuple

import torch
import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import (
    LeRobotDatasetMetadata,
    LeRobotDataset,
)

from vla_scratch.utils.paths import REPO_ROOT
from vla_scratch.transforms.data_keys import (
    PROCESSED_STATE_KEY,
    PROCESSED_ACTION_KEY,
    PROCESSED_IMAGE_KEY,
    PROCESSED_IMAGE_MASK_KEY,
    TASK_KEY,
    GENERATION_PROMPT_KEY,
    GENERATION_ANSWER_KEY,
)
from vla_scratch.datasets.utils.paligemma_bbox_format import (
    paligemma_detect_answer,
    paligemma_detect_prompt,
    use_paligemma_tokens_enabled,
)

if TYPE_CHECKING:
    from vla_scratch.datasets.libero.config import LiberoConfig


def _load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


class LIBERODataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config: "LiberoConfig",
    ):
        self.action_horizon = action_horizon = config.action_horizon
        self.state_history = state_history = config.state_history

        root = getattr(config, "root_path", None)
        repo_id = config.repo_id
        meta_root = None
        if root:
            root_path = Path(root).expanduser()
            if not root_path.is_absolute():
                root_path = (REPO_ROOT / root_path).resolve()
            meta_root = root_path / repo_id

        meta_data = LeRobotDatasetMetadata(repo_id=repo_id, root=meta_root)
        fps = meta_data.fps

        features = meta_data.features

        self.cmd_keys: list[str] = [key for key in features.keys() if "cmd" in key]
        # LeRobot action column naming differs across datasets:
        # - Some provide both "actions" and "actions_orig"
        # - Some provide only "actions"
        # Keep timestamps aligned to what the dataset actually contains.
        self.cmd_keys.append("actions")
        if "actions_orig" in features:
            self.cmd_keys.append("actions_orig")
        self.state_keys: list[str] = [
            key for key in features.keys() if "state" in key
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
            repo_id=repo_id,
            root=meta_root,
            delta_timestamps=delta_timestamps,
            video_backend=config.video_backend,
        )
        assert fps == self.dataset.fps

        self.bbox_only = bool(getattr(config, "bbox_only", False))
        self.remove_bbox = bool(getattr(config, "remove_bbox", False))
        assert not (self.bbox_only and self.remove_bbox), (
            "Cannot set both bbox_only and remove_bbox to True."
        )

        self._bbox_records: List[dict] = []
        self._bbox_idx_map: dict[Tuple[int, int], int] = {}
        self._filtered_indices: List[int] | None = None

        bbox_path = meta_data.root / "meta" / "bboxes.jsonl"
        if bbox_path.exists():
            records = _load_jsonl(bbox_path)
            records.sort(
                key=lambda r: (int(r["episode_index"]), int(r["frame_index"]))
            )
            self._bbox_records = records
            for bbox_idx, r in enumerate(records):
                key = (int(r["episode_index"]), int(r["frame_index"]))
                self._bbox_idx_map.setdefault(key, bbox_idx)

            if self.bbox_only:
                episodes_obj = meta_data.episodes
                if hasattr(episodes_obj, "keys"):
                    episode_ids = list(episodes_obj.keys())
                else:
                    episode_ids = list(range(len(episodes_obj)))
                episode_ids = sorted(int(x) for x in episode_ids)

                episode_lengths = [
                    int(episodes_obj[ep_id]["length"]) for ep_id in episode_ids
                ]
                episode_start_indices = np.cumsum([0] + episode_lengths)[:-1]
                episode_to_start = dict(zip(episode_ids, episode_start_indices))
                self._filtered_indices = [
                    episode_to_start[int(r["episode_index"])]
                    + int(r["frame_index"])
                    for r in self._bbox_records
                    if int(r["episode_index"]) in episode_to_start
                ]

    def __len__(self):
        if self._filtered_indices is not None:
            return len(self._filtered_indices)
        return len(self.dataset)

    def __getitem__(self, idx):
        if self._filtered_indices is not None:
            idx = int(self._filtered_indices[idx])
        item = self.dataset[idx]
        # v3.0 LIBERO LeRobot format uses `image` and `wrist_image`.
        if "image" in item and "wrist_image" in item:
            img = torch.stack([item["image"], item["wrist_image"]], dim=0)
        else:
            # Backward/alternate formats may use explicit camera feature names.
            img = torch.stack(
                [item["images.cam_front"], item["images.cam_wrist"]], dim=0
            )
        img = (img * 255).to(torch.uint8)
        img_mask = torch.ones((img.shape[0], 1), dtype=torch.bool)

        if "state" in item:
            state = item["state"]
        else:
            state = torch.cat(
                [
                    item["arm_state_cart_pos"],
                    item["arm_state_cart_rot"],
                    item["gripper_state_qpos"],
                ],
                dim=-1,
            )
        state = state[1:]

        if self.bbox_only:
            actions = None
        else:
            if "actions_orig" in item:
                actions = item["actions_orig"]
            elif "actions" in item:
                actions = item["actions"]
            else:
                raise KeyError(
                    "Expected LeRobot dataset to contain an action column named "
                    "'actions_orig' or 'actions', but neither was found."
                )

        prompt = ""
        answer = ""
        if not self.remove_bbox and self._bbox_records:
            ep_idx_t = item.get("episode_index")
            frame_idx_t = item.get("frame_index")
            if ep_idx_t is not None and frame_idx_t is not None:
                ep_idx = int(ep_idx_t.item())
                frame_idx = int(frame_idx_t.item())
                bbox_idx = self._bbox_idx_map.get((ep_idx, frame_idx), -1)
                if 0 <= bbox_idx < len(self._bbox_records):
                    bbox = self._bbox_records[bbox_idx].get("bbox") or []
                    labels = [d["label"] for d in bbox]
                    if use_paligemma_tokens_enabled():
                        prompt = paligemma_detect_prompt()
                        answer = paligemma_detect_answer(
                            [d["bbox_normalized"] for d in bbox], labels
                        )
                    else:
                        bbox_coords = [
                            [int(x * 1000) for x in d["bbox_normalized"]]
                            for d in bbox
                        ]
                        bbox_out = [
                            {"bbox_2d": coords, "label": label}
                            for coords, label in zip(bbox_coords, labels)
                        ]
                        prompt = (
                            "Please return bounding boxes for all task-relevant objects in JSON format as"
                            '[{"bbox_2d": [x1, y1, x2, y2], "label": "<object_name>"}]'
                        )
                        answer = json.dumps(bbox_out)

        processed = {
            PROCESSED_IMAGE_KEY: img,
            PROCESSED_IMAGE_MASK_KEY: img_mask,
            PROCESSED_STATE_KEY: state,
            PROCESSED_ACTION_KEY: actions,
            TASK_KEY: item.get("task"),
            GENERATION_PROMPT_KEY: prompt,
            GENERATION_ANSWER_KEY: answer,
        }
        return processed
