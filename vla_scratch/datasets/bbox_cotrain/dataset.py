import re
import json
from typing import TYPE_CHECKING, Iterable, List, Set, Tuple, Union

import torch
import numpy as np
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)

from vla_scratch.transforms.data_keys import (
    PROCESSED_ACTION_KEY,
    PROCESSED_IMAGE_KEY,
    PROCESSED_IMAGE_MASK_KEY,
    PROCESSED_STATE_KEY,
    TASK_KEY,
    GENERATION_PROMPT_KEY,
    GENERATION_ANSWER_KEY,
)

if TYPE_CHECKING:
    from .config import CoTrainConfig


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


def _select_episodes(
    meta: LeRobotDatasetMetadata, split_patterns: List[str]
) -> List[int] | None:
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


def _token_to_episodes(token: str) -> List[int]:
    if "-" in token:
        start, end = token.split("-", 1)
        start_i, end_i = int(start), int(end)
        step = 1 if start_i <= end_i else -1
        return list(range(start_i, end_i + step, step))
    if ":" in token:
        start, end = token.split(":", 1)
        start_i, end_i = int(start), int(end)
        step = 1 if start_i <= end_i else -1
        return list(range(start_i, end_i + step, step))
    return [int(token)]


def _parse_episode_str(value: Union[List[int], str, None]) -> List[int] | None:
    """
    Accept explicit episode lists or simple range strings (e.g. "0-128").
    """
    if value is None:
        return None
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        episodes: List[int] = []
        for entry in value:
            if isinstance(entry, (str, bytes)):
                tokens = [
                    tok.strip() for tok in str(entry).split(",") if tok.strip()
                ]
                if not tokens:
                    continue
                for token in tokens:
                    episodes.extend(_token_to_episodes(token))
            else:
                episodes.append(int(entry))
        return episodes

    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            cleaned = cleaned[1:-1]
        tokens = [tok.strip() for tok in cleaned.split(",") if tok.strip()]
        if not tokens and cleaned:
            tokens = [cleaned]
        episodes: List[int] = []
        for token in tokens:
            episodes.extend(_token_to_episodes(token))
        return episodes
    return None


def _build_bbox_index(
    path, episodes: Set[int] | None
) -> Tuple[dict[Tuple[int, int], int], List[Tuple[int, int]]]:
    """
    Build a (episode_index, frame_index) -> byte offset index for jsonl bboxes.
    Only stores offsets (and keys) to avoid loading bbox payloads into RAM.
    """
    index: dict[Tuple[int, int], int] = {}
    keys: List[Tuple[int, int]] = []
    with open(path, "rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            ep_idx = int(record["episode_index"])
            if episodes is not None and ep_idx not in episodes:
                continue
            frame_idx = int(record["frame_index"])
            key = (ep_idx, frame_idx)
            if key not in index:
                index[key] = offset
                keys.append(key)
    return index, keys


def load_bbox_jsonl(path) -> List[dict]:
    """
    Load bbox jsonl into memory; intended for debugging/benchmark scripts.
    """
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _resolve_episodes(
    meta: LeRobotDatasetMetadata, episodes: List[int] | None
) -> List[int]:
    if episodes is None:
        return list(meta.episodes.keys())
    return episodes


class CoTrainDataset(torch.utils.data.Dataset):
    """
    Minimal LeRobot dataset wrapper for Bbox Cotrain

    - Keeps actions as-is (no extra transforms).
    - Emits processed keys defined in `vla_scratch.transforms.data_keys`.
    - If `meta/bboxes.jsonl` is present, resolves (episode_index, frame_index) -> bbox list.
    """

    def __init__(self, config: "CoTrainConfig"):
        root = config.root_path
        repo_id = config.repo_id

        meta_root = root / repo_id if root else None
        meta = LeRobotDatasetMetadata(repo_id=repo_id, root=meta_root)

        # If explicit episodes are provided, they take precedence.
        episodes_override = _parse_episode_str(config.episodes)
        if episodes_override is not None:
            episodes = episodes_override
        else:
            episodes = _select_episodes(meta, config.splits)

        fps = meta.fps
        self.action_horizon = config.action_horizon
        self.state_history = config.state_history
        delta_timestamps = {
            "actions": (
                np.linspace(
                    0, self.action_horizon - 1, self.action_horizon, dtype=int
                )
                / fps
            ).tolist(),
        }

        # _prefetch_required_chunks(meta, episodes)

        import time

        time_start = time.time()
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            root=meta_root,
            delta_timestamps=delta_timestamps,
            episodes=episodes,
            video_backend=config.video_backend,
        )
        time_end = time.time()
        print(
            f"Initialized LeRobotDataset with {len(self.dataset)} samples in {time_end - time_start:.2f}s"
        )
        # Expose selection metadata for downstream inspection utilities.
        self.episodes = episodes
        self.metadata = meta

        self._bbox_index: dict[Tuple[int, int], int] = {}
        self._bbox_keys: List[Tuple[int, int]] = []
        self._bbox_file = None
        bbox_path = meta.root / "meta" / "bboxes.jsonl"
        if bbox_path.exists():
            print(f"Loading bounding boxes from {bbox_path}")
            time_start = time.time()
            episodes_set = set(episodes) if episodes is not None else None
            self._bbox_index, self._bbox_keys = _build_bbox_index(
                bbox_path, episodes_set
            )
            self._bbox_path = bbox_path
            time_end = time.time()
            print(
                f"Loaded {len(self._bbox_index)} bounding box records in {time_end - time_start:.2f}s"
            )
        else:
            self._bbox_path = None

        self.bbox_only = config.bbox_only
        self.remove_bbox = config.remove_bbox
        assert not (
            self.bbox_only and self.remove_bbox
        ), "Cannot set both bbox_only and remove_bbox to True."
        if config.bbox_only:
            print(
                "Filtering dataset to only include frames with bounding boxes."
            )
            episode_list = _resolve_episodes(meta, episodes)
            episode_lengths = [
                meta.episodes[ep_id]["length"] for ep_id in episode_list
            ]
            episode_start_indices = np.cumsum([0] + episode_lengths)[:-1]
            episode_to_start = dict(zip(episode_list, episode_start_indices))
            self.filtered_indices = [
                episode_to_start[ep_idx] + frame_idx
                for ep_idx, frame_idx in self._bbox_keys
                if ep_idx in episode_to_start
            ]
            print(f"Filtered dataset size: {len(self.filtered_indices)}")
            self.size = len(self.filtered_indices)
        else:
            self.filtered_indices = None
            self.size = len(self.dataset)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.filtered_indices is not None:
            idx = int(self.filtered_indices[idx])
        item = self.dataset[idx]

        # Images: single camera, add camera dimension and convert to uint8.
        img = item["observation.images.image"]
        actions = item["actions"]  # shape: (action_horizon, action_dim)

        state_len = self.state_history

        ep_idx = int(item["episode_index"].item())
        frame_idx = int(item["frame_index"].item())
        record = self._read_bbox_record(ep_idx, frame_idx)
        if record is not None and not self.remove_bbox:
            bbox = record.get("bbox")
            bbox_coords = [
                [int(x * 1000) for x in d["bbox_normalized"]] for d in bbox
            ]
            labels = [d["label"] for d in bbox]
            bbox = [
                {"bbox_2d": coords, "label": label}
                for coords, label in zip(bbox_coords, labels)
            ]
            prompt = (
                "Please return bounding boxes for all task-relevant objects in JSON format as"
                '[{"bbox_2d": [x1, y1, x2, y2], "label": "<object_name>"}]'
            )
            answer = json.dumps(bbox)
        else:
            prompt = ""
            answer = ""

        import torchvision.transforms.functional as F

        img = (img * 255).to(torch.uint8)
        img = F.resize(img, (480, 640))

        processed = {
            PROCESSED_IMAGE_KEY: img.unsqueeze(0),
            PROCESSED_IMAGE_MASK_KEY: torch.ones((1, 1), dtype=torch.bool),
            PROCESSED_ACTION_KEY: actions if not self.bbox_only else None,
            PROCESSED_STATE_KEY: torch.randn(
                (state_len, 1), dtype=torch.float32
            ),
            TASK_KEY: item.get("task"),
            GENERATION_PROMPT_KEY: prompt,
            GENERATION_ANSWER_KEY: answer,
        }
        return processed

    def _read_bbox_record(self, ep_idx: int, frame_idx: int) -> dict | None:
        if not self._bbox_index:
            return None
        offset = self._bbox_index.get((ep_idx, frame_idx))
        if offset is None or self._bbox_path is None:
            return None
        if self._bbox_file is None:
            self._bbox_file = open(self._bbox_path, "rb")
        self._bbox_file.seek(offset)
        line = self._bbox_file.readline()
        if not line:
            return None
        return json.loads(line)


def _prefetch_required_chunks(
    meta: LeRobotDatasetMetadata,
    download_videos: bool = True,
) -> None:
    chunk_ids: List[int] = sorted(
        {meta.get_episode_chunk(ep_idx) for ep_idx in meta.episodes.keys()}
    )
    if not chunk_ids:
        return

    allow_patterns: List[str] = [
        f"data/chunk-{chunk_id:03d}/*" for chunk_id in chunk_ids
    ]
    if download_videos and meta.video_keys:
        for chunk_id in chunk_ids:
            for video_key in meta.video_keys:
                allow_patterns.append(
                    f"videos/chunk-{chunk_id:03d}/{video_key}/*"
                )

    from huggingface_hub import snapshot_download, get_token

    print(f"Prefetching with token: {get_token()}")
    snapshot_download(
        repo_id=meta.repo_id,
        repo_type="dataset",
        revision=meta.revision,
        local_dir=meta.root,
        allow_patterns=allow_patterns,
        ignore_patterns=None if download_videos else ["videos/"],
        token=get_token(),
    )
