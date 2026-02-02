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
from vla_scratch.datasets.utils.paligemma_bbox_format import (
    paligemma_detect_answer,
    paligemma_detect_prompt,
    use_paligemma_tokens_enabled,
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


def _load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


class DontBlindDataset(torch.utils.data.Dataset):
    """
    Minimal LeRobot dataset wrapper for BlindVLA.

    - Keeps actions as-is (no extra transforms).
    - Emits processed keys defined in `vla_scratch.transforms.data_keys`.
    - If `meta/bboxes.jsonl` is present, resolves (episode_index, frame_index) -> bbox list.
    """

    def __init__(self, config: "DontBlindConfig"):
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

        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            root=meta_root,
            delta_timestamps=delta_timestamps,
            episodes=episodes,
        )
        # Expose selection metadata for downstream inspection utilities.
        self.episodes = episodes
        self.metadata = meta

        self._bbox_records: List[dict] = []
        self._bbox_idx_map: dict[Tuple[int, int], int] = {}
        bbox_path = meta.root / "meta" / "bboxes.jsonl"
        if bbox_path.exists():
            print(f"Loading bounding boxes from {bbox_path}")
            records = _load_jsonl(bbox_path)
            records.sort(
                key=lambda r: (int(r["episode_index"]), int(r["frame_index"]))
            )
            self._bbox_records = [
                r for r in records if (int(r["episode_index"]) in episodes)
            ]
            for bbox_idx, r in enumerate(records):
                key = (int(r["episode_index"]), int(r["frame_index"]))
                self._bbox_idx_map.setdefault(key, bbox_idx)

        if config.bbox_only:
            print(
                "Filtering dataset to only include frames with bounding boxes."
            )
            episode_lengths = [
                meta.episodes[ep_id]["length"] for ep_id in episodes
            ]
            episode_start_indices = np.cumsum([0] + episode_lengths)[:-1]
            self.filtered_indices = [
                episode_start_indices[episodes.index(int(r["episode_index"]))]
                + int(r["frame_index"])
                for r in self._bbox_records
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

        state_len = self.state_history + 1

        ep_idx = int(item["episode_index"].item())
        frame_idx = int(item["frame_index"].item())
        bbox_idx = self._bbox_idx_map.get((ep_idx, frame_idx), -1)
        if 0 <= bbox_idx < len(self._bbox_records):
            bbox = self._bbox_records[bbox_idx].get("bbox")
            labels = [d["label"] for d in bbox]
            if use_paligemma_tokens_enabled():
                prompt = paligemma_detect_prompt()
                answer = paligemma_detect_answer(
                    [d["bbox_normalized"] for d in bbox], labels
                )
            else:
                bbox_coords = [
                    [int(x * 1000) for x in d["bbox_normalized"]] for d in bbox
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
        else:
            prompt = ""
            answer = ""

        processed = {
            PROCESSED_IMAGE_KEY: (img * 255).to(torch.uint8).unsqueeze(0),
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
