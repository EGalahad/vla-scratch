#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _ensure_empty_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output path already exists: {path}. Pass --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def _iter_task_dirs(er_root: Path) -> List[Path]:
    # ER dataset layout: <root>/<group>/<task_id>/
    out: List[Path] = []
    for group in sorted(er_root.iterdir()):
        if not group.is_dir():
            continue
        for task in sorted(group.iterdir()):
            if not task.is_dir():
                continue
            if (task / "bounding_box.json").exists() and (task / "images").exists():
                out.append(task)
    return out


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_from_image_name(name: str) -> Optional[int]:
    m = re.search(r"seed(\d+)", name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _infer_image_size(image_path: Path) -> Tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(image_path) as im:
            w, h = im.size
            return int(w), int(h)
    except Exception:
        # ER images are usually 256x256; keep a safe fallback.
        return 256, 256


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


@dataclass(frozen=True)
class EpisodeItem:
    episode_index: int
    task_text: str
    task_dir: str
    seed: int
    src_image: Path
    dst_image: Path


def _collect_episodes(
    er_root: Path,
    *,
    max_episodes: Optional[int] = None,
) -> List[EpisodeItem]:
    episodes: List[EpisodeItem] = []
    ep_idx = 0
    for task_path in _iter_task_dirs(er_root):
        obj = _load_json(task_path / "bounding_box.json")
        task_text = str(obj.get("task_description") or "").strip()
        if not task_text:
            task_text = task_path.name

        # Build seed->image mapping from images folder.
        images = sorted((task_path / "images").glob("*.png"))
        seed_to_image: Dict[int, Path] = {}
        for p in images:
            seed = _seed_from_image_name(p.name)
            if seed is None:
                continue
            seed_to_image[seed] = p

        # Ensure we only create episodes for seeds that exist in bbox json.
        bps = obj.get("bboxes_per_step")
        if not isinstance(bps, list):
            raise RuntimeError(f"Unexpected bboxes_per_step format in {task_path}")

        for step in bps:
            if not isinstance(step, dict):
                continue
            seed_raw = step.get("seed")
            try:
                seed = int(seed_raw)
            except Exception:
                continue
            img = seed_to_image.get(seed)
            if img is None:
                # Skip if image is missing (shouldn't happen, but keep robust).
                continue

            # Destination filled later by caller (depends on output root).
            episodes.append(
                EpisodeItem(
                    episode_index=ep_idx,
                    task_text=task_text,
                    task_dir=str(task_path.relative_to(er_root)),
                    seed=seed,
                    src_image=img,
                    dst_image=Path(""),
                )
            )
            ep_idx += 1
            if max_episodes is not None and len(episodes) >= max_episodes:
                return episodes
    return episodes


def _build_tasks(episodes: List[EpisodeItem]) -> Tuple[List[str], Dict[str, int]]:
    # Stable ordering: sort by text.
    uniq = sorted({e.task_text.strip().lower(): e.task_text.strip() for e in episodes}.values())
    task_to_index = {t: i for i, t in enumerate(uniq)}
    return uniq, task_to_index


def _make_features(
    *,
    image_h: int,
    image_w: int,
    state_dim: int,
    action_dim: int,
    fps: int,
) -> dict:
    # Use 'video' dtype to match the standard LeRobot v3.0 format
    # (frames are stored in videos/<key>/chunk-*/file-*.mp4 and decoded on-the-fly).
    # Important: LeRobotDataset passes `features` to `datasets.Dataset.from_parquet`.
    # The datasets library casts parquet tables to this exact schema, so we must
    # include every column we write in parquet (not just state/actions).
    return {
        "image": {
            "dtype": "video",
            "shape": [image_h, image_w, 3],
            "names": ["height", "width", "rgb"],
            "info": {
                "video.height": image_h,
                "video.width": image_w,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        },
        "wrist_image": {
            "dtype": "video",
            "shape": [image_h, image_w, 3],
            "names": ["height", "width", "rgb"],
            "info": {
                "video.height": image_h,
                "video.width": image_w,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        },
        "state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": None,
            "fps": fps,
        },
        "actions": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": None,
            "fps": fps,
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": fps},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
        "index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
        "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
    }


def _write_tasks_parquet(tasks: List[str], out_root: Path) -> None:
    import pandas as pd
    from lerobot.datasets.utils import write_tasks

    df = pd.DataFrame({"task_index": list(range(len(tasks)))}, index=tasks)
    write_tasks(df, out_root)


def _write_info_json(
    *,
    repo_id: str,
    out_root: Path,
    robot_type: str,
    fps: int,
    features: dict,
    chunks_size: int,
    data_files_size_in_mb: int,
    video_files_size_in_mb: int,
) -> None:
    # Use official helper to ensure schema-compatible info.json.
    from lerobot.datasets.utils import create_empty_dataset_info, write_json, INFO_PATH
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION

    info = create_empty_dataset_info(
        CODEBASE_VERSION,
        fps,
        features,
        use_videos=True,
        robot_type=robot_type,
        chunks_size=chunks_size,
        data_files_size_in_mb=data_files_size_in_mb,
        video_files_size_in_mb=video_files_size_in_mb,
    )
    # A couple of fields are informative only; keep consistent with other datasets.
    info["total_episodes"] = 0
    info["total_frames"] = 0
    info["total_tasks"] = 0
    write_json(info, out_root / INFO_PATH)


def _write_modality_json(out_root: Path, *, state_dim: int, action_dim: int) -> None:
    # Minimal modality mapping compatible with our training code.
    # Keep keys aligned with existing libero modality.json structure.
    payload = {
        "state": {"state": {"start": 0, "end": state_dim}},
        "action": {"action": {"start": 0, "end": action_dim}},
        "video": {
            "primary_image": {"original_key": "image"},
            "wrist_image": {"original_key": "wrist_image"},
        },
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    }
    (out_root / "meta").mkdir(parents=True, exist_ok=True)
    (out_root / "meta" / "modality.json").write_text(
        json.dumps(payload, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_stats_json(
    *,
    out_root: Path,
    num_frames: int,
    state_dim: int,
    action_dim: int,
    fps: int,
    num_episodes: Optional[int] = None,
    num_tasks: Optional[int] = None,
) -> None:
    # Keep minimal numeric stats so downstream normalization code won't crash if it expects file.
    from lerobot.datasets.utils import write_stats

    zeros_state = [0.0] * state_dim
    zeros_action = [0.0] * action_dim
    stats = {
        "state": {
            "min": zeros_state,
            "max": zeros_state,
            "mean": zeros_state,
            "std": zeros_state,
            "count": [num_frames],
        },
        "actions": {
            "min": zeros_action,
            "max": zeros_action,
            "mean": zeros_action,
            "std": zeros_action,
            "count": [num_frames],
        },
        "timestamp": {
            "min": [0.0],
            "max": [0.0],
            "mean": [0.0],
            "std": [0.0],
            "count": [num_frames],
        },
        "frame_index": {
            "min": [0],
            "max": [0],
            "mean": [0.0],
            "std": [0.0],
            "count": [num_frames],
        },
        "episode_index": {
            "min": [0],
            "max": [max(0, int((num_episodes or num_frames) - 1))],
            "mean": [0.0],
            "std": [0.0],
            "count": [num_frames],
        },
        "index": {
            "min": [0],
            "max": [max(0, num_frames - 1)],
            "mean": [0.0],
            "std": [0.0],
            "count": [num_frames],
        },
        "task_index": {
            "min": [0],
            "max": [max(0, int((num_tasks or 1) - 1))],
            "mean": [0.0],
            "std": [0.0],
            "count": [num_frames],
        },
        "fps": fps,
    }
    write_stats(stats, out_root)


def _write_data_parquet(
    *,
    er_root: Path,
    out_root: Path,
    repo_id: str,
    episodes: List[EpisodeItem],
    task_to_index: Dict[str, int],
    state_dim: int,
    action_dim: int,
    fps: int,
    chunks_size: int,
    overwrite: bool,
) -> Tuple[List[EpisodeItem], int]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    # For video-based datasets, images are not stored in the parquet file; only numeric modalities are.
    fixed_episodes: List[EpisodeItem] = [
        EpisodeItem(
            episode_index=e.episode_index,
            task_text=e.task_text,
            task_dir=e.task_dir,
            seed=e.seed,
            src_image=e.src_image,
            dst_image=e.src_image,
        )
        for e in episodes
    ]

    num_frames = len(fixed_episodes)
    data_dir = out_root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Important: write this parquet WITHOUT HuggingFace schema metadata.
    # LeRobot reference datasets (e.g., libero_spatial_no_noops_lerobot) store data parquet
    # with only pandas metadata, not huggingface metadata. Writing via `datasets.Dataset.to_parquet`
    # adds huggingface schema metadata that can cause cast failures on nested list columns.
    state_arr = pa.array(
        np.zeros((num_frames, state_dim), dtype=np.float32).tolist(),
        type=pa.list_(pa.float32()),
    )
    actions_arr = pa.array(
        np.zeros((num_frames, action_dim), dtype=np.float32).tolist(),
        type=pa.list_(pa.float32()),
    )
    table = pa.Table.from_pydict(
        {
            "state": state_arr,
            "actions": actions_arr,
            # IMPORTANT: LeRobotDataset treats parquet `timestamp` as episode-relative time (seconds),
            # then shifts it by the episode video's `from_timestamp`. If we store global timestamps
            # here (i/fps), they get added twice, causing out-of-range frame indices at decode time.
            "timestamp": np.zeros((num_frames,), dtype=np.float32),
            "frame_index": np.zeros((num_frames,), dtype=np.int64),
            "episode_index": np.asarray(
                [e.episode_index for e in fixed_episodes], dtype=np.int64
            ),
            "index": np.arange(num_frames, dtype=np.int64),
            "task_index": np.asarray(
                [task_to_index[e.task_text] for e in fixed_episodes], dtype=np.int64
            ),
        }
    )
    pq.write_table(
        table,
        data_dir / "file-000.parquet",
        compression="snappy",
        use_dictionary=True,
    )
    return fixed_episodes, num_frames


def _write_videos(
    *,
    out_root: Path,
    episodes: List[EpisodeItem],
    fps: int,
    vcodec: str = "libsvtav1",
) -> None:
    from lerobot.datasets.video_utils import encode_video_frames

    # We store all frames sequentially into a single mp4 shard (chunk-000/file-000.mp4)
    # and use per-episode from/to timestamps to locate each episode segment.
    tmp_dir = out_root / "tmp_frames"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    frames_dir = tmp_dir / "image"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Frame order follows dataset index (which we set to episode_index order).
    for global_idx, e in enumerate(episodes):
        dst = frames_dir / f"frame-{global_idx:06d}.png"
        _link_or_copy(e.src_image, dst)

    video_path = out_root / "videos" / "image" / "chunk-000" / "file-000.mp4"
    encode_video_frames(frames_dir, video_path, fps=int(fps), vcodec=vcodec, overwrite=True)

    # wrist_image duplicates image: copy the mp4 to avoid re-encoding.
    wrist_path = out_root / "videos" / "wrist_image" / "chunk-000" / "file-000.mp4"
    wrist_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video_path, wrist_path)

    shutil.rmtree(tmp_dir, ignore_errors=True)


def _write_episodes_parquet(
    *,
    out_root: Path,
    episodes: List[EpisodeItem],
    num_frames: int,
    fps: int,
) -> None:
    from datasets import Dataset, Features, Sequence, Value
    from lerobot.datasets.utils import DEFAULT_EPISODES_PATH

    ep_indices = [e.episode_index for e in episodes]
    length = [1 for _ in episodes]
    dataset_from_index = [i for i in range(num_frames)]
    dataset_to_index = [i + 1 for i in range(num_frames)]
    tasks = [[e.task_text] for e in episodes]

    # Data stored in a single parquet shard.
    data_chunk_index = [0 for _ in episodes]
    data_file_index = [0 for _ in episodes]

    # Videos stored in a single shard too.
    vid_chunk_index = [0 for _ in episodes]
    vid_file_index = [0 for _ in episodes]
    # Each episode has length 1; map to the single frame at global index i.
    # Use from/to timestamp in the concatenated mp4.
    from_ts = [i / float(fps) for i in range(num_frames)]
    to_ts = [(i + 1) / float(fps) for i in range(num_frames)]

    # Provide meta/episodes chunk/file indices for compatibility with existing datasets.
    meta_chunk_index = [0 for _ in episodes]
    meta_file_index = [0 for _ in episodes]

    features = Features(
        {
            "episode_index": Value("int64"),
            "data/chunk_index": Value("int64"),
            "data/file_index": Value("int64"),
            "dataset_from_index": Value("int64"),
            "dataset_to_index": Value("int64"),
            "videos/image/chunk_index": Value("int64"),
            "videos/image/file_index": Value("int64"),
            "videos/image/from_timestamp": Value("float64"),
            "videos/image/to_timestamp": Value("float64"),
            "videos/wrist_image/chunk_index": Value("int64"),
            "videos/wrist_image/file_index": Value("int64"),
            "videos/wrist_image/from_timestamp": Value("float64"),
            "videos/wrist_image/to_timestamp": Value("float64"),
            "tasks": Sequence(Value("string")),
            "length": Value("int64"),
            "meta/episodes/chunk_index": Value("int64"),
            "meta/episodes/file_index": Value("int64"),
        }
    )
    ds = Dataset.from_dict(
        {
            "episode_index": ep_indices,
            "data/chunk_index": data_chunk_index,
            "data/file_index": data_file_index,
            "dataset_from_index": dataset_from_index,
            "dataset_to_index": dataset_to_index,
            "videos/image/chunk_index": vid_chunk_index,
            "videos/image/file_index": vid_file_index,
            "videos/image/from_timestamp": from_ts,
            "videos/image/to_timestamp": to_ts,
            "videos/wrist_image/chunk_index": vid_chunk_index,
            "videos/wrist_image/file_index": vid_file_index,
            "videos/wrist_image/from_timestamp": from_ts,
            "videos/wrist_image/to_timestamp": to_ts,
            "tasks": tasks,
            "length": length,
            "meta/episodes/chunk_index": meta_chunk_index,
            "meta/episodes/file_index": meta_file_index,
        },
        features=features,
    )

    ep_path = out_root / DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0)
    ep_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(ep_path))


def _write_mapping_jsonl(out_root: Path, episodes: List[EpisodeItem]) -> Path:
    path = out_root / "meta" / "er_mapping.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in episodes:
            f.write(
                json.dumps(
                    {
                        "episode_index": e.episode_index,
                        "frame_index": 0,
                        "task": e.task_text,
                        "task_dir": e.task_dir,
                        "seed": e.seed,
                        "src_image": str(e.src_image),
                        "image": str(e.dst_image),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def _write_bboxes_jsonl(
    *,
    er_root: Path,
    out_root: Path,
    episodes: List[EpisodeItem],
    filter_unknown: bool,
    require_camera_main: bool,
) -> Path:
    # Build a fast lookup: (task_dir, seed) -> (episode_index, image_path)
    key_to_episode: Dict[Tuple[str, int], EpisodeItem] = {(e.task_dir, e.seed): e for e in episodes}
    out_path = out_root / "meta" / "bboxes.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for task_path in _iter_task_dirs(er_root):
            task_dir = str(task_path.relative_to(er_root))
            obj = _load_json(task_path / "bounding_box.json")
            bps = obj.get("bboxes_per_step")
            if not isinstance(bps, list):
                continue
            # Seed->image for size inference (use src image to avoid depending on dst naming).
            seed_to_img: Dict[int, Path] = {}
            for p in sorted((task_path / "images").glob("*.png")):
                s = _seed_from_image_name(p.name)
                if s is not None:
                    seed_to_img[s] = p

            for step in bps:
                if not isinstance(step, dict):
                    continue
                try:
                    seed = int(step.get("seed"))
                except Exception:
                    continue
                ep = key_to_episode.get((task_dir, seed))
                if ep is None:
                    continue

                bbox_list = step.get("bboxes") or []
                if not isinstance(bbox_list, list) or not bbox_list:
                    continue

                img_path = seed_to_img.get(seed)
                if img_path is None:
                    # fallback to dst image if available
                    img_path = ep.dst_image if ep.dst_image else None
                if img_path is None:
                    continue
                w, h = _infer_image_size(img_path)
                if w <= 0 or h <= 0:
                    continue

                out_bbox: List[Dict[str, Any]] = []
                for b in bbox_list:
                    if not isinstance(b, dict):
                        continue
                    label = str(b.get("label") or "").strip()
                    if not label:
                        continue
                    if filter_unknown:
                        ll = label.strip().lower()
                        # ER bbox labels often look like "unknown" or "unknown 5".
                        if ll == "unknown" or ll.startswith("unknown "):
                            continue
                    if require_camera_main:
                        cam = str(b.get("camera") or "").strip().lower()
                        if cam and cam != "main":
                            continue
                    box = b.get("box")
                    if not isinstance(box, (list, tuple)) or len(box) != 4:
                        continue
                    x1 = _safe_float(box[0])
                    y1 = _safe_float(box[1])
                    x2 = _safe_float(box[2])
                    y2 = _safe_float(box[3])
                    if None in (x1, y1, x2, y2):
                        continue
                    bn = [
                        _clip01(float(x1) / float(w)),
                        _clip01(float(y1) / float(h)),
                        _clip01(float(x2) / float(w)),
                        _clip01(float(y2) / float(h)),
                    ]
                    out_bbox.append({"label": label, "bbox_normalized": bn})

                if not out_bbox:
                    continue

                out_f.write(
                    json.dumps(
                        {
                            "episode_index": int(ep.episode_index),
                            "frame_index": 0,
                            "bbox": out_bbox,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert ER ECOT dataset to LeRobot v3.0-like format (seed -> episode, length=1).",
    )
    parser.add_argument(
        "--er-ecot-root",
        type=Path,
        default=Path("data/er_libero_ecot_data"),
        help="Path to ER ECOT root (contains er_goal/ er_object/ ...).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output dataset parent directory (will create <output-root>/<repo-id>/).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="LeRobot repo_id folder name to create under output-root.",
    )
    parser.add_argument("--robot-type", type=str, default="franka")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--chunks-size", type=int, default=1000)
    parser.add_argument("--data-files-size-mb", type=int, default=100)
    parser.add_argument("--video-files-size-mb", type=int, default=200)
    parser.add_argument("--video-codec", type=str, default="libsvtav1")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--keep-unknown",
        action="store_true",
        help="If set, do NOT filter label=='unknown' in bboxes.",
    )
    parser.add_argument(
        "--allow-non-main-camera",
        action="store_true",
        help="If set, keep bboxes from cameras other than 'main'.",
    )
    args = parser.parse_args()

    er_root = args.er_ecot_root.expanduser().resolve()
    if not er_root.exists():
        raise FileNotFoundError(f"ER ECOT root not found: {er_root}")

    out_parent = args.output_root.expanduser().resolve()
    out_ds_root = out_parent / args.repo_id
    _ensure_empty_dir(out_ds_root, overwrite=bool(args.overwrite))
    (out_ds_root / "meta").mkdir(parents=True, exist_ok=True)

    episodes = _collect_episodes(er_root, max_episodes=args.max_episodes)
    if not episodes:
        raise RuntimeError(f"No episodes discovered under {er_root}")

    # Infer image size from first episode source image.
    w, h = _infer_image_size(episodes[0].src_image)
    features = _make_features(
        image_h=h,
        image_w=w,
        state_dim=int(args.state_dim),
        action_dim=int(args.action_dim),
        fps=int(args.fps),
    )

    tasks, task_to_index = _build_tasks(episodes)

    _write_info_json(
        repo_id=args.repo_id,
        out_root=out_ds_root,
        robot_type=args.robot_type,
        fps=int(args.fps),
        features=features,
        chunks_size=int(args.chunks_size),
        data_files_size_in_mb=int(args.data_files_size_mb),
        video_files_size_in_mb=int(args.video_files_size_mb),
    )
    _write_modality_json(
        out_ds_root, state_dim=int(args.state_dim), action_dim=int(args.action_dim)
    )
    _write_tasks_parquet(tasks, out_ds_root)

    fixed_episodes, num_frames = _write_data_parquet(
        er_root=er_root,
        out_root=out_ds_root,
        repo_id=args.repo_id,
        episodes=episodes,
        task_to_index=task_to_index,
        state_dim=int(args.state_dim),
        action_dim=int(args.action_dim),
        fps=int(args.fps),
        chunks_size=int(args.chunks_size),
        overwrite=bool(args.overwrite),
    )

    _write_videos(
        out_root=out_ds_root,
        episodes=fixed_episodes,
        fps=int(args.fps),
        vcodec=str(args.video_codec),
    )
    _write_episodes_parquet(
        out_root=out_ds_root,
        episodes=fixed_episodes,
        num_frames=num_frames,
        fps=int(args.fps),
    )
    _write_stats_json(
        out_root=out_ds_root,
        num_frames=num_frames,
        state_dim=int(args.state_dim),
        action_dim=int(args.action_dim),
        fps=int(args.fps),
        num_episodes=len(fixed_episodes),
        num_tasks=len(tasks),
    )
    _write_mapping_jsonl(out_ds_root, fixed_episodes)
    _write_bboxes_jsonl(
        er_root=er_root,
        out_root=out_ds_root,
        episodes=fixed_episodes,
        filter_unknown=not bool(args.keep_unknown),
        require_camera_main=not bool(args.allow_non_main_camera),
    )

    # Patch info.json totals now that we know counts.
    info_path = out_ds_root / "meta" / "info.json"
    # lerobot uses <root>/meta/info.json (INFO_PATH), which is "meta/info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["total_episodes"] = len(fixed_episodes)
        info["total_frames"] = num_frames
        info["total_tasks"] = len(tasks)
        # Many training pipelines require a split; default everything to train.
        info["splits"] = {"train": f"0:{len(fixed_episodes)}"}
        info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote LeRobot-like dataset to: {out_ds_root}")
    print(f"Episodes: {len(fixed_episodes)}  Frames: {num_frames}  Tasks: {len(tasks)}")
    print(f"Mapping: {out_ds_root / 'meta' / 'er_mapping.jsonl'}")
    print(f"BBoxes:   {out_ds_root / 'meta' / 'bboxes.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
