#!/usr/bin/env python3
"""
Generate a LeRobot-compatible bbox jsonl file (meta/bboxes.jsonl) from an ECOT bbox dataset.

This script performs:
  1) Task matching (primarily via ECOT first-level folder name).
  2) Within-task 1:1 demo<->episode matching using action fingerprints (first N frames).
  3) Conversion of ECOT per-step pixel bboxes -> LeRobot jsonl records with bbox_normalized.

Output:
  Writes <lerobot_root>/meta/bboxes.jsonl

Notes:
  - Fingerprinting uses actions only (states can differ in dimensionality between ECOT and LeRobot).
  - Frame count used for fingerprinting is min(max_frames, trajectory_length).
  - By default, fingerprint mismatches are treated as errors (no length fallback).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pyarrow.dataset as ds


def _norm_task_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    # LIBERO-90 ECOT folder names may include scene prefixes like:
    # "LIVING ROOM SCENE1 ..." / "KITCHEN SCENE10 ..." / "STUDY SCENE2 ..."
    # LeRobot task strings typically omit these prefixes, so strip them.
    text = re.sub(r"^(kitchen|living room|study) scene\d+\s+", "", text)
    return text


def _iter_parquet_files(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.rglob("*.parquet") if p.is_file()])


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _quantize_to_int16(arr: np.ndarray, *, scale: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    q = np.round(arr * scale).astype(np.int32)
    q = np.clip(q, -32768, 32767).astype(np.int16)
    return q


def _fingerprint_actions(actions: np.ndarray, *, quant_scale: float) -> str:
    actions = np.asarray(actions)
    if actions.ndim != 2:
        raise ValueError(f"Expected 2D actions array, got actions.ndim={actions.ndim}")
    q_action = _quantize_to_int16(actions, scale=quant_scale)
    h = hashlib.sha256()
    h.update(np.array([q_action.shape[0], q_action.shape[1]], dtype=np.int32).tobytes())
    h.update(q_action.tobytes(order="C"))
    return h.hexdigest()


def _load_ecot_actions_prefix(demo_path: Path, n_frames: int) -> np.ndarray:
    actions_path = demo_path / "actions.npy"
    if not actions_path.exists():
        raise FileNotFoundError(f"Missing actions.npy under {demo_path}")
    actions = np.load(actions_path)
    n = min(n_frames, len(actions))
    return actions[:n]


@dataclass(frozen=True)
class LeRobotEpisode:
    episode_index: int
    task_text: str
    length: int
    fp: Optional[str] = None


@dataclass(frozen=True)
class EcotDemo:
    task_text: str
    task_dir: str
    demo_dir: str
    demo_id: int
    total_steps: int
    bboxes_per_step_len: int
    step_idx_max: Optional[int]
    step_idx_min: Optional[int]
    step_idx_coverage_ok: bool
    bbox_items_total: int
    task_from_dir: str
    demo_path: Path
    bbox_path: Path


def _load_lerobot_episodes(lerobot_root: Path) -> List[LeRobotEpisode]:
    episodes_dir = lerobot_root / "meta" / "episodes"
    parquet_files = _iter_parquet_files(episodes_dir)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {episodes_dir}")
    dataset = ds.dataset([str(p) for p in parquet_files], format="parquet")
    table = dataset.to_table(columns=["episode_index", "tasks", "length"])
    episode_index_arr = table["episode_index"].to_pylist()
    tasks_arr = table["tasks"].to_pylist()
    length_arr = table["length"].to_pylist()

    episodes: List[LeRobotEpisode] = []
    for ep_i, tasks, length in zip(episode_index_arr, tasks_arr, length_arr):
        task_text = ""
        if isinstance(tasks, list) and tasks:
            task_text = str(tasks[0])
        episodes.append(LeRobotEpisode(int(ep_i), task_text, int(length), fp=None))
    episodes.sort(key=lambda e: e.episode_index)
    return episodes


def _build_task_to_episode_list(episodes: Iterable[LeRobotEpisode]) -> Dict[str, List[LeRobotEpisode]]:
    mapping: Dict[str, List[LeRobotEpisode]] = defaultdict(list)
    for ep in episodes:
        mapping[_norm_task_text(ep.task_text)].append(ep)
    for k in list(mapping.keys()):
        mapping[k].sort(key=lambda e: e.episode_index)
    return dict(mapping)


def _compute_lerobot_episode_fingerprints(
    lerobot_root: Path,
    episodes: List[LeRobotEpisode],
    *,
    max_frames: int,
    quant_scale: float,
) -> Dict[int, str]:
    data_dir = lerobot_root / "data"
    parquet_files = _iter_parquet_files(data_dir)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")
    dataset = ds.dataset([str(p) for p in parquet_files], format="parquet")
    table = dataset.to_table(
        columns=["episode_index", "frame_index", "actions"],
        filter=ds.field("frame_index") < max_frames,
    )
    ep_idx = np.asarray(table["episode_index"].to_numpy(zero_copy_only=False))
    frame_idx = np.asarray(table["frame_index"].to_numpy(zero_copy_only=False))
    actions_list = table["actions"].to_pylist()

    per_ep_rows: Dict[int, List[Tuple[int, List[float]]]] = defaultdict(list)
    for ei, fi, ac in zip(ep_idx, frame_idx, actions_list):
        per_ep_rows[int(ei)].append((int(fi), ac))

    ep_len_map = {e.episode_index: e.length for e in episodes}
    fp_map: Dict[int, str] = {}
    for ep in episodes:
        rows = per_ep_rows.get(ep.episode_index, [])
        if not rows:
            continue
        rows.sort(key=lambda t: t[0])
        ep_len = ep_len_map.get(ep.episode_index, len(rows))
        n = min(max_frames, ep_len, len(rows))
        if n <= 0:
            continue
        ac = np.asarray([rows[i][1] for i in range(n)], dtype=np.float32)
        fp_map[ep.episode_index] = _fingerprint_actions(ac, quant_scale=quant_scale)
    return fp_map


def _discover_ecot_bbox_files(ecot_root: Path) -> List[Path]:
    if not ecot_root.exists():
        raise FileNotFoundError(f"ECOT root does not exist: {ecot_root}")
    files = sorted([p for p in ecot_root.rglob("bounding_box.json") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No bounding_box.json found under {ecot_root}")
    return files


def _parse_ecot_bbox_file(bbox_path: Path, ecot_root_resolved: Path) -> EcotDemo:
    obj = json.loads(bbox_path.read_text())

    task_text = str(obj.get("task_description") or "")
    demo_id = _safe_int(obj.get("episode_id"))
    if demo_id is None:
        m = re.search(r"demo_(\d+)$", str(bbox_path.parent.name))
        if m:
            demo_id = int(m.group(1))
        else:
            raise ValueError(f"Cannot determine demo_id for {bbox_path}")

    total_steps = _safe_int(obj.get("total_steps"))
    if total_steps is None:
        total_steps = _safe_int(obj.get("num_steps")) or -1

    bboxes_per_step = obj.get("bboxes_per_step")
    bboxes_per_step_len = len(bboxes_per_step) if isinstance(bboxes_per_step, list) else -1

    step_idxs: List[int] = []
    bbox_items_total = 0
    if isinstance(bboxes_per_step, list):
        for i, step in enumerate(bboxes_per_step):
            if isinstance(step, dict):
                si = _safe_int(step.get("step_idx"))
                if si is None:
                    si = i
                step_idxs.append(si)
                bboxes = step.get("bboxes")
                if isinstance(bboxes, list):
                    bbox_items_total += len(bboxes)
            else:
                step_idxs.append(i)

    if step_idxs:
        step_idx_min = min(step_idxs)
        step_idx_max = max(step_idxs)
        expected = set(range(len(step_idxs)))
        step_idx_coverage_ok = set(step_idxs) == expected
    else:
        step_idx_min = None
        step_idx_max = None
        step_idx_coverage_ok = False

    try:
        rel = bbox_path.resolve().relative_to(ecot_root_resolved)
        task_dir = rel.parts[0] if len(rel.parts) >= 1 else ""
        demo_dir = rel.parts[1] if len(rel.parts) >= 2 else ""
    except Exception:
        task_dir = bbox_path.parent.parent.name if bbox_path.parent.parent else ""
        demo_dir = bbox_path.parent.name

    task_from_dir = task_dir
    task_from_dir = re.sub(r"_demo$", "", task_from_dir)
    task_from_dir = task_from_dir.replace("_", " ").strip()

    return EcotDemo(
        task_text=task_text,
        task_dir=task_dir,
        demo_dir=demo_dir,
        demo_id=int(demo_id),
        total_steps=int(total_steps),
        bboxes_per_step_len=int(bboxes_per_step_len),
        step_idx_max=step_idx_max,
        step_idx_min=step_idx_min,
        step_idx_coverage_ok=bool(step_idx_coverage_ok),
        bbox_items_total=int(bbox_items_total),
        task_from_dir=task_from_dir,
        demo_path=bbox_path.parent,
        bbox_path=bbox_path,
    )


def _desired_demo_length(demo: EcotDemo) -> Optional[int]:
    if demo.bboxes_per_step_len >= 0:
        return demo.bboxes_per_step_len
    if demo.total_steps >= 0:
        return demo.total_steps
    return None


def _task_key_candidates(demo: EcotDemo) -> List[str]:
    keys: List[str] = []
    if demo.task_from_dir:
        keys.append(_norm_task_text(demo.task_from_dir))
    if demo.task_text:
        keys.append(_norm_task_text(demo.task_text))
    out: List[str] = []
    seen = set()
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _match_task_key(
    demo: EcotDemo, task_to_eps: Mapping[str, List[LeRobotEpisode]]
) -> Optional[str]:
    for k in _task_key_candidates(demo):
        if k in task_to_eps:
            return k
    return None


def _match_demos_to_episodes_for_task(
    demos: List[EcotDemo],
    episodes: List[LeRobotEpisode],
    *,
    max_frames: int,
    quant_scale: float,
    max_abs_length_delta: int,
    allow_length_fallback: bool,
) -> Dict[Path, LeRobotEpisode]:
    """
    Return mapping: ecot_demo_path -> matched LeRobotEpisode
    """
    remaining_eps: Dict[int, LeRobotEpisode] = {e.episode_index: e for e in episodes}
    fp_to_eps: Dict[str, List[int]] = defaultdict(list)
    for e in episodes:
        if e.fp:
            fp_to_eps[e.fp].append(e.episode_index)

    def demo_sort_key(d: EcotDemo) -> Tuple[int, int, int]:
        demo_len = _desired_demo_length(d) or -1
        return (-demo_len, -d.bbox_items_total, d.demo_id)

    assignment: Dict[Path, LeRobotEpisode] = {}
    for demo in sorted(demos, key=demo_sort_key):
        if demo.bboxes_per_step_len >= 0 and not demo.step_idx_coverage_ok:
            raise RuntimeError(f"Invalid step_idx coverage for {demo.bbox_path}")
        desired_len = _desired_demo_length(demo)
        if desired_len is None:
            raise RuntimeError(f"Missing demo length for {demo.bbox_path}")

        n_fp = min(max_frames, desired_len)
        demo_actions = _load_ecot_actions_prefix(demo.demo_path, n_fp)
        demo_fp = _fingerprint_actions(demo_actions, quant_scale=quant_scale)

        candidate_ep_ids: List[int] = []
        for ep_id in fp_to_eps.get(demo_fp, []):
            ep = remaining_eps.get(ep_id)
            if ep is None:
                continue
            if demo.step_idx_max is not None and demo.step_idx_max >= ep.length:
                continue
            if abs(desired_len - ep.length) > max_abs_length_delta:
                continue
            candidate_ep_ids.append(ep_id)

        if len(candidate_ep_ids) == 1:
            ep_id = candidate_ep_ids[0]
            ep = remaining_eps.pop(ep_id)
            if ep.fp:
                fp_to_eps[ep.fp] = [x for x in fp_to_eps[ep.fp] if x != ep_id]
            assignment[demo.demo_path] = ep
            continue

        if len(candidate_ep_ids) > 1:
            raise RuntimeError(
                f"Ambiguous fingerprint match for {demo.bbox_path}: candidates={len(candidate_ep_ids)}"
            )

        if not allow_length_fallback:
            raise RuntimeError(
                f"Fingerprint not found in task episodes for {demo.bbox_path} (task={demo.task_from_dir})"
            )

        # Optional fallback: closest length among remaining episodes
        candidates: List[Tuple[int, LeRobotEpisode]] = []
        for ep in remaining_eps.values():
            if demo.step_idx_max is not None and demo.step_idx_max >= ep.length:
                continue
            candidates.append((abs(desired_len - ep.length), ep))
        if not candidates:
            raise RuntimeError(f"No remaining episode satisfies step range for {demo.bbox_path}")
        candidates.sort(key=lambda t: (t[0], t[1].length, t[1].episode_index))
        best_delta, best_ep = candidates[0]
        if best_delta > max_abs_length_delta:
            raise RuntimeError(
                f"Length mismatch for {demo.bbox_path}: ecot={desired_len}, best_lerobot={best_ep.length}, delta={best_delta}"
            )
        remaining_eps.pop(best_ep.episode_index, None)
        assignment[demo.demo_path] = best_ep

    return assignment


def _infer_bbox_image_size(demo_path: Path) -> Tuple[int, int]:
    """
    Best-effort image size inference for pixel bbox normalization.
    Falls back to (256, 256).
    """
    marked = demo_path / "images"
    if marked.exists():
        for p in sorted(marked.glob("*.png"))[:3]:
            try:
                from PIL import Image

                with Image.open(p) as im:
                    return int(im.size[0]), int(im.size[1])
            except Exception:
                pass
    return 256, 256


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _convert_ecot_bbox_json_to_records(
    demo: EcotDemo,
    episode_index: int,
) -> List[Dict[str, Any]]:
    """
    Convert ECOT bounding_box.json into LeRobot bbox jsonl records.
    Returns a list of dicts (one per frame that has at least one bbox).
    """
    obj = json.loads(demo.bbox_path.read_text())
    bboxes_per_step = obj.get("bboxes_per_step")
    if not isinstance(bboxes_per_step, list):
        raise RuntimeError(f"Unexpected bboxes_per_step format in {demo.bbox_path}")

    width, height = _infer_bbox_image_size(demo.demo_path)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid inferred image size for {demo.demo_path}: {(width, height)}")

    records: List[Dict[str, Any]] = []
    for i, step in enumerate(bboxes_per_step):
        if not isinstance(step, dict):
            continue
        frame_idx = _safe_int(step.get("step_idx"))
        if frame_idx is None:
            frame_idx = i

        bbox_list = step.get("bboxes", [])
        if not isinstance(bbox_list, list) or not bbox_list:
            continue

        out_bbox: List[Dict[str, Any]] = []
        for b in bbox_list:
            if not isinstance(b, dict):
                continue
            label = str(b.get("label") or "").strip()
            box = b.get("box")
            if not label or not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            # Normalize pixel coords to [0,1] then clip.
            bn = [
                _clip01(x1 / width),
                _clip01(y1 / height),
                _clip01(x2 / width),
                _clip01(y2 / height),
            ]
            out_bbox.append({"label": label, "bbox_normalized": bn})

        if not out_bbox:
            continue
        records.append(
            {
                "episode_index": int(episode_index),
                "frame_index": int(frame_idx),
                "bbox": out_bbox,
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate LeRobot meta/bboxes.jsonl by aligning ECOT demos to LeRobot episodes.",
    )
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--ecot-root", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--quant-scale", type=float, default=1000.0)
    parser.add_argument("--max-abs-length-delta", type=int, default=5)
    parser.add_argument(
        "--allow-length-fallback",
        action="store_true",
        help="If set, fall back to length-based matching when fingerprint matching fails.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing meta/bboxes.jsonl if present.",
    )
    args = parser.parse_args()

    lerobot_root = args.lerobot_root
    ecot_root = args.ecot_root
    max_frames = int(args.max_frames)
    if max_frames < 100:
        raise ValueError("--max-frames must be >= 100 (or set to 100); script uses min(max_frames, trajectory_length).")

    # Load LeRobot episodes + fingerprints
    episodes = _load_lerobot_episodes(lerobot_root)
    fp_map = _compute_lerobot_episode_fingerprints(
        lerobot_root,
        episodes,
        max_frames=max_frames,
        quant_scale=float(args.quant_scale),
    )
    episodes = [
        LeRobotEpisode(
            episode_index=e.episode_index,
            task_text=e.task_text,
            length=e.length,
            fp=fp_map.get(e.episode_index),
        )
        for e in episodes
    ]
    task_to_eps = _build_task_to_episode_list(episodes)

    # Load ECOT demos
    bbox_files = _discover_ecot_bbox_files(ecot_root)
    ecot_root_resolved = ecot_root.resolve()
    demos = [_parse_ecot_bbox_file(p, ecot_root_resolved) for p in bbox_files]

    # Group demos by matched task key
    demos_by_task: Dict[str, List[EcotDemo]] = defaultdict(list)
    missing_task: List[EcotDemo] = []
    for d in demos:
        task_key = _match_task_key(d, task_to_eps)
        if task_key is None:
            missing_task.append(d)
        else:
            demos_by_task[task_key].append(d)

    if missing_task:
        sample = missing_task[0]
        raise RuntimeError(
            f"{len(missing_task)} ECOT demos have tasks not found in LeRobot. "
            f"Example task_from_dir={sample.task_from_dir!r}, bbox={sample.bbox_path}"
        )

    # Match demos to episodes within each task
    demo_to_episode: Dict[Path, LeRobotEpisode] = {}
    for task_key, group in demos_by_task.items():
        eps = task_to_eps[task_key]
        mapping = _match_demos_to_episodes_for_task(
            group,
            eps,
            max_frames=max_frames,
            quant_scale=float(args.quant_scale),
            max_abs_length_delta=int(args.max_abs_length_delta),
            allow_length_fallback=bool(args.allow_length_fallback),
        )
        demo_to_episode.update(mapping)

    # Convert bbox json to jsonl records
    records: List[Dict[str, Any]] = []
    for d in demos:
        ep = demo_to_episode.get(d.demo_path)
        if ep is None:
            raise RuntimeError(f"Internal error: no matched episode for demo {d.demo_path}")
        records.extend(_convert_ecot_bbox_json_to_records(d, ep.episode_index))

    # Sort for stable output
    records.sort(key=lambda r: (int(r["episode_index"]), int(r["frame_index"])))

    out_path = lerobot_root / "meta" / "bboxes.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} exists. Re-run with --overwrite to replace it.")

    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Wrote bbox jsonl:")
    print(f"  path: {out_path}")
    print(f"  records: {len(records)}")
    per_ep = Counter(int(r["episode_index"]) for r in records)
    print(f"  episodes_with_bboxes: {len(per_ep)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

