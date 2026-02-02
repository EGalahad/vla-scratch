#!/usr/bin/env python3
"""
Check whether an ECOT-style bbox dataset can be aligned 1:1 with a LeRobot dataset.

This script is intentionally non-destructive: it only reads inputs and reports.

Expected LeRobot layout (root points to the dataset folder):
  <lerobot_root>/
    meta/
      episodes/
        chunk-*/file-*.parquet
      tasks.parquet   (optional for this checker)
    data/
      chunk-*/file-*.parquet

Expected ECOT layout (as produced by split_bbox_and_pointing_to_demos.py style):
  <ecot_root>/
    <instruction_or_file_stem>/
      demo_<k>/
        bounding_box.json

Important:
  We match tasks primarily from ECOT's first-level folder name (file stem / instruction slug),
  because it is often more reliable than bounding_box.json's task_description. Then, within each
  task, we align demos to LeRobot episodes by length/step constraints (not by demo_id index).
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

import pyarrow as pa
import pyarrow.dataset as ds
import numpy as np


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


def _load_lerobot_episodes(lerobot_root: Path) -> List[LeRobotEpisode]:
    episodes_dir = lerobot_root / "meta" / "episodes"
    parquet_files = _iter_parquet_files(episodes_dir)
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found under {episodes_dir} (expected LeRobot meta episodes parquet)."
        )

    dataset = ds.dataset([str(p) for p in parquet_files], format="parquet")
    # tasks is stored as a list/ndarray-like in parquet; read as list<string>.
    table = dataset.to_table(columns=["episode_index", "tasks", "length"])

    # Convert with minimal overhead; tasks can be list with 1 element.
    episode_index_arr = table["episode_index"].to_pylist()
    tasks_arr = table["tasks"].to_pylist()
    length_arr = table["length"].to_pylist()

    episodes: List[LeRobotEpisode] = []
    for ep_i, tasks, length in zip(episode_index_arr, tasks_arr, length_arr):
        task_text = ""
        if isinstance(tasks, list) and tasks:
            task_text = str(tasks[0])
        episodes.append(
            LeRobotEpisode(
                episode_index=int(ep_i),
                task_text=task_text,
                length=int(length),
            )
        )
    episodes.sort(key=lambda e: e.episode_index)
    return episodes


def _discover_ecot_demos(ecot_root: Path) -> List[Path]:
    if not ecot_root.exists():
        raise FileNotFoundError(f"ECOT root does not exist: {ecot_root}")
    demos: List[Path] = []
    for bbox_path in ecot_root.rglob("bounding_box.json"):
        if bbox_path.is_file():
            demos.append(bbox_path)
    demos.sort()
    if not demos:
        raise FileNotFoundError(
            f"No bounding_box.json found under {ecot_root} (expected ECOT layout)."
        )
    return demos


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _parse_ecot_bbox_file(bbox_path: Path) -> EcotDemo:
    obj = json.loads(bbox_path.read_text())

    task_text = str(obj.get("task_description") or "")
    demo_id = _safe_int(obj.get("episode_id"))
    if demo_id is None:
        # fallback: parse demo_<k> from path
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
                # Unexpected format; still count coverage by list index.
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

    # Task folder is immediate child of ecot_root. Use resolved paths to avoid
    # relative-vs-absolute mismatches.
    ecot_root_resolved = getattr(_parse_ecot_bbox_file, "_ecot_root_resolved", None)
    if ecot_root_resolved is None:
        raise RuntimeError("Internal error: ECOT root not set for parser.")
    try:
        rel = bbox_path.resolve().relative_to(ecot_root_resolved)
        task_dir = rel.parts[0] if len(rel.parts) >= 1 else ""
        demo_dir = rel.parts[1] if len(rel.parts) >= 2 else ""
    except Exception:
        task_dir = bbox_path.parent.parent.name if bbox_path.parent.parent else ""
        demo_dir = bbox_path.parent.name

    # Derive task string from first-level folder name:
    # e.g. "open_the_middle_drawer_of_the_cabinet_demo" -> "open the middle drawer of the cabinet"
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
    )


def _build_task_to_episode_list(episodes: Iterable[LeRobotEpisode]) -> Dict[str, List[LeRobotEpisode]]:
    mapping: Dict[str, List[LeRobotEpisode]] = defaultdict(list)
    for ep in episodes:
        mapping[_norm_task_text(ep.task_text)].append(ep)
    for k in list(mapping.keys()):
        mapping[k].sort(key=lambda e: e.episode_index)
    return dict(mapping)


@dataclass
class MatchResult:
    ecot: EcotDemo
    matched: bool
    reason: str
    matched_episode_index: Optional[int] = None
    lerobot_episode_length: Optional[int] = None
    length_delta: Optional[int] = None


def _task_key_candidates(demo: EcotDemo) -> List[Tuple[str, str]]:
    """
    Return (source, normalized_task_key) in priority order.
    Primary key is derived from ECOT first-level folder name.
    """
    keys: List[Tuple[str, str]] = []
    if demo.task_from_dir:
        keys.append(("ecot_dir", _norm_task_text(demo.task_from_dir)))
    if demo.task_text:
        keys.append(("ecot_json", _norm_task_text(demo.task_text)))
    # Deduplicate while preserving order
    seen = set()
    out: List[Tuple[str, str]] = []
    for src, k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append((src, k))
    return out


def _desired_demo_length(demo: EcotDemo) -> Optional[int]:
    if demo.bboxes_per_step_len is not None and demo.bboxes_per_step_len >= 0:
        return demo.bboxes_per_step_len
    if demo.total_steps is not None and demo.total_steps >= 0:
        return demo.total_steps
    return None


def _quantize_to_int16(arr: np.ndarray, *, scale: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    q = np.round(arr * scale).astype(np.int32)
    q = np.clip(q, -32768, 32767).astype(np.int16)
    return q


def _fingerprint_actions(
    actions: np.ndarray,
    *,
    quant_scale: float,
) -> str:
    """
    Fingerprint a trajectory prefix using quantized action bytes only.

    Rationale:
      - Across variants, ECOT "states.npy" may have different dimensionality than
        LeRobot parquet "state" (e.g. ECOT contains rich simulator state while
        LeRobot stores a compact robot state). Actions are consistently aligned
        and sufficient to disambiguate identical natural-language tasks across scenes.
    """
    actions = np.asarray(actions)
    if actions.ndim != 2:
        raise ValueError(f"Expected 2D actions array, got actions.ndim={actions.ndim}")

    q_action = _quantize_to_int16(actions, scale=quant_scale)
    h = hashlib.sha256()
    h.update(np.array([q_action.shape[0], q_action.shape[1]], dtype=np.int32).tobytes())
    h.update(q_action.tobytes(order="C"))
    return h.hexdigest()


def _load_ecot_state_action_prefix(demo_path: Path, n_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    states_path = demo_path / "states.npy"
    actions_path = demo_path / "actions.npy"
    if not states_path.exists() or not actions_path.exists():
        raise FileNotFoundError(
            f"Missing states/actions npy under {demo_path} (need states.npy and actions.npy)."
        )
    states = np.load(states_path)
    actions = np.load(actions_path)
    n = min(n_frames, len(states), len(actions))
    return states[:n], actions[:n]


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
        raise FileNotFoundError(
            f"No parquet files found under {data_dir} (expected LeRobot data parquet)."
        )
    dataset = ds.dataset([str(p) for p in parquet_files], format="parquet")
    filt = ds.field("frame_index") < max_frames
    table = dataset.to_table(
        columns=["episode_index", "frame_index", "actions"],
        filter=filt,
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
        fp_map[ep.episode_index] = _fingerprint_actions(
            ac,
            quant_scale=quant_scale,
        )
    return fp_map


def _match_within_task(
    demos: List[EcotDemo],
    episodes: List[LeRobotEpisode],
    *,
    max_abs_length_delta: int,
    max_frames: int,
    quant_scale: float,
    require_fingerprint: bool,
) -> List[MatchResult]:
    """
    Align demos to episodes in a 1:1 way within a single task.

    Strategy:
      - Prefer matching by closest length (|demo_len - episode_len|).
      - Enforce step index constraints (max_step_idx < episode_len when available).
      - Greedy assignment to avoid duplicates, deterministic tie-breakers.
    """
    remaining_eps = {ep.episode_index: ep for ep in episodes}

    fp_to_eps: Dict[str, List[int]] = defaultdict(list)
    for ep in episodes:
        if ep.fp:
            fp_to_eps[ep.fp].append(ep.episode_index)

    # Sort demos by "difficulty": longer sequences first, then higher bbox count, then demo_id.
    def demo_sort_key(d: EcotDemo) -> Tuple[int, int, int]:
        demo_len = _desired_demo_length(d) or -1
        return (-demo_len, -d.bbox_items_total, d.demo_id)

    results: List[MatchResult] = []
    for demo in sorted(demos, key=demo_sort_key):
        if demo.bboxes_per_step_len >= 0 and not demo.step_idx_coverage_ok:
            results.append(MatchResult(demo, False, "step_idx_not_0_to_n_minus_1"))
            continue

        desired_len = _desired_demo_length(demo)
        if desired_len is None:
            results.append(MatchResult(demo, False, "missing_demo_length"))
            continue

        # Prefer exact disambiguation via state/action fingerprints.
        demo_fp: Optional[str] = None
        try:
            n_fp = min(max_frames, desired_len)
            st_prefix, ac_prefix = _load_ecot_state_action_prefix(demo.demo_path, n_fp)
            demo_fp = _fingerprint_actions(np.asarray(ac_prefix, dtype=np.float32), quant_scale=quant_scale)
        except Exception:
            demo_fp = None

        if demo_fp is None:
            if require_fingerprint:
                results.append(MatchResult(demo, False, "missing_fingerprint"))
                continue
        elif not fp_to_eps:
            if require_fingerprint:
                results.append(MatchResult(demo, False, "no_lerobot_fingerprints"))
                continue
        else:
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
                results.append(
                    MatchResult(
                        demo,
                        True,
                        "ok",
                        matched_episode_index=ep.episode_index,
                        lerobot_episode_length=ep.length,
                        length_delta=abs(desired_len - ep.length),
                    )
                )
                continue
            if len(candidate_ep_ids) > 1:
                results.append(
                    MatchResult(
                        demo,
                        False,
                        f"ambiguous_fingerprint_match:candidates={len(candidate_ep_ids)}",
                    )
                )
                continue
            if require_fingerprint:
                results.append(MatchResult(demo, False, "fingerprint_not_found_in_task"))
                continue

        # Optional fallback (disabled when require_fingerprint=True):
        # Build candidate list among remaining episodes.
        candidate_eps: List[Tuple[int, LeRobotEpisode]] = []
        for ep in remaining_eps.values():
            if demo.step_idx_max is not None and demo.step_idx_max >= ep.length:
                continue
            delta = abs(desired_len - ep.length)
            candidate_eps.append((delta, ep))

        if not candidate_eps:
            results.append(MatchResult(demo, False, "no_episode_satisfies_step_range"))
            continue

        candidate_eps.sort(key=lambda t: (t[0], t[1].length, t[1].episode_index))
        best_delta, best_ep = candidate_eps[0]

        if best_delta > max_abs_length_delta:
            results.append(
                MatchResult(
                    demo,
                    False,
                    f"length_mismatch_best:ecot={desired_len},best_lerobot={best_ep.length},delta={best_delta}",
                    matched_episode_index=best_ep.episode_index,
                    lerobot_episode_length=best_ep.length,
                    length_delta=best_delta,
                )
            )
            continue

        # Assign
        remaining_eps.pop(best_ep.episode_index, None)
        results.append(
            MatchResult(
                demo,
                True,
                "ok",
                matched_episode_index=best_ep.episode_index,
                lerobot_episode_length=best_ep.length,
                length_delta=best_delta,
            )
        )

    return results


def _summarize_results(results: List[MatchResult]) -> Dict[str, Any]:
    total = len(results)
    ok = sum(1 for r in results if r.matched)
    bad = total - ok
    reasons = Counter(r.reason for r in results if not r.matched)
    task_counts = Counter(_norm_task_text(r.ecot.task_text) for r in results)
    matched_eps = [r.matched_episode_index for r in results if r.matched and r.matched_episode_index is not None]
    dup_eps = len(matched_eps) - len(set(matched_eps))
    return {
        "total_ecot_demos": total,
        "matched_ok": ok,
        "mismatched": bad,
        "mismatch_reasons": dict(reasons.most_common()),
        "unique_ecot_tasks": len(task_counts),
        "top_ecot_tasks": dict(task_counts.most_common(10)),
        "matched_episode_index_duplicates": dup_eps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether ECOT bbox demos can be aligned 1:1 with a LeRobot dataset.",
    )
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        required=True,
        help="Path to the LeRobot dataset root (contains meta/ and data/).",
    )
    parser.add_argument(
        "--ecot-root",
        type=Path,
        required=True,
        help="Path to the ECOT dataset root (contains *_demo/demo_*/bounding_box.json).",
    )
    parser.add_argument(
        "--max-abs-length-delta",
        type=int,
        default=0,
        help="Max allowed |demo_steps - episode_length| for a match.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Frames to use for fingerprinting (min with trajectory length; must be >= 100 for your requirement).",
    )
    parser.add_argument(
        "--quant-scale",
        type=float,
        default=1000.0,
        help="Quantization scale for float->int16 fingerprinting (1000 => ~1e-3 resolution).",
    )
    parser.add_argument(
        "--allow-length-fallback",
        action="store_true",
        help="If set, fall back to length matching when fingerprint matching fails.",
    )
    parser.add_argument(
        "--max-print-mismatches",
        type=int,
        default=30,
        help="Max number of mismatches to print verbosely.",
    )
    args = parser.parse_args()

    lerobot_root = args.lerobot_root
    ecot_root = args.ecot_root

    episodes = _load_lerobot_episodes(lerobot_root)
    # Compute LeRobot fingerprints for disambiguation across duplicate task instructions.
    max_frames = int(args.max_frames)
    if max_frames < 100:
        raise ValueError("--max-frames must be >= 100 (or set to 100); script will use min(max_frames, episode_length).")
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

    bbox_files = _discover_ecot_demos(ecot_root)
    # Configure parser with resolved ecot root for reliable relpath extraction.
    setattr(_parse_ecot_bbox_file, "_ecot_root_resolved", ecot_root.resolve())
    demos = [_parse_ecot_bbox_file(p) for p in bbox_files]

    # Group ECOT demos by best-effort task key, preferring directory-derived instruction.
    ecot_task_groups: Dict[str, List[EcotDemo]] = defaultdict(list)
    ecot_task_key_source: Dict[str, str] = {}
    unmatched_task_demos: List[EcotDemo] = []

    for d in demos:
        key_candidates = _task_key_candidates(d)
        matched_key = None
        matched_src = None
        for src, k in key_candidates:
            if k in task_to_eps:
                matched_key = k
                matched_src = src
                break
        if matched_key is None:
            # Keep for reporting; use dir-based key for grouping display.
            fallback_key = key_candidates[0][1] if key_candidates else ""
            ecot_task_groups[fallback_key].append(d)
            ecot_task_key_source.setdefault(fallback_key, key_candidates[0][0] if key_candidates else "unknown")
            unmatched_task_demos.append(d)
        else:
            ecot_task_groups[matched_key].append(d)
            ecot_task_key_source.setdefault(matched_key, matched_src or "unknown")

    results: List[MatchResult] = []

    # First, handle tasks not found in LeRobot.
    for d in unmatched_task_demos:
        # Use the best candidate key for messaging.
        key_candidates = _task_key_candidates(d)
        best_key = key_candidates[0][1] if key_candidates else ""
        results.append(MatchResult(d, False, f"task_not_found_in_lerobot:{best_key}"))

    # Then, for tasks that exist, match within each task with 1:1 assignment.
    for task_key, group in ecot_task_groups.items():
        if task_key not in task_to_eps:
            continue
        eps = task_to_eps[task_key]
        results.extend(
            _match_within_task(
                group,
                eps,
                max_abs_length_delta=int(args.max_abs_length_delta),
                max_frames=max_frames,
                quant_scale=float(args.quant_scale),
                require_fingerprint=not bool(args.allow_length_fallback),
            )
        )

    summary = _summarize_results(results)

    print("== LeRobot / ECOT Alignment Check ==")
    print(f"LeRobot root: {lerobot_root}")
    print(f"ECOT root:    {ecot_root}")
    print()
    print("Summary:")
    for k, v in summary.items():
        print(f"  - {k}: {v}")

    # Print mismatches with context.
    mismatches = [r for r in results if not r.matched]
    if mismatches:
        print()
        print(f"First {min(len(mismatches), args.max_print_mismatches)} mismatches:")
        for r in mismatches[: args.max_print_mismatches]:
            d = r.ecot
            print(
                "  - "
                + json.dumps(
                    {
                        "task_from_dir": d.task_from_dir,
                        "task_json": d.task_text,
                        "task_dir": d.task_dir,
                        "demo_id": d.demo_id,
                        "demo_dir": d.demo_dir,
                        "total_steps": d.total_steps,
                        "bboxes_per_step_len": d.bboxes_per_step_len,
                        "step_idx_min": d.step_idx_min,
                        "step_idx_max": d.step_idx_max,
                        "reason": r.reason,
                        "matched_episode_index": r.matched_episode_index,
                        "lerobot_episode_length": r.lerobot_episode_length,
                    },
                    ensure_ascii=False,
                )
            )

    # Non-zero exit if anything mismatched OR if duplicates indicate non-1:1 mapping.
    if summary["mismatched"] > 0:
        return 2
    if summary["matched_episode_index_duplicates"] > 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
