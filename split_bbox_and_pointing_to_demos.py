#!/usr/bin/env python3
"""
将 LIBERO 的 bbox / pointing 结果 (results_*.json) 按 demo 拆分，
写入对应 ECOT demo 目录下的 bounding_box.json / pointing.json。

典型用法:
  python scripts/xinye/split_bbox_and_pointing_to_demos.py \\
    --ecot-root data/libero_90_no_noops_ecot_data \\
    --bbox-root data/libero_bbox_outputs/libero_90/bboxes_and_points \\
    --points-root data/libero_segmentation_outputs/libero_90/points_and_masks \\
    --num-workers 8

支持:
  - 多进程并行处理 (--num-workers)
  - dry-run 模式 (--dry-run): 只打印统计信息和示例路径, 不实际写文件

Assumptions:
  - results_*.json 的结构与 embodied_vqa/data_loaders.py 中使用的一致:
      {
        "<hdf5_file_path>": {
          "<episode_id>": {
            "episode_id": "...",
            "file_path": "<hdf5_file_path>",
            ...,
            "bboxes_per_step": [...],  # bbox root 下
            "points_per_step": [...],  # points root 下
          },
          ...
        },
        ...
      }
  - ECOT demo 目录结构:
      <ecot_root>/<file_name>/<demo_id>/
    其中:
      file_name == Path(<hdf5_file_path>).stem
      demo_id   == f"demo_{episode_id}"
"""

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class WorkerSummary:
    """Per-results.json processing summary (single modality)."""

    modality: str
    json_path: str
    total_episodes: int = 0
    written: int = 0
    missing_demo_dirs: int = 0
    existing_overwritten: int = 0
    samples: List[str] = None
    # Detailed failure records for unmatched items (for debugging)
    failures: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "json_path": self.json_path,
            "total_episodes": self.total_episodes,
            "written": self.written,
            "missing_demo_dirs": self.missing_demo_dirs,
            "existing_overwritten": self.existing_overwritten,
            "samples": self.samples or [],
            "failures": self.failures or [],
        }


def _process_results_file(
    json_path: str,
    ecot_root: str,
    modality: str,
    dry_run: bool,
    max_samples: int = 10,
) -> WorkerSummary:
    """Worker function: process a single results_*.json file.

    Args:
        json_path: Path to results_*.json
        ecot_root: Root directory of ECOT demos
        modality: "bbox" or "point"
        dry_run: If True, do not write any files
        max_samples: Max number of example paths to record for logging
    """
    ecot_root_path = Path(ecot_root)
    summary = WorkerSummary(
        modality=modality,
        json_path=json_path,
        total_episodes=0,
        written=0,
        missing_demo_dirs=0,
        existing_overwritten=0,
        samples=[],
        failures=[],
    )

    json_path_obj = Path(json_path)
    try:
        with json_path_obj.open("r") as f:
            data = json.load(f)
    except Exception as e:
        msg = f"[{modality}] Failed to load {json_path}: {e}"
        print(msg)
        summary.failures.append(msg)
        return summary

    output_filename = "bounding_box.json" if modality == "bbox" else "pointing.json"

    for file_path, episodes in data.items():
        file_name = Path(file_path).stem  # e.g. KITCHEN_SCENE1_...
        for episode_id, episode_data in episodes.items():
            summary.total_episodes += 1

            demo_id = f"demo_{episode_id}"
            demo_dir = ecot_root_path / file_name / demo_id
            output_path = demo_dir / output_filename

            if not demo_dir.exists():
                summary.missing_demo_dirs += 1
                fail_msg = (
                    f"[{modality}] demo dir not found for file_path={file_path}, "
                    f"episode_id={episode_id}, expected_dir={demo_dir}"
                )
                summary.failures.append(fail_msg)
                if len(summary.samples) < max_samples:
                    summary.samples.append(f"missing demo dir: {demo_dir}")
                continue

            if dry_run:
                summary.written += 1
                if len(summary.samples) < max_samples:
                    summary.samples.append(f"would write: {output_path}")
                continue

            if output_path.exists():
                summary.existing_overwritten += 1

            try:
                demo_dir.mkdir(parents=True, exist_ok=True)
                # Use indentation to keep files human-readable and close to the
                # style of the original results_*.json (multi-line, nested).
                with output_path.open("w") as f_out:
                    json.dump(
                        episode_data,
                        f_out,
                        indent=2,
                        ensure_ascii=False,
                    )
                summary.written += 1
            except Exception as e:
                msg = (
                    f"[{modality}] Failed to write {output_path} "
                    f"(file_path={file_path}, episode_id={episode_id}): {e}"
                )
                print(msg)
                summary.failures.append(msg)

    return summary


def _run_for_root(
    root: Optional[str],
    ecot_root: str,
    modality: str,
    num_workers: int,
    dry_run: bool,
) -> List[WorkerSummary]:
    """Run processing for a single modality root (bbox or point)."""
    if root is None:
        return []

    root_path = Path(root)
    if not root_path.exists():
        print(f"[{modality}] Root not found, skipping: {root_path}")
        return []

    json_files = sorted(root_path.glob("results_*.json"))
    if not json_files:
        print(f"[{modality}] No results_*.json found under {root_path}, nothing to do.")
        return []

    print(f"[{modality}] Found {len(json_files)} results_*.json under {root_path}")

    summaries: List[WorkerSummary] = []

    # num_workers <= 1: fall back to sequential processing
    if num_workers <= 1:
        print(f"[{modality}] Using sequential processing (num_workers={num_workers})")
        for jp in json_files:
            summary = _process_results_file(
                json_path=str(jp),
                ecot_root=ecot_root,
                modality=modality,
                dry_run=dry_run,
            )
            summaries.append(summary)
        return summaries

    print(f"[{modality}] Using ProcessPoolExecutor with {num_workers} workers")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _process_results_file,
                str(jp),
                ecot_root,
                modality,
                dry_run,
            )
            for jp in json_files
        ]

        for fut in as_completed(futures):
            try:
                summary = fut.result()
                summaries.append(summary)
            except Exception as e:
                print(f"[{modality}] Worker raised exception: {e}")

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split LIBERO bbox / pointing results_*.json into per-demo JSON files.\n"
            "For each (file_path, episode_id), write:\n"
            "  <ecot_root>/<file_name>/demo_<episode_id>/bounding_box.json\n"
            "  <ecot_root>/<file_name>/demo_<episode_id>/pointing.json"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ecot-root",
        type=str,
        required=True,
        help="ECOT demo root directory (contains <file_name>/demo_*/).",
    )
    parser.add_argument(
        "--bbox-root",
        type=str,
        default=None,
        help="Root directory containing bbox results_*.json (e.g. .../bboxes_and_points).",
    )
    parser.add_argument(
        "--points-root",
        type=str,
        default=None,
        help="Root directory containing pointing results_*.json (e.g. .../points_and_masks).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, multiprocessing.cpu_count() // 2),
        help="Number of worker processes to use.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run mode: do not write any files, only print statistics and examples.",
    )

    args = parser.parse_args()

    if args.bbox_root is None and args.points_root is None:
        raise SystemExit("At least one of --bbox-root or --points-root must be provided.")

    ecot_root = Path(args.ecot_root)
    if not ecot_root.exists():
        raise SystemExit(f"ECOT root does not exist: {ecot_root}")

    print("============================================================")
    print("Split bbox / pointing results into per-demo JSON files")
    print("============================================================")
    print(f"ECOT root:   {ecot_root}")
    if args.bbox_root:
        print(f"BBox root:   {args.bbox_root}")
    if args.points_root:
        print(f"Points root: {args.points_root}")
    print(f"num_workers: {args.num_workers}")
    print(f"dry_run:     {args.dry_run}")

    all_summaries: List[WorkerSummary] = []

    # Process bbox results
    bbox_summaries = _run_for_root(
        root=args.bbox_root,
        ecot_root=str(ecot_root),
        modality="bbox",
        num_workers=args.num_workers,
        dry_run=args.dry_run,
    )
    all_summaries.extend(bbox_summaries)

    # Process points results
    point_summaries = _run_for_root(
        root=args.points_root,
        ecot_root=str(ecot_root),
        modality="point",
        num_workers=args.num_workers,
        dry_run=args.dry_run,
    )
    all_summaries.extend(point_summaries)

    # Print aggregated stats
    if not all_summaries:
        print("\nNo summaries produced (nothing processed).")
        return

    print("\n==================== Aggregate Summary ====================")
    agg: Dict[Tuple[str, str], Dict[str, int]] = {}
    total_episodes = 0
    total_written = 0
    all_failures: List[str] = []

    for s in all_summaries:
        key = (s.modality, s.json_path)
        total_episodes += s.total_episodes
        total_written += s.written
        if s.failures:
            all_failures.extend(s.failures)
        agg[key] = {
            "total_episodes": s.total_episodes,
            "written": s.written,
            "missing_demo_dirs": s.missing_demo_dirs,
            "existing_overwritten": s.existing_overwritten,
        }

    total_failed = max(0, total_episodes - total_written)
    print(
        f"Total attempts (episodes): {total_episodes}, "
        f"matched (written): {total_written}, "
        f"unmatched: {total_failed}"
    )

    for (modality, json_path), stats in sorted(agg.items(), key=lambda kv: kv[0][1]):
        print(
            f"[{modality}] {json_path}: "
            f"episodes={stats['total_episodes']}, "
            f"written={stats['written']}, "
            f"missing_demo_dirs={stats['missing_demo_dirs']}, "
            f"overwritten={stats['existing_overwritten']}"
        )

    # Show a few example paths and detailed failure reasons
    print("\nExample paths / warnings:")
    shown = 0
    for s in all_summaries:
        if not s.samples:
            continue
        print(f"\n[{s.modality}] {s.json_path}:")
        for sample in s.samples:
            print(f"  {sample}")
        shown += 1
        if shown >= 10:
            break

    if all_failures:
        print("\nUnmatched items with reasons (first 50):")
        for msg in all_failures[:50]:
            print(f"  {msg}")
        if len(all_failures) > 50:
            print(f"  ... {len(all_failures) - 50} more failures not shown")


if __name__ == "__main__":
    main()
