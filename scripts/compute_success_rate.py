#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SUCCESS_RE = re.compile(
    r"(?:^|[_\-.])success(?:=|[_\-.])(?P<val>true|false)(?:[_\-.]|$)",
    re.IGNORECASE,
)
EPISODE_RE = re.compile(r"(?:^|[_\-.])episode_(?P<id>\d+)(?:[_\-.]|$)", re.IGNORECASE)


@dataclass(frozen=True)
class Sample:
    path: Path
    key: str
    success: bool


def _iter_files(input_dir: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from (p for p in input_dir.rglob("*") if p.is_file())
    else:
        yield from (p for p in input_dir.iterdir() if p.is_file())


def _extract_success_from_name(filename: str) -> bool | None:
    match = SUCCESS_RE.search(filename)
    if not match:
        return None
    return match.group("val").lower() == "true"


def _make_key(path: Path, dedupe: str) -> str:
    name = path.name
    if dedupe == "none":
        return str(path)
    if dedupe == "stem":
        return path.stem
    if dedupe == "episode":
        match = EPISODE_RE.search(name)
        if match:
            return f"episode_{int(match.group('id'))}"
        return path.stem
    raise ValueError(f"Unknown dedupe mode: {dedupe}")


def collect_samples(input_dir: Path, recursive: bool, dedupe: str) -> tuple[list[Sample], list[Path]]:
    matched: list[Sample] = []
    ignored: list[Path] = []
    for path in _iter_files(input_dir, recursive=recursive):
        success = _extract_success_from_name(path.name)
        if success is None:
            ignored.append(path)
            continue
        matched.append(Sample(path=path, key=_make_key(path, dedupe=dedupe), success=success))
    return matched, ignored


def compute_success_rate(samples: list[Sample]) -> tuple[int, int, int, dict[str, list[Sample]]]:
    by_key: dict[str, list[Sample]] = {}
    for s in samples:
        by_key.setdefault(s.key, []).append(s)

    total = len(by_key)
    success = 0
    failure = 0
    for key, items in by_key.items():
        outcomes = {s.success for s in items}
        if outcomes == {True}:
            success += 1
        elif outcomes == {False}:
            failure += 1
        else:
            # Mixed results for same key; count as success if any success, but surface it via warnings.
            success += 1
            sys.stderr.write(
                f"[warn] Mixed success flags for {key}: "
                + ", ".join(f"{p.path.name}={p.success}" for p in items)
                + "\n"
            )
    return total, success, failure, by_key


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute simulation success rate from filenames containing success=True/False."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing simulation outputs (e.g., mp4 files with *_success_True/False*).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories (default: on).",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Do not recurse into subdirectories.",
    )
    parser.set_defaults(recursive=True)
    parser.add_argument(
        "--dedupe",
        choices=["episode", "stem", "none"],
        default="episode",
        help="How to deduplicate multiple files per sample (default: episode).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary.",
    )
    parser.add_argument(
        "--print-examples",
        type=int,
        default=0,
        help="Print N example matched filenames for quick inspection.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    input_dir: Path = args.input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        sys.stderr.write(f"input-dir is not a directory: {input_dir}\n")
        return 2

    samples, ignored = collect_samples(input_dir, recursive=args.recursive, dedupe=args.dedupe)
    total, success, failure, by_key = compute_success_rate(samples)

    if total == 0:
        sys.stderr.write(
            "No files matched the success pattern. Expected filenames containing success=True/False.\n"
        )
        sys.stderr.write(f"Ignored files: {len(ignored)}\n")
        return 1

    rate = success / total
    print(f"input_dir: {input_dir}")
    print(f"dedupe: {args.dedupe} | recursive: {args.recursive}")
    print(f"samples_total: {total}")
    print(f"samples_success: {success}")
    print(f"samples_failure: {failure}")
    print(f"success_rate: {rate:.4f} ({rate*100:.2f}%)")
    print(f"matched_files: {len(samples)} | ignored_files: {len(ignored)}")

    if args.print_examples and args.print_examples > 0:
        print("\nexamples:")
        for s in samples[: args.print_examples]:
            print(f"- {s.path.relative_to(input_dir)}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_dir": str(input_dir),
            "recursive": bool(args.recursive),
            "dedupe": str(args.dedupe),
            "samples_total": int(total),
            "samples_success": int(success),
            "samples_failure": int(failure),
            "success_rate": float(rate),
            "matched_files": int(len(samples)),
            "ignored_files": int(len(ignored)),
            "mixed_keys": sorted([k for k, v in by_key.items() if len({s.success for s in v}) > 1]),
        }
        args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote_json: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
