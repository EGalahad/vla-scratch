from __future__ import annotations

import os
from typing import Iterable, Sequence, Tuple


def use_paligemma_tokens_enabled() -> bool:
    val = os.getenv("VLA_USE_PALIGEMMA_TOKENS", "").strip().lower()
    return val in {"1", "true", "yes", "y", "on"}


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _to_loc_index(x: float) -> int:
    # Map normalized [0, 1] -> discrete [0, 1023] with floor.
    x = _clip01(float(x))
    idx = int(x * 1024.0)
    if idx < 0:
        return 0
    if idx > 1023:
        return 1023
    return idx


def loc_token(x: float) -> str:
    return f"<loc{_to_loc_index(x):04d}>"


def bbox_xyxy_to_loc_tokens(
    bbox_normalized_xyxy: Sequence[float],
) -> Tuple[str, str, str, str]:
    if len(bbox_normalized_xyxy) != 4:
        raise ValueError(
            f"Expected bbox_normalized to have 4 values (x1,y1,x2,y2), got {bbox_normalized_xyxy!r}"
        )
    x1, y1, x2, y2 = bbox_normalized_xyxy
    # Output order required by user: y1, x1, y2, x2
    return (loc_token(y1), loc_token(x1), loc_token(y2), loc_token(x2))


def paligemma_detect_prompt() -> str:
    return (
        "Detect all task-relevant objects. "
        "Output format: detect label1; label2; ...; "
        "<locYYYY><locXXXX><locYYYY><locXXXX> label1; "
        "<locYYYY><locXXXX><locYYYY><locXXXX> label2; ... "
        "Each <loc####> is a discretized coordinate (0-1023) in order y1, x1, y2, x2."
    )


def paligemma_detect_answer(
    bboxes_normalized_xyxy: Iterable[Sequence[float]],
    labels: Sequence[str],
) -> str:
    labels = list(labels)
    bboxes = list(bboxes_normalized_xyxy)
    if len(bboxes) != len(labels):
        raise ValueError(
            f"bboxes/labels length mismatch: {len(bboxes)} vs {len(labels)}"
        )
    if not labels:
        return ""

    header_parts = ["detect " + f"{labels[0]}"]
    for label in labels[1:]:
        header_parts.append(f"{label}")

    bbox_parts = []
    for bbox_norm, label in zip(bboxes, labels):
        y1, x1, y2, x2 = bbox_xyxy_to_loc_tokens(bbox_norm)
        bbox_parts.append(f"{y1}{x1}{y2}{x2} {label}")

    return "; ".join(header_parts + bbox_parts)
