from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

import torch

from matplotlib import cm
from matplotlib import colors as mpl_colors
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from vla_scratch.datasets.math_utils import matrix_from_quat


def _to_numpy(array: Iterable) -> np.ndarray:
    """Convert supported inputs to a numpy array on CPU."""
    if isinstance(array, np.ndarray):
        return array
    if torch is not None and isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _to_tensor(array: Iterable) -> "torch.Tensor":
    """Convert supported inputs to a torch tensor on CPU."""
    if torch is None:  # pragma: no cover - torch is expected to be available
        raise ImportError("plot_pose_trajectory requires PyTorch to be installed.")
    if isinstance(array, torch.Tensor):
        return array.detach().cpu()
    return torch.as_tensor(np.asarray(array), dtype=torch.float32)


def plot_pose_trajectory(
    ax,
    positions: Iterable,
    quaternions: Iterable,
    *,
    axis_length: float = 0.05,
    stride: int = 1,
    line_kwargs: Mapping | None = None,
    frame_kwargs: Mapping | None = None,
    label: str | None = None,
    axis_colors: Sequence[str] = ("r", "g", "b"),
    progress_cmap: str = "viridis",
    start_marker_kwargs: Mapping | None = None,
    end_marker_kwargs: Mapping | None = None,
) -> None:
    """Plot positions and orientation frames for a trajectory on a 3D axis.

    Args:
        ax: A matplotlib 3D axis (e.g. from `fig.add_subplot(..., projection="3d")`).
        positions: Iterable of xyz positions with shape (N, 3).
        quaternions: Iterable of quaternions in (w, x, y, z) with shape (N, 4).
        axis_length: Length of the orientation axes to draw for each pose.
        stride: Draw orientation frames every `stride` poses to reduce clutter.
        line_kwargs: Optional matplotlib kwargs passed to `ax.plot`.
        frame_kwargs: Optional matplotlib kwargs passed to `ax.quiver`.
        label: Optional label for the trajectory line.
        axis_colors: Sequence of three colors for x/y/z axes respectively.
        progress_cmap: Name of matplotlib colormap used to encode time progress.
        start_marker_kwargs: Optional kwargs for the start scatter marker.
        end_marker_kwargs: Optional kwargs for the end scatter marker.
    """
    if stride < 1:
        raise ValueError("stride must be a positive integer.")

    pos_np = _to_numpy(positions)
    quat_tensor = _to_tensor(quaternions)

    if pos_np.shape[-1] != 3:
        raise ValueError(f"Expected positions with shape (N, 3); received {pos_np.shape}.")
    if quat_tensor.shape[-1] != 4:
        raise ValueError(f"Expected quaternions with shape (N, 4); received {quat_tensor.shape}.")
    if pos_np.shape[0] != quat_tensor.shape[0]:
        raise ValueError(
            f"Number of positions ({pos_np.shape[0]}) does not match number of quaternions ({quat_tensor.shape[0]})."
        )

    line_kwargs = dict(line_kwargs or {})
    frame_kwargs = dict(frame_kwargs or {})
    start_marker_kwargs = dict(start_marker_kwargs or {})
    end_marker_kwargs = dict(end_marker_kwargs or {})

    linewidth = line_kwargs.pop("linewidth", None)
    if linewidth is not None:
        line_kwargs.setdefault("linewidths", linewidth)
    linestyle = line_kwargs.pop("linestyle", None)
    if linestyle is not None:
        line_kwargs.setdefault("linestyles", linestyle)
    # Remove label now that it is handled via set_label
    line_kwargs.pop("label", None)

    frame_kwargs.setdefault("arrow_length_ratio", 0.15)
    frame_kwargs.setdefault("linewidth", 1.0)
    trajectory_len = pos_np.shape[0]

    cmap = cm.get_cmap(progress_cmap)
    norm = mpl_colors.Normalize(vmin=0.5, vmax=max(trajectory_len - 1, 1))

    if trajectory_len == 1:
        point_color = tuple(cmap(norm(0))[:3])
        start_marker_kwargs.setdefault("color", point_color)
        start_marker_kwargs.setdefault("s", 40)
        if label is not None:
            start_marker_kwargs.setdefault("label", f"{label} start")
        ax.scatter(
            pos_np[0, 0],
            pos_np[0, 1],
            pos_np[0, 2],
            **start_marker_kwargs,
        )
    else:
        segments = np.stack([pos_np[:-1], pos_np[1:]], axis=1)
        segment_colors = cmap(norm(np.arange(trajectory_len - 1)))
        line_collection = Line3DCollection(
            segments,
            colors=segment_colors,
            **line_kwargs,
        )
        if label is not None:
            line_collection.set_label(label)
        ax.add_collection3d(line_collection)

        start_color = tuple(cmap(norm(0))[:3])
        end_color = tuple(cmap(norm(trajectory_len - 1))[:3])
        start_marker_kwargs.setdefault("color", start_color)
        start_marker_kwargs.setdefault("s", 40)
        start_marker_kwargs.setdefault("marker", "o")
        start_marker_kwargs.setdefault("label", f"{label} start" if label else None)
        end_marker_kwargs.setdefault("color", end_color)
        end_marker_kwargs.setdefault("s", 40)
        end_marker_kwargs.setdefault("marker", "X")
        end_marker_kwargs.setdefault("label", f"{label} end" if label else None)

        ax.scatter(
            pos_np[0, 0],
            pos_np[0, 1],
            pos_np[0, 2],
            **{k: v for k, v in start_marker_kwargs.items() if v is not None},
        )
        ax.scatter(
            pos_np[-1, 0],
            pos_np[-1, 1],
            pos_np[-1, 2],
            **{k: v for k, v in end_marker_kwargs.items() if v is not None},
        )

    rotation_mats = matrix_from_quat(quat_tensor).detach().cpu().numpy()

    for idx in range(0, pos_np.shape[0], stride):
        origin = pos_np[idx]
        rot = rotation_mats[idx]
        for axis_idx, axis_color in enumerate(axis_colors):
            direction = rot[:, axis_idx] * axis_length
            ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                direction[0],
                direction[1],
                direction[2],
                color=axis_color,
                **frame_kwargs,
            )

    orientation_endpoints = (
        pos_np[:, None, :] + rotation_mats.transpose(0, 2, 1) * axis_length
    ).reshape(-1, 3)
    all_points = np.concatenate([pos_np, orientation_endpoints], axis=0)

    mins = np.min(all_points, axis=0)
    maxs = np.max(all_points, axis=0)
    center = (maxs + mins) / 2.0
    span = max(maxs - mins)
    if span == 0:
        span = max(axis_length, 1e-3)
    half_span = span / 2.0

    ax.set_xlim(center[0] - half_span, center[0] + half_span)
    ax.set_ylim(center[1] - half_span, center[1] + half_span)
    ax.set_zlim(center[2] - half_span, center[2] + half_span)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
