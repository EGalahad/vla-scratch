from typing import Dict

import torch

from vla_scratch.datasets.transforms import TransformFn
from vla_scratch.datasets.common import (
    PROCESSED_ACTION_KEY,
    PROCESSED_IMAGE_KEY,
    PROCESSED_IMAGE_MASK_KEY,
    PROCESSED_STATE_KEY,
)
from vla_scratch.datasets.libero.common import (
    ACTION_KEY,
    FUTURE_STATE_KEY,
    STATE_KEY,
    TASK_KEY,
    IMAGE_KEY,
    WRIST_IMAGE_KEY,
)
from vla_scratch.datasets.math_utils import (
    matrix_from_quat,
    quat_apply_inverse,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    unscale_transform,
)


def _rotation_matrix_to_6d(rotation: torch.Tensor) -> torch.Tensor:
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(
            f"Rotation matrix must be (..., 3, 3); received {rotation.shape}"
        )
    return rotation[..., :2, :].reshape(*rotation.shape[:-2], 6)


class LiberoState(TransformFn):
    """Convert history of Libero states into relative 6D pose deltas."""

    def compute(self, sample: Dict) -> Dict:
        state_seq: torch.Tensor = sample[STATE_KEY]
        history = state_seq.shape[0] - 1

        pos_w = state_seq[:, 0:3]
        rotvec_seq = state_seq[:, 3:6]
        angle = torch.linalg.norm(rotvec_seq, dim=-1)
        axis = rotvec_seq / (angle.unsqueeze(-1) + 1e-8)
        quat_w = quat_from_angle_axis(angle, axis)

        current_pos_w = pos_w[-1]
        current_quat_w = quat_w[-1]

        history_pos_w = pos_w[:history]
        current_quat_w_hist = current_quat_w.unsqueeze(0).expand(history, -1)
        history_dpos = quat_apply_inverse(
            current_quat_w_hist, history_pos_w - current_pos_w
        )

        history_quat_w = quat_w[:history]
        history_dquat = quat_mul(quat_conjugate(current_quat_w_hist), history_quat_w)
        history_drotmat = matrix_from_quat(history_dquat)
        history_dori6d = _rotation_matrix_to_6d(history_drotmat)

        history_grippers = state_seq[:history, 6:8]
        state_vec = torch.cat([history_dpos, history_dori6d, history_grippers], dim=-1)
        sample[PROCESSED_STATE_KEY] = state_vec
        return sample


class LiberoAction(TransformFn):
    """
    Convert Libero actions into relative 6D pose deltas.
    """

    actions_low = torch.tensor([-0.05, -0.05, -0.05, -0.5, -0.5, -0.5, -1.0])
    actions_high = torch.tensor([0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 1.0])

    def __init__(self, eef_frame: bool = True) -> None:
        self.eef_frame = eef_frame

    def compute(self, sample: Dict) -> Dict:
        future_states: torch.Tensor = sample[FUTURE_STATE_KEY]
        actions: torch.Tensor = sample[ACTION_KEY]

        actions = self._unscale(actions)

        future_pos_w = future_states[:, 0:3]
        future_rotvec_w = future_states[:, 3:6]
        angle = torch.linalg.norm(future_rotvec_w, dim=-1)
        axis = future_rotvec_w / (angle.unsqueeze(-1) + 1e-8)
        future_quat_w = quat_from_angle_axis(angle, axis)

        cmd_dpos = actions[:, 0:3]
        cmd_drotvec = actions[:, 3:6]
        cmd_dangle = torch.linalg.norm(cmd_drotvec, dim=-1)
        cmd_daxis = cmd_drotvec / (cmd_dangle.unsqueeze(-1) + 1e-8)
        cmd_dquat = quat_from_angle_axis(cmd_dangle, cmd_daxis)

        target_pos_w = future_pos_w + cmd_dpos
        target_quat_w = quat_mul(cmd_dquat, future_quat_w)
        target_dori6d = _rotation_matrix_to_6d(matrix_from_quat(target_quat_w))

        horizon = actions.shape[0]
        current_pos_w = future_pos_w[-horizon]
        current_quat_w = future_quat_w[-horizon]
        current_quat_w_expand = current_quat_w.unsqueeze(0).expand_as(target_quat_w)
        target_dpos = quat_apply_inverse(
            current_quat_w_expand, target_pos_w - current_pos_w
        )
        target_dquat = quat_mul(quat_conjugate(current_quat_w_expand), target_quat_w)
        target_drotmat = matrix_from_quat(target_dquat)
        target_rel_ori6d = _rotation_matrix_to_6d(target_drotmat)

        cmd_grippers = actions[:, 6:7]
        if self.eef_frame:
            actions_out = torch.cat([target_dpos, target_rel_ori6d, cmd_grippers], dim=-1)
        else:
            actions_out = torch.cat([target_pos_w, target_dori6d, cmd_grippers], dim=-1)

        sample[PROCESSED_ACTION_KEY] = actions_out
        return sample

    def _unscale(self, actions: torch.Tensor) -> torch.Tensor:
        return unscale_transform(
            actions,
            self.actions_low.to(actions.device, actions.dtype),
            self.actions_high.to(actions.device, actions.dtype),
        )


class StructurePrompt(TransformFn):
    """Format the task string into a language prompt."""

    def compute(self, sample: Dict) -> Dict:
        task_prompt: str = sample[TASK_KEY]
        sample["prompt"] = f"<bos>Task: {task_prompt}; \n Action:"
        return sample


class LiberoImages(TransformFn):
    """Stack Libero camera streams into a standard tensor layout."""

    image_keys = (IMAGE_KEY, WRIST_IMAGE_KEY)

    def compute(self, sample: Dict) -> Dict:
        images = []
        for key in self.image_keys:
            if key not in sample:
                continue
            img = sample[key]
            if img.ndim == 3 and img.shape[-1] == 3 and img.shape[0] != 3:
                img = img.permute(2, 0, 1)
            images.append(img)
        if not images:
            raise KeyError("No images found in sample for LiberoImages transform.")
        stacked = torch.stack(images, dim=0).to(torch.uint8)
        sample[PROCESSED_IMAGE_KEY] = stacked
        mask = torch.ones((stacked.shape[0], 1), dtype=torch.bool, device=stacked.device)
        sample[PROCESSED_IMAGE_MASK_KEY] = mask
        return sample
