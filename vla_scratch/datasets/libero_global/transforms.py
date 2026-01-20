import torch
from typing import Dict, Tuple


from vla_scratch.transforms.base import TransformFn
from vla_scratch.transforms.data_keys import (
    PROCESSED_ACTION_KEY,
    PROCESSED_IMAGE_KEY,
    PROCESSED_IMAGE_MASK_KEY,
    PROCESSED_STATE_KEY,
)
from vla_scratch.datasets.libero.data_keys import (
    ARM_CMD_CART_POS_KEY,
    ARM_CMD_CART_ROT_KEY,
    ARM_STATE_CART_POS_KEY,
    ARM_STATE_CART_ROT_KEY,
    CAM_FRONT_KEY,
    CAM_WRIST_KEY,
    GRIPPER_CMD_ACTION_KEY,
    GRIPPER_STATE_QPOS_KEY,
)
from vla_scratch.utils.math import (
    matrix_from_quat,
    quat_apply_inverse,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    quat_from_matrix,
    axis_angle_from_quat,
    quat_apply,
    rotation_matrix_to_6d,
    rotation_6d_to_matrix,
)


def _history_components(
    sample: Dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos_seq = torch.as_tensor(sample[ARM_STATE_CART_POS_KEY])
    rot_seq = torch.as_tensor(sample[ARM_STATE_CART_ROT_KEY])
    grip_seq = torch.as_tensor(sample[GRIPPER_STATE_QPOS_KEY])
    return pos_seq, rot_seq, grip_seq


class LiberoStateToLocal(TransformFn):
    """Convert history of Libero states into relative 6D pose deltas."""

    def compute(self, sample: Dict) -> Dict:
        pos_w, rotvec_seq, grip_seq = _history_components(sample)
        history = pos_w.shape[0] - 1

        angle = torch.linalg.norm(rotvec_seq, dim=-1)
        quat_w = quat_from_angle_axis(angle, rotvec_seq)

        current_pos_w = pos_w[-1]
        current_quat_w = quat_w[-1]

        history_pos_w = pos_w[:history]
        current_quat_w_hist = current_quat_w.unsqueeze(0).expand(history, -1)
        history_pos_local = quat_apply_inverse(
            current_quat_w_hist, history_pos_w - current_pos_w
        )

        history_quat_w = quat_w[:history]
        history_quat_local = quat_mul(
            quat_conjugate(current_quat_w_hist), history_quat_w
        )
        history_mat_local = matrix_from_quat(history_quat_local)
        history_ori6d_local = rotation_matrix_to_6d(history_mat_local)

        history_grippers = grip_seq[:history]
        state_vec = torch.cat(
            [history_pos_local, history_ori6d_local, history_grippers], dim=-1
        )
        sample[PROCESSED_STATE_KEY] = state_vec
        return sample


class LiberoGlobalState(TransformFn):
    """Convert history of Libero states into global 6D pose."""

    def compute(self, sample: Dict) -> Dict:
        pos_w, rotvec_w, grippers = _history_components(sample)
        pos_w = pos_w[1:]
        rotvec_w = rotvec_w[1:]
        grippers = grippers[1:]

        angle = torch.linalg.norm(rotvec_w, dim=-1)
        quat_w = quat_from_angle_axis(angle, rotvec_w)
        ori6d_w = rotation_matrix_to_6d(matrix_from_quat(quat_w))

        state_vec = torch.cat([pos_w, ori6d_w, grippers], dim=-1)
        sample[PROCESSED_STATE_KEY] = state_vec
        return sample


class LiberoActionToLocal(TransformFn):
    """
    Convert Libero actions into relative 6D pose deltas.
    """

    def compute(self, sample: Dict) -> Dict:
        cmd_pos_w = sample[ARM_CMD_CART_POS_KEY]
        cmd_rot_w = sample[ARM_CMD_CART_ROT_KEY]
        cmd_angle = torch.linalg.norm(cmd_rot_w, dim=-1)
        cmd_quat_w = quat_from_angle_axis(cmd_angle, cmd_rot_w)

        current_pos_w = sample[ARM_STATE_CART_POS_KEY][-1]
        current_rotvec_w = sample[ARM_STATE_CART_ROT_KEY][-1]
        current_angle = torch.linalg.norm(current_rotvec_w)
        current_quat_w = quat_from_angle_axis(current_angle, current_rotvec_w)

        # transform cmd into local frame of current pose
        current_quat_w_expand = current_quat_w.unsqueeze(0).expand_as(
            cmd_quat_w
        )
        cmd_pos_local = quat_apply_inverse(
            current_quat_w_expand, cmd_pos_w - current_pos_w
        )
        cmd_quat_local = quat_mul(
            quat_conjugate(current_quat_w_expand), cmd_quat_w
        )
        cmd_mat_local = matrix_from_quat(cmd_quat_local)
        cmd_ori6d_local = rotation_matrix_to_6d(cmd_mat_local)

        cmd_grippers = sample[GRIPPER_CMD_ACTION_KEY]
        actions_out = torch.cat(
            [cmd_pos_local, cmd_ori6d_local, cmd_grippers], dim=-1
        )

        sample[PROCESSED_ACTION_KEY] = actions_out
        return sample


class LiberoActionToGlobal(TransformFn):
    """Invert network action targets back to dataset action format.

    Expects `sample["actions"]` to contain per-step targets of shape [K, 10]:
      [Δpos_to_target(3), Δori6d_to_target(6), gripper(1)] expressed relative to
    the current pose (last state in window), and un-normalized via UnNormalizeAction.

    Produces dataset-like actions [K, 7]: [dpos(3), axis_angle(3), gripper(1)],
    scaled to dataset action range using the same bounds as training.
    """

    def compute(self, sample: Dict) -> Dict:
        actions: torch.Tensor = sample[PROCESSED_ACTION_KEY]

        current_pos_w = torch.as_tensor(sample[ARM_STATE_CART_POS_KEY])[-1]
        current_rotvec_w = torch.as_tensor(sample[ARM_STATE_CART_ROT_KEY])[-1]
        angle = torch.linalg.norm(current_rotvec_w)
        current_quat_w = quat_from_angle_axis(angle, current_rotvec_w)

        cmd_pos_local = actions[:, 0:3]
        cmd_ori6d_local = actions[:, 3:9]
        cmd_mat_local = rotation_6d_to_matrix(cmd_ori6d_local)
        cmd_quat_local = quat_from_matrix(cmd_mat_local)

        current_quat_w_expand = current_quat_w.unsqueeze(0).expand_as(
            cmd_quat_local
        )
        cmd_pos_w = current_pos_w.unsqueeze(0) + quat_apply(
            current_quat_w_expand, cmd_pos_local
        )
        cmd_quat_w = quat_mul(current_quat_w_expand, cmd_quat_local)
        cmd_rotvec_w = axis_angle_from_quat(cmd_quat_w)

        grip = actions[:, 9:10]
        sample[ARM_CMD_CART_POS_KEY] = cmd_pos_w
        sample[ARM_CMD_CART_ROT_KEY] = cmd_rotvec_w
        sample[GRIPPER_CMD_ACTION_KEY] = grip
        return sample


class LiberoImages(TransformFn):
    """Stack Libero camera streams into a standard tensor layout."""

    image_keys = (CAM_FRONT_KEY, CAM_WRIST_KEY)
    mask = torch.ones((len(image_keys), 1), dtype=torch.bool)

    def compute(self, sample: Dict) -> Dict:
        images = [sample[key] for key in self.image_keys]
        stacked = torch.stack(images, dim=0)
        stacked = (stacked * 255).type(torch.uint8)
        # shape: (num_cameras, C, H, W)
        sample[PROCESSED_IMAGE_KEY] = stacked
        sample[PROCESSED_IMAGE_MASK_KEY] = self.mask
        return sample
