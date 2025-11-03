# this file hosts the LeRobotDataset class for loading libero dataset from IPEC-COMMUNITY

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from vla_scratch.datasets.libero.common import *
from vla_scratch.datasets.libero.config import LiberoIPECConfig


class IPECDataset(Dataset):
    # norm_stats_path = "normalization_stats/libero_proprio_stats.npz"
    def __init__(
        self,
        config: LiberoIPECConfig,
    ):
        repo_id = Path(config.repo_id)
        self.action_horizon = action_horizon = config.action_horizon
        self.state_history = state_history = config.state_history

        meta_data = LeRobotDatasetMetadata(repo_id)
        fps = meta_data.fps

        delta_timestamps = {
            "action": (
                np.linspace(0, action_horizon - 1, action_horizon, dtype=int) / fps
            ).tolist(),
            "observation.state": (
                np.linspace(
                    -state_history,
                    action_horizon - 1,
                    state_history + action_horizon,
                    dtype=int,
                )
                / fps
            ).tolist(),
        }
        self.lerobot_dataset = LeRobotDataset(
            repo_id=repo_id, delta_timestamps=delta_timestamps
        )
        assert fps == self.lerobot_dataset.fps

    def __len__(self):
        return len(self.lerobot_dataset)

    def __getitem__(self, idx):
        item = self.lerobot_dataset[idx]
        full_state = item.pop("observation.state")
        # shape: [state_history + action_horizon, state_dim]

        item[STATE_KEY] = full_state[: self.state_history + 1]
        # shape: [state_history + 1, state_dim]
        item[FUTURE_STATE_KEY] = full_state[self.state_history :]
        # shape: [action_horizon, state_dim]

        actions = item.pop("action")
        actions[:, -1] = 1 - 2 * actions[:, -1]  # convert [0, 1] to [1, -1]
        item[ACTION_KEY] = actions

        item[IMAGE_KEY] = item.pop("observation.images.image")
        item[WRIST_IMAGE_KEY] = item.pop("observation.images.wrist_image")
        item[TASK_KEY] = item.pop("task")
        return item


def test_episode():
    repo_id = "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot"
    dataset = IPECDataset(repo_id=repo_id, action_horizon=30, state_history=10)

    episode_dict = dataset.lerobot_dataset.meta.episodes
    episode_lengths = [ep["length"] for ep in episode_dict.values()]
    episode_tasks = [ep["tasks"] for ep in episode_dict.values()]
    episode_starts = np.cumsum([0] + episode_lengths[:-1])

    # go through 2 6 8 episodes and see if the task names match
    for ep_idx in [1, 2, 4, 6, 8]:
        start_idx = episode_starts[ep_idx]
        end_idx = start_idx + episode_lengths[ep_idx]
        tasks = episode_tasks[ep_idx][0]
        for idx in range(start_idx, end_idx):
            item = dataset[idx]
            assert item["task"] == tasks

        print(f"Episode {ep_idx} passed task name check.")
        breakpoint()


def test_visualize(integrate_mode="close_loop"):
    from vla_scratch.datasets.math_utils import (
        quat_apply,
        quat_from_euler_xyz,
        quat_from_angle_axis,
        matrix_from_quat,
        quat_mul,
        unscale_transform,
    )
    from vla_scratch.datasets.visualize import plot_pose_trajectory
    import matplotlib.pyplot as plt
    import torch

    # get huggingface hub cache path
    repo_id = "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot"
    dataset = IPECDataset(repo_id=repo_id, action_horizon=150)

    item = dataset[0]

    # lerobot format
    # state: [pos(3), rot_vec(3), gripper_qpos(2)]
    # action: [delta_pos(3), delta_rotvec(3), gripper_cmd(1)]

    action_low = torch.tensor([-0.05, -0.05, -0.05, -0.5, -0.5, -0.5, -1.0])
    action_high = torch.tensor([0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 1.0])

    future_states = item["future_state"]  # shape: [horizon, state_dim]
    actions = item["actions"]  # shape: [horizon, action_dim]
    actions = unscale_transform(actions, action_low, action_high)

    future_pos_w = future_states[:, 0:3]
    future_rotvec_w = future_states[:, 3:6]
    angle = torch.linalg.norm(future_rotvec_w, dim=-1)
    axis = future_rotvec_w / (angle.unsqueeze(-1) + 1e-8)
    future_quat_w = quat_from_angle_axis(angle, axis)
    # future_rpy_w = future_states[:, 3:6]
    # future_quat_w = quat_from_euler_xyz(*future_rpy_w.unbind(-1))

    cmd_dpos = actions[:, 0:3]
    cmd_drotvec = actions[:, 3:6]
    cmd_dangle = torch.linalg.norm(cmd_drotvec, dim=-1)
    cmd_daxis = cmd_drotvec / (cmd_dangle.unsqueeze(-1) + 1e-8)
    cmd_dquat = quat_from_angle_axis(cmd_dangle, cmd_daxis)
    # cmd_drpy = actions[:, 3:6]
    # cmd_dquat = quat_from_euler_xyz(*cmd_drpy.unbind(-1))

    if integrate_mode == "open_loop":
        current_pos_w = future_pos_w[0]
        current_quat_w = future_quat_w[0]
        target_pos_w_list = []
        target_quat_w_list = []
        for i in range(actions.shape[0]):
            target_pos_w_ = current_pos_w + cmd_dpos[i]
            target_quat_w_ = quat_mul(cmd_dquat[i], current_quat_w)
            target_pos_w_list.append(target_pos_w_)
            target_quat_w_list.append(target_quat_w_)
            current_pos_w = target_pos_w_
            current_quat_w = target_quat_w_
        target_pos_w = torch.stack(target_pos_w_list, dim=0)
        target_quat_w = torch.stack(target_quat_w_list, dim=0)
    elif integrate_mode == "close_loop":
        target_pos_w = future_pos_w + cmd_dpos
        target_quat_w = quat_mul(cmd_dquat, future_quat_w)
    else:
        raise ValueError(f"Unknown integrate_mode: {integrate_mode}")

    future_pos_w = future_pos_w[1:]
    future_quat_w = future_quat_w[1:]
    target_pos_w = target_pos_w[:-1]
    target_quat_w = target_quat_w[:-1]

    stride = max(1, future_pos_w.shape[0] // 20)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    plot_pose_trajectory(
        ax,
        future_pos_w,
        future_quat_w,
        axis_length=0.01,
        stride=stride,
        line_kwargs={"linewidth": 4.0, "linestyle": "--"},
        label="future pose",
        progress_cmap="Blues",
        start_marker_kwargs={"label": "future start"},
        end_marker_kwargs={"label": "future end"},
    )
    plot_pose_trajectory(
        ax,
        target_pos_w,
        target_quat_w,
        axis_length=0.01,
        stride=stride,
        line_kwargs={"linewidth": 4.0},
        label="target pose",
        progress_cmap="Oranges",
        start_marker_kwargs={"label": "target start"},
        end_marker_kwargs={"label": "target end"},
    )

    ax.legend()
    ax.set_title("Future vs. Target Pose Trajectories")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_visualize(integrate_mode="close_loop")
    test_visualize(integrate_mode="open_loop")
    test_episode()
