#!/usr/bin/env python3
"""
Run bbox_cotrain ManiSkill environments and query a vla-scratch policy over ZMQ with action chunking.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, cast

import gymnasium as gym
import hydra
import imageio.v2 as imageio
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

import sys
sys.path.append(".")
from vla_scratch.serving.zmq_policy_client import ZmqPolicyClient
from vla_scratch.transforms.data_keys import (
    PROCESSED_ACTION_KEY,
    PROCESSED_IMAGE_KEY,
    PROCESSED_IMAGE_MASK_KEY,
    PROCESSED_STATE_KEY,
    TASK_KEY,
    GENERATION_PROMPT_KEY,
    GENERATION_ANSWER_KEY,
)
# from vla_scratch.datasets.libero.data_keys import (
#     ARM_STATE_CART_POS_KEY,
#     ARM_STATE_CART_ROT_KEY,
#     GRIPPER_STATE_QPOS_KEY,
# )


from mani_skill.envs import *  # noqa: F401,F403

@dataclass
class BboxCotrainEvalConfig:
    defaults: list[Any] = field(default_factory=lambda: ["_self_"])

    host: str = "127.0.0.1"
    port: int = 8000
    env_id: str = "PutOnPlateInScene25MultiCarrot2-v1"
    episodes: int = 10
    max_steps: int = 80
    action_chunk_size: int = 5
    seed: int = 0
    state_history: int = 0
    obj_set: str = "test"
    episode_id: Optional[int] = None
    same_init: bool = False
    render: bool = False
    sim_backend: str = "gpu"
    shader: str = "default"
    video_path: Optional[str] = None


cs = ConfigStore.instance()
cs.store(name="eval_bbox_cotrain", node=BboxCotrainEvalConfig)


def build_payload(
    obs: Dict[str, Any],
    instruction: str,
    *,
    state_history: int,
) -> Dict[str, Any]:
    img = obs["sensor_data"]["3rd_view_camera"]["rgb"]
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)
    if img.ndim == 3:
        img = img[None, ...]
    img_hwc = img  # (B, H, W, 3)
    img_chw = np.transpose(img_hwc, (0, 3, 1, 2))  # (B, 3, H, W)
    return {
        PROCESSED_IMAGE_KEY: img_chw[0, None, ...],  # (3, H, W)
        PROCESSED_IMAGE_MASK_KEY: np.ones((1, 1), dtype=bool),
        PROCESSED_STATE_KEY: np.zeros((state_history + 1, 1), dtype=np.float32),
        TASK_KEY: instruction,
        GENERATION_PROMPT_KEY: "",
        GENERATION_ANSWER_KEY: "",
        # ARM_STATE_CART_POS_KEY: np.zeros((batch, state_history + 1, 3), dtype=np.float32),
        # ARM_STATE_CART_ROT_KEY: np.zeros((batch, state_history + 1, 3), dtype=np.float32),
        # GRIPPER_STATE_QPOS_KEY: np.zeros((batch, state_history + 1, 1), dtype=np.float32),
    }

def _build_env(args: BboxCotrainEvalConfig) -> gym.Env:
    new_width, new_height = 640, 480
    scale_x = new_width / 640.0
    scale_y = new_height / 480.0
    new_intrinsic = np.array(
        [
            [623.588 * scale_x, 0, 319.501 * scale_x],
            [0, 623.588 * scale_y, 239.545 * scale_y],
            [0, 0, 1],
        ]
    )
    env_kwargs = dict(
        id=args.env_id,
        num_envs=1,
        obs_mode="rgb+segmentation",
        control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        sim_backend=args.sim_backend,
        sim_config={"sim_freq": 500, "control_freq": 5},
        max_episode_steps=args.max_steps,
        sensor_configs={
            "shader_pack": args.shader,
            "3rd_view_camera": {
                "width": new_width,
                "height": new_height,
                "intrinsic": new_intrinsic,
            },
        },
        render_mode="human" if args.render else None,
        render_backend="cpu",
        enable_shadow=True,
        parallel_in_single_scene=False,
    )
    return gym.make(**env_kwargs)


def _maybe_tensor_action(action: np.ndarray, env: gym.Env) -> torch.Tensor | np.ndarray:
    if not hasattr(env.unwrapped, "device"):
        return action
    action_tensor = torch.as_tensor(action, dtype=torch.float32, device=env.unwrapped.device)
    if action_tensor.ndim == 1:
        action_tensor = action_tensor.unsqueeze(0)
    return action_tensor


def _is_done(terminated: Any, truncated: Any) -> bool:
    if isinstance(terminated, torch.Tensor) or isinstance(truncated, torch.Tensor):
        return bool(torch.any(terminated) or torch.any(truncated))
    return bool(np.any(terminated) or np.any(truncated))

def _frame_from_obs(obs: Dict[str, Any]) -> np.ndarray:
    frame = obs["sensor_data"]["3rd_view_camera"]["rgb"]
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0.0, 1.0)
        frame = (frame * 255.0).astype(np.uint8)
    return frame


@hydra.main(config_name="eval_bbox_cotrain", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    args = cast(BboxCotrainEvalConfig, OmegaConf.to_object(cfg))

    env = _build_env(args)
    client = ZmqPolicyClient(host=args.host, port=args.port)
    shared_episode_id: Optional[int] = None
    if args.same_init:
        shared_episode_id = int(np.random.RandomState(args.seed).randint(0, 1_000_000_000))

    if args.video_path:
        os.makedirs(args.video_path, exist_ok=True)

    for ep in range(args.episodes):
        reset_options: Dict[str, Any] = {"obj_set": args.obj_set}
        episode_id = args.episode_id if args.episode_id is not None else shared_episode_id
        if episode_id is not None:
            reset_options["episode_id"] = episode_id
        obs, info = env.reset(seed=[args.seed + ep], options=reset_options)
        instruction_raw = env.unwrapped.get_language_instruction()
        instruction = instruction_raw[0] if isinstance(instruction_raw, (list, tuple)) else instruction_raw
        print(f"[Episode {ep}] Instruction: {instruction}")

        frames: List[np.ndarray] = []
        if args.video_path:
            frames.append(_frame_from_obs(obs))

        action_queue: List[np.ndarray] = []
        for step in range(args.max_steps):
            payload = build_payload(obs, instruction, state_history=args.state_history)
            if len(action_queue) == 0:
                start_inf = time.monotonic()
                resp = client.infer(payload)
                infer_time = time.monotonic() - start_inf
                model_time = resp["server_timing"]["infer_s"]
                print(f"inference time {infer_time:.3f}s, model time {model_time:.3f}s")

                actions = np.asarray(resp[PROCESSED_ACTION_KEY])

                # actions = np.ones((args.action_chunk_size, env.action_space.shape[0]), dtype=np.float32) * 0.01

                action_list = [np.asarray(a, dtype=np.float32).reshape(-1) for a in actions]
                action_queue.extend(action_list[: args.action_chunk_size])

            action_step = action_queue.pop(0).reshape(1, -1)
            action_step = _maybe_tensor_action(action_step, env)
            obs, _reward, terminated, truncated, info = env.step(action_step)

            if args.video_path:
                frames.append(_frame_from_obs(obs))

            if args.render:
                env.render()

            if _is_done(terminated, truncated):
                if isinstance(info, dict) and "success" in info:
                    success = info["success"]
                    if isinstance(success, torch.Tensor):
                        success = success.detach().cpu().numpy()
                    print(f"[Episode {ep}] success={success}")
                break

        if args.video_path and frames:
            video_file = os.path.join(args.video_path, f"episode_{ep}_task_{instruction}_success_{success}.mp4")
            with imageio.get_writer(video_file, fps=5) as writer:
                for frame in frames:
                    writer.append_data(frame)

    client.close()
    env.close()


if __name__ == "__main__":
    main()
