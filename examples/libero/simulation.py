#!/usr/bin/env python3
"""
Roll out the LIBERO benchmark while querying a remote policy over ZMQ (Hydra-configured).

- Configures the LIBERO repo (paths + imports) locally.
- Runs each task in the selected suite, forwarding observations to the policy
  server with the same keys/layout used in training (lerobot_ipec).
- Applies the returned action chunk's first step to the simulator.
"""

import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple, cast, List

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

# os.environ.setdefault("MUJOCO_GL", "glfw")

import imageio.v2 as imageio
import numpy as np
import robosuite as suite
import robosuite.utils.transform_utils as T

from vla_scratch.utils.serving.zmq_policy_client import ZmqPolicyClient
from vla_scratch.transforms.data_keys import (
    PROCESSED_IMAGE_KEY,
    PROCESSED_IMAGE_MASK_KEY,
    PROCESSED_STATE_KEY,
    TASK_KEY,
)


@dataclass
class LiberoEvalConfig:
    defaults: list[Any] = field(default_factory=lambda: ["_self_"])

    host: str = "127.0.0.1"
    port: int = 8000
    libero_task_suite: str = "libero_spatial"

    state_history: int = 1
    action_chunk_size: int = 5

    max_steps: int = 250

    headless: bool = True
    render_every: int = 1
    camera_resolution: int = 256
    render_camera: str = "frontview"

    rotate_images: bool = True
    settle_steps: int = 5
    seed: int = 0
    episodes_per_task: int = 1
    video_path: Optional[str] = None


cs = ConfigStore.instance()
cs.store(name="eval_libero", node=LiberoEvalConfig)


def get_libero_dummy_action() -> list[float]:
    """No-op action used to let the environment settle."""
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def _image_to_chw(image: np.ndarray, *, rotate: bool = True) -> np.ndarray:
    """Convert env RGB image (H, W, 3) to CHW float32 in [0, 1]."""
    img = np.asarray(image, dtype=np.float32)
    if rotate:
        img = img[::-1, ::-1]
    img = np.transpose(img, (2, 0, 1)) / 255.0
    return img


def _state_from_obs(
    obs: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
    rotvec = np.asarray(T.quat2axisangle(quat), dtype=np.float32)
    grip = np.asarray(
        obs.get("robot0_gripper_qpos", np.zeros(2, dtype=np.float32)),
        dtype=np.float32,
    )
    return pos, rotvec, grip


def _frame_from_obs(obs: Dict[str, Any], *, rotate_images: bool) -> np.ndarray:
    frame = np.asarray(obs["agentview_image"])
    if frame.ndim == 3 and frame.shape[0] == 3 and frame.shape[-1] != 3:
        frame = np.transpose(frame, (1, 2, 0))
    if rotate_images:
        frame = frame[::-1, ::-1]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0.0, 1.0)
        frame = (frame * 255.0).astype(np.uint8)
    return frame


def _build_policy_sample(
    obs: Dict[str, Any],
    task_description: str,
    state_buffers: Tuple[
        Deque[np.ndarray], Deque[np.ndarray], Deque[np.ndarray]
    ],
    rotate_images: bool,
) -> Dict[str, Any]:
    pos_hist, rot_hist, grip_hist = state_buffers
    cam_front = _image_to_chw(obs["agentview_image"], rotate=rotate_images)
    cam_wrist = _image_to_chw(
        obs["robot0_eye_in_hand_image"], rotate=rotate_images
    )
    img = np.stack([cam_front, cam_wrist], axis=0)
    img = (img * 255).astype(np.uint8, copy=False)
    img_mask = np.ones((img.shape[0], 1), dtype=bool)
    pos_stack = np.stack(list(pos_hist), axis=0).astype(np.float32, copy=False)
    rot_stack = np.stack(list(rot_hist), axis=0).astype(np.float32, copy=False)
    grip_stack = np.stack(list(grip_hist), axis=0).astype(
        np.float32, copy=False
    )
    state = np.concatenate([pos_stack, rot_stack, grip_stack], axis=-1)[1:]
    return {
        PROCESSED_IMAGE_KEY: img,
        PROCESSED_IMAGE_MASK_KEY: img_mask,
        PROCESSED_STATE_KEY: state,
        TASK_KEY: task_description,
    }


def _to_action_list(resp: Dict[str, Any]) -> List[np.ndarray]:
    actions = np.asarray(resp["action_chunk.actions"], dtype=np.float32)
    return [actions[i] for i in range(actions.shape[0])]


def rollout_task(
    client: ZmqPolicyClient,
    env,
    task_description: str,
    cfg: LiberoEvalConfig,
    episode_idx: int,
) -> bool:
    state_len = cfg.state_history + 1
    pos_hist: Deque[np.ndarray] = deque(maxlen=state_len)
    rot_hist: Deque[np.ndarray] = deque(maxlen=state_len)
    grip_hist: Deque[np.ndarray] = deque(maxlen=state_len)

    obs = env.reset()

    # Allow environment to settle
    for _ in range(cfg.settle_steps):
        dummy = get_libero_dummy_action()
        obs, _, _, _ = env.step(dummy)

    pos, rot, grip = _state_from_obs(obs)
    for _ in range(state_len):
        pos_hist.append(pos)
        rot_hist.append(rot)
        grip_hist.append(grip)

    action_queue = []
    frames: List[np.ndarray] = []
    success = False
    start_time = time.monotonic()
    for step in range(cfg.max_steps):
        sample = _build_policy_sample(
            obs,
            task_description,
            (pos_hist, rot_hist, grip_hist),
            cfg.rotate_images,
        )
        if cfg.video_path:
            frames.append(_frame_from_obs(obs, rotate_images=cfg.rotate_images))

        if len(action_queue) == 0:
            start_inf_time = time.monotonic()
            resp = client.infer(sample)
            infer_time = time.monotonic() - start_inf_time
            relative_time = time.monotonic() - start_time
            model_time = resp["server_timing"]["infer_s"]
            print(
                f"[{relative_time:.3f}s from start] inference time {infer_time:.3f}s, "
                f"model time {model_time:.3f}s"
            )
            action_list = _to_action_list(resp)
            action_queue.extend(action_list[: cfg.action_chunk_size])
        obs, _, done, info = env.step(action_queue.pop(0))

        if (
            not cfg.headless
            and (cfg.render_every > 0)
            and (step % cfg.render_every == 0)
        ):
            if hasattr(env, "render"):
                env.render()
            else:
                env.env.render()

        pos, rot, grip = _state_from_obs(obs)
        pos_hist.append(pos)
        rot_hist.append(rot)
        grip_hist.append(grip)

        if done or info.get("success"):
            success = True
            break

    if cfg.video_path and frames:
        safe_task = task_description.replace("/", "_")
        video_file = os.path.join(
            cfg.video_path,
            f"episode_{episode_idx}_task_{safe_task}_success_{success}.mp4",
        )
        with imageio.get_writer(video_file, fps=20) as writer:
            for frame in frames:
                writer.append_data(frame)
    return success


@hydra.main(config_name="eval_libero", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    args = cast(LiberoEvalConfig, OmegaConf.to_object(cfg))

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.env_wrapper import ControlEnv

    class OnScreenRenderEnv(ControlEnv):
        """Enable on-screen rendering while keeping offscreen buffers for images."""

        def __init__(self, **kwargs):
            kwargs["has_renderer"] = True
            kwargs["has_offscreen_renderer"] = True
            kwargs.setdefault("render_camera", "frontview")
            super().__init__(**kwargs)

    task_suite = benchmark.get_benchmark_dict()[args.libero_task_suite]()
    client = ZmqPolicyClient(host=args.host, port=args.port)

    print(
        f"Running LIBERO suite {args.libero_task_suite} against policy server at {args.host}:{args.port}"
    )

    if args.video_path:
        os.makedirs(args.video_path, exist_ok=True)

    successes = []
    episode_idx = 0
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        env_args = {
            "bddl_file_name": os.path.join(
                get_libero_path("bddl_files"),
                task.problem_folder,
                task.bddl_file,
            ),
            "camera_heights": args.camera_resolution,
            "camera_widths": args.camera_resolution,
            "has_renderer": not args.headless,
            "has_offscreen_renderer": True,
            "render_camera": args.render_camera,
        }
        env_cls = OffScreenRenderEnv if args.headless else OnScreenRenderEnv
        env = env_cls(**env_args)
        env.seed(args.seed)

        task_desc = task.language
        for i in range(args.episodes_per_task):
            print(f"[task {task_id} ep {i:02d}]: {task_desc}")
            success = rollout_task(
                client,
                env,
                task_desc,
                args,
                episode_idx=episode_idx,
            )
            successes.append(success)
            episode_idx += 1
            print(f"  success={success}")
        env.close()
    success_rate = sum(successes) / len(successes) if successes else 0.0
    print(f"Success rate: {success_rate:.2f}")


if __name__ == "__main__":
    main()
