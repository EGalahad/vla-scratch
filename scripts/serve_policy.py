#!/usr/bin/env python3
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, cast

import numpy as np
import torch

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, MISSING, OmegaConf

from vla_scratch.transforms.data_keys import PROCESSED_ACTION_KEY
from vla_scratch.datasets.config import DataConfig
from vla_scratch.policies.config import PolicyConfig, create_policy

from vla_scratch.transforms.data_types import DataSample
from vla_scratch.transforms.base import TransformFn
from vla_scratch.helpers.data import (
    build_input_transforms,
    build_output_transforms,
)
from vla_scratch.utils.checkpoint import (
    find_latest_checkpoint,
    load_model_from_checkpoint,
)
from vla_scratch.utils.config import merge_cfg_from_checkpoint
from vla_scratch.datasets.libero.data_keys import (
    ARM_STATE_CART_POS_KEY,
    ARM_STATE_CART_ROT_KEY,
    CAM_FRONT_KEY,
    CAM_WRIST_KEY,
    GRIPPER_STATE_QPOS_KEY,
    TASK_NAME_KEY,
)

from vla_scratch.serving.zmq_policy_server import ZmqPolicyServer
from vla_scratch.transforms.common import ToTorch, ToNumpy

logger = logging.getLogger(__name__)


@dataclass
class ServeConfig:
    defaults: list[Any] = field(
        default_factory=lambda: ["_self_", {"policy": "pi"}, {"data": "libero-ipec"}]
    )

    # server
    host: str = "0.0.0.0"
    port: int = 8000
    inference_steps: int = 10

    # configs
    data: DataConfig = MISSING
    policy: PolicyConfig = MISSING
    checkpoint_path: Optional[str] = None


cs = ConfigStore.instance()
cs.store(name="serve", node=ServeConfig())


def _state_tensors_from_obs(obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    def _as_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().clone().type(torch.float32)
        return torch.from_numpy(value).type(torch.float32)

    return {
        ARM_STATE_CART_POS_KEY: _as_tensor(obs[ARM_STATE_CART_POS_KEY]) if ARM_STATE_CART_POS_KEY in obs else None,
        ARM_STATE_CART_ROT_KEY: _as_tensor(obs[ARM_STATE_CART_ROT_KEY]) if ARM_STATE_CART_ROT_KEY in obs else None,
        GRIPPER_STATE_QPOS_KEY: _as_tensor(obs[GRIPPER_STATE_QPOS_KEY]) if GRIPPER_STATE_QPOS_KEY in obs else None,
    }


class ServePolicy:
    def __init__(
        self,
        model: torch.nn.Module,
        input_transforms: Sequence[TransformFn],
        output_transforms: Sequence[TransformFn],
        inference_steps: int = 10,
    ) -> None:
        self._model = model
        self._num_steps = inference_steps
        self._device = next(model.parameters()).device
        self._input_transforms = input_transforms
        self._output_transforms = output_transforms

    @torch.inference_mode()
    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        data_sample = obs
        for transform in self._input_transforms:
            data_sample = transform.compute(data_sample)
        data_sample: DataSample = data_sample.to(self._device).unsqueeze(0)

        actions = self._model.sample_actions(data_sample.observation, num_steps=self._num_steps)

        output = {
            PROCESSED_ACTION_KEY: actions.squeeze(0).cpu(),
            **_state_tensors_from_obs(obs),
        }
        for transform in self._output_transforms:
            output = transform.compute(output)
        return output

    def reset(self) -> None:
        pass


class ReplayPolicy:
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
        input_transforms: Sequence[TransformFn],
        output_transforms: Sequence[TransformFn],
        inference_steps: int = 10,
    ) -> None:
        self._dataset = dataset
        self._counter = 0

        self._model = model
        self._num_steps = inference_steps
        self._device = next(model.parameters()).device
        self._num_steps = inference_steps
        self._input_transforms = input_transforms
        self._output_transforms = output_transforms

    @torch.inference_mode()
    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        data_sample, _ = self._dataset[self._counter]
        actions = data_sample.action_chunk.actions.unsqueeze(0)
        self._counter += 1
        print(self._counter)

        state_payload = {
            key: tensor.unsqueeze(0) for key, tensor in _state_tensors_from_obs(obs).items()
        }
        output = {
            PROCESSED_ACTION_KEY: actions.squeeze(0).cpu(),
            **state_payload,
        }
        for transform in self._output_transforms:
            output = transform.compute(output)
        return output

    def reset(self) -> None:
        self._counter = 0


@hydra.main(config_name="serve", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # Optionally override fields with saved configs from checkpoint dir
    serve_cfg = cast(ServeConfig, OmegaConf.to_object(cfg))
    if serve_cfg.checkpoint_path is not None:
        cfg = merge_cfg_from_checkpoint(cfg, serve_cfg.checkpoint_path)
    serve_cfg = cast(ServeConfig, OmegaConf.to_object(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Create model from policy config
    print("Initializing model...")
    with torch.device(device):
        model = create_policy(serve_cfg.policy)
    print("Model initialized.")

    # Load latest checkpoint
    if serve_cfg.checkpoint_path is not None:
        ckpt = find_latest_checkpoint(Path(serve_cfg.checkpoint_path))
        if ckpt is None:
            raise FileNotFoundError(
                f"No checkpoint found under {serve_cfg.checkpoint_path}"
            )
        print(f"Loading checkpoint: {ckpt}")
        missing, unexpected = load_model_from_checkpoint(model, ckpt, device, strict=False)
        print("Checkpoint loaded.")
        if missing:
            logger.warning("Missing keys when loading checkpoint: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys when loading checkpoint: %s", unexpected)

    model.eval()

    # Build transforms
    for i, spec in enumerate(list(serve_cfg.data.input_transforms or [])):
        if (
            isinstance(spec, dict)
            and "enable_aug" in spec
        ):
            spec.update({"enable_aug": False})
            serve_cfg.data.input_transforms[i] = spec

    input_transforms = [ToTorch()] + build_input_transforms(serve_cfg.data, serve_cfg.policy)
    output_transforms = build_output_transforms(serve_cfg.data, serve_cfg.policy) + [ToNumpy()]

    # Wrap into serving policy
    policy = ServePolicy(
        model,
        input_transforms=input_transforms,
        output_transforms=output_transforms,
        inference_steps=serve_cfg.inference_steps,
    )
    # policy = ReplayPolicy(
    #     dataset,
    #     model,
    #     input_transforms=input_transforms,
    #     output_transforms=output_transforms,
    #     inference_steps=serve_cfg.inference_steps,
    # )

    metadata = {
        "policy": serve_cfg.policy._target_.split(".")[-1],
        "device": str(device),
    }

    # Warmup once to trigger initialization
    # warmup = True
    warmup = False
    if warmup:
        history_len = serve_cfg.policy.state_history + 1
        observation_in = {
            CAM_FRONT_KEY: np.random.randint(0, 255, size=(3, 256, 256), dtype=np.uint8),
            CAM_WRIST_KEY: np.random.randint(0, 255, size=(3, 256, 256), dtype=np.uint8),
            ARM_STATE_CART_POS_KEY: np.random.rand(history_len, 3).astype(np.float32),
            ARM_STATE_CART_ROT_KEY: np.random.rand(history_len, 3).astype(np.float32),
            GRIPPER_STATE_QPOS_KEY: np.random.rand(history_len, 2).astype(np.float32),
            TASK_NAME_KEY: "Pick up the red block and place it on the green block.",
        }
        policy.infer(observation_in)
        policy.reset()

    server = ZmqPolicyServer(host=serve_cfg.host, port=serve_cfg.port, metadata=metadata)
    # server = WebsocketPolicyServer(policy=policy, host=serve_cfg.host, port=serve_cfg.port, metadata=metadata)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(
        f"Serving policy {metadata.get('policy')} on {serve_cfg.host}:{serve_cfg.port} (host={hostname} ip={local_ip})",
    )

    try:
        while True:
            request = server.wait_for_request()
            if request is None:
                continue
            msg_type = request.get("type", "infer")
            if msg_type == "reset":
                policy.reset()
                server.send_response({"status": "ok"})
                continue

            obs = {k: v for k, v in request.items() if k != "type"}
            # for key, value in obs.items():
            #     if isinstance(value, list):
            #         obs[key] = np.array(value)
            #     print(f"obs[{key}]: {type(obs[key])} {obs[key].shape if isinstance(obs[key], np.ndarray) else ''}")
            # print("---")
            t0 = time.monotonic()
            action = policy.infer(obs)
            infer_s = time.monotonic() - t0
            response = dict(action)
            response["server_timing"] = {"infer_s": infer_s}
            server.send_response(response)
    except KeyboardInterrupt:
        logger.info("Shutting down server loop.")
    finally:
        server.close()


if __name__ == "__main__":
    main()
