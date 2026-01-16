#!/usr/bin/env python3
import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, cast, TYPE_CHECKING
from setproctitle import setproctitle

import art
import emoji
import torch

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, MISSING, OmegaConf

from vla_scratch.transforms.data_keys import PROCESSED_ACTION_KEY
from vla_scratch.datasets.config import DataConfig
from vla_scratch.policies.config import PolicyConfig, create_policy

from vla_scratch.helpers.data import (
    build_input_transforms,
    build_output_transforms,
    create_dataset,
)
from vla_scratch.utils.checkpoint import (
    find_latest_checkpoint,
    load_model_from_checkpoint,
    merge_policy_cfg_from_checkpoint,
)

from vla_scratch.utils.serving.zmq_policy_server import ZmqPolicyServer
from vla_scratch.transforms.common import ToTorch, ToNumpy

if TYPE_CHECKING:
    from vla_scratch.transforms.data_types import DataSample
    from vla_scratch.policies.base import BasePolicy
    from vla_scratch.transforms.base import TransformFn

logger = logging.getLogger(__name__)


@dataclass
class ServeConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"policy": "pi-qwen"},
            {"data": "libero-spatial"},
        ]
    )

    # server
    host: str = "0.0.0.0"
    port: int = 8000
    inference_steps: int = 10

    # configs
    data: DataConfig = MISSING
    policy: PolicyConfig = MISSING
    checkpoint_path: Optional[str] = None
    merge_policy_cfg: bool = False


cs = ConfigStore.instance()
cs.store(name="serve", node=ServeConfig())


def _initialize_policy_dims(
    data_cfg: DataConfig, policy_cfg: PolicyConfig
) -> None:
    data_cfg.action_horizon = policy_cfg.action_horizon
    data_cfg.state_history = policy_cfg.state_history

    dataset = create_dataset(
        data_cfg,
        policy_cfg,
    )
    if len(dataset) == 0:
        raise ValueError(
            "Dataset is empty; unable to infer action/state dimensions."
        )

    data_sample: "DataSample" = dataset[0][0]
    action_tensor = (
        data_sample.action_chunk.actions
        if data_sample.action_chunk is not None
        else None
    )
    if action_tensor is None:
        raise ValueError(
            "Dataset sample has no actions; unable to infer action_dim."
        )
    if data_sample.observation.state is None:
        raise ValueError(
            "Dataset sample has no state; unable to infer state_dim."
        )

    action_dim = int(action_tensor.shape[-1])
    state_dim = int(data_sample.observation.state.shape[-1])

    if policy_cfg.action_dim is None:
        policy_cfg.action_dim = action_dim
    elif policy_cfg.action_dim != action_dim:
        logger.warning(
            "Policy action_dim=%s differs from dataset action_dim=%s; keeping policy value.",
            policy_cfg.action_dim,
            action_dim,
        )

    if policy_cfg.state_dim is None:
        policy_cfg.state_dim = state_dim
    elif policy_cfg.state_dim != state_dim:
        logger.warning(
            "Policy state_dim=%s differs from dataset state_dim=%s; keeping policy value.",
            policy_cfg.state_dim,
            state_dim,
        )
    return dataset


class ServePolicy:
    def __init__(
        self,
        model: "BasePolicy",
        input_transforms: Sequence["TransformFn"],
        output_transforms: Sequence["TransformFn"],
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
        data_sample: "DataSample" = data_sample.to(self._device).unsqueeze(0)

        actions = self._model.sample_actions(
            data_sample.observation, num_steps=self._num_steps
        )

        output = {
            PROCESSED_ACTION_KEY: actions.squeeze(0).cpu(),
        }
        for transform in self._output_transforms:
            output = transform.compute(output)
        return output

    def reset(self) -> None:
        pass


@hydra.main(config_name="serve", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    art.tprint("VLA-SCRATCH", font="big")
    setproctitle("vla-serve")
    if (checkpoint_path := cfg.get("checkpoint_path")) is not None:
        cfg.checkpoint_path = find_latest_checkpoint(checkpoint_path)
    if cfg.get("merge_policy_cfg", False):
        cfg = merge_policy_cfg_from_checkpoint(cfg, cfg.get("checkpoint_path"))
        OmegaConf.resolve(cfg)

    serve_cfg = cast(ServeConfig, OmegaConf.to_object(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Create model from policy config
    for i, spec in enumerate(list(serve_cfg.data.input_transforms or [])):
        if isinstance(spec, dict) and "enable_aug" in spec:
            spec.update({"enable_aug": False})
            serve_cfg.data.input_transforms[i] = spec

    dataset = _initialize_policy_dims(serve_cfg.data, serve_cfg.policy)
    with torch.device(device):
        model = create_policy(serve_cfg.policy)

    # Load latest checkpoint
    if (ckpt := serve_cfg.checkpoint_path) is not None:
        print(emoji.emojize(":package: Loading checkpoint..."))
        missing, unexpected = load_model_from_checkpoint(
            model, ckpt, device, strict=False
        )
        print(emoji.emojize(":package: Checkpoint loaded."))
        if missing:
            logger.warning("Missing keys when loading checkpoint: %s", missing)
        if unexpected:
            logger.warning(
                "Unexpected keys when loading checkpoint: %s", unexpected
            )

    model.eval()

    # Build transforms
    input_transforms = build_input_transforms(serve_cfg.data, serve_cfg.policy)
    output_transforms = build_output_transforms(
        serve_cfg.data, serve_cfg.policy
    )
    input_transforms = [ToTorch()] + input_transforms
    output_transforms = output_transforms + [ToNumpy()]

    # Wrap into serving policy
    policy = ServePolicy(
        model,
        input_transforms=input_transforms,
        output_transforms=output_transforms,
        inference_steps=serve_cfg.inference_steps,
    )

    # Warmup once to trigger initialization
    warmup = True
    # warmup = False
    if warmup:
        print(emoji.emojize(":fire: Warmup pass..."))
        observation_in = dataset.base_dataset[0]
        policy.infer(observation_in)

    policy.reset()
    server = ZmqPolicyServer(host=serve_cfg.host, port=serve_cfg.port)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(
        emoji.emojize(
            f":rocket: Server listening at tcp://{serve_cfg.host}:{serve_cfg.port} "
        )
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
