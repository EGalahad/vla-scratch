from typing import Any, Dict

import numpy as np
import torch

from vla_scratch.transforms.data_keys import (
    PROCESSED_ACTION_KEY,
    PROCESSED_IMAGE_KEY,
    PROCESSED_IMAGE_MASK_KEY,
    PROCESSED_STATE_KEY,
    TASK_KEY,
)
from vla_scratch.transforms.data_types import ActionChunk, Observation, DataSample
from vla_scratch.transforms.base import TransformFn


class ToDataSample(TransformFn):
    def compute(self, sample: Dict[str, torch.Tensor]) -> DataSample:  # type: ignore[override]
        observation = Observation(
            images=sample[PROCESSED_IMAGE_KEY],
            image_masks=sample[PROCESSED_IMAGE_MASK_KEY],
            state=sample[PROCESSED_STATE_KEY],
            task=sample[TASK_KEY],
        )
        if (actions := sample.get(PROCESSED_ACTION_KEY)) is not None:
            action = ActionChunk(actions=actions)
        else:
            action = None
        return DataSample(observation=observation, action_chunk=action)



class ToTorch(TransformFn):
    def compute(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {
            key: torch.from_numpy(val).to(torch.float32) if isinstance(val, np.ndarray) else val
            for key, val in sample.items()
        }


class ToNumpy(TransformFn):
    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                out[key] = value.cpu().numpy()
            else:
                out[key] = value
        return out
