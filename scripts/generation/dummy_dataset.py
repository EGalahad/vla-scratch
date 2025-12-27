from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

from vla_scratch.transforms.data_keys import (
    PROCESSED_IMAGE_KEY,
    PROCESSED_IMAGE_MASK_KEY,
    PROCESSED_STATE_KEY,
    TASK_KEY,
)


def _gradient_image(height: int = 384, width: int = 384) -> torch.Tensor:
    """Make a simple RGB gradient tensor in [0, 255] matching build_demo_image."""
    red = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    green = red[::-1]
    blue = np.full_like(red, 128, dtype=np.uint8)
    stacked = np.stack([red, green, blue], axis=0)  # [3, H, W]
    return torch.from_numpy(stacked).float()


def _car_image() -> torch.Tensor:
    """Create a deterministic RGB gradient so the script works without external assets."""
    img = Image.open("/mnt/vepfs01/output/haoyang/quickstart/mozbrain/car.jpg")
    return torch.from_numpy(np.array(img).transpose(2, 0, 1)).float()

class DummyDataset(Dataset):
    """Single-sample dataset providing the fields expected by ToDataSample."""

    def __init__(self, cfg) -> None:
        self.state_dim = getattr(cfg, "state_dim", 8)
        self.task = getattr(cfg, "task_text", "Describe the image in English.")
        self.task = getattr(cfg, "task_text", "describe en")
        self.image = _gradient_image()
        # self.image = _car_image()

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        sample = {
            PROCESSED_IMAGE_KEY: self.image.unsqueeze(0),  # [num_cam=1, 3, H, W]
            PROCESSED_IMAGE_MASK_KEY: torch.ones(1, 1, dtype=torch.bool),
            PROCESSED_STATE_KEY: torch.zeros(1, self.state_dim),  # [state_history=1, state_dim]
            TASK_KEY: self.task,
        }
        return sample
