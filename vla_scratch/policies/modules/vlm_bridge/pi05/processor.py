from typing import Tuple, TYPE_CHECKING

import torch
from tensordict import TensorClass
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as vision_fn

from vla_scratch.transforms.base import TransformFn
from vla_scratch.policies.modules.vlm_bridge.base import TARGET_IGNORE_ID

if TYPE_CHECKING:
    from vla_scratch.transforms.data_types import DataSample
    from transformers import PaliGemmaProcessor


PI05_PROCESSOR_REVISION = "35e4f46485b4d07967e7e9935bc3786aad50687c"


class Pi05PolicyInput(TensorClass):
    pixel_values: torch.FloatTensor
    input_ids: torch.LongTensor
    attention_mask: torch.BoolTensor
    target_ids: torch.LongTensor
    obs_register_att_mask: torch.BoolTensor


class Pi05Processor(TransformFn):
    """Prepare π0.5 image and task-prefix inputs."""

    def __init__(
        self,
        max_length: int = 200,
        truncation: bool = True,
        padding: str = "max_length",
        target_size: Tuple[int, int] = (224, 224),
        augment_images: bool = False,
        front_camera_indices: Tuple[int, ...] = (0,),
    ) -> None:
        self.target_size = tuple(int(s) for s in target_size)
        self.augment_images = augment_images
        self.front_camera_indices = frozenset(front_camera_indices)
        self.training = False
        from transformers import PaliGemmaProcessor

        self.processor: "PaliGemmaProcessor" = (
            PaliGemmaProcessor.from_pretrained(
                "google/paligemma-3b-pt-224",
                revision=PI05_PROCESSOR_REVISION,
            )
        )
        self.tokenizer = self.processor.tokenizer

        self.truncation = truncation
        self.padding = padding
        self.max_length = max_length

    def compute(self, sample: "DataSample") -> "DataSample":
        images = sample.observation.images.type(torch.uint8)
        images = torch.stack(
            [
                vision_fn.resize(image, list(self.target_size), antialias=True)
                for image in images
            ]
        )
        if self.augment_images and self.training:
            invalid_indices = self.front_camera_indices - set(range(len(images)))
            if invalid_indices:
                raise ValueError(
                    "front_camera_indices contains unavailable cameras: "
                    f"{sorted(invalid_indices)}"
                )
            images = torch.stack(
                [
                    self._augment_image(
                        image,
                        is_wrist=index not in self.front_camera_indices,
                    )
                    for index, image in enumerate(images)
                ]
            )
        images_list = [[img for img in images]]
        num_images = images.shape[0]
        prompt = (
            sample.observation.task.lower()
            .strip()
            .replace("_", " ")
            .replace("\n", " ")
        )
        prompt = "".join(["<image>"] * num_images + [prompt])
        max_length = (
            num_images * self.processor.image_seq_length + self.max_length
        )
        encoded = self.processor(
            text=prompt,
            images=images_list,
            suffix=None,
            max_length=max_length,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        target_ids = encoded.get("labels")
        if target_ids is None:
            target_ids = input_ids.clone()
            target_ids.fill_(TARGET_IGNORE_ID)
        else:
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is not None:
                target_ids[input_ids == eos_token_id] = TARGET_IGNORE_ID

        # print("Decoded input ids:")
        # print(self.tokenizer.decode(input_ids[0], skip_special_tokens=False))

        # print("Decoded input ids after replacing image token id with 0:")
        # input_ids[input_ids == self.processor.image_token_id] = 0
        # print(self.tokenizer.decode(input_ids[0], skip_special_tokens=False))

        # print("target ids:")
        # target_ids[target_ids == TARGET_IGNORE_ID] = 0
        # print(self.tokenizer.decode(target_ids[0], skip_special_tokens=False))
        # breakpoint()

        policy_td = Pi05PolicyInput(
            pixel_values=encoded["pixel_values"],
            input_ids=input_ids.squeeze(0).long(),
            attention_mask=attention_mask.squeeze(0).bool(),
            target_ids=target_ids.squeeze(0).long(),
            obs_register_att_mask=attention_mask.squeeze(0).bool(),
        )
        sample.observation.policy_input = policy_td
        return sample

    def set_training(self, enabled: bool) -> None:
        self.training = enabled

    @staticmethod
    def _augment_image(image: torch.Tensor, *, is_wrist: bool) -> torch.Tensor:
        """Apply OpenPI augmentation with independent randomness per image."""
        if not is_wrist:
            height, width = image.shape[-2:]
            crop_height, crop_width = int(height * 0.95), int(width * 0.95)
            top = int(torch.randint(height - crop_height + 1, ()).item())
            left = int(torch.randint(width - crop_width + 1, ()).item())
            image = vision_fn.resized_crop(
                image,
                top,
                left,
                crop_height,
                crop_width,
                [height, width],
                antialias=True,
            )
            angle = float(torch.empty(()).uniform_(-5.0, 5.0).item())
            image = vision_fn.rotate(image, angle)
        return ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)(
            image
        )
