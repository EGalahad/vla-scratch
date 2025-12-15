import importlib
from typing import Tuple, TYPE_CHECKING

import torch
import torch.nn.functional as F
from tensordict import TensorClass

from vla_scratch.transforms.base import TransformFn

if TYPE_CHECKING:
    from vla_scratch.transforms.data_types import DataSample
    from transformers import PaliGemmaProcessor


class PaligemmaPolicyInput(TensorClass):
    pixel_values: torch.FloatTensor
    input_ids: torch.LongTensor
    attention_mask: torch.BoolTensor


class PaligemmaProcessor(TransformFn):
    """Prepare image + prompt inputs for PaliGemma VLM bridges."""

    def __init__(
        self,
        processor_class: str,
        model_id: str,
        max_length: int = 256,
        truncation: bool = True,
        padding: str = "max_length",
        target_size: Tuple[int, int] = (224, 224),
    ) -> None:
        self.target_size = tuple(int(s) for s in target_size)
        processors = importlib.import_module("transformers")
        processor_cls = getattr(processors, processor_class)
        self.processor: "PaliGemmaProcessor" = processor_cls.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer

        self.truncation = truncation
        self.padding = padding
        self.max_length = max_length

    def compute(self, sample: "DataSample") -> "DataSample":
        # images = F.interpolate(
        #     sample.observation.images.type(torch.float32),
        #     size=self.target_size,
        #     mode="bilinear",
        #     align_corners=False,
        # ).type(torch.uint8)
        # images = (images / 255.0 - 0.5) / 0.5
        images = sample.observation.images.type(torch.uint8)
        pixel_values = self.processor.image_processor(images, return_tensors="pt")["pixel_values"]

        task_prompt: str = sample.observation.task
        # prompt: str = f"<bos>Task: {task_prompt}; \n Action:"
        prompt: str = f"<bos>{task_prompt}\n"
        # prompt: str = f"{task_prompt}"
        encoded = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors="pt",
        )
        policy_td = PaligemmaPolicyInput(
            # images=images,
            pixel_values=pixel_values,
            input_ids=encoded["input_ids"].squeeze(0).long(),
            attention_mask=encoded["attention_mask"].squeeze(0).bool(),
        )
        sample.observation.policy_input = policy_td
        return sample
