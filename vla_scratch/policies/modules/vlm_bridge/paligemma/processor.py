import importlib
from typing import Tuple, TYPE_CHECKING

import torch
import torch.nn.functional as F
from tensordict import TensorClass

from vla_scratch.transforms.base import TransformFn

if TYPE_CHECKING:
    from vla_scratch.transforms.data_types import DataSample


class PaligemmaPolicyInput(TensorClass):
    images: torch.FloatTensor
    input_ids: torch.LongTensor
    attention_mask: torch.BoolTensor


class PaligemmaProcessor(TransformFn):
    """Prepare image + prompt inputs for PaliGemma VLM bridges."""

    def __init__(
        self,
        processor_class: str,
        model_id: str,
        max_length: int = 256,
        target_size: Tuple[int, int] = (224, 224),
    ) -> None:
        self.target_size = tuple(int(s) for s in target_size)
        processors = importlib.import_module("transformers")
        processor_cls = getattr(processors, processor_class)
        processor = processor_cls.from_pretrained(model_id)
        self.tokenizer = processor.tokenizer
        self.max_length = max_length

    def compute(self, sample: "DataSample") -> "DataSample":
        images_orig = sample.observation.images
        images = F.interpolate(
            images_orig.type(torch.float32),
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        )
        images = (images / 255.0 - 0.5) / 0.5

        task_prompt: str = sample.observation.task
        prompt: str = f"<bos>Task: {task_prompt}; \n Action:"
        encoded = self.tokenizer(
            prompt,
            max_length=self.max_length,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        )
        policy_td = PaligemmaPolicyInput(
            images=images,
            input_ids=encoded["input_ids"].squeeze(0).long(),
            attention_mask=encoded["attention_mask"].squeeze(0).bool(),
        )
        sample.observation.policy_input = policy_td
        return sample
