from typing import Dict

import importlib

import torch
import torch.nn.functional as F

from vla_scratch.datasets.transforms import TransformFn
from vla_scratch.datasets.common import (
    PROCESSED_IMAGE_KEY,
    PROCESSED_IMAGE_MASK_KEY,
    TOKENIZED_KEY,
    TOKENIZED_MASK_KEY,
)


class PreprocessImage(TransformFn):
    def __init__(self, target_size: tuple[int, int] = (224, 224)) -> None:
        self.target_size = target_size

    def compute(self, sample: Dict) -> Dict:
        images: torch.Tensor = sample[PROCESSED_IMAGE_KEY].to(torch.float32)
        images = F.interpolate(
            images,
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        )
        images = (images / 255. - 0.5) / 0.5
        sample[PROCESSED_IMAGE_KEY] = images
        if PROCESSED_IMAGE_MASK_KEY not in sample:
            mask = torch.ones((images.shape[0], 1), dtype=torch.bool, device=images.device)
            sample[PROCESSED_IMAGE_MASK_KEY] = mask
        return sample


class TokenizePrompt(TransformFn):
    def __init__(
        self,
        processor_class: str,
        model_id: str,
        max_length: int = 256,
    ) -> None:
        # processors = importlib.import_module("transformers")
        # processor_cls = getattr(processors, processor_class)
        # processor = processor_cls.from_pretrained(model_id)
        from transformers import PaliGemmaProcessor
        processor = PaliGemmaProcessor.from_pretrained(model_id)
        self.tokenizer = processor.tokenizer
        self.max_length = max_length

    def compute(self, sample: Dict) -> Dict:
        prompt: str = sample["prompt"]
        encoded = self.tokenizer(
            prompt,
            max_length=self.max_length,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        )
        sample[TOKENIZED_KEY] = encoded["input_ids"].squeeze(0).long()
        sample[TOKENIZED_MASK_KEY] = encoded["attention_mask"].squeeze(0).bool()
        return sample
