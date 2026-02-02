import importlib
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

import torch
from tensordict import TensorClass

from vla_scratch.transforms.base import TransformFn
from vla_scratch.policies.modules.vlm_bridge.base import TARGET_IGNORE_ID
from vla_scratch.utils.paths import REPO_ROOT

if TYPE_CHECKING:
    from vla_scratch.transforms.data_types import DataSample
    from transformers import PaliGemmaProcessor


class PaligemmaPolicyInput(TensorClass):
    pixel_values: torch.FloatTensor
    input_ids: torch.LongTensor
    attention_mask: torch.BoolTensor
    target_ids: torch.LongTensor
    obs_register_att_mask: torch.BoolTensor


class PaligemmaProcessor(TransformFn):
    """Prepare image + prompt inputs for PaliGemma VLM bridges."""

    @staticmethod
    def _resolve_model_id(model_id: str) -> tuple[str, bool]:
        """Resolve relative local paths even after Hydra changes cwd.

        Returns (resolved_id, is_local_path).
        """
        p = Path(model_id).expanduser()
        if p.exists():
            return str(p.resolve()), True
        p2 = (REPO_ROOT / p).resolve()
        if p2.exists():
            return str(p2), True
        return model_id, False

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
        resolved_id, is_local = self._resolve_model_id(model_id)
        self.processor: "PaliGemmaProcessor" = processor_cls.from_pretrained(
            resolved_id,
            local_files_only=is_local,
        )
        self.tokenizer = self.processor.tokenizer

        self.truncation = truncation
        self.padding = padding
        self.max_length = max_length
        self.prompt_sep_text = "<<<PROMPT_SEP>>>"
        self.prompt_sep_ids = self.tokenizer.encode(
            self.prompt_sep_text, add_special_tokens=False
        )

    def compute(self, sample: "DataSample") -> "DataSample":
        images = sample.observation.images.type(torch.uint8)
        images_list = [[img for img in images]]
        prompt = f"{sample.observation.task}{self.prompt_sep_text}{sample.observation.generation_prompt}"
        num_images = images.shape[0]
        prompt = "".join(["<image>"] * num_images + [prompt])
        suffix = sample.observation.generation_answer
        encoded = self.processor(
            text=prompt,
            images=images_list,
            suffix=suffix,
            max_length=self.max_length,
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

        obs_register_att_mask = self._build_obs_register_att_mask(
            input_ids, attention_mask
        )
        policy_td = PaligemmaPolicyInput(
            pixel_values=encoded["pixel_values"],
            input_ids=input_ids.squeeze(0).long(),
            attention_mask=attention_mask.squeeze(0).bool(),
            target_ids=target_ids.squeeze(0).long(),
            obs_register_att_mask=obs_register_att_mask.squeeze(0).bool(),
        )
        sample.observation.policy_input = policy_td
        return sample

    def _build_obs_register_att_mask(
        self, input_ids: torch.LongTensor, attention_mask: torch.Tensor
    ) -> torch.BoolTensor:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)

        bsz, seqlen = input_ids.shape
        mask = torch.zeros(
            (bsz, seqlen), dtype=torch.bool, device=input_ids.device
        )
        for b in range(bsz):
            sep_start = self._find_subsequence(
                input_ids[b].tolist(), self.prompt_sep_ids
            )
            if sep_start is None:
                mask[b] = attention_mask[b].bool()
            else:
                mask[b, :sep_start] = attention_mask[b, :sep_start].bool()
        return mask

    @staticmethod
    def _find_subsequence(
        sequence: list[int], subsequence: list[int]
    ) -> Optional[int]:
        if not subsequence:
            return None
        max_start = len(sequence) - len(subsequence)
        for i in range(max_start + 1):
            if sequence[i : i + len(subsequence)] == subsequence:
                return i
        return None
