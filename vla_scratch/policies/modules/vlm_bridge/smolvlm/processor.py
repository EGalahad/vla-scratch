import importlib
from typing import Dict, List, Optional, TYPE_CHECKING

import torch
from tensordict import TensorClass

from vla_scratch.transforms.base import TransformFn
from vla_scratch.policies.modules.vlm_bridge.base import TARGET_IGNORE_ID

if TYPE_CHECKING:
    from vla_scratch.transforms.data_types import DataSample
    from transformers import SmolVLMProcessor


class SmolVLMPolicyInput(TensorClass):
    input_ids: torch.LongTensor
    target_ids: torch.LongTensor
    attention_mask: torch.BoolTensor
    obs_register_att_mask: torch.BoolTensor
    pixel_values: torch.FloatTensor


class SmolVLMProcessor(TransformFn):
    """Tokenize prompt using SmolVLM chat template and produce policy inputs."""

    def __init__(
        self,
        processor_class: str,
        model_id: str,
        max_length: int = 256,
        padding: str | bool = "max_length",
        image_size_longest_edge: int | None = None,
        max_image_size_longest_edge: int | None = None,
    ) -> None:
        processors = importlib.import_module("transformers")
        processor_cls = getattr(processors, processor_class)
        self.processor: "SmolVLMProcessor" = processor_cls.from_pretrained(model_id)
        self.max_length = max_length
        self.padding = padding
        if image_size_longest_edge is not None:
            self.processor.image_processor.size = {
                "longest_edge": int(image_size_longest_edge)
            }
        if max_image_size_longest_edge is not None:
            self.processor.image_processor.max_image_size = {
                "longest_edge": int(max_image_size_longest_edge)
            }

        tokenizer = self.processor.tokenizer
        tokenizer.padding_side = "left"
        self.prompt_sep_text = "<<<PROMPT_SEP>>>"
        self.prompt_sep_ids = tokenizer.encode(self.prompt_sep_text, add_special_tokens=False)
        self.assistant_header_ids = tokenizer.encode(
            "\nAssistant:", add_special_tokens=False
        )
        self.end_of_utterance_id = tokenizer.convert_tokens_to_ids(
            tokenizer.eos_token
        )

    def compute(self, sample: "DataSample") -> "DataSample":
        images = sample.observation.images

        user_content: List[Dict] = [{"type": "image", "image": img} for img in images]
        user_content.append({"type": "text", "text": sample.observation.task})
        user_content.append({"type": "text", "text": self.prompt_sep_text})
        user_content.append(
            {"type": "text", "text": sample.observation.generation_prompt}
        )

        gpt_content = [{"type": "text", "text": sample.observation.generation_answer}]
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": gpt_content},
        ]

        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            padding=self.padding,
            truncation=False,
            max_length=self.max_length,
        )

        assistant_mask = self.assistant_content_mask(
            encoded,
            assistant_header_ids=self.assistant_header_ids,
            end_of_utterance_id=self.end_of_utterance_id,
        )
        target_ids = encoded["input_ids"].clone()
        target_ids[~assistant_mask] = TARGET_IGNORE_ID
        # target_ids[~assistant_mask] = 0
        # print("Decoded target ids:", self.processor.tokenizer.decode(target_ids[0], skip_special_tokens=False))
        # breakpoint()

        obs_register_att_mask = self._build_obs_register_att_mask(encoded)

        policy_td = SmolVLMPolicyInput(
            input_ids=encoded["input_ids"].squeeze(0).long(),
            target_ids=target_ids.squeeze(0).long(),
            attention_mask=encoded["attention_mask"].squeeze(0).bool(),
            obs_register_att_mask=obs_register_att_mask.squeeze(0).bool(),
            pixel_values=encoded["pixel_values"],
        )
        sample.observation.policy_input = policy_td
        return sample

    def _build_obs_register_att_mask(self, encoded: dict) -> torch.BoolTensor:
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        bsz, seqlen = input_ids.shape
        mask = torch.zeros((bsz, seqlen), dtype=torch.bool, device=input_ids.device)
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
    def _find_subsequence(sequence: list[int], subsequence: list[int]) -> Optional[int]:
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.tolist()
        if not subsequence:
            return None
        max_start = len(sequence) - len(subsequence)
        for i in range(max_start + 1):
            if sequence[i : i + len(subsequence)] == subsequence:
                return i
        return None

    @staticmethod
    def assistant_content_mask(
        encoded: dict,
        assistant_header_ids: list[int],
        end_of_utterance_id: int,
    ) -> torch.BoolTensor:
        input_ids = encoded["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        bsz, seqlen = input_ids.shape
        mask = torch.zeros((bsz, seqlen), dtype=torch.bool, device=input_ids.device)
        for b in range(bsz):
            ids_list = input_ids[b].tolist()
            header_start = SmolVLMProcessor._find_subsequence(
                ids_list, assistant_header_ids
            )
            if header_start is None:
                continue
            for j in range(header_start + len(assistant_header_ids), seqlen):
                mask[b, j] = True
                if input_ids[b, j].item() == end_of_utterance_id:
                    break

        if "attention_mask" in encoded:
            attention_mask = encoded["attention_mask"]
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
            mask = mask & attention_mask.bool()

        return mask
