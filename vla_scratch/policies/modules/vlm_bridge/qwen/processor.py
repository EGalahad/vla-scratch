import importlib
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
from tensordict import TensorClass

from vla_scratch.transforms.base import TransformFn

if TYPE_CHECKING:
    from vla_scratch.transforms.data_types import DataSample


class QwenPolicyInput(TensorClass):
    input_ids: torch.LongTensor
    attention_mask: torch.BoolTensor
    pixel_values: torch.FloatTensor
    image_grid_thw: torch.LongTensor
    image_grid_thw_list: List[Tuple[int, int, int]]
    position_ids: torch.LongTensor
    mrope_position_deltas: torch.LongTensor


class QwenProcessor(TransformFn):
    """Tokenize prompt using Qwen3-VL chat template and produce policy inputs."""

    def __init__(
        self,
        processor_class: str,
        model_id: str,
        max_length: int = 256,
        add_generation_prompt: bool = True,
        padding: str | bool = "max_length",
    ) -> None:
        processors = importlib.import_module("transformers")
        processor_cls = getattr(processors, processor_class)
        self.processor = processor_cls.from_pretrained(model_id)
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
        self.max_length = max_length
        self.add_generation_prompt = add_generation_prompt
        self.padding = padding

    def compute(self, sample: "DataSample") -> "DataSample":
        content: List[Dict] = [
            {"type": "image", "image": img} for img in sample.observation.images
        ]
        content.append({"type": "text", "text": sample.observation.task})
        messages = [{"role": "user", "content": content}]

        encoded = self.processor.apply_chat_template(
            [messages],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            text_kwargs={
                "max_length": self.max_length,
                "truncation": True,
                "padding": self.padding,
                "return_tensors": "pt",
            },
        )

        position_ids, mrope_position_deltas = self.get_rope_index(
            input_ids=encoded["input_ids"],
            image_grid_thw=encoded["image_grid_thw"],
            attention_mask=encoded["attention_mask"],
        )
        policy_td = QwenPolicyInput(
            input_ids=encoded["input_ids"].squeeze(0).long(),
            attention_mask=encoded["attention_mask"].squeeze(0).bool(),
            pixel_values=encoded["pixel_values"],
            image_grid_thw=encoded["image_grid_thw"],
            image_grid_thw_list=[
                tuple(thw) for thw in encoded["image_grid_thw"].tolist()
            ],
            position_ids=position_ids,
            mrope_position_deltas=mrope_position_deltas,
        )
        sample.observation.policy_input = policy_td

        return sample

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(
                video_grid_thw, video_grid_thw[:, 0], dim=0
            )
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.processor.image_processor.merge_size
        image_token_id = self.processor.image_token_id
        video_token_id = self.processor.video_token_id
        vision_start_token_id = self.processor.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)

            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = (
                    position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                )
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas
