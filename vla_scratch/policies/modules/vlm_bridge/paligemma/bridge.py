from __future__ import annotations

import importlib
from typing import List, Optional, Tuple, TYPE_CHECKING, Dict

import einops
import torch

from vla_scratch.policies.utils.training import (
    apply_checkpoint_when_training,
    fully_shard_layers,
)
from vla_scratch.policies.modules.vlm_bridge.base import VLMBridge
from vla_scratch.policies.modules.vlm_bridge.paligemma.processor import (
    PaligemmaPolicyInput,
)
from vla_scratch.policies.modules.vlm_bridge.paligemma.utils import (
    replace_paligemma_forward,
)
from vla_scratch.policies.utils.transformers import make_att_2d_masks
from vla_scratch.policies.modules.vlm_bridge.data_types import VLMOutputs

if TYPE_CHECKING:
    from transformers.models.paligemma.modeling_paligemma import (
        PaliGemmaForConditionalGeneration,
    )
    from transformers.models.gemma.modeling_gemma import GemmaModel
    from vla_scratch.transforms.data_types import Observation


class PaligemmaBridge(VLMBridge):
    def __init__(self, *, model_id: str, vlm_type: str, max_length: int = 64):
        super().__init__()
        self.max_length = max_length

        tfm = importlib.import_module("transformers")
        try:
            vlm_cls = getattr(tfm, vlm_type)
        except AttributeError as e:
            raise ImportError(f"transformers has no class named '{vlm_type}'.") from e

        self.causal_model: "PaliGemmaForConditionalGeneration" = (
            vlm_cls.from_pretrained(
                model_id,
                attn_implementation="sdpa",
                trust_remote_code=True,
                device_map=torch.cuda.current_device(),
            )
        )

        PaliGemmaProcessor = getattr(tfm, "PaliGemmaProcessor")
        self.processor = PaliGemmaProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer

        replace_paligemma_forward()

    def apply_fsdp(self, mp_policy, mesh):
        fully_shard_layers(
            self.causal_model.vision_tower.vision_model.encoder.layers, mesh, mp_policy
        )
        fully_shard_layers(self.causal_model.language_model.layers, mesh, mp_policy)

    def get_text_dims(self) -> Tuple[int, int, int]:
        cfg = self.causal_model.config.text_config
        return (
            cfg.num_hidden_layers,
            cfg.head_dim,
            cfg.num_key_value_heads,
            cfg.hidden_size,
        )

    def encode(
        self,
        observation: "Observation",
        *,
        extra_embs: Optional[torch.Tensor] = None,
        extra_pad_masks: Optional[torch.Tensor] = None,
        extra_att_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[VLMOutputs, Dict]:
        policy_td = observation.policy_input
        if not isinstance(policy_td, PaligemmaPolicyInput):
            raise TypeError("Observation policy_input must be PaligemmaPolicyInput")
        pixel_values = policy_td.pixel_values
        image_masks = observation.image_masks
        input_ids = policy_td.input_ids
        input_pad_masks = policy_td.attention_mask

        device = observation.device
        lm: "GemmaModel" = self.causal_model.model.language_model

        torch.cuda.nvtx.range_push("embed_text_img")
        b, n_cam = pixel_values.shape[0], pixel_values.shape[1]
        images_flat = einops.rearrange(pixel_values, "b n c h w -> (b n) c h w")
        img_emb_flat = apply_checkpoint_when_training(
            self, self.causal_model.model.get_image_features, images_flat
        )
        img_emb = einops.rearrange(img_emb_flat, "(b n) t d -> b (n t) d", b=b, n=n_cam)
        img_mask_repeat = einops.repeat(
            image_masks, "b n 1 -> b (n t)", t=img_emb_flat.shape[1]
        )

        lang_emb = apply_checkpoint_when_training(self, lm.embed_tokens, input_ids)
        torch.cuda.nvtx.range_pop()

        embs = [img_emb, lang_emb]
        pad_masks = [img_mask_repeat, input_pad_masks]
        att_masks = [
            torch.zeros(img_emb.shape[1], dtype=torch.bool, device=device),
            torch.zeros(lang_emb.shape[1], dtype=torch.bool, device=device),
        ]

        if extra_embs is not None:
            embs.append(extra_embs)
            pad_masks.append(extra_pad_masks)
            att_masks.append(extra_att_masks)

        embs = torch.cat(embs, dim=1)
        prefix_pad_masks = torch.cat(pad_masks, dim=1)
        prefix_att_masks_1d = torch.cat(att_masks, dim=0).expand(b, -1)

        torch.cuda.nvtx.range_push("build_attn_mask")
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks_1d)
        prefix_att_mask = einops.rearrange(prefix_att_2d, "b i j -> b 1 i j")
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("pos_emb")
        position_ids = torch.cumsum(prefix_pad_masks, dim=1)
        pos_emb = lm.rotary_emb.forward(embs, position_ids)
        torch.cuda.nvtx.range_pop()

        hidden_states = embs * (embs.shape[-1] ** 0.5)
        kv_cache_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
        encoder_hidden_states_list: List[torch.Tensor] = []
        for i, layer in enumerate(lm.layers):
            torch.cuda.nvtx.range_push(f"layer_{i}")
            hidden_states, (k, v) = apply_checkpoint_when_training(
                self,
                layer,
                hidden_states,
                prefix_att_mask,
                pos_emb,
            )
            torch.cuda.nvtx.range_pop()

            kv_cache_list.append((k, v))
            encoder_hidden_states_list.append(hidden_states)

        hidden_states = lm.norm(hidden_states)

        vlm_outputs = VLMOutputs(
            last_hidden_state=hidden_states,
            prefix_pad_masks=prefix_pad_masks,
            hidden_state_list=tuple(encoder_hidden_states_list),
            kv_cache_list=tuple(kv_cache_list),
        )
        # mean along seq dim
        padding_ratio = policy_td.attention_mask.float().mean(dim=-1)
        log_dict = {
            "padding_ratio/mean": padding_ratio.mean(),
            "padding_ratio/std": padding_ratio.std(),
            "padding_ratio/min": padding_ratio.min(),
            "padding_ratio/max": padding_ratio.max(),
        }
        return vlm_outputs, log_dict
