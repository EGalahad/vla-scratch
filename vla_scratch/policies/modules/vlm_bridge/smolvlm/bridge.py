from __future__ import annotations

import importlib
from typing import Dict, Optional, Tuple, TYPE_CHECKING
from copy import copy

import torch

from vla_scratch.policies.modules.vlm_bridge.smolvlm.utils import replace_context
from vla_scratch.policies.utils.training import (
    apply_checkpoint_when_training,
    fully_shard_layers,
)
from vla_scratch.policies.modules.vlm_bridge.base import (
    VLMBridge,
    VLMOutputs,
    TARGET_IGNORE_ID,
)
from vla_scratch.policies.modules.vlm_bridge.smolvlm.processor import SmolVLMPolicyInput
from vla_scratch.policies.utils.transformers import make_att_2d_masks

if TYPE_CHECKING:
    from transformers.models.smolvlm.modeling_smolvlm import (
        SmolVLMForConditionalGeneration,
    )
    from vla_scratch.transforms.data_types import Observation


class SmolVLMBridge(VLMBridge):
    def __init__(self, *, model_id: str, vlm_type: str):
        super().__init__()
        tfm = importlib.import_module("transformers")
        try:
            vlm_cls = getattr(tfm, vlm_type)
        except AttributeError as e:
            raise ImportError(f"transformers has no class named '{vlm_type}'.") from e

        self.causal_model: "SmolVLMForConditionalGeneration" = vlm_cls.from_pretrained(
            model_id,
            attn_implementation="sdpa",
            trust_remote_code=True,
            device_map=torch.cuda.current_device(),
        )

        SmolVLMProcessor = getattr(tfm, "SmolVLMProcessor")
        self.processor = SmolVLMProcessor.from_pretrained(model_id)
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

    def apply_fsdp(self, mp_policy, mesh):
        fully_shard_layers(
            self.causal_model.model.vision_model.encoder.layers, mesh, mp_policy
        )
        fully_shard_layers(self.causal_model.model.text_model.layers, mesh, mp_policy)

    def get_text_dims(self) -> Tuple[int, int, int]:
        cfg = self.causal_model.config.text_config
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        return cfg.num_hidden_layers, head_dim, cfg.num_key_value_heads, cfg.hidden_size

    @replace_context()
    def encode(
        self,
        observation: "Observation",
        *,
        extra_embs: Optional[torch.Tensor] = None,
        extra_pad_masks: Optional[torch.Tensor] = None,
        extra_att_masks: Optional[torch.Tensor] = None,
        zero_pos_id_for_extra: bool = False,
        extra_attention_mask: bool = False,
    ) -> Tuple[torch.Tensor, VLMOutputs, Dict]:
        policy_td: "SmolVLMPolicyInput" = observation.policy_input
        if not isinstance(policy_td, SmolVLMPolicyInput):
            raise TypeError("Observation policy_input must be SmolVLMPolicyInput")

        input_ids = policy_td.input_ids
        input_pad_masks = policy_td.attention_mask
        target_ids = policy_td.target_ids
        pixel_values = policy_td.pixel_values
        if pixel_values.ndim == 6:
            bsz, num_images, num_frames, channels, height, width = pixel_values.shape
            pixel_values = pixel_values.reshape(
                bsz, num_images * num_frames, channels, height, width
            )
        bsz = input_ids.shape[0]

        torch.cuda.nvtx.range_push("embed_text")
        text_model = self.causal_model.model.text_model
        inputs_embeds = text_model.get_input_embeddings()(input_ids)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("embed_image")
        image_hidden_states = self.causal_model.model.get_image_features(pixel_values)
        image_hidden_states = image_hidden_states.to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        torch.cuda.nvtx.range_push("merge_inputs")
        merged_embeds = self.causal_model.model.inputs_merger(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_hidden_states=image_hidden_states,
        )
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()

        embs = [merged_embeds]
        pad_masks = [input_pad_masks]
        att_masks = [
            torch.ones(
                merged_embeds.shape[1],
                dtype=torch.bool,
                device=merged_embeds.device,
            ),
        ]

        extra_len = 0
        if extra_embs is not None:
            embs.append(extra_embs)
            pad_masks.append(extra_pad_masks)
            att_masks.append(extra_att_masks)
            extra_len = extra_embs.shape[1]

        embs = torch.cat(embs, dim=1)
        prefix_pad_masks = torch.cat(pad_masks, dim=1)
        prefix_att_masks_1d = torch.cat(att_masks, dim=0).expand(bsz, -1)

        # position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # position_ids.masked_fill_(~prefix_pad_masks.bool(), 0)
        position_ids = torch.arange(embs.shape[1], device=embs.device)
        position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
        if extra_len > 0 and zero_pos_id_for_extra:
            position_ids[:, -extra_len:] = 0

        attention_mask = prefix_pad_masks
        if extra_embs is not None or extra_attention_mask:
            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks_1d)
            if extra_embs is not None and extra_attention_mask:
                obs_reg_att_mask = policy_td.obs_register_att_mask
                prefix_len = input_pad_masks.shape[1]
                obs_reg_att_mask = obs_reg_att_mask[:, None, :].expand(
                    bsz, extra_len, prefix_len
                )
                prefix_att_2d[:, -extra_len:, :prefix_len] = obs_reg_att_mask

            attn_mask = torch.zeros(
                prefix_att_2d.shape, device=embs.device, dtype=embs.dtype
            )
            attn_mask.masked_fill_(~prefix_att_2d, torch.finfo(attn_mask.dtype).min)
            attention_mask = attn_mask[:, None, :, :]

        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import create_causal_mask

        cache_position = torch.arange(embs.shape[1], device=embs.device)
        past_key_values = DynamicCache()
        causal_mask = create_causal_mask(
            config=text_model.config,
            input_embeds=embs,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        position_embeddings = text_model.rotary_emb(embs, position_ids)

        hidden_states = embs
        kv_cache_list = []
        encoder_hidden_states_list = []
        for layer_idx, decoder_layer in enumerate(text_model.layers):
            torch.cuda.nvtx.range_push(f"layer_{layer_idx}")
            past_key_values_this_layer = copy(past_key_values)
            hidden_states = apply_checkpoint_when_training(
                self,
                decoder_layer,
                hidden_states,
                attention_mask=causal_mask,
                past_key_values=past_key_values_this_layer,
                position_embeddings=position_embeddings,
            )
            torch.cuda.nvtx.range_pop()
            layer_cache = past_key_values_this_layer.layers.pop(-1)
            kv_cache_list.append((layer_cache.keys, layer_cache.values))
            encoder_hidden_states_list.append(hidden_states)

        hidden_states = text_model.norm(hidden_states)
        hidden_state_list = torch.stack(encoder_hidden_states_list, dim=1)

        pred_logits = self.causal_model.lm_head(
            hidden_states[:, : hidden_states.shape[1] - extra_len]
        )
        pred_logits = pred_logits[:, :-1]
        pred_logits = pred_logits.reshape(-1, pred_logits.shape[-1])
        target_ids = target_ids[:, 1:].reshape(-1)
        ce_loss_sum = torch.nn.functional.cross_entropy(
            pred_logits, target_ids, ignore_index=TARGET_IGNORE_ID, reduction="sum"
        )
        num_correct_tokens = (pred_logits.argmax(dim=-1) == target_ids).sum()
        total = (target_ids != TARGET_IGNORE_ID).sum().clamp(min=1)
        ce_loss = ce_loss_sum / total
        accuracy = num_correct_tokens.float() / total

        key_states = torch.stack([k for k, _v in kv_cache_list], dim=1)
        value_states = torch.stack([v for _k, v in kv_cache_list], dim=1)

        vlm_outputs = VLMOutputs(
            last_hidden_state=hidden_states,
            prefix_pad_masks=prefix_pad_masks,
            key_states=key_states,
            value_states=value_states,
            hidden_state_list=hidden_state_list,
            batch_size=[bsz],
        )
        padding_ratio = policy_td.attention_mask.float().mean(dim=-1)
        log_dict = {
            "padding_ratio/mean": padding_ratio.mean(),
            "padding_ratio/std": padding_ratio.std(),
            "padding_ratio/min": padding_ratio.min(),
            "padding_ratio/max": padding_ratio.max(),
            "loss/ce_loss": ce_loss.detach(),
            "loss/accuracy": accuracy.detach(),
        }
        return ce_loss, vlm_outputs, log_dict
