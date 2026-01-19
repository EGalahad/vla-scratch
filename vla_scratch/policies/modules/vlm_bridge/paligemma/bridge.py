from __future__ import annotations

import importlib
from typing import List, Optional, Tuple, TYPE_CHECKING, Dict

import einops
import torch

from vla_scratch.policies.utils.training import (
    apply_checkpoint_when_training,
    fully_shard_layers,
)
from vla_scratch.policies.modules.vlm_bridge.base import (
    VLMBridge,
    VLMOutputs,
    TARGET_IGNORE_ID,
)
from vla_scratch.policies.modules.vlm_bridge.paligemma.processor import (
    PaligemmaPolicyInput,
)
from vla_scratch.policies.modules.vlm_bridge.paligemma.utils import (
    replace_paligemma_forward,
)
from vla_scratch.policies.utils.transformers import make_att_2d_masks

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
            raise ImportError(
                f"transformers has no class named '{vlm_type}'."
            ) from e

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
            self.causal_model.vision_tower.vision_model.encoder.layers,
            mesh,
            mp_policy,
        )
        fully_shard_layers(
            self.causal_model.language_model.layers, mesh, mp_policy
        )

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
        zero_pos_id_for_extra: bool = False,
        extra_attention_mask: bool = False,
    ) -> Tuple[torch.Tensor, VLMOutputs, Dict]:
        policy_td: "PaligemmaPolicyInput" = observation.policy_input
        if not isinstance(policy_td, PaligemmaPolicyInput):
            raise TypeError(
                "Observation policy_input must be PaligemmaPolicyInput"
            )
        pixel_values = policy_td.pixel_values
        input_ids = policy_td.input_ids
        input_pad_masks = policy_td.attention_mask
        target_ids = policy_td.target_ids

        lm: "GemmaModel" = self.causal_model.model.language_model

        torch.cuda.nvtx.range_push("embed_text_img")
        image_token_id = self.causal_model.config.image_token_id
        if image_token_id >= self.causal_model.model.vocab_size:
            special_image_mask = input_ids == image_token_id
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        inputs_embeds = apply_checkpoint_when_training(
            self, lm.embed_tokens, llm_input_ids
        )

        bsz = pixel_values.shape[0]
        images_flat = einops.rearrange(pixel_values, "b n c h w -> (b n) c h w")
        image_features = apply_checkpoint_when_training(
            self, self.causal_model.model.get_image_features, images_flat
        )
        image_features = image_features.to(
            inputs_embeds.device, inputs_embeds.dtype
        )

        image_token_mask = input_ids == self.causal_model.config.image_token_id
        special_image_mask = image_token_mask.unsqueeze(-1).expand_as(
            inputs_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(
            special_image_mask, image_features
        )
        torch.cuda.nvtx.range_pop()

        embs = [inputs_embeds]
        pad_masks = [input_pad_masks]
        att_masks = [
            torch.ones(
                inputs_embeds.shape[1],
                dtype=torch.bool,
                device=inputs_embeds.device,
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

        torch.cuda.nvtx.range_push("build_attn_mask")
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks_1d)
        prefix_att_mask = einops.rearrange(prefix_att_2d, "b i j -> b 1 i j")
        if extra_embs is not None and extra_attention_mask:
            obs_reg_att_mask = policy_td.obs_register_att_mask
            prefix_len = input_pad_masks.shape[1]
            obs_reg_att_mask = einops.repeat(
                obs_reg_att_mask, "b s -> b 1 extra_len s", extra_len=extra_len
            )
            prefix_att_mask[:, :, -extra_len:, :prefix_len] = obs_reg_att_mask
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("pos_emb")
        position_ids = torch.cumsum(prefix_pad_masks, dim=1)
        if extra_len > 0 and zero_pos_id_for_extra:
            position_ids[:, -extra_len:] = 0
        position_emb = lm.rotary_emb.forward(embs, position_ids)
        hidden_states = embs * (embs.shape[-1] ** 0.5)
        torch.cuda.nvtx.range_pop()

        kv_cache_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
        encoder_hidden_states_list: List[torch.Tensor] = []
        for layer_idx, decoder_layer in enumerate(lm.layers):
            torch.cuda.nvtx.range_push(f"layer_{layer_idx}")
            outputs = apply_checkpoint_when_training(
                self,
                decoder_layer,
                hidden_states,
                prefix_att_mask,
                position_emb,
            )
            hidden_states, (k, v) = outputs
            torch.cuda.nvtx.range_pop()

            kv_cache_list.append((k, v))
            encoder_hidden_states_list.append(hidden_states)

        hidden_states = lm.norm(hidden_states)

        pred_logits = self.causal_model.lm_head(
            hidden_states[:, : hidden_states.shape[1] - extra_len]
        )
        pred_logits = einops.rearrange(pred_logits[:, :-1], "b s v -> (b s) v")
        target_ids = einops.rearrange(target_ids[:, 1:], "b s -> (b s)")
        ce_loss_sum = torch.nn.functional.cross_entropy(
            pred_logits,
            target_ids,
            ignore_index=TARGET_IGNORE_ID,
            reduction="sum",
        )
        num_correct_tokens = (pred_logits.argmax(dim=-1) == target_ids).sum()
        total = (target_ids != TARGET_IGNORE_ID).sum().clamp(min=1)
        ce_loss = ce_loss_sum / total
        accuracy = num_correct_tokens.float() / total

        key_states = torch.stack([k for k, v in kv_cache_list], dim=1)
        value_states = torch.stack([v for k, v in kv_cache_list], dim=1)
        hidden_state_list = torch.stack(encoder_hidden_states_list, dim=1)

        vlm_outputs = VLMOutputs(
            last_hidden_state=hidden_states,
            prefix_pad_masks=prefix_pad_masks,
            key_states=key_states,
            value_states=value_states,
            hidden_state_list=hidden_state_list,
            batch_size=[bsz],
        )
        # mean along seq dim
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
