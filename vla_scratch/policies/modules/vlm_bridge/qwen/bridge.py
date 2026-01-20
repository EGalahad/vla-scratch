from __future__ import annotations

import importlib
from copy import copy
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
from vla_scratch.policies.modules.vlm_bridge.qwen.processor import (
    QwenPolicyInput,
)
from vla_scratch.policies.modules.vlm_bridge.qwen.utils import (
    is_qwen3vl_forward_replaced,
    replace_qwen3vl_forward,
)
from vla_scratch.policies.utils.transformers import make_att_2d_masks

if TYPE_CHECKING:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextModel,
        Qwen3VLVisionModel,
        Qwen3VLForConditionalGeneration,
    )
    from vla_scratch.transforms.data_types import Observation

# for performance ablation
QWEN3_VL_USE_GRID_THW_LIST = True
QWEN3_VL_RECOMPUTE_POS_IDS = False
QWEN3_VL_MASKED_ADD_STACK = True


class Qwen3VLBridge(VLMBridge):
    def __init__(self, *, model_id: str, vlm_type: str):
        super().__init__()
        tfm = importlib.import_module("transformers")
        try:
            vlm_cls = getattr(tfm, vlm_type)
        except AttributeError as e:
            raise ImportError(
                f"transformers has no class named '{vlm_type}'."
            ) from e

        self.causal_model: "Qwen3VLForConditionalGeneration" = (
            vlm_cls.from_pretrained(
                model_id,
                attn_implementation="sdpa",
                trust_remote_code=True,
                device_map=torch.cuda.current_device(),
            )
        )

        from transformers import Qwen3VLProcessor

        self.processor = Qwen3VLProcessor.from_pretrained(model_id)
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

        replace_qwen3vl_forward()
        visual: "Qwen3VLVisionModel" = self.causal_model.model.visual
        visual.prepared_freq_table = visual.rotary_pos_emb(128)

    def apply_fsdp(self, mp_policy, mesh):
        fully_shard_layers(
            self.causal_model.model.visual.blocks, mesh, mp_policy
        )
        fully_shard_layers(
            self.causal_model.model.language_model.layers, mesh, mp_policy
        )

    def get_text_dims(self) -> Tuple[int, int, int]:
        cfg = self.causal_model.config.text_config
        head_dim = cfg.head_dim
        num_kv_heads = cfg.num_key_value_heads
        hidden = cfg.hidden_size
        return cfg.num_hidden_layers, head_dim, num_kv_heads, hidden

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
        device = observation.device
        bsz = observation.shape[0]
        REPLACED = is_qwen3vl_forward_replaced()

        assert isinstance(observation.policy_input, QwenPolicyInput)
        policy_td = observation.policy_input

        torch.cuda.nvtx.range_push("embed_text_img")
        torch.cuda.nvtx.range_push("embed_text")
        lm: "Qwen3VLTextModel" = self.causal_model.language_model
        inputs_embeds: torch.Tensor = lm.embed_tokens(policy_td.input_ids)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("embed_images")
        pixel_values = einops.rearrange(
            policy_td.pixel_values, "b grid patch -> (b grid) patch"
        )
        if REPLACED and QWEN3_VL_USE_GRID_THW_LIST:
            grid_thw_list = policy_td.image_grid_thw_list
            grid_thw_list = sum(grid_thw_list, [])
            assert all(
                grid_thw == grid_thw_list[0] for grid_thw in grid_thw_list
            ), "All grid_thw must match for the optimized function."

            image_embeds, deepstack_image_embeds = (
                self.causal_model.model.visual(pixel_values, grid_thw_list)
            )
        else:
            grid_thw_tensor = policy_td.image_grid_thw.reshape(-1, 3)
            image_embeds, deepstack_image_embeds = (
                self.causal_model.model.visual(pixel_values, grid_thw_tensor)
            )
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("scatter_img")
        image_mask = (
            policy_td.input_ids == self.causal_model.model.config.image_token_id
        )
        inputs_embeds.masked_scatter_(image_mask.unsqueeze(-1), image_embeds)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()

        input_pad_mask = policy_td.attention_mask
        position_ids = einops.rearrange(
            policy_td.position_ids, "b plane 1 s -> plane b s"
        )
        if QWEN3_VL_RECOMPUTE_POS_IDS:
            position_ids, _ = self.causal_model.model.get_rope_index(
                input_ids=policy_td.input_ids,
                image_grid_thw=policy_td.image_grid_thw.flatten(0, 1),
                attention_mask=policy_td.attention_mask,
            )

        embs = [inputs_embeds]
        pad_masks = [input_pad_mask]
        att_masks = [
            torch.ones(inputs_embeds.shape[1], dtype=torch.bool, device=device)
        ]
        extra_len = 0
        if extra_embs is not None:
            torch.cuda.nvtx.range_push("extend_extra_tokens")
            embs.append(extra_embs)
            pad_masks.append(extra_pad_masks)
            att_masks.append(extra_att_masks)

            extra_len = extra_embs.shape[1]
            valid_lengths = input_pad_mask.sum(dim=1)
            gather_idx = (valid_lengths - 1).clamp(min=0)
            last_pos = position_ids[0][
                torch.arange(bsz, device=device), gather_idx
            ]

            increments = torch.arange(
                1, extra_len + 1, device=device, dtype=last_pos.dtype
            ).unsqueeze(0)
            extra_text_pos = last_pos.unsqueeze(1) + increments
            extra_pos_3d = extra_text_pos.unsqueeze(0).expand(3, -1, -1)
            if zero_pos_id_for_extra:
                extra_pos_3d.zero_()
            position_ids = torch.cat([position_ids, extra_pos_3d], dim=-1)

            extra_visual = torch.zeros(
                (bsz, extra_len), dtype=image_mask.dtype, device=device
            )
            image_mask = torch.cat([image_mask, extra_visual], dim=1)
            torch.cuda.nvtx.range_pop()

        embs = torch.cat(embs, dim=1)
        prefix_pad_masks = torch.cat(pad_masks, dim=1)
        prefix_att_masks_1d = torch.cat(att_masks, dim=0).expand(bsz, -1)

        torch.cuda.nvtx.range_push("build_attn_mask")
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks_1d)
        prefix_att_mask = einops.rearrange(prefix_att_2d, "b i j -> b 1 i j")
        if extra_embs is not None and extra_attention_mask:
            obs_reg_att_mask = policy_td.obs_register_att_mask
            # assert torch.eq(obs_reg_att_mask & input_pad_mask, obs_reg_att_mask).all(), "obs_register_att_mask must be a subset of input_pad_mask"
            prefix_len = input_pad_mask.shape[1]
            obs_reg_att_mask = einops.repeat(
                obs_reg_att_mask, "b s -> b 1 extra_len s", extra_len=extra_len
            )
            prefix_att_mask[:, :, -extra_len:, :prefix_len] = obs_reg_att_mask
        torch.cuda.nvtx.range_pop()

        position_emb = lm.rotary_emb.forward(embs, position_ids)
        hidden_states = embs

        kv_cache_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
        encoder_hidden_states_list: List[torch.Tensor] = []
        if not REPLACED:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                DynamicCache,
            )

            past_key_values = DynamicCache()

        for layer_idx, decoder_layer in enumerate(lm.layers):
            torch.cuda.nvtx.range_push(f"layer_{layer_idx}")
            past_key_values_this_layer = (
                copy(past_key_values) if not REPLACED else None
            )
            outputs = apply_checkpoint_when_training(
                self,
                decoder_layer,
                hidden_states,
                attention_mask=prefix_att_mask,
                position_embeddings=position_emb,
                past_key_values=past_key_values_this_layer,
            )
            if REPLACED:
                hidden_states, (k, v) = outputs
            else:
                hidden_states = outputs
                layer = past_key_values_this_layer.layers.pop(-1)
                k, v = layer.keys, layer.values
            torch.cuda.nvtx.range_pop()

            kv_cache_list.append((k, v))
            encoder_hidden_states_list.append(hidden_states)

            if deepstack_image_embeds is not None and layer_idx in range(
                len(deepstack_image_embeds)
            ):
                torch.cuda.nvtx.range_push("deepstack_inject")
                if QWEN3_VL_MASKED_ADD_STACK:
                    delta = torch.zeros_like(hidden_states)
                    delta.masked_scatter_(
                        image_mask.unsqueeze(-1),
                        deepstack_image_embeds[layer_idx],
                    )
                    hidden_states.add_(delta)
                else:
                    local_this = (
                        hidden_states[image_mask, :].clone()
                        + deepstack_image_embeds[layer_idx]
                    )
                    hidden_states[image_mask, :] = local_this
                torch.cuda.nvtx.range_pop()

        # compute ce loss
        hidden_states = lm.norm(hidden_states)
        pred_logits = self.causal_model.lm_head(
            hidden_states[:, : hidden_states.shape[1] - extra_len]
        )
        pred_logits = einops.rearrange(pred_logits[:, :-1], "b s v -> (b s) v")
        target_ids = einops.rearrange(
            policy_td.target_ids[:, 1:], "b s -> (b s)"
        )
        ce_loss_sum = torch.nn.functional.cross_entropy(
            pred_logits,
            target_ids,
            ignore_index=TARGET_IGNORE_ID,
            reduction="sum",
        )
        # compute corect tokens
        num_correct_tokens = (pred_logits.argmax(dim=-1) == target_ids).sum()

        total = (target_ids != TARGET_IGNORE_ID).sum().clamp(min=1)
        ce_loss = ce_loss_sum / total
        accuracy = num_correct_tokens.float() / total

        key_states = torch.stack([k for k, v in kv_cache_list], dim=1)
        value_states = torch.stack([v for k, v in kv_cache_list], dim=1)
        hidden_state_list = torch.stack(encoder_hidden_states_list, dim=1)

        # construct VLMOutputs
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
