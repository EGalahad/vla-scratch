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
from vla_scratch.policies.modules.vlm_bridge.base import VLMBridge, TARGET_IGNORE_ID
from vla_scratch.policies.modules.vlm_bridge.qwen.processor import QwenPolicyInput
from vla_scratch.policies.modules.vlm_bridge.qwen.utils import (
    is_qwen3vl_forward_replaced,
    replace_qwen3vl_forward,
)
from vla_scratch.policies.utils.transformers import make_att_2d_masks
from vla_scratch.policies.modules.vlm_bridge.data_types import VLMOutputs

if TYPE_CHECKING:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    from vla_scratch.transforms.data_types import Observation
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel


class Qwen3VLBridge(VLMBridge):
    def __init__(self, *, model_id: str, vlm_type: str):
        super().__init__()
        tfm = importlib.import_module("transformers")
        try:
            vlm_cls = getattr(tfm, vlm_type)
        except AttributeError as e:
            raise ImportError(f"transformers has no class named '{vlm_type}'.") from e

        self.causal_model = vlm_cls.from_pretrained(
            model_id,
            attn_implementation="sdpa",
            trust_remote_code=True,
            device_map=torch.cuda.current_device(),
        )

        from transformers import Qwen3VLProcessor

        self.processor = Qwen3VLProcessor.from_pretrained(model_id)
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

        replace_qwen3vl_forward()
        visual: "Qwen3VLVisionModel" = self.causal_model.model.visual
        visual.prepared_freq_table = visual.rotary_pos_emb(128)

    def apply_fsdp(self, mp_policy, mesh):
        fully_shard_layers(self.causal_model.model.visual.blocks, mesh, mp_policy)
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
    ) -> Tuple[VLMOutputs, Dict]:
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
        if REPLACED:
            grid_thw_list = policy_td.image_grid_thw_list
            grid_thw_list = sum(grid_thw_list, [])
            assert all(
                grid_thw == grid_thw_list[0] for grid_thw in grid_thw_list
            ), "All grid_thw must match for the optimized function."

            image_embeds, deepstack_image_embeds = self.causal_model.model.visual(
                pixel_values, grid_thw_list
            )
        else:
            image_grid_thw = policy_td.image_grid_thw.reshape(-1, 3)
            image_embeds, deepstack_image_embeds = self.causal_model.model.visual(
                pixel_values, image_grid_thw
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
            last_pos = position_ids[0][torch.arange(bsz, device=device), gather_idx]

            increments = torch.arange(
                1, extra_len + 1, device=device, dtype=last_pos.dtype
            ).unsqueeze(0)
            extra_text_pos = last_pos.unsqueeze(1) + increments
            extra_pos_3d = extra_text_pos.unsqueeze(0).expand(3, -1, -1)
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
        torch.cuda.nvtx.range_pop()

        position_embeddings = lm.rotary_emb(embs, position_ids)
        hidden_states = embs

        kv_cache_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
        encoder_hidden_states_list: List[torch.Tensor] = []
        if not REPLACED:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import DynamicCache

            past_key_values = DynamicCache()

        if deepstack_image_embeds is not None:
            deepstack_image_embeds_deltas = []
            for layer_emb in deepstack_image_embeds:
                delta = torch.zeros_like(embs)
                delta.masked_scatter_(image_mask.unsqueeze(-1), layer_emb)
                deepstack_image_embeds_deltas.append(delta)

        for layer_idx, decoder_layer in enumerate(lm.layers):
            torch.cuda.nvtx.range_push(f"layer_{layer_idx}")
            past_key_values_this_layer = copy(past_key_values) if not REPLACED else None
            outputs = apply_checkpoint_when_training(
                self,
                decoder_layer,
                hidden_states,
                attention_mask=prefix_att_mask,
                position_embeddings=position_embeddings,
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
                hidden_states.add_(deepstack_image_embeds_deltas[layer_idx])
                torch.cuda.nvtx.range_pop()

        # compute ce loss
        hidden_states = lm.norm(hidden_states)
        predicted_probs = self.causal_model.lm_head(hidden_states[:, :-extra_len])
        predicted_probs = einops.rearrange(predicted_probs[:, :-1], "b s v -> (b s) v")
        target_ids = einops.rearrange(policy_td.target_ids[:, 1:], "b s -> (b s)")
        ce_loss = torch.nn.functional.cross_entropy(
            predicted_probs, target_ids, ignore_index=TARGET_IGNORE_ID, reduction="sum"
        )
        ce_loss = ce_loss / (target_ids != TARGET_IGNORE_ID).sum().clamp(min=1)

        # construct VLMOutputs
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
            "loss/ce_loss": ce_loss.detach(),
        }
        return ce_loss, vlm_outputs, log_dict
