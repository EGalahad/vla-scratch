from __future__ import annotations

from types import MethodType
from typing import List, Optional, Tuple, TYPE_CHECKING, Dict

import einops
import torch
from torch.distributed.fsdp._fully_shard import register_fsdp_forward_method

from vla_scratch.policies.utils.training import (
    apply_checkpoint_when_training,
    fully_shard_layers,
)
from vla_scratch.policies.modules.vlm_bridge.base import (
    VLMBridge,
    VLMOutputs,
)
from vla_scratch.policies.modules.vlm_bridge.pi05.processor import (
    Pi05PolicyInput,
)
from vla_scratch.policies.modules.vlm_bridge.pi05.utils import (
    _siglip_layer_forward,
    gemma_decoder_layer_forward,
    load_pi05_weights,
    pi05_image_features,
)
from vla_scratch.policies.utils.transformers import make_att_2d_masks

if TYPE_CHECKING:
    from transformers.models.gemma.modeling_gemma import GemmaModel
    from vla_scratch.transforms.data_types import Observation


class Pi05Bridge(VLMBridge):
    def __init__(self, *, compile_layers: bool = True):
        super().__init__()
        from transformers import (
            PaliGemmaConfig,
            PaliGemmaForConditionalGeneration,
        )

        config = PaliGemmaConfig()
        config._vocab_size = 257152
        config.image_token_index = 257152
        config.text_config.vocab_size = 257152
        config.text_config.hidden_activation = "gelu_pytorch_tanh"
        config.vision_config.intermediate_size = 4304
        config.vision_config.projection_dim = 2048
        config.vision_config.projector_hidden_act = "gelu_fast"
        config._attn_implementation = "sdpa"
        self.causal_model = PaliGemmaForConditionalGeneration(config)
        self.weight_metadata = load_pi05_weights(self.causal_model)

        for layer in self.causal_model.model.language_model.layers:
            layer.bridge_forward = MethodType(  # type: ignore[attr-defined]
                gemma_decoder_layer_forward, layer
            )
            if compile_layers:
                layer.bridge_forward = torch.compile(  # type: ignore[attr-defined]
                    layer.bridge_forward, dynamic=False
                )
        vision_layers = (
            self.causal_model.vision_tower.vision_model.encoder.layers
        )
        for layer in vision_layers:
            layer.bridge_forward = MethodType(  # type: ignore[attr-defined]
                _siglip_layer_forward, layer
            )

    def apply_fsdp(self, mp_policy, mesh):
        layer_groups = (
            self.causal_model.vision_tower.vision_model.encoder.layers,
            self.causal_model.language_model.layers,
        )
        for layers in layer_groups:
            fully_shard_layers(layers, mesh, mp_policy)
            for layer in layers:
                register_fsdp_forward_method(layer, "bridge_forward")

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
        policy_td: "Pi05PolicyInput" = observation.policy_input
        if not isinstance(policy_td, Pi05PolicyInput):
            raise TypeError("Observation policy_input must be Pi05PolicyInput")
        pixel_values = policy_td.pixel_values
        input_ids = policy_td.input_ids
        input_pad_masks = policy_td.attention_mask
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
        image_features = pi05_image_features(
            self,
            self.causal_model,
            images_flat,
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
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        if extra_len > 0 and zero_pos_id_for_extra:
            position_ids[:, -extra_len:] = 0
        position_emb = lm.rotary_emb.forward(embs, position_ids)
        hidden_states = embs * (embs.shape[-1] ** 0.5)
        torch.cuda.nvtx.range_pop()

        kv_cache_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
        encoder_hidden_states_list: List[torch.Tensor] = []
        for layer_idx, decoder_layer in enumerate(lm.layers):
            torch.cuda.nvtx.range_push(f"layer_{layer_idx}")
            hidden_states, (k, v) = apply_checkpoint_when_training(
                self,
                decoder_layer.bridge_forward,
                hidden_states,
                prefix_att_mask,
                position_emb,
            )
            torch.cuda.nvtx.range_pop()

            kv_cache_list.append((k, v))
            encoder_hidden_states_list.append(hidden_states)

        hidden_states = lm.norm(hidden_states)

        ce_loss = hidden_states.new_zeros(())
        accuracy = hidden_states.new_zeros(())

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
