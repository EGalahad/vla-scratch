import math
import jaxtyping as at
import einops
import time
from typing import List, Tuple, Literal


import torch
import torch.nn as nn
import torch.nn.functional as F


from vla_scratch.datasets.data_types import Observation
from vla_scratch.policies.data_types import *
from vla_scratch.policies.modules.dit import DiTModel, get_config
from vla_scratch.policies.pi.utils import (
    make_att_2d_masks,
    attention_fill_false_to_inf,
    create_sinusoidal_pos_embedding,
)
from vla_scratch.policies.utils import sample_noise
from vla_scratch.policies.pi.config import PiConfig


class PiPolicy(nn.Module):
    def __init__(self, config: PiConfig):
        super().__init__()
        self.config = config

        if config.action_dim is None or config.state_dim is None:
            raise ValueError(
                "PiConfig.action_dim and PiConfig.state_dim must be set before "
                "initializing PiPolicy."
            )

        start_time = time.time()
        from transformers import PaliGemmaForConditionalGeneration

        self.paligemma = PaliGemmaForConditionalGeneration.from_pretrained(
            config.model_id,
            trust_remote_code=True,
            device_map=torch.cuda.current_device(),
        )
        end_time = time.time()
        print(f"PaliGemma model initialized in {end_time - start_time:.2f} seconds.")

        # number of hidden layers and head dim must match to do cross-attention at each layer
        vlm_hf_config = self.paligemma.config
        action_expert_config = get_config(config.action_expert_variant)
        assert (
            vlm_hf_config.text_config.num_hidden_layers
            == action_expert_config.num_hidden_layers
        ), f"VLM and action expert must have the same number of layers, got {vlm_hf_config.text_config.num_hidden_layers} and {action_expert_config.num_hidden_layers}"
        assert (
            vlm_hf_config.text_config.head_dim == action_expert_config.head_dim
        ), f"VLM and action expert must have the same head dim, got {vlm_hf_config.text_config.head_dim} and {action_expert_config.head_dim}"

        start_time = end_time
        self.gemma_expert = DiTModel(config=action_expert_config)
        end_time = time.time()
        print(f"Gemma expert model initialized in {end_time - start_time:.2f} seconds.")

        action_width = action_expert_config.hidden_size
        self.action_in_proj = nn.Linear(config.action_dim, action_width)
        self.action_out_proj = nn.Linear(action_width, config.action_dim)
        self.state_in_proj = nn.Linear(config.state_dim, action_width)

        self.time_mlp = nn.Sequential(
            nn.Linear(action_width, action_width),
            nn.SiLU(),
            nn.Linear(action_width, action_width),
            nn.SiLU(),
        )

        # register buffers
        if config.use_state:
            suffix_len = config.action_horizon + config.state_history
        else:
            suffix_len = config.action_horizon
        suffix_pad_mask = torch.ones(suffix_len, dtype=torch.bool)
        suffix_att_mask = torch.zeros(suffix_len, dtype=torch.bool)
        suffix_att_mask[0] = 1
        # create a new attention block for the suffix, prefix should not attend to suffix

        self.register_buffer("suffix_pad_mask", suffix_pad_mask, persistent=False)
        self.register_buffer("suffix_att_mask", suffix_att_mask, persistent=False)
        self.suffix_pad_mask: at.Bool[torch.Tensor, "action_horizon"]
        self.suffix_att_mask: at.Bool[torch.Tensor, "action_horizon"]

    def embed_prefix(
        self,
        images: at.Float[torch.Tensor, "b n_cam c h w"],
        img_masks: at.Bool[torch.Tensor, "b n_cam 1"],
        lang_tokens: at.Int64[torch.Tensor, "b l"],
        lang_masks: at.Bool[torch.Tensor, "b l"],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        torch.cuda.nvtx.range_push("image_embedding")
        images_flat = einops.rearrange(images, "b n_cam c h w -> (b n_cam) c h w")
        img_emb_flat: at.Float[torch.Tensor, "(b n_cam) n_emb img_emb_dim"] = (
            self._apply_checkpoint(
                self.paligemma.model.get_image_features,
                images_flat,
            )
        )
        img_emb = einops.rearrange(
            img_emb_flat,
            " (b n_cam) n_emb img_emb_dim -> b (n_cam n_emb) img_emb_dim",
            b=images.shape[0],
            n_cam=images.shape[1],
        )
        img_mask_repeat = einops.repeat(
            img_masks, "b n_cam 1 -> b (n_cam n_emb)", n_emb=img_emb_flat.shape[1]
        )
        torch.cuda.nvtx.range_pop()

        embs.append(img_emb)
        pad_masks.append(img_mask_repeat)
        att_masks.append(
            torch.zeros(img_emb.shape[1], dtype=torch.bool, device=img_emb.device)
        )

        torch.cuda.nvtx.range_push("language_embedding")
        lang_emb = self._apply_checkpoint(
            self.paligemma.language_model.embed_tokens, lang_tokens
        )
        torch.cuda.nvtx.range_pop()

        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks.append(
            torch.zeros(lang_emb.shape[1], dtype=torch.bool, device=lang_emb.device)
        )

        torch.cuda.nvtx.range_push("concat_embeddings")
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.cat(att_masks, dim=0).expand(images.shape[0], -1)
        torch.cuda.nvtx.range_pop()
        return embs, pad_masks, att_masks

    def embed_suffix(
        self,
        state: at.Float[torch.Tensor, "*batch_size state_history state_dim"],
        noisy_actions: at.Float[torch.Tensor, "*batch_size action_horizon action_dim"],
        time: at.Float[torch.Tensor, "*batch_size"],
    ) -> Tuple[
        at.Float[torch.Tensor, "*batch_size action_horizon hidden_dim"],
        at.Bool[torch.Tensor, "*batch_size action_horizon"],
        at.Bool[torch.Tensor, "*batch_size action_horizon"],
        at.Float[torch.Tensor, "*batch_size hidden_dim"],
    ]:
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            time,
            dimension=self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=time.device,
            dtype=time.dtype,
        )
        time_emb = self._apply_checkpoint(self.time_mlp.forward, time_emb)

        action_emb = self._apply_checkpoint(self.action_in_proj.forward, noisy_actions)
        if self.config.use_state:
            state_emb = self._apply_checkpoint(self.state_in_proj.forward, state)
            suffix_emb = torch.cat([state_emb, action_emb], dim=1)
        else:
            suffix_emb = action_emb

        bsize = action_emb.shape[0]
        pad_mask = einops.repeat(
            self.suffix_pad_mask, "action_horizon -> b action_horizon", b=bsize
        )
        att_mask = einops.repeat(
            self.suffix_att_mask, "action_horizon -> b action_horizon", b=bsize
        )

        return suffix_emb, pad_mask, att_mask, time_emb

    def encode_prefix(
        self, observation: Observation
    ) -> Tuple[HiddenState, PrefixPadMask, List[KVCache]]:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        torch.cuda.nvtx.range_push("embed_prefix")
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            observation.images,
            observation.image_masks,
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
        )
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("attention_mask")
        prefix_att_2d_mask = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_mask = attention_fill_false_to_inf(prefix_att_2d_mask)[:, None, :, :]
        # shape: [batch_size, 1, prefix_len, prefix_len]

        hidden_size = prefix_embs.shape[-1]
        hidden_states = prefix_embs * math.sqrt(hidden_size)
        torch.cuda.nvtx.range_pop()

        from transformers.models.gemma.modeling_gemma import (
            GemmaModel,
            GemmaDecoderLayer,
            apply_rotary_pos_emb,
        )

        model: GemmaModel = self.paligemma.language_model

        torch.cuda.nvtx.range_push("rotary_embedding")
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        position_embeddings = model.rotary_emb.forward(prefix_embs, prefix_position_ids)
        torch.cuda.nvtx.range_pop()

        def compute_layer(layer: GemmaDecoderLayer, hidden_states):
            torch.cuda.nvtx.range_push(f"input_layernorm")
            pre_att = layer.input_layernorm(hidden_states)
            torch.cuda.nvtx.range_pop()

            input_shape = hidden_states.shape[:-1]  # [batch_size, seq_len]
            head_shape = (*input_shape, -1, layer.self_attn.head_dim)

            # attention
            torch.cuda.nvtx.range_push(f"project_qkv")
            q = layer.self_attn.q_proj(pre_att).view(head_shape)  # .transpose(1, 2)
            k = layer.self_attn.k_proj(pre_att).view(head_shape)  # .transpose(1, 2)
            v = layer.self_attn.v_proj(pre_att).view(head_shape)  # .transpose(1, 2)
            q = einops.rearrange(q, "b seq head dim -> b head seq dim")
            k = einops.rearrange(k, "b seq head dim -> b head seq dim")
            v = einops.rearrange(v, "b seq head dim -> b head seq dim")
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push(f"rotary_embedding")
            cos, sin = position_embeddings
            q_rotate, k_rotate = apply_rotary_pos_emb(q, k, cos, sin)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push(f"attention")
            out_att = F.scaled_dot_product_attention(
                q_rotate,
                k_rotate,
                v,
                attn_mask=prefix_att_mask,
                scale=layer.self_attn.scaling,
            )
            out_att = einops.rearrange(
                out_att, "b head seq dim -> b seq (head dim)"
            ).contiguous()
            out_att = layer.self_attn.o_proj(out_att)
            res_att = hidden_states + out_att
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push(f"mlp")
            pre_mlp = layer.post_attention_layernorm(res_att)
            out_mlp = layer.mlp(pre_mlp)
            res_mlp = res_att + out_mlp
            torch.cuda.nvtx.range_pop()
            return res_mlp, (k_rotate, v)

        kv_cache_list: List[KVCache] = []
        for i, layer in enumerate(model.layers):
            # print(f"layer {i}: {hidden_states}")
            torch.cuda.nvtx.range_push(f"layer_{i}")
            hidden_states, (k, v) = self._apply_checkpoint(
                compute_layer, layer, hidden_states
            )
            torch.cuda.nvtx.range_pop()

            kv_cache_list.append((k, v))

        hidden_states = model.norm(hidden_states)
        return hidden_states, prefix_pad_masks, kv_cache_list

    def predict_suffix(
        self,
        state,
        prefix_pad_masks,
        prefix_key_values: List[KVCache],
        noisy_actions,
        time,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        torch.cuda.nvtx.range_push("embed_suffix")
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(state, noisy_actions, time)
        )
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("attention_mask")
        suffix_len = suffix_pad_masks.shape[1]
        prefix_pad_mask = einops.repeat(prefix_pad_masks, "b p -> b s p", s=suffix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_mask = torch.cat([prefix_pad_mask, suffix_att_2d_masks], dim=2)
        full_att_mask = attention_fill_false_to_inf(full_att_2d_mask)[:, None, :, :]
        # shape: [batch_size, 1, suffix_len, prefix_len + suffix_len]
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("position_ids")
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        torch.cuda.nvtx.range_pop()

        suffix_out, _ = self.gemma_expert.forward(
            inputs_embeds=suffix_embs,
            position_ids=position_ids,
            adarms_cond=adarms_cond,
            attention_mask=full_att_mask,
            past_key_values=prefix_key_values,
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :, :]
        return self.action_out_proj(suffix_out)

    @torch.inference_mode()
    def sample_actions(
        self, observation: Observation, num_steps=10
    ) -> at.Float[torch.Tensor, "*batch_size chunk_size action_dim"]:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        _, prefix_pad_masks, prefix_key_values = self.encode_prefix(observation)

        bsize = observation.images.shape[0]
        device = observation.images.device
        dtype = observation.images.dtype

        actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
        noise = sample_noise(actions_shape, device, dtype)

        dt = torch.tensor(1 / num_steps, dtype=dtype, device=device)
        time = torch.tensor(1.0, dtype=dtype, device=device)

        x_t = noise
        while time >= dt / 2:
            v_t = self.predict_suffix(
                observation.state,
                prefix_pad_masks,
                prefix_key_values,
                x_t,
                time.expand(bsize),
            )

            x_t = x_t - dt * v_t
            time -= dt
        return x_t

    def forward(self, part: Literal["prefix", "suffix", "sample"], *args, **kwargs):
        if part == "prefix":
            return self.encode_prefix(*args, **kwargs)
        elif part == "suffix":
            return self.predict_suffix(*args, **kwargs)
        elif part == "sample":
            return self.sample_actions(*args, **kwargs)
        else:
            raise ValueError(f"Unknown part: {part}")

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)
