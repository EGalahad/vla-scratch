from __future__ import annotations

import time
from typing import List, Tuple, TYPE_CHECKING

import einops
import jaxtyping as at
import torch
import torch.nn as nn
from torch.distributed.fsdp._fully_shard import (
    MixedPrecisionPolicy,
    fully_shard,
    register_fsdp_forward_method,
)

from vla_scratch.policies.base import BasePolicy
from vla_scratch.policies.modules.action_expert.dit import DiTModel
from vla_scratch.policies.modules.vlm_bridge.paligemma.bridge import PaligemmaBridge
from vla_scratch.policies.modules.vlm_bridge.qwen.bridge import Qwen3VLBridge
from vla_scratch.policies.utils.training import (
    apply_checkpoint_when_training,
    fully_shard_layers,
)
from vla_scratch.policies.utils.transformers import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    sample_noise,
)

if TYPE_CHECKING:
    from vla_scratch.policies.pi.config import PiConfig
    from vla_scratch.policies.utils.data_types import (
        HiddenState,
        KVCache,
        PrefixPadMask,
    )
    from vla_scratch.transforms.data_types import Observation


class PiPolicy(BasePolicy):
    def __init__(self, config: "PiConfig"):
        super().__init__()
        self.config = config

        if config.action_dim is None or config.state_dim is None:
            raise ValueError(
                "PiConfig.action_dim and PiConfig.state_dim must be set before "
                "initializing PiPolicy."
            )

        start_time = time.time()
        # Build a bridge wrapper for this VLM; wrappers instantiate models internally
        # if "PaliGemma" in config.vlm_type:
        if config.vlm_type == "PaliGemmaForConditionalGeneration":
            self.vlm_bridge = PaligemmaBridge(
                model_id=config.model_id,
                vlm_type=config.vlm_type,
                max_length=config.max_prompt_length,
            )
        elif config.vlm_type == "Qwen3VLForConditionalGeneration":
            self.vlm_bridge = Qwen3VLBridge(
                model_id=config.model_id,
                vlm_type=config.vlm_type,
                max_length=config.max_prompt_length,
            )
        else:
            raise NotImplementedError(
                f"Unsupported VLM type for PiPolicy: {config.vlm_type}"
            )

        end_time = time.time()
        print(
            f"VLM model initialized in {end_time - start_time:.2f} seconds: {config.vlm_type}"
        )

        # number of hidden layers and head dim must match to do cross-attention at each layer
        text_layers, text_head_dim, text_num_kv_heads, vlm_hidden_size = (
            self.vlm_bridge.get_text_dims()
        )

        self.use_obs_register = config.num_obs_registers > 0
        if self.use_obs_register:
            # add a learnable token to the VLM for observation register
            self.obs_registers = nn.Parameter(
                torch.zeros(config.num_obs_registers, vlm_hidden_size)
            )
            self.obs_registers_pad_masks = torch.ones(
                config.num_obs_registers, dtype=torch.bool
            )
            self.obs_registers_att_masks = torch.zeros(
                config.num_obs_registers, dtype=torch.bool
            )
        else:
            assert (
                not config.expert_only_use_register
            ), "expert_only_use_register must be False when num_obs_registers is 0."
        action_expert_config = config.action_expert_cfg
        if action_expert_config.head_dim != text_head_dim:
            print(
                f"Warning: Overriding DiT head_dim {action_expert_config.head_dim} "
                f"to match VLM text head_dim {text_head_dim}."
            )
            action_expert_config.head_dim = text_head_dim
        if action_expert_config.num_key_value_heads != text_num_kv_heads:
            print(
                f"Warning: Overriding DiT num_key_value_heads {action_expert_config.num_key_value_heads} "
                f"to match VLM text num_key_value_heads {text_num_kv_heads}."
            )
            action_expert_config.num_key_value_heads = text_num_kv_heads

        start_time = time.time()
        self.action_expert = DiTModel(config=action_expert_config)
        end_time = time.time()
        self.action_expert_layers = action_expert_config.num_hidden_layers
        print(f"Action expert initialized in {end_time - start_time:.2f} seconds.")

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

    def apply_fsdp(self, param_type, reduce_type, output_dtype, mesh):
        """Helper function to apply FSDP to a module with given mixed precision policy."""

        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_type,
            reduce_dtype=reduce_type,
            cast_forward_inputs=True,
        )
        self.vlm_bridge.apply_fsdp(mp_policy, mesh)

        fully_shard_layers(self.action_expert.blocks, mesh, mp_policy)

        mp_policy_root = MixedPrecisionPolicy(
            param_dtype=param_type,
            reduce_dtype=reduce_type,
            output_dtype=output_dtype,
            cast_forward_inputs=True,
        )
        fully_shard(self, mesh=mesh, mp_policy=mp_policy_root)
        register_fsdp_forward_method(self, "encode_prefix")
        register_fsdp_forward_method(self, "predict_suffix")
        register_fsdp_forward_method(self, "sample_actions")

    def encode_prefix(
        self, observation: "Observation"
    ) -> Tuple["HiddenState", "PrefixPadMask", List["KVCache"]]:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        # Prepare extra observation register tokens if configured
        extra_embs = None
        extra_pad = None
        extra_att = None
        if self.use_obs_register:
            bsize = observation.shape[0]
            extra_embs = einops.repeat(self.obs_registers, "s d -> b s d", b=bsize)
            extra_pad = einops.repeat(self.obs_registers_pad_masks, "s -> b s", b=bsize)
            extra_att = self.obs_registers_att_masks

        # Bridge handles model-specific preprocessing + transformer forward
        hidden_states, prefix_pad_masks, kv_cache_list = self.vlm_bridge.encode_prefix(
            observation=observation,
            extra_embs=extra_embs,
            extra_pad_masks=extra_pad,
            extra_att_masks=extra_att,
        )
        return hidden_states, prefix_pad_masks, kv_cache_list

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
        time_emb = apply_checkpoint_when_training(self, self.time_mlp, time_emb)

        action_emb = apply_checkpoint_when_training(
            self, self.action_in_proj, noisy_actions
        )
        if self.config.use_state:
            state_emb = apply_checkpoint_when_training(self, self.state_in_proj, state)
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

    def predict_suffix(
        self,
        state,
        prefix_pad_masks,
        prefix_key_values: List["KVCache"],
        noisy_actions,
        time,
    ):
        """Apply one denoising step of `noisy_actions` at a given timestep."""
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
        full_att_mask = einops.rearrange(full_att_2d_mask, "b i j -> b 1 i j")
        # shape: [batch_size, 1, suffix_len, prefix_len + suffix_len]
        torch.cuda.nvtx.range_pop()

        prefix_key_values = prefix_key_values[-self.action_expert_layers :]
        # only use the last num_obs_registers tokens from the prefix for the expert
        if self.config.expert_only_use_register:
            torch.cuda.nvtx.range_push("select_obs_registers")
            num_registers = self.config.num_obs_registers
            prefix_key_values = [
                (
                    kv[0][..., -num_registers:, :],
                    kv[1][..., -num_registers:, :],
                )
                for kv in prefix_key_values
            ]
            full_att_mask = full_att_mask[..., -(num_registers + suffix_len) :]
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("position_ids")
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        torch.cuda.nvtx.range_pop()

        suffix_out, _, disp_loss = self.action_expert.forward(
            inputs_embeds=suffix_embs,
            position_ids=position_ids,
            adarms_cond=adarms_cond,
            attention_mask=full_att_mask,
            past_key_values=prefix_key_values,
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :, :]
        return self.action_out_proj(suffix_out), disp_loss

    @torch.inference_mode()
    def sample_actions(
        self, observation: "Observation", num_steps=10
    ) -> at.Float[torch.Tensor, "*batch_size chunk_size action_dim"]:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        _, prefix_pad_masks, prefix_key_values = self.encode_prefix(observation)

        bsize = observation.shape[0]
        device = observation.device
        dtype = observation.state.dtype

        actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
        noise = sample_noise(actions_shape, device, dtype)

        dt_float = 1.0 / num_steps
        time_float = 1.0
        dt = torch.tensor(dt_float, dtype=dtype, device=device)
        time = torch.tensor(time_float, dtype=dtype, device=device)

        x_t = noise
        while time_float >= dt_float / 2:
            v_t, _ = self.predict_suffix(
                observation.state,
                prefix_pad_masks,
                prefix_key_values,
                x_t,
                time.expand(bsize),
            )

            x_t = x_t - dt * v_t
            time -= dt
            time_float -= dt_float
        return x_t
