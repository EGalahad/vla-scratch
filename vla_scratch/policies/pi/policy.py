from __future__ import annotations

import time
from typing import Tuple, TYPE_CHECKING, Dict

import torch
import torch.nn as nn
from torch.distributed.fsdp._fully_shard import (
    MixedPrecisionPolicy,
    fully_shard,
    register_fsdp_forward_method,
)

from tensordict import TensorClass

import einops
import jaxtyping as at

from vla_scratch.policies.base import BasePolicy
from vla_scratch.policies.modules.action_expert.cross_attention_dit import (
    DiTModel,
)
from vla_scratch.policies.modules.vlm_bridge.paligemma.bridge import (
    PaligemmaBridge,
)
from vla_scratch.policies.modules.vlm_bridge.qwen.bridge import Qwen3VLBridge
from vla_scratch.policies.modules.vlm_bridge.smolvlm.bridge import SmolVLMBridge

from vla_scratch.policies.utils.training import (
    apply_checkpoint_when_training,
    fully_shard_layers,
)
from vla_scratch.policies.utils.diffusion import (
    build_beta_time_dist,
    sample_clamped_time,
    repeat_batch,
    sample_noise,
)
from vla_scratch.policies.utils.transformers import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
)

if TYPE_CHECKING:
    from vla_scratch.policies.modules.vlm_bridge.base import VLMOutputs
    from vla_scratch.policies.pi.config import PiConfig
    from vla_scratch.transforms.data_types import Observation, DataSample


class SuffixInput(TensorClass):
    prefix_pad_masks: at.Bool[torch.Tensor, "*b seq_len"]
    hidden_state_list: at.Float[torch.Tensor, "*b n_layer seq_len hidden"]


class PiPolicy(BasePolicy):
    suffix_pad_mask: at.Bool[torch.Tensor, "action_horizon"]
    suffix_att_mask: at.Bool[torch.Tensor, "action_horizon"]

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
            )
        elif config.vlm_type == "Qwen3VLForConditionalGeneration":
            self.vlm_bridge = Qwen3VLBridge(
                model_id=config.model_id,
                vlm_type=config.vlm_type,
            )
        elif config.vlm_type == "SmolVLMForConditionalGeneration":
            self.vlm_bridge = SmolVLMBridge(
                model_id=config.model_id,
                vlm_type=config.vlm_type,
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
            self.obs_registers_att_masks[0] = (
                1  # prevent prefix from attending to registers
            )
        else:
            assert (
                not config.expert_only_use_register
            ), "expert_only_use_register must be False when num_obs_registers is 0."
        action_expert_config = config.action_expert_cfg
        start_time = time.time()
        self.action_expert = DiTModel(config=action_expert_config)
        end_time = time.time()
        print(
            f"Action expert initialized in {end_time - start_time:.2f} seconds."
        )

        action_expert_width = action_expert_config.hidden_size
        self.action_in_proj = nn.Linear(config.action_dim, action_expert_width)
        self.action_out_proj = nn.Linear(action_expert_width, config.action_dim)
        self.state_in_proj = nn.Linear(config.state_dim, action_expert_width)

        self.time_mlp = nn.Sequential(
            nn.Linear(action_expert_width, action_expert_width),
            nn.SiLU(),
            nn.Linear(action_expert_width, action_expert_width),
            nn.SiLU(),
        )

        # register buffers
        if config.use_state:
            suffix_len = config.action_horizon + config.state_history
        else:
            suffix_len = config.action_horizon
        self.suffix_len = suffix_len
        suffix_pad_mask = torch.ones(suffix_len, dtype=torch.bool)
        suffix_att_mask = torch.zeros(suffix_len, dtype=torch.bool)
        # create a new attention block for the suffix, prefix should not attend to suffix
        suffix_att_mask[0] = 1
        self.register_buffer(
            "suffix_pad_mask", suffix_pad_mask, persistent=False
        )
        self.register_buffer(
            "suffix_att_mask", suffix_att_mask, persistent=False
        )

        if config.suffix_add_pos_emb:
            pos_emb_state = torch.zeros(
                config.state_history, action_expert_width
            )
            self.position_embedding_state = nn.Parameter(pos_emb_state)
            pos_emb_action = torch.zeros(
                config.action_horizon, action_expert_width
            )
            self.position_embedding_action = nn.Parameter(pos_emb_action)

        param_device = next(self.parameters()).device
        self.time_dist = build_beta_time_dist(
            self.config.time_dist_alpha,
            self.config.time_dist_beta,
            device=param_device,
        )

    def initialize_weights(self):
        if self.config.suffix_add_pos_emb:
            nn.init.normal_(
                self.position_embedding_state,
                mean=0.0,
                std=self.config.suffix_pos_emb_init_gain,
            )
            nn.init.normal_(
                self.position_embedding_action,
                mean=0.0,
                std=self.config.suffix_pos_emb_init_gain,
            )
        if self.use_obs_register:
            nn.init.normal_(
                self.obs_registers,
                mean=0.0,
                std=self.config.obs_register_init_gain,
            )
        self.action_expert.initialize_weights()

    def apply_fsdp(self, param_type, reduce_type, output_dtype, mesh):
        """Helper function to apply FSDP to a module with given mixed precision policy."""

        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_type,
            reduce_dtype=reduce_type,
            cast_forward_inputs=True,
        )
        self.vlm_bridge.apply_fsdp(mp_policy, mesh)

        fully_shard_layers(
            self.action_expert.blocks, mesh, mp_policy, num_to_prefetch=6
        )

        mp_policy_root = MixedPrecisionPolicy(
            param_dtype=param_type,
            reduce_dtype=reduce_type,
            output_dtype=output_dtype,
            cast_forward_inputs=True,
        )
        fully_shard(self, mesh=mesh, mp_policy=mp_policy_root)
        register_fsdp_forward_method(self, "compute_loss")
        # will c10 error if below is not registered
        register_fsdp_forward_method(self, "encode_prefix")
        register_fsdp_forward_method(self, "predict_suffix")
        register_fsdp_forward_method(self, "sample_actions")
        return self

    def encode_prefix(
        self, observation: "Observation"
    ) -> Tuple[torch.Tensor, "VLMOutputs", Dict]:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        # Prepare extra observation register tokens if configured
        extra_embs = None
        extra_pad = None
        extra_att = None
        if self.use_obs_register:
            bsize = observation.shape[0]
            extra_embs = einops.repeat(
                self.obs_registers, "s d -> b s d", b=bsize
            )
            extra_pad = einops.repeat(
                self.obs_registers_pad_masks, "s -> b s", b=bsize
            )
            extra_att = self.obs_registers_att_masks

        # Bridge handles model-specific preprocessing + transformer forward
        ce_loss, vlm_outputs, log_dict = self.vlm_bridge.encode(
            observation=observation,
            extra_embs=extra_embs,
            extra_pad_masks=extra_pad,
            extra_att_masks=extra_att,
            zero_pos_id_for_extra=self.config.zero_pos_id_for_obs_register,
            extra_attention_mask=self.config.causal_mask_obs_register,
        )
        return ce_loss, vlm_outputs, log_dict

    def construct_suffix_input(self, vlm_outputs: "VLMOutputs") -> SuffixInput:
        """Construct SuffixInput from VLMOutputs for caching purposes."""
        # only retain last N layers for action expert
        prefix_pad_masks = vlm_outputs.prefix_pad_masks
        action_expert_layers = self.config.action_expert_cfg.num_hidden_layers
        hidden_state_list = vlm_outputs.hidden_state_list[
            :, -action_expert_layers:
        ]
        # only use the last num_obs_registers tokens from the prefix for the expert
        if self.config.expert_only_use_register:
            torch.cuda.nvtx.range_push("select_obs_registers")
            num_registers = self.config.num_obs_registers
            prefix_pad_masks = prefix_pad_masks[:, -num_registers:]
            hidden_state_list = hidden_state_list[:, :, -num_registers:, :]
            torch.cuda.nvtx.range_pop()

        suffix_input = SuffixInput(
            prefix_pad_masks=prefix_pad_masks,
            hidden_state_list=hidden_state_list,
            batch_size=vlm_outputs.shape,
        )
        return suffix_input

    def predict_suffix(
        self,
        state: at.Float[torch.Tensor, "b horizon dim"],
        suffix_input: SuffixInput,
        noisy_actions,
        time,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Apply one denoising step of `noisy_actions` at a given timestep."""
        prefix_pad_masks = suffix_input.prefix_pad_masks

        torch.cuda.nvtx.range_push("embed_suffix")
        suffix_embs, suffix_pad_masks, suffix_att_masks, time_emb = (
            self._embed_suffix(state, noisy_actions, time)
        )
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("attention_mask")
        suffix_att_2d_masks = make_att_2d_masks(
            suffix_pad_masks, suffix_att_masks
        )
        prefix_pad_mask = einops.repeat(
            prefix_pad_masks, "b p -> b s p", s=self.suffix_len
        )
        full_att_2d_mask = torch.cat(
            [prefix_pad_mask, suffix_att_2d_masks], dim=2
        )
        full_att_mask = einops.rearrange(full_att_2d_mask, "b i j -> b 1 i j")
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("position_ids")
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = (
            prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        )
        torch.cuda.nvtx.range_pop()

        encoder_hidden_states = suffix_input.hidden_state_list.unbind(dim=1)
        disp_loss, suffix_out, log_dict = self.action_expert.forward(
            inputs_embeds=suffix_embs,
            position_ids=position_ids,
            adarms_cond=time_emb,
            attention_mask=full_att_mask,
            encoder_hidden_states=encoder_hidden_states,
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :, :]
        return disp_loss, self.action_out_proj(suffix_out), log_dict

    def compute_loss(
        self,
        data_sample: "DataSample",
    ) -> Tuple[torch.Tensor, Dict]:
        torch.cuda.nvtx.range_push("Model Encode Prefix")
        ce_loss, vlm_outputs, log_dict_prefix = self.encode_prefix(
            observation=data_sample.observation
        )
        torch.cuda.nvtx.range_pop()

        if data_sample.action_chunk is not None:
            suffix_input = self.construct_suffix_input(vlm_outputs)
            if self.config.detach_encoder_output:
                torch.cuda.nvtx.range_push("Detach KV Cache")
                suffix_input = suffix_input.detach()
                torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("Expand Data Sample")
            data_sample = repeat_batch(
                data_sample, self.config.num_noise_per_sample
            )
            suffix_input = repeat_batch(
                suffix_input, self.config.num_noise_per_sample
            )
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("Apply Noise")
            actions = data_sample.action_chunk.actions
            selected_noise = sample_noise(
                actions.shape, actions.device, actions.dtype
            )
            u_t = selected_noise - actions
            timestep = sample_clamped_time(self.time_dist, data_sample.shape)
            noisy_actions = actions + timestep[:, None, None] * u_t
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("Model Predict Suffix")
            _, v_t, log_dict_suffix = self.predict_suffix(
                state=data_sample.observation.state,
                suffix_input=suffix_input,
                noisy_actions=noisy_actions,
                time=timestep,
            )
            torch.cuda.nvtx.range_pop()

            flow_mse = torch.nn.functional.mse_loss(
                u_t.to(v_t.dtype), v_t, reduction="none"
            ).mean()
            log_dict_suffix["loss/flow_mse"] = flow_mse.detach()
        else:
            flow_mse = 0.0
            log_dict_suffix = {}

        loss = flow_mse + self.config.ce_loss_weight * ce_loss

        log_dict = {**log_dict_prefix, **log_dict_suffix}

        return loss, log_dict

    @torch.inference_mode()
    def sample_actions(
        self, observation: "Observation", num_steps=10
    ) -> at.Float[torch.Tensor, "*batch_size chunk_size action_dim"]:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        _, vlm_output, log_dict = self.encode_prefix(observation)
        suffix_input = self.construct_suffix_input(vlm_output)

        bsize = observation.shape[0]
        device = observation.device
        dtype = observation.state.dtype

        actions_shape = (
            bsize,
            self.config.action_horizon,
            self.config.action_dim,
        )
        noise = sample_noise(actions_shape, device, dtype)

        dt_float = 1.0 / num_steps
        time_float = 1.0
        dt = torch.tensor(dt_float, dtype=dtype, device=device)
        time = torch.tensor(time_float, dtype=dtype, device=device)

        x_t = noise
        while time_float >= dt_float / 2:
            _, v_t, _ = self.predict_suffix(
                observation.state,
                suffix_input=suffix_input,
                noisy_actions=x_t,
                time=time.expand(bsize),
            )

            x_t = x_t - dt * v_t
            time -= dt
            time_float -= dt_float
        return x_t

    def _embed_suffix(
        self,
        state: at.Float[torch.Tensor, "*batch_size state_history state_dim"],
        noisy_actions: at.Float[
            torch.Tensor, "*batch_size action_horizon action_dim"
        ],
        time: at.Float[torch.Tensor, "*batch_size"],
    ) -> Tuple[
        at.Float[torch.Tensor, "*batch_size action_horizon hidden_dim"],
        at.Bool[torch.Tensor, "*batch_size action_horizon"],
        at.Bool[torch.Tensor, "*batch_size action_horizon"],
        at.Float[torch.Tensor, "*batch_size hidden_dim"],
    ]:
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]

        # use float 32 for sinusoidal embedding
        time_fp32 = time.to(torch.float32)
        time_emb_fp32 = create_sinusoidal_pos_embedding(
            time_fp32,
            dimension=self.config.action_expert_cfg.hidden_size,
            min_period=4e-3,
            max_period=4.0,
            device=time.device,
            dtype=time_fp32.dtype,
        )
        time_emb = time_emb_fp32.to(time.dtype)

        time_emb = apply_checkpoint_when_training(self, self.time_mlp, time_emb)

        action_emb = apply_checkpoint_when_training(
            self, self.action_in_proj, noisy_actions
        )

        if self.config.use_state:
            state_emb = apply_checkpoint_when_training(
                self, self.state_in_proj, state
            )
            if self.config.suffix_add_pos_emb:
                state_emb = (
                    state_emb + self.position_embedding_state[None, :, :]
                )
                action_emb = (
                    action_emb + self.position_embedding_action[None, :, :]
                )
            suffix_emb = torch.cat([state_emb, action_emb], dim=1)
        else:
            if self.config.suffix_add_pos_emb:
                action_emb = (
                    action_emb + self.position_embedding_action[None, :, :]
                )
            suffix_emb = action_emb

        bsize = action_emb.shape[0]
        pad_mask = einops.repeat(
            self.suffix_pad_mask, "action_horizon -> b action_horizon", b=bsize
        )
        att_mask = einops.repeat(
            self.suffix_att_mask, "action_horizon -> b action_horizon", b=bsize
        )

        return suffix_emb, pad_mask, att_mask, time_emb
