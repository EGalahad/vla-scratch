from dataclasses import dataclass, replace
from typing import Optional
from hydra.core.config_store import ConfigStore
from vla_scratch.policies.config import PolicyConfig
from vla_scratch.policies.modules.action_expert.dit import DiTConfig


@dataclass
class PiConfig(PolicyConfig):
    vlm_type: str
    model_id: str
    max_prompt_length: int

    action_expert_cfg: DiTConfig

    use_state: bool = True

    num_obs_registers: int = 0
    expert_only_use_register: bool = False


default_pi_config = PiConfig(
    _target_="vla_scratch.policies.pi.policy.PiPolicy",
    model_id="google/paligemma-3b-mix-224",
    vlm_type="PaliGemmaForConditionalGeneration",
    action_expert_cfg=DiTConfig(
        hidden_size=1024,
        num_hidden_layers=12,
        intermediate_size=4096,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=256,
    ),
    max_prompt_length=64,
    state_history=1,
    action_horizon=20,
    transforms=[
        {
            "_target_": "vla_scratch.policies.modules.vlm_bridge.paligemma.processor.PaligemmaProcessor",
            "processor_class": "PaliGemmaProcessor",
            "model_id": "google/paligemma-3b-mix-224",
            "max_length": 64,
            "target_size": (224, 224),
        }
    ],
)

Cs = ConfigStore.instance()
Cs.store(
    name="pi",
    node=default_pi_config,
    group="policy",
)
Cs.store(
    name="pi-qwen",
    node=replace(
        default_pi_config,
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        vlm_type="Qwen3VLForConditionalGeneration",
        max_prompt_length=512,
        transforms=[
            {
                "_target_": "vla_scratch.policies.modules.vlm_bridge.qwen.processor.QwenProcessor",
                "processor_class": "Qwen3VLProcessor",
                "model_id": "Qwen/Qwen3-VL-2B-Instruct",
                "max_length": 512,
                "add_generation_prompt": True,
                "padding": "max_length",
            }
        ],
        action_expert_cfg=DiTConfig(
            hidden_size=1024,
            num_hidden_layers=12,
            intermediate_size=4096,
            num_attention_heads=8,
            num_key_value_heads=8,
            head_dim=256,
        ),
    ),
    group="policy",
)
