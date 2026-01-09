from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from vla_scratch.policies.config import PolicyConfig
from vla_scratch.policies.modules.action_expert.cross_attention_dit import DiTConfig


@dataclass
class PiConfig(PolicyConfig):
    vlm_type: str
    model_id: str

    action_expert_cfg: DiTConfig = field(
        default_factory=lambda: DiTConfig(
            # hidden size
            hidden_size=1024,
            intermediate_size=4096,
            # attention size
            num_attention_heads=32,
            num_key_value_heads=32,
            head_dim=128,
            # layers
            num_hidden_layers=12,
            cross_attention_every=2,
            only_attend_to_final_layer=True,
        )
    )

    # architecture
    num_obs_registers: int = 4
    expert_only_use_register: bool = True
    suffix_add_pos_emb: bool = True
    use_state: bool = False

    # noising
    num_noise_per_sample: int = 2
    time_dist_alpha: float = 1.0
    time_dist_beta: float = 1.5

    # training
    detach_encoder_output: bool = False
    ce_loss_weight: float = 0.1

    # misc
    obs_register_init_gain: float = 0.02
    suffix_pos_emb_init_gain: float = 0.02
    zero_pos_id_for_obs_register: bool = True
    causal_mask_obs_register: bool = True


pi_paligemma_config = PiConfig(
    _target_="vla_scratch.policies.pi.policy.PiPolicy",
    model_id="google/paligemma-3b-mix-224",
    vlm_type="PaliGemmaForConditionalGeneration",
    state_history=1,
    action_horizon=10,
    transforms=[
        {
            "_target_": "vla_scratch.policies.modules.vlm_bridge.paligemma.processor.PaligemmaProcessor",
            "processor_class": "PaliGemmaProcessor",
            "model_id": "google/paligemma-3b-mix-224",
            "max_length": 550,
            "target_size": (224, 224),
        }
    ],
)

pi_paligemma2_config = PiConfig(
    _target_="vla_scratch.policies.pi.policy.PiPolicy",
    model_id="google/paligemma2-3b-mix-224",
    vlm_type="PaliGemmaForConditionalGeneration",
    state_history=1,
    action_horizon=10,
    transforms=[
        {
            "_target_": "vla_scratch.policies.modules.vlm_bridge.paligemma.processor.PaligemmaProcessor",
            "processor_class": "PaliGemmaProcessor",
            "model_id": "google/paligemma2-3b-mix-224",
            "max_length": 550,
            "target_size": (224, 224),
        }
    ],
)

pi_qwen_config = PiConfig(
    _target_="vla_scratch.policies.pi.policy.PiPolicy",
    state_history=1,
    action_horizon=10,
    model_id="Qwen/Qwen3-VL-2B-Instruct",
    vlm_type="Qwen3VLForConditionalGeneration",
    transforms=[
        {
            "_target_": "vla_scratch.policies.modules.vlm_bridge.qwen.processor.QwenProcessor",
            "processor_class": "Qwen3VLProcessor",
            "model_id": "Qwen/Qwen3-VL-2B-Instruct",
            "max_length": 180,
            # WARN: select this based on your image sizes and prompt lengths, try to make it minimum as possible because if impacts iteration time a lot!
            "padding": "max_length",
        }
    ],
)

cs = ConfigStore.instance()
cs.store(name="pi-paligemma", node=pi_paligemma_config, group="policy")
cs.store(name="pi-paligemma2", node=pi_paligemma2_config, group="policy")
cs.store(name="pi-qwen", node=pi_qwen_config, group="policy")
