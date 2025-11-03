from dataclasses import dataclass, field
from typing import Optional, Any, List
from hydra.core.config_store import ConfigStore
from vla_scratch.policies.config import PolicyConfig

def _default_pi_transforms() -> list[Any]:
    return [
        {
            "_target_": "vla_scratch.policies.pi.transforms.TokenizePrompt",
            "processor_class": "PaliGemmaProcessor",
            "model_id": "google/paligemma-3b-mix-224",
            "max_length": 64,
        },
        {
            "_target_": "vla_scratch.policies.pi.transforms.PreprocessImage",
            "target_size": (224, 224),
        },
    ]


@dataclass
class PiConfig(PolicyConfig):
    _target_: str = "vla_scratch.policies.pi.policy.PiPolicy"

    transforms: List[Any] = field(default_factory=_default_pi_transforms)

    action_expert_variant: str = "300m"
    model_id: str = "google/paligemma-3b-mix-224"

    state_dim: Optional[int] = None
    action_dim: Optional[int] = None
    state_history: int = 10
    action_horizon: int = 30

    use_state: bool = True


Cs = ConfigStore.instance()
Cs.store(name="pi", node=PiConfig, group="policy")
