from dataclasses import dataclass
from typing import Any, List
from vla_scratch.utils.config import locate_class


@dataclass
class PolicyConfig:
    _target_: str
    transforms: List[Any]

    state_history: int
    action_horizon: int
    state_dim: int = None
    action_dim: int = None

    def instantiate(self) -> Any:
        policy_cls = locate_class(self._target_)
        return policy_cls(self)

def create_policy(policy_config: PolicyConfig) -> Any:
    policy_cls = locate_class(policy_config._target_)
    return policy_cls(policy_config)
