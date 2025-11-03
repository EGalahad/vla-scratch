from dataclasses import dataclass, MISSING
from typing import Any, List
import importlib


@dataclass
class PolicyConfig:
    _target_: str
    transforms: List[Any]

    state_history: int = MISSING
    action_horizon: int = MISSING

def _locate_class(target: str) -> type:
    module_name, _, attr_name = target.rpartition(".")
    if not module_name:
        raise ValueError(f"Target '{target}' must be a fully-qualified path.")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ImportError(f"Cannot import '{attr_name}' from '{module_name}'.") from exc


def create_policy(policy_config: PolicyConfig) -> Any:
    policy_cls = _locate_class(policy_config._target_)
    return policy_cls(policy_config)
