from dataclasses import dataclass
from typing import Any, List, Optional
from vla_scratch.utils.config import locate_class


@dataclass(kw_only=True)
class PolicyConfig:
    _target_: str
    transforms: List[Any]

    state_history: int
    action_horizon: int
    state_dim: Optional[int] = None
    action_dim: Optional[int] = None
    use_paligemma_tokens: bool = False

    def instantiate(self) -> Any:
        policy_cls = locate_class(self._target_)
        return policy_cls(self)
