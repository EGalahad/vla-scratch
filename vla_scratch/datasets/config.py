from dataclasses import dataclass, field
from typing import Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from vla_scratch.utils.config import locate_class


@dataclass
class DataConfig:
    _target_: str
    root_path: Optional["Path"] = None
    action_horizon: Optional[int] = None
    state_history: Optional[int] = None
    # Structured transform lists
    input_transforms: List[Any] = field(default_factory=list)
    output_transforms: List[Any] = field(default_factory=list)
    output_inv_transforms: List[Any] = field(default_factory=list)
    norm_stats_path: Optional[str] = None
    noise_cfg: Optional[dict] = None

    def instantiate(self, *args, **kwargs) -> Any:
        dataset_cls = locate_class(self._target_)
        return dataset_cls(self, *args, **kwargs)
