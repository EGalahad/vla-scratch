from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional
from collections.abc import MutableMapping

from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

from vla_scratch.utils.config import locate_class


@dataclass
class DataConfig:
    _target_: str
    action_horizon: Optional[int] = None
    state_history: Optional[int] = None
    video_backend: Optional[str] = None
    # Structured transform lists
    input_transforms: List[Any] = field(default_factory=list)
    output_transforms: List[Any] = field(default_factory=list)
    output_inv_transforms: List[Any] = field(default_factory=list)
    norm_stats_path: Optional[str] = None
    noise_cfg: Optional[dict] = None

    def instantiate(self, *args, **kwargs) -> Any:
        dataset_cls = locate_class(self._target_)
        return dataset_cls(self, *args, **kwargs)


@dataclass
class EvalDatasetCfg:
    data: DataConfig
    eval_fraction: float = 1.0
    eval_type: str = "sample_mse"


@dataclass
class TrainDatasetCfg:
    data: DataConfig
    batch_size: int = MISSING


@dataclass
class EvalDataCfg(MutableMapping[str, EvalDatasetCfg]):
    datasets: dict[str, EvalDatasetCfg] = field(default_factory=dict)

    def __getitem__(self, key: str) -> EvalDatasetCfg:
        return self.datasets[key]

    def __setitem__(self, key: str, value: EvalDatasetCfg) -> None:
        self.datasets[key] = value

    def __delitem__(self, key: str) -> None:
        del self.datasets[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.datasets)

    def __len__(self) -> int:
        return len(self.datasets)

    def __bool__(self) -> bool:
        return bool(self.datasets)


@dataclass
class TrainDataCfg(MutableMapping[str, TrainDatasetCfg]):
    datasets: dict[str, TrainDatasetCfg] = field(default_factory=dict)

    def __getitem__(self, key: str) -> TrainDatasetCfg:
        return self.datasets[key]

    def __setitem__(self, key: str, value: TrainDatasetCfg) -> None:
        self.datasets[key] = value

    def __delitem__(self, key: str) -> None:
        del self.datasets[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.datasets)

    def __len__(self) -> int:
        return len(self.datasets)

    def __bool__(self) -> bool:
        return bool(self.datasets)


cs = ConfigStore.instance()
cs.store(name="none", node=TrainDataCfg(), group="train_data")
cs.store(name="none", node=EvalDataCfg(), group="eval_data")
