import time
from typing import Any, Dict, Sequence, SupportsIndex, Tuple

import torch
from tensordict import TensorDict


class TransformFn:
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def compute(self, sample: Any) -> Any:
        raise NotImplementedError


class TransformedDataset(torch.utils.data.Dataset):
    @staticmethod
    def collate_fn(batch: Sequence[Tuple[Any, TensorDict]]) -> Sequence[Any]:
        return tuple(torch.stack(items) for items in zip(*batch))

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        transforms: Sequence[TransformFn],
    ):
        self.base_dataset = dataset
        self.transforms = list(transforms)
        self._log_names = [tr.__repr__() for tr in self.transforms]

    def __getitem__(self, index: SupportsIndex) -> Tuple[Any, TensorDict]:
        perf: Dict[str, float] = {}
        sample = self.base_dataset[index]
        for transform, name in zip(self.transforms, self._log_names):
            start = time.perf_counter()
            sample = transform.compute(sample)
            perf[name] = time.perf_counter() - start
        return sample, TensorDict(perf)

    def __len__(self) -> int:
        return len(self.base_dataset)
