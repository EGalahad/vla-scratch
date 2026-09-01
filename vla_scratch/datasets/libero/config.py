from dataclasses import dataclass

from vla_scratch.datasets.config import DataConfig
from hydra.core.config_store import ConfigStore


@dataclass
class LiberoConfig(DataConfig):
    _target_: str = "vla_scratch.datasets.libero.dataset.LIBERODataset"
    repo_id: str = "elijahgalahad/libero_spatial_noops_v30"
    root: str | None = None
    norm_stats_path: str = "hf:elijahgalahad/norm_stats-libero-spatial"


libero_spatial_config = LiberoConfig()
libero_spatial_ipec_config = LiberoConfig(
    _target_="vla_scratch.datasets.libero.dataset.LIBEROIpecDataset",
    repo_id="IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot",
    norm_stats_path="normalization_stats/libero/ipec-spatial",
)

cs = ConfigStore.instance()
cs.store(name="libero-spatial", node=libero_spatial_config, group="data")
cs.store(
    name="libero-spatial-ipec",
    node=libero_spatial_ipec_config,
    group="data",
)
