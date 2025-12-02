from pathlib import Path
from typing import Any, Optional
from omegaconf import DictConfig, OmegaConf
import importlib
import re

from vla_scratch.utils.checkpoint import find_latest_checkpoint


def locate_class(target: str) -> type:
    """Import and return a class/function given a fully-qualified path string.

    Example: "vla_scratch.datasets.spirit.transforms.SpiritImages"
    """
    module_name, _, attr_name = target.rpartition(".")
    if not module_name:
        raise ValueError(f"Target '{target}' must be a fully-qualified path.")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ImportError(f"Cannot import '{attr_name}' from '{module_name}'.") from exc


def resolve_config_placeholders(
    template: str | Path | None,
    *,
    data_cfg: Any,
    policy_cfg: Optional[Any] = None,
) -> Optional[str]:
    """Resolve placeholders like '{data.attr}' or '{policy.attr}' in a string/Path.

    Unknown placeholders are left untouched; only 'data.*' and 'policy.*' are resolved
    by simple attribute lookup.
    """
    if template is None:
        return None
    s = str(template)

    def _replace_data(match: re.Match[str]) -> str:
        attr = match.group(1)
        return str(getattr(data_cfg, attr, match.group(0)))

    def _replace_policy(match: re.Match[str]) -> str:
        if policy_cfg is None:
            return match.group(0)
        attr = match.group(1)
        return str(getattr(policy_cfg, attr, match.group(0)))

    s = re.sub(r"\{data\.([a-zA-Z0-9_]+)\}", _replace_data, s)
    s = re.sub(r"\{policy\.([a-zA-Z0-9_]+)\}", _replace_policy, s)
    return s


def save_train_config(cfg: DictConfig, run_dir: Path) -> None:
    OmegaConf.save(cfg, run_dir / "train-cfg.yaml")


def merge_cfg_from_checkpoint(
    cfg: DictConfig,
    checkpoint_path: Optional[Path | str],
) -> DictConfig:
    """Merge saved train config from a checkpoint run directory into `cfg`."""
    if checkpoint_path is None:
        return cfg
    try:
        ckpt = find_latest_checkpoint(checkpoint_path)
    except Exception:
        ckpt = None
    if ckpt is None:
        return cfg

    run_dir = Path(ckpt).parent
    train_cfg_path = run_dir / "train-cfg.yaml"
    if not train_cfg_path.exists():
        return cfg
    try:
        saved_cfg = OmegaConf.load(train_cfg_path)
    except Exception:
        return cfg
    saved_policy = saved_cfg["policy"]
    saved_data = saved_cfg["data"]
    cfg["policy"] = OmegaConf.merge(cfg["policy"], saved_policy)
    cfg["data"] = OmegaConf.merge(cfg["data"], saved_data)
    return cfg
    
    
