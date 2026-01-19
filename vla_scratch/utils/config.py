from pathlib import Path
from typing import Any, Optional
import importlib
import re


def locate_class(target: str) -> type:
    """Import and return a class/function given a fully-qualified path string."""
    module_name, _, attr_name = target.rpartition(".")
    if not module_name:
        raise ValueError(f"Target '{target}' must be a fully-qualified path.")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ImportError(
            f"Cannot import '{attr_name}' from '{module_name}'."
        ) from exc


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
