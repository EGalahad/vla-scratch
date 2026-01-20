import logging
import torch
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    StateDictOptions,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from omegaconf import DictConfig, OmegaConf

_HF_PREFIX = "hf:"
_LOGGER = logging.getLogger(__name__)


def _resolve_hf_checkpoint_path(path: str) -> Path:
    """Resolve an hf: checkpoint path to a local Hugging Face cache path.

    Format: hf:org/repo[/optional/subpath] or hf:org/repo@revision[/subpath]
    """
    from huggingface_hub import snapshot_download, get_token

    raw = path[len(_HF_PREFIX) :]
    if not raw:
        raise ValueError("Expected hf:org/repo[/subpath] in checkpoint_path.")

    parts = raw.split("/", 2)
    if len(parts) >= 2:
        repo_id = "/".join(parts[:2])
        subpath = parts[2] if len(parts) == 3 else ""
    else:
        repo_id = raw
        subpath = ""

    revision = None
    if "@" in repo_id:
        repo_id, revision = repo_id.split("@", 1)

    _LOGGER.info("Downloading Hugging Face checkpoint: %s", repo_id)
    snapshot_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        token=get_token(),
    )
    local_path = Path(snapshot_dir)
    if subpath:
        local_path = local_path / subpath
    _LOGGER.info("Resolved Hugging Face checkpoint path: %s", local_path)
    return local_path


def find_latest_checkpoint(
    path: Path | str, desired_iter: Optional[int] = None
) -> Optional[Path]:
    """Resolve a checkpoint path to a concrete checkpoint location.

    Supports checkpoint directories (checkpoint_*/model.pt) and hf: paths.

    Returns:
    - If `path` is a file, returns it.
    - If `path` is a checkpoint directory (contains model.pt), returns the dir.
    - If `path` is a run directory, returns the newest checkpoint directory.
    - None if nothing is found.
    """
    if isinstance(path, str):
        if path.startswith(_HF_PREFIX):
            path = _resolve_hf_checkpoint_path(path)
        else:
            path = Path(path).expanduser().resolve()
    p = Path(path)
    if p.is_file():
        # If this is model.pt inside a checkpoint dir, prefer returning the dir
        if p.name == "model.pt" and p.parent.name.startswith("checkpoint_"):
            return p.parent
        return p
    if not p.exists():
        return None

    # If the path itself looks like a checkpoint directory
    if p.is_dir() and (p / "model.pt").exists():
        return p

    # Gather new-style checkpoint directories under this directory
    def _epoch_num_from_name(name: str) -> int:
        try:
            return int(name.split("_")[-1])
        except Exception:
            return -1

    dir_candidates = [
        d
        for d in p.glob("checkpoint_*")
        if d.is_dir() and (d / "model.pt").exists()
    ]
    if dir_candidates:

        def _score(d: Path) -> int:
            ep = _epoch_num_from_name(d.name)
            if desired_iter is not None and ep == desired_iter:
                return 10**9
            return ep

        dir_candidates.sort(key=_score)
        return dir_candidates[-1]

    return None


def merge_policy_cfg_from_checkpoint(
    cfg: DictConfig,
    checkpoint_path: Optional[Path | str],
) -> DictConfig:
    """Merge saved cfg.yaml from a checkpoint run directory into `cfg`.

    Only `policy` and `data` groups are merged to keep runtime overrides intact.
    """
    if checkpoint_path is None:
        return cfg
    run_dir = Path(checkpoint_path)
    cfg_path = run_dir.parent / "cfg.yaml"
    saved_cfg = OmegaConf.load(cfg_path)

    if "policy" in saved_cfg:
        cfg["policy"] = OmegaConf.merge(cfg.get("policy"), saved_cfg["policy"])
    return cfg


def load_model_from_checkpoint(
    model: torch.nn.Module,
    path: Path | str,
    device: torch.device | str = "cpu",
    *,
    strict: bool = False,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Load a checkpoint into `model`.

    `path` is a checkpoint directory containing `model.pt`, or a direct
    path to `model.pt`. Returns (missing_keys, unexpected_keys).
    """
    if isinstance(path, str) and path.startswith(_HF_PREFIX):
        resolved = find_latest_checkpoint(path)
        if resolved is None:
            raise FileNotFoundError(f"No checkpoint found under {path}")
        path = resolved
    p = Path(path)
    if p.is_dir():
        p = p / "model.pt"
    ckpt = torch.load(p, map_location=device)
    model_state = ckpt
    missing, unexpected = model.load_state_dict(model_state, strict=strict)
    return tuple(missing), tuple(unexpected)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint: str | Path,
    global_rank: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Distributed-aware checkpoint load for directory checkpoints.

    `checkpoint` is a directory with `model.pt` and `optimizer.pt`. Returns
    (missing_keys, unexpected_keys) from set_model_state_dict.
    """
    if isinstance(checkpoint, str) and checkpoint.startswith(_HF_PREFIX):
        resolved = find_latest_checkpoint(checkpoint)
        if resolved is None:
            raise FileNotFoundError(f"No checkpoint found under {checkpoint}")
        checkpoint = resolved
    p = Path(checkpoint)
    # Only rank 0 reads from disk; others pass empty dicts and receive via broadcast
    if global_rank == 0:
        model_sd: Dict[str, Any] = {}
        optim_sd: Dict[str, Any] = {}
        mp = p / "model.pt"
        op = p / "optimizer.pt"
        if mp.exists():
            model_sd = torch.load(
                mp, map_location="cpu", mmap=True, weights_only=False
            )
        if optimizer is not None and op.exists():
            optim_sd = torch.load(
                op, map_location="cpu", mmap=True, weights_only=False
            )
    else:
        model_sd = {}
        optim_sd = {}

    options = StateDictOptions(
        full_state_dict=True, broadcast_from_rank0=True, strict=False
    )
    missing, unexpected = set_model_state_dict(
        model=model,
        model_state_dict=model_sd,
        options=options,
    )

    if optimizer is not None:
        # If the optimizer state dict uses FQNs in param_groups, make sure
        # every referenced FQN has an entry in the state map to avoid KeyErrors
        # during torch.distributed.checkpoint restore.
        if global_rank == 0:
            groups = optim_sd["param_groups"]
            state = optim_sd.get("state", {})
            # Detect FQN-style groups
            uses_fqn = False
            if isinstance(groups, list) and groups:
                first_params = (
                    groups[0].get("params", [])
                    if isinstance(groups[0], dict)
                    else []
                )
                if first_params and isinstance(first_params[0], str):
                    uses_fqn = True
            if uses_fqn and isinstance(state, dict):
                for g in groups:
                    params = g.get("params", []) if isinstance(g, dict) else []
                    for fqn in params:
                        if fqn not in state:
                            state[fqn] = {}
        # Load optimizer state in a best-effort manner. Checkpoints created
        # before adding new params (e.g., 'obs_registers') may not contain
        # optimizer slots for newly introduced parameters. In that case, fall
        # back to skipping optimizer load instead of raising.
        set_optimizer_state_dict(
            model=model,
            optimizers=optimizer,
            optim_state_dict=optim_sd,
            options=options,
        )
    return tuple(missing), tuple(unexpected)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_rank: int,
    filename: str | Path,
):
    """Save checkpoint as a directory with model.pt and optimizer.pt.

    `filename` should be the checkpoint directory name, e.g.,
    `checkpoint_5`. If a file extension is provided, it will be stripped.
    """

    def _cast_float_tensors_to_bfloat16(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return (
                obj.to(dtype=torch.bfloat16) if obj.is_floating_point() else obj
            )
        if isinstance(obj, dict):
            return {
                k: _cast_float_tensors_to_bfloat16(v) for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [_cast_float_tensors_to_bfloat16(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_cast_float_tensors_to_bfloat16(v) for v in obj)
        return obj

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    model_state_dict, optim_state_dict = get_state_dict(
        model,
        optimizers=optimizer,
        options=options,
    )

    if global_rank == 0:
        model_state_dict = _cast_float_tensors_to_bfloat16(model_state_dict)
        optim_state_dict = _cast_float_tensors_to_bfloat16(optim_state_dict)
        base = Path(filename)
        # Strip extension if provided (for backward compatibility)
        if base.suffix:
            base = base.with_suffix("")
        base.mkdir(parents=True, exist_ok=True)
        model_file = base / "model.pt"
        optim_file = base / "optimizer.pt"
        torch.save(model_state_dict, model_file)
        torch.save(optim_state_dict, optim_file)
        print(f"Saved checkpoint to {base} (model.pt, optimizer.pt)")


def save_cfg_yaml(saved_cfg: DictConfig, run_dir: Path | str) -> Path:
    """Save a structured config to cfg.yaml in run_dir."""
    run_path = Path(run_dir)
    cfg_path = run_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        OmegaConf.save(saved_cfg, f)
    return cfg_path
