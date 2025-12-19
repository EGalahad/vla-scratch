import torch
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    StateDictOptions,
    set_model_state_dict,
    set_optimizer_state_dict,
)


def find_latest_checkpoint(path: Path | str, desired_iter: Optional[int] = None) -> Optional[Path]:
    """Resolve a checkpoint path to a concrete checkpoint location.

    Supports both legacy single-file checkpoints (checkpoint_*.pth) and the
    new directory style (checkpoint_*/model.pt, optimizer.pt).

    Returns:
    - If `path` is a file, returns it (legacy compatible).
    - If `path` is a checkpoint directory (contains model.pt), returns the dir.
    - If `path` is a run directory, returns the newest checkpoint directory
      if available, otherwise the newest legacy .pth file.
    - None if nothing is found.
    """
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

    dir_candidates = [d for d in p.glob("checkpoint_*") if d.is_dir() and (d / "model.pt").exists()]
    if dir_candidates:
        def _score(d: Path) -> int:
            ep = _epoch_num_from_name(d.name)
            if desired_iter is not None and ep == desired_iter:
                return 10**9
            return ep
        dir_candidates.sort(key=_score)
        return dir_candidates[-1]

    # Fallback: legacy single-file checkpoints
    file_candidates = sorted(p.glob("checkpoint_*.pth"))
    if not file_candidates:
        return None

    def _score_file(cp: Path) -> int:
        stem = cp.stem  # checkpoint_X
        try:
            epoch = int(stem.split("_")[-1])
            if desired_iter is not None and epoch == desired_iter:
                return 10**9
            return epoch
        except Exception:
            return -1

    file_candidates.sort(key=_score_file)
    return file_candidates[-1]


def load_model_from_checkpoint(
    model: torch.nn.Module,
    path: Path | str,
    device: torch.device | str = "cpu",
    *,
    strict: bool = False,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Load a checkpoint into `model` supporting both dir and file formats.

    - New format: `path` is a checkpoint directory containing `model.pt` saved
      as a full model state_dict (from FSDP get_state_dict or plain).
    - Legacy: `path` is a single file containing either a raw state_dict or a
      dict with key "model".
    Returns (missing_keys, unexpected_keys).
    """
    p = Path(path)
    if p.is_dir():
        p = p / "model.pt"
    ckpt = torch.load(p, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model_state = ckpt["model"]
    else:
        model_state = ckpt
    missing, unexpected = model.load_state_dict(model_state, strict=strict)
    return tuple(missing), tuple(unexpected)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint: str | Path,
    global_rank: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Distributed-aware checkpoint load supporting dir and file formats.

    - New format: `checkpoint` is a directory with `model.pt` and `optimizer.pt`.
    - Legacy: `checkpoint` is a single .pth file with keys {"model","optimizer"}
      or a raw model state_dict.
    Returns (missing_keys, unexpected_keys) from set_model_state_dict.
    """
    p = Path(checkpoint)
    # Only rank 0 reads from disk; others pass empty dicts and receive via broadcast
    if global_rank == 0:
        model_sd: Dict[str, Any] = {}
        optim_sd: Dict[str, Any] = {}
        if p.is_dir():
            mp = p / "model.pt"
            op = p / "optimizer.pt"
            if mp.exists():
                model_sd = torch.load(mp, map_location="cpu", mmap=True, weights_only=False)
            if optimizer is not None and op.exists():
                optim_sd = torch.load(op, map_location="cpu", mmap=True, weights_only=False)
        else:
            # Legacy single-file checkpoint
            state = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
            if isinstance(state, dict):
                model_sd = state.get("model", state)
                if optimizer is not None:
                    optim_sd = state.get("optimizer", {})
            else:
                model_sd = state
    else:
        model_sd = {}
        optim_sd = {}

    options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=False)
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
                first_params = groups[0].get("params", []) if isinstance(groups[0], dict) else []
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
            return obj.to(dtype=torch.bfloat16) if obj.is_floating_point() else obj
        if isinstance(obj, dict):
            return {k: _cast_float_tensors_to_bfloat16(v) for k, v in obj.items()}
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
