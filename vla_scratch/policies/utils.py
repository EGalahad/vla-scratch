import torch
from typing import Iterable
import torch.distributed as dist
from torch.distributed.distributed_c10d import ProcessGroup
import jaxtyping as at


def get_beta_dist(
    alpha: float, beta: float, device
) -> torch.distributions.Distribution:
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(beta_t, alpha_t)
    return dist


def sample_time(
    time_dist: torch.distributions.Distribution, bsize: torch.Size
) -> at.Float[torch.Tensor, "b"]:
    return time_dist.sample(bsize) * 0.999 + 0.001


def sample_noise(shape, device, dtype):
    return torch.normal(
        mean=0.0,
        std=1.0,
        size=shape,
        dtype=dtype,
        device=device,
    )


@torch.compile
@torch.no_grad()
def clip_grad_norm_(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pg: ProcessGroup | None = None,
) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)

    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(
        grads, norm_type, error_if_nonfinite, foreach
    )
    if pg is not None:
        total_norm **= norm_type
        dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pg)
        total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm
