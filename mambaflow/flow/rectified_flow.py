from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .mixture_consistency import project_sources_to_mixture, project_velocity_zero_sum


@dataclass
class FlowConfig:
    consistency: str = "every_step"  # none | final | every_step
    noise_scale: float = 1.0
    num_steps: int = 8


def sample_training_tuple(target_sources: torch.Tensor, mixture: torch.Tensor, noise_scale: float = 1.0):
    """Return x_t, t, target velocity for rectified flow in mixture-consistent subspace.

    target_sources: [B,2,L] where channels are [target, residual].
    mixture: [B,L]
    """
    z = torch.randn_like(target_sources) * noise_scale
    z = project_sources_to_mixture(z, mixture)
    y = project_sources_to_mixture(target_sources, mixture)
    b = y.shape[0]
    t = torch.rand(b, device=y.device, dtype=y.dtype)
    shape = (b,) + (1,) * (y.ndim - 1)
    xt = (1.0 - t.view(shape)) * z + t.view(shape) * y
    v = y - z
    v = project_velocity_zero_sum(v)
    return xt, t, v


@torch.no_grad()
def euler_sample(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    mixture: torch.Tensor,
    shape: torch.Size,
    cfg: FlowConfig,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    device = device or mixture.device
    dtype = dtype or mixture.dtype
    x = torch.randn(shape, device=device, dtype=dtype) * cfg.noise_scale
    if cfg.consistency in {"final", "every_step"}:
        x = project_sources_to_mixture(x, mixture)
    n = int(cfg.num_steps)
    for i in range(n):
        t = torch.full((shape[0],), i / max(n, 1), device=device, dtype=dtype)
        v = velocity_fn(x, t)
        if cfg.consistency in {"final", "every_step"}:
            v = project_velocity_zero_sum(v)
        x = x + v / max(n, 1)
        if cfg.consistency == "every_step":
            x = project_sources_to_mixture(x, mixture)
    if cfg.consistency in {"final", "every_step"}:
        x = project_sources_to_mixture(x, mixture)
    return x
