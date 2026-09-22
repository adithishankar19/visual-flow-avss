from __future__ import annotations

import torch


def project_sources_to_mixture(sources: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
    """Project [B, S, L] sources so they sum exactly to mixture [B, L] or [B,1,L]."""
    if mixture.ndim == 2:
        mixture = mixture.unsqueeze(1)
    if sources.ndim != 3:
        raise ValueError(f"Expected sources [B,S,L], got {tuple(sources.shape)}")
    if mixture.shape[-1] != sources.shape[-1]:
        l = min(mixture.shape[-1], sources.shape[-1])
        mixture = mixture[..., :l]
        sources = sources[..., :l]
    err = mixture - sources.sum(dim=1, keepdim=True)
    return sources + err / sources.shape[1]


def project_velocity_zero_sum(velocity: torch.Tensor) -> torch.Tensor:
    """Project velocity so source-channel sum is exactly zero."""
    return velocity - velocity.mean(dim=1, keepdim=True)
