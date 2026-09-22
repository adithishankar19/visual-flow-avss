from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    if half == 0:
        return t[:, None]
    freqs = torch.exp(torch.linspace(0, math.log(10000), half, device=t.device, dtype=t.dtype) * -1)
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class FiLMResidualBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int, dilation: int = 1) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8 if channels >= 8 else 1, channels)
        self.conv1 = nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(8 if channels >= 8 else 1, channels)
        self.conv2 = nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation)
        self.film = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        h = self.conv1(F.silu(self.norm1(x)))
        h = h * (1 + gamma) + beta
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class WaveformFlowHead(nn.Module):
    """Small waveform-domain velocity head conditioned on MambaVoice encoders.

    Inputs:
      x_t: [B, 2, L] current target/residual state
      mixture: [B, L]
      cond: [B, D] MambaVoice audio+video conditioning
      t: [B] flow time
    Output:
      velocity: [B, 2, L]
    """

    def __init__(self, cond_dim: int = 256, hidden: int = 128, depth: int = 8, time_dim: int = 128) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim + time_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )
        self.in_conv = nn.Conv1d(3, hidden, 7, padding=3)
        dilations = [2 ** (i % 6) for i in range(depth)]
        self.blocks = nn.ModuleList([FiLMResidualBlock(hidden, cond_dim, d) for d in dilations])
        self.out_norm = nn.GroupNorm(8 if hidden >= 8 else 1, hidden)
        self.out_conv = nn.Conv1d(hidden, 2, 7, padding=3)

    def forward(self, x_t: torch.Tensor, mixture: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if mixture.ndim == 2:
            mixture = mixture.unsqueeze(1)
        if mixture.shape[-1] != x_t.shape[-1]:
            l = min(mixture.shape[-1], x_t.shape[-1])
            mixture = mixture[..., :l]
            x_t = x_t[..., :l]
        temb = sinusoidal_embedding(t, self.time_dim)
        c = self.cond_proj(torch.cat([cond, temb], dim=-1))
        h = self.in_conv(torch.cat([x_t, mixture], dim=1))
        for block in self.blocks:
            h = block(h, c)
        return self.out_conv(F.silu(self.out_norm(h)))
