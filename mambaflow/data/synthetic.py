from __future__ import annotations

import math
from typing import Dict

import torch
from torch.utils.data import Dataset


class SyntheticAcapellaDataset(Dataset):
    """Tiny debug dataset with target+interferer sine mixtures and fake face landmarks."""

    def __init__(self, length: int = 64, samples: int = 16384 * 2, video_frames: int = 100, seed: int = 0) -> None:
        self.length = length
        self.samples = samples
        self.video_frames = video_frames
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(self.seed + idx)
        t = torch.linspace(0, 1, self.samples)
        f0 = 120 + 80 * torch.rand((), generator=g)
        f1 = 200 + 120 * torch.rand((), generator=g)
        vibrato = 0.02 * torch.sin(2 * math.pi * 5 * t)
        target = 0.35 * torch.sin(2 * math.pi * f0 * t + vibrato)
        interferer = 0.30 * torch.sin(2 * math.pi * f1 * t + 0.3)
        mixture = target + interferer + 0.005 * torch.randn(self.samples, generator=g)
        face = torch.randn(self.video_frames, 3, 68, generator=g) * 0.05
        # encode weak mouth motion correlated with target amplitude envelope
        env = target.abs().unfold(0, max(1, self.samples // self.video_frames), max(1, self.samples // self.video_frames)).mean(-1)
        env = torch.nn.functional.interpolate(env.view(1, 1, -1), size=self.video_frames, mode="linear", align_corners=False).view(-1)
        face[:, 1, :20] += env[:, None]
        return {"mixture": mixture, "target": target, "face": face}
