from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class STFTConfig:
    n_fft: int = 1024
    hop_length: int = 256
    win_length: Optional[int] = None
    center: bool = True
    normalized: bool = False

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "STFTConfig":
        cfg = dict(cfg or {})
        return cls(
            n_fft=int(cfg.get("n_fft", 1024)),
            hop_length=int(cfg.get("hop_length", 256)),
            win_length=None if cfg.get("win_length", None) is None else int(cfg.get("win_length")),
            center=bool(cfg.get("center", True)),
            normalized=bool(cfg.get("normalized", False)),
        )

    @property
    def window_length(self) -> int:
        return int(self.win_length or self.n_fft)


def _window(cfg: STFTConfig, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.hann_window(cfg.window_length, device=device, dtype=dtype)


def stft_waveform(wave: torch.Tensor, cfg: STFTConfig) -> torch.Tensor:
    """Convert waveform [B,L] or [B,S,L] to complex STFT [...,F,T]."""
    if wave.ndim not in {2, 3}:
        raise ValueError(f"Expected waveform [B,L] or [B,S,L], got {tuple(wave.shape)}")
    orig_shape = wave.shape[:-1]
    flat = wave.reshape(-1, wave.shape[-1])
    win = _window(cfg, device=flat.device, dtype=flat.dtype)
    spec = torch.stft(
        flat,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.window_length,
        window=win,
        center=cfg.center,
        normalized=cfg.normalized,
        return_complex=True,
    )
    return spec.reshape(*orig_shape, spec.shape[-2], spec.shape[-1])


def istft_waveform(spec: torch.Tensor, cfg: STFTConfig, *, length: int) -> torch.Tensor:
    """Convert complex STFT [...,F,T] to waveform [...,L]."""
    if not spec.is_complex():
        raise ValueError("istft_waveform expects a complex tensor")
    orig_shape = spec.shape[:-2]
    flat = spec.reshape(-1, spec.shape[-2], spec.shape[-1])
    # STFT real/imag channels may occasionally become non-contiguous after view_as_complex.
    flat = flat.contiguous()
    dtype = torch.float32 if flat.dtype == torch.complex64 else torch.float64
    win = _window(cfg, device=flat.device, dtype=dtype)
    wave = torch.istft(
        flat,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.window_length,
        window=win,
        center=cfg.center,
        normalized=cfg.normalized,
        length=int(length),
    )
    return wave.reshape(*orig_shape, wave.shape[-1])


def complex_to_ri(x: torch.Tensor) -> torch.Tensor:
    """Complex [...,F,T] -> real/imag channels [...,2,F,T]."""
    if not x.is_complex():
        raise ValueError("complex_to_ri expects complex input")
    return torch.view_as_real(x).movedim(-1, -3).contiguous()


def ri_to_complex(x: torch.Tensor) -> torch.Tensor:
    """Real/imag channels [...,2,F,T] -> complex [...,F,T]."""
    if x.shape[-3] != 2:
        raise ValueError(f"Expected real/imag channel dimension of 2 at -3, got {tuple(x.shape)}")
    y = x.movedim(-3, -1).contiguous()
    return torch.view_as_complex(y)
