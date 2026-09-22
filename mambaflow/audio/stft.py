from __future__ import annotations

import torch


def multires_stft_distance(x: torch.Tensor, y: torch.Tensor, fft_sizes=(512, 1024, 2048), hop_ratio: float = 0.25) -> torch.Tensor:
    """Optional perceptual diagnostic loss for waveform estimates."""
    loss = x.new_tensor(0.0)
    for n_fft in fft_sizes:
        hop = int(n_fft * hop_ratio)
        win = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
        X = torch.stft(x, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
        Y = torch.stft(y, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
        loss = loss + (X.abs().log1p() - Y.abs().log1p()).abs().mean()
    return loss / len(fft_sizes)
