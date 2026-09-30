from __future__ import annotations

import torch
import torch.nn.functional as F


def si_sdr(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    reference = reference - reference.mean(dim=-1, keepdim=True)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    scale = (estimate * reference).sum(dim=-1, keepdim=True) / (reference.pow(2).sum(dim=-1, keepdim=True) + eps)
    target = scale * reference
    noise = estimate - target
    return 10 * torch.log10((target.pow(2).sum(dim=-1) + eps) / (noise.pow(2).sum(dim=-1) + eps))


_MRSTFT_WINDOWS: dict[tuple, torch.Tensor] = {}


def _mrstft_window(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (n, device, dtype)
    win = _MRSTFT_WINDOWS.get(key)
    if win is None:
        win = torch.hann_window(n, device=device, dtype=dtype)
        _MRSTFT_WINDOWS[key] = win
    return win


def multi_resolution_stft_loss(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    *,
    fft_sizes: tuple[int, ...] = (2048, 1024, 512, 256),
    hop_ratio: float = 0.25,
    log_eps: float = 1e-5,
) -> torch.Tensor:
    """Multi-resolution STFT loss L_MR: spectral convergence + log magnitude.

    The log-magnitude term weights all frequency bins roughly equally, and the
    spectral-convergence term keeps the loud bins accurate.  Both are
    scale-sensitive, so neither can be satisfied by a quiet but well-correlated
    prediction.

    estimate, reference: [B, L] waveforms.
    """
    if estimate.shape != reference.shape:
        raise ValueError(
            f"estimate {tuple(estimate.shape)} and reference {tuple(reference.shape)} must match"
        )
    est = estimate.reshape(-1, estimate.shape[-1]).float()
    ref = reference.reshape(-1, reference.shape[-1]).float()

    total = est.new_zeros(())
    for n_fft in fft_sizes:
        if est.shape[-1] < n_fft:
            continue
        hop = max(1, int(n_fft * hop_ratio))
        win = _mrstft_window(n_fft, est.device, est.dtype)
        kw = dict(n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win,
                  center=True, return_complex=True)
        mag_e = torch.stft(est, **kw).abs()
        mag_r = torch.stft(ref, **kw).abs()
        # Spectral convergence: relative Frobenius error, dominated by loud bins.
        sc = torch.linalg.vector_norm(mag_r - mag_e, dim=(-2, -1)) / (
            torch.linalg.vector_norm(mag_r, dim=(-2, -1)) + log_eps
        )
        # Log magnitude: roughly uniform weight per bin across the spectrum.
        lm = F.l1_loss(
            torch.log(mag_e + log_eps), torch.log(mag_r + log_eps)
        )
        total = total + sc.mean() + lm
    return total / max(1, len(fft_sizes))
