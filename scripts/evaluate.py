#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mambaflow.losses import si_sdr
from mambaflow.trainer import (
    build_dataset,
    collate_dict,
    load_config,
    load_model_from_checkpoint,
    seed_everything,
    _parse_optional_int,
)


def waveform_sdr(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    reference = reference.float()
    estimate = estimate.float()

    if reference.ndim > 2:
        reference = reference.reshape(reference.shape[0], -1)
        estimate = estimate.reshape(estimate.shape[0], -1)

    noise = reference - estimate
    ref_energy = torch.sum(reference ** 2, dim=1)
    noise_energy = torch.sum(noise ** 2, dim=1)

    return 10.0 * torch.log10((ref_energy + eps) / (noise_energy + eps))



def mambavoice_sdr(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    reference = reference.float()
    estimate = estimate.float()

    if reference.ndim > 2:
        reference = reference.reshape(reference.shape[0], -1)
        estimate = estimate.reshape(estimate.shape[0], -1)

    reference = reference - reference.mean(dim=1, keepdim=True)
    estimate = estimate - estimate.mean(dim=1, keepdim=True)

    alpha = torch.sum(reference * estimate, dim=1, keepdim=True) / torch.sum(reference ** 2, dim=1, keepdim=True).clamp(min=eps)
    projection = alpha * reference
    noise = estimate - projection

    return 10.0 * torch.log10(
        torch.sum(projection ** 2, dim=1).clamp(min=eps)
        / torch.sum(noise ** 2, dim=1).clamp(min=eps)
    )


def torchmetrics_si_sdr(reference: torch.Tensor, estimate: torch.Tensor, zero_mean: bool = False) -> torch.Tensor:
    from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

    reference = reference.float()
    estimate = estimate.float()

    if reference.ndim == 1:
        reference = reference.unsqueeze(0)
        estimate = estimate.unsqueeze(0)

    if reference.ndim > 2:
        reference = reference.reshape(reference.shape[0], -1)
        estimate = estimate.reshape(estimate.shape[0], -1)

    return scale_invariant_signal_distortion_ratio(
        preds=estimate,
        target=reference,
        zero_mean=zero_mean,
    )


def fastbss_si_sdr(reference: torch.Tensor, estimate: torch.Tensor, zero_mean: bool = False) -> torch.Tensor:
    import fast_bss_eval

    reference = reference.float()
    estimate = estimate.float()

    if reference.ndim == 1:
        reference = reference.reshape(1, 1, -1)
        estimate = estimate.reshape(1, 1, -1)
    else:
        batch = reference.shape[0]
        reference = reference.reshape(batch, 1, -1)
        estimate = estimate.reshape(batch, 1, -1)

    score = fast_bss_eval.si_sdr(
        reference,
        estimate,
        zero_mean=zero_mean,
    )

    if isinstance(score, tuple):
        score = score[0]

    if not isinstance(score, torch.Tensor):
        score = torch.as_tensor(score, device=reference.device)

    return score.reshape(-1)

def fastbss_sdr(reference: torch.Tensor, estimate: torch.Tensor, zero_mean: bool = False, filter_length: int = 512) -> torch.Tensor:
    import fast_bss_eval

    reference = reference.float()
    estimate = estimate.float()

    if reference.ndim == 1:
        reference = reference.reshape(1, 1, -1)
        estimate = estimate.reshape(1, 1, -1)
    else:
        batch = reference.shape[0]
        reference = reference.reshape(batch, 1, -1)
        estimate = estimate.reshape(batch, 1, -1)

    score = fast_bss_eval.sdr(
        reference,
        estimate,
        filter_length=filter_length,
        zero_mean=zero_mean,
    )

    if isinstance(score, tuple):
        score = score[0]

    if not isinstance(score, torch.Tensor):
        score = torch.as_tensor(score, device=reference.device)

    return score.reshape(-1)

def torchmetrics_sdr(reference: torch.Tensor, estimate: torch.Tensor, zero_mean: bool = False, filter_length: int = 512) -> torch.Tensor:
    from torchmetrics.functional.audio import signal_distortion_ratio

    reference = reference.float()
    estimate = estimate.float()

    if reference.ndim == 1:
        reference = reference.reshape(1, 1, -1)
        estimate = estimate.reshape(1, 1, -1)
    else:
        batch = reference.shape[0]
        reference = reference.reshape(batch, 1, -1)
        estimate = estimate.reshape(batch, 1, -1)

    score = signal_distortion_ratio(
        preds=estimate,
        target=reference,
        filter_length=filter_length,
        zero_mean=zero_mean,
    )

    return score.reshape(-1)
def _alter_face(
    face: torch.Tensor | None,
    mode: str,
    donor: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if face is None or mode == "correct":
        return face
    if mode == "zero":
        return torch.zeros_like(face)
    if mode == "shift":
        # Shift temporal dimension. Handles [B,T,2,68], [B,2,T,68], [B,2,T,68,1].
        if face.ndim == 5:
            dim = 2 if face.shape[1] == 2 else 1
        elif face.ndim == 4:
            dim = 2 if face.shape[1] == 2 else 1
        else:
            dim = 1
        return torch.roll(face, shifts=max(1, face.shape[dim] // 2), dims=dim)
    if mode == "wrong":
        # A genuinely different speaker's landmarks. `donor` comes from another
        # item in the split.
        #
        # This used to fall back to a temporal shift whenever batch_size == 1 --
        # which is the default in many evaluation wrappers -- so the
        # "wrong video" ablation silently measured the same thing as "shift"
        # in every evaluation run so far.
        if donor is not None:
            d = donor.to(device=face.device, dtype=face.dtype)
            if d.shape == face.shape:
                return d
        if face.shape[0] > 1:
            return face[torch.roll(torch.arange(face.shape[0], device=face.device), 1)]
        raise RuntimeError(
            "visual_mode='wrong' needs a donor face from another sample. "
            "This is a bug: _wrong_face_donor should have supplied one."
        )
    raise ValueError(f"Unknown visual_mode={mode!r}")


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for the mean."""
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _compute_row(idx: int, mode: str, out: dict, batch: dict) -> dict:
    target = batch["target"]
    mixture = batch["mixture"]
    pred = out["target"]
    residual = out.get("residual", mixture - pred)
    l = min(pred.shape[-1], residual.shape[-1], target.shape[-1], mixture.shape[-1])
    pred_l = pred[..., :l]
    residual_l = residual[..., :l]
    target_l = target[..., :l]
    mixture_l = mixture[..., :l]

    sdr = si_sdr(target_l, pred_l).mean().item()
    sdr_res = si_sdr(target_l, residual_l).mean().item()
    sdr_mix = si_sdr(target_l, mixture_l).mean().item()
    wav_sdr = waveform_sdr(target_l, pred_l).mean().item()
    wav_sdr_res = waveform_sdr(target_l, residual_l).mean().item()
    wav_sdr_mix = waveform_sdr(target_l, mixture_l).mean().item()
    mv_sdr = mambavoice_sdr(target_l, pred_l).mean().item()
    mv_sdr_res = mambavoice_sdr(target_l, residual_l).mean().item()
    mv_sdr_mix = mambavoice_sdr(target_l, mixture_l).mean().item()
    tm_si_sdr = torchmetrics_si_sdr(target_l, pred_l, zero_mean=False).mean().item()
    tm_si_sdr_res = torchmetrics_si_sdr(target_l, residual_l, zero_mean=False).mean().item()
    tm_si_sdr_mix = torchmetrics_si_sdr(target_l, mixture_l, zero_mean=False).mean().item()
    tm_si_sdr_zm = torchmetrics_si_sdr(target_l, pred_l, zero_mean=True).mean().item()
    tm_si_sdr_zm_res = torchmetrics_si_sdr(target_l, residual_l, zero_mean=True).mean().item()
    tm_si_sdr_zm_mix = torchmetrics_si_sdr(target_l, mixture_l, zero_mean=True).mean().item()
    fb_si_sdr = fastbss_si_sdr(target_l, pred_l, zero_mean=False).mean().item()
    fb_si_sdr_res = fastbss_si_sdr(target_l, residual_l, zero_mean=False).mean().item()
    fb_si_sdr_mix = fastbss_si_sdr(target_l, mixture_l, zero_mean=False).mean().item()
    fb_si_sdr_zm = fastbss_si_sdr(target_l, pred_l, zero_mean=True).mean().item()
    fb_si_sdr_zm_res = fastbss_si_sdr(target_l, residual_l, zero_mean=True).mean().item()
    fb_si_sdr_zm_mix = fastbss_si_sdr(target_l, mixture_l, zero_mean=True).mean().item()
    fb_sdr = fastbss_sdr(target_l, pred_l, zero_mean=False, filter_length=512).mean().item()
    fb_sdr_res = fastbss_sdr(target_l, residual_l, zero_mean=False, filter_length=512).mean().item()
    fb_sdr_mix = fastbss_sdr(target_l, mixture_l, zero_mean=False, filter_length=512).mean().item()
    fb_sdr_zm = fastbss_sdr(target_l, pred_l, zero_mean=True, filter_length=512).mean().item()
    fb_sdr_zm_res = fastbss_sdr(target_l, residual_l, zero_mean=True, filter_length=512).mean().item()
    fb_sdr_zm_mix = fastbss_sdr(target_l, mixture_l, zero_mean=True, filter_length=512).mean().item()
    tm_sdr = torchmetrics_sdr(target_l, pred_l, zero_mean=False, filter_length=512).mean().item()
    tm_sdr_res = torchmetrics_sdr(target_l, residual_l, zero_mean=False, filter_length=512).mean().item()
    tm_sdr_mix = torchmetrics_sdr(target_l, mixture_l, zero_mean=False, filter_length=512).mean().item()
    tm_sdr_zm = torchmetrics_sdr(target_l, pred_l, zero_mean=True, filter_length=512).mean().item()
    tm_sdr_zm_res = torchmetrics_sdr(target_l, residual_l, zero_mean=True, filter_length=512).mean().item()
    tm_sdr_zm_mix = torchmetrics_sdr(target_l, mixture_l, zero_mean=True, filter_length=512).mean().item()
    target_energy = target_l.pow(2).sum(dim=-1) + 1e-8
    target_gain = ((pred_l * target_l).sum(dim=-1) / target_energy).mean().item()
    residual_gain = ((residual_l * target_l).sum(dim=-1) / target_energy).mean().item()
    residual_ref = mixture_l - target_l
    residual_ref_energy = residual_ref.pow(2).sum(dim=-1) + 1e-8
    interferer_gain_abs = ((pred_l * residual_ref).sum(dim=-1) / residual_ref_energy).abs().mean().item()
    pred_rms = pred_l.pow(2).mean(dim=-1).sqrt().mean().item()
    target_rms = target_l.pow(2).mean(dim=-1).sqrt().mean().item()
    pred_target_rms_ratio = pred_rms / (target_rms + 1e-8)
    return {
        "idx": idx,
        "visual_mode": mode,
        "si_sdr": sdr,
        "si_sdr_residual_to_target": sdr_res,
        "si_sdr_mixture_to_target": sdr_mix,
        "sdr": wav_sdr,
        "sdr_residual_to_target": wav_sdr_res,
        "sdr_mixture_to_target": wav_sdr_mix,
        "mambavoice_sdr": mv_sdr,
        "mambavoice_sdr_residual_to_target": mv_sdr_res,
        "mambavoice_sdr_mixture_to_target": mv_sdr_mix,
        "torchmetrics_si_sdr": tm_si_sdr,
        "torchmetrics_si_sdr_residual_to_target": tm_si_sdr_res,
        "torchmetrics_si_sdr_mixture_to_target": tm_si_sdr_mix,
        "torchmetrics_si_sdr_zm": tm_si_sdr_zm,
        "torchmetrics_si_sdr_zm_residual_to_target": tm_si_sdr_zm_res,
        "torchmetrics_si_sdr_zm_mixture_to_target": tm_si_sdr_zm_mix,
        "fastbss_si_sdr": fb_si_sdr,
        "fastbss_si_sdr_residual_to_target": fb_si_sdr_res,
        "fastbss_si_sdr_mixture_to_target": fb_si_sdr_mix,
        "fastbss_si_sdr_zm": fb_si_sdr_zm,
        "fastbss_si_sdr_zm_residual_to_target": fb_si_sdr_zm_res,
        "fastbss_si_sdr_zm_mixture_to_target": fb_si_sdr_zm_mix,
        "fastbss_sdr": fb_sdr,
        "fastbss_sdr_residual_to_target": fb_sdr_res,
        "fastbss_sdr_mixture_to_target": fb_sdr_mix,
        "fastbss_sdr_zm": fb_sdr_zm,
        "fastbss_sdr_zm_residual_to_target": fb_sdr_zm_res,
        "fastbss_sdr_zm_mixture_to_target": fb_sdr_zm_mix,
        "torchmetrics_sdr": tm_sdr,
        "torchmetrics_sdr_residual_to_target": tm_sdr_res,
        "torchmetrics_sdr_mixture_to_target": tm_sdr_mix,
        "torchmetrics_sdr_zm": tm_sdr_zm,
        "torchmetrics_sdr_zm_residual_to_target": tm_sdr_zm_res,
        "torchmetrics_sdr_zm_mixture_to_target": tm_sdr_zm_mix,
        "target_gain": target_gain,
        "residual_target_gain": residual_gain,
        "pred_interferer_gain_abs": interferer_gain_abs,
        "target_left_in_residual": float(sdr_res > sdr),
        "pred_rms": pred_rms,
        "target_rms": target_rms,
        "pred_target_rms_ratio": pred_target_rms_ratio,
        "mixture_abs_error": out["mixture_error"].abs().mean().item(),
    }


def _print_summary(rows: list[dict], mode: str) -> None:
    rows_m = [r for r in rows if r["visual_mode"] == mode]
    if not rows_m:
        return
    vals = np.array([r["si_sdr"] for r in rows_m], dtype=np.float64)
    residual_vals = np.array([r["si_sdr_residual_to_target"] for r in rows_m], dtype=np.float64)
    sdr_vals = np.array([r["sdr"] for r in rows_m], dtype=np.float64)
    sdr_residual_vals = np.array([r["sdr_residual_to_target"] for r in rows_m], dtype=np.float64)
    mv_sdr_vals = np.array([r["mambavoice_sdr"] for r in rows_m], dtype=np.float64)
    mv_sdr_residual_vals = np.array([r["mambavoice_sdr_residual_to_target"] for r in rows_m], dtype=np.float64)
    tm_si_sdr_vals = np.array([r["torchmetrics_si_sdr"] for r in rows_m], dtype=np.float64)
    tm_si_sdr_residual_vals = np.array([r["torchmetrics_si_sdr_residual_to_target"] for r in rows_m], dtype=np.float64)
    tm_si_sdr_zm_vals = np.array([r["torchmetrics_si_sdr_zm"] for r in rows_m], dtype=np.float64)
    tm_si_sdr_zm_residual_vals = np.array([r["torchmetrics_si_sdr_zm_residual_to_target"] for r in rows_m], dtype=np.float64)
    fb_si_sdr_vals = np.array([r["fastbss_si_sdr"] for r in rows_m], dtype=np.float64)
    fb_si_sdr_residual_vals = np.array([r["fastbss_si_sdr_residual_to_target"] for r in rows_m], dtype=np.float64)
    fb_si_sdr_zm_vals = np.array([r["fastbss_si_sdr_zm"] for r in rows_m], dtype=np.float64)
    fb_si_sdr_zm_residual_vals = np.array([r["fastbss_si_sdr_zm_residual_to_target"] for r in rows_m], dtype=np.float64)
    fb_sdr_vals = np.array([r["fastbss_sdr"] for r in rows_m], dtype=np.float64)
    fb_sdr_residual_vals = np.array([r["fastbss_sdr_residual_to_target"] for r in rows_m], dtype=np.float64)
    fb_sdr_zm_vals = np.array([r["fastbss_sdr_zm"] for r in rows_m], dtype=np.float64)
    fb_sdr_zm_residual_vals = np.array([r["fastbss_sdr_zm_residual_to_target"] for r in rows_m], dtype=np.float64)
    tm_sdr_vals = np.array([r["torchmetrics_sdr"] for r in rows_m], dtype=np.float64)
    tm_sdr_residual_vals = np.array([r["torchmetrics_sdr_residual_to_target"] for r in rows_m], dtype=np.float64)
    tm_sdr_zm_vals = np.array([r["torchmetrics_sdr_zm"] for r in rows_m], dtype=np.float64)
    tm_sdr_zm_residual_vals = np.array([r["torchmetrics_sdr_zm_residual_to_target"] for r in rows_m], dtype=np.float64)
    # ---- Headline block: improvement over the mixture, with uncertainty. ----
    # Absolute SI-SDR/SDR depends entirely on how the mixtures were built, so it
    # is not comparable to other papers. The improvement over the input mixture
    # is, and it is the number this literature reports.
    mix_si = np.array([r["si_sdr_mixture_to_target"] for r in rows_m], dtype=np.float64)
    fb_si_mix = np.array([r["fastbss_si_sdr_mixture_to_target"] for r in rows_m], dtype=np.float64)
    fb_sdr_mix_v = np.array([r["fastbss_sdr_mixture_to_target"] for r in rows_m], dtype=np.float64)
    fb_si = np.array([r["fastbss_si_sdr"] for r in rows_m], dtype=np.float64)
    fb_sdr_v = np.array([r["fastbss_sdr"] for r in rows_m], dtype=np.float64)
    n = len(rows_m)
    print(f"[{mode}] === headline (n={n}) ===")
    print(f"[{mode}] input_si_sdr_mean={mix_si.mean():.3f}  input_fastbss_si_sdr_mean={fb_si_mix.mean():.3f}")
    for label, est, ref in (
        ("si_sdri", vals, mix_si),
        ("fastbss_si_sdri", fb_si, fb_si_mix),
        ("fastbss_sdri", fb_sdr_v, fb_sdr_mix_v),
    ):
        imp = est - ref
        lo, hi = _bootstrap_ci(imp)
        print(
            f"[{mode}] {label}_mean={imp.mean():.3f} [95% CI {lo:.3f}, {hi:.3f}] "
            f"median={np.median(imp):.3f}"
        )
    lo, hi = _bootstrap_ci(vals)
    print(f"[{mode}] si_sdr_mean={vals.mean():.3f} [95% CI {lo:.3f}, {hi:.3f}]")
    lo, hi = _bootstrap_ci(fb_sdr_v)
    print(
        f"[{mode}] fastbss_sdr_mean={fb_sdr_v.mean():.3f} [95% CI {lo:.3f}, {hi:.3f}] "
        "(NOTE: BSS-Eval SDR with a 512-tap allowed distortion filter; "
        "systematically higher than SI-SDR and not interchangeable with it)"
    )
    print(f"[{mode}] === full metric dump ===")

    print(f"[{mode}] mean_si_sdr={vals.mean():.3f}")
    print(f"[{mode}] median_si_sdr={np.median(vals):.3f}")
    print(f"[{mode}] p10_si_sdr={np.percentile(vals, 10):.3f}")
    print(f"[{mode}] p90_si_sdr={np.percentile(vals, 90):.3f}")
    print(f"[{mode}] failure_rate_lt_0db={100.0 * np.mean(vals < 0):.2f}%")
    print(f"[{mode}] mean_sdr={sdr_vals.mean():.3f}")
    print(f"[{mode}] median_sdr={np.median(sdr_vals):.3f}")
    print(f"[{mode}] p10_sdr={np.percentile(sdr_vals, 10):.3f}")
    print(f"[{mode}] p90_sdr={np.percentile(sdr_vals, 90):.3f}")
    print(f"[{mode}] sdr_failure_rate_lt_0db={100.0 * np.mean(sdr_vals < 0):.2f}%")
    print(f"[{mode}] mean_mambavoice_sdr={mv_sdr_vals.mean():.3f}")
    print(f"[{mode}] median_mambavoice_sdr={np.median(mv_sdr_vals):.3f}")
    print(f"[{mode}] p10_mambavoice_sdr={np.percentile(mv_sdr_vals, 10):.3f}")
    print(f"[{mode}] p90_mambavoice_sdr={np.percentile(mv_sdr_vals, 90):.3f}")
    print(f"[{mode}] mambavoice_sdr_failure_rate_lt_0db={100.0 * np.mean(mv_sdr_vals < 0):.2f}%")
    print(f"[{mode}] mean_torchmetrics_si_sdr={tm_si_sdr_vals.mean():.3f}")
    print(f"[{mode}] median_torchmetrics_si_sdr={np.median(tm_si_sdr_vals):.3f}")
    print(f"[{mode}] p10_torchmetrics_si_sdr={np.percentile(tm_si_sdr_vals, 10):.3f}")
    print(f"[{mode}] p90_torchmetrics_si_sdr={np.percentile(tm_si_sdr_vals, 90):.3f}")
    print(f"[{mode}] torchmetrics_si_sdr_failure_rate_lt_0db={100.0 * np.mean(tm_si_sdr_vals < 0):.2f}%")
    print(f"[{mode}] mean_torchmetrics_si_sdr_zm={tm_si_sdr_zm_vals.mean():.3f}")
    print(f"[{mode}] median_torchmetrics_si_sdr_zm={np.median(tm_si_sdr_zm_vals):.3f}")
    print(f"[{mode}] p10_torchmetrics_si_sdr_zm={np.percentile(tm_si_sdr_zm_vals, 10):.3f}")
    print(f"[{mode}] p90_torchmetrics_si_sdr_zm={np.percentile(tm_si_sdr_zm_vals, 90):.3f}")
    print(f"[{mode}] torchmetrics_si_sdr_zm_failure_rate_lt_0db={100.0 * np.mean(tm_si_sdr_zm_vals < 0):.2f}%")
    print(f"[{mode}] mean_fastbss_si_sdr={fb_si_sdr_vals.mean():.3f}")
    print(f"[{mode}] median_fastbss_si_sdr={np.median(fb_si_sdr_vals):.3f}")
    print(f"[{mode}] p10_fastbss_si_sdr={np.percentile(fb_si_sdr_vals, 10):.3f}")
    print(f"[{mode}] p90_fastbss_si_sdr={np.percentile(fb_si_sdr_vals, 90):.3f}")
    print(f"[{mode}] fastbss_si_sdr_failure_rate_lt_0db={100.0 * np.mean(fb_si_sdr_vals < 0):.2f}%")
    print(f"[{mode}] mean_fastbss_si_sdr_zm={fb_si_sdr_zm_vals.mean():.3f}")
    print(f"[{mode}] median_fastbss_si_sdr_zm={np.median(fb_si_sdr_zm_vals):.3f}")
    print(f"[{mode}] p10_fastbss_si_sdr_zm={np.percentile(fb_si_sdr_zm_vals, 10):.3f}")
    print(f"[{mode}] p90_fastbss_si_sdr_zm={np.percentile(fb_si_sdr_zm_vals, 90):.3f}")
    print(f"[{mode}] fastbss_si_sdr_zm_failure_rate_lt_0db={100.0 * np.mean(fb_si_sdr_zm_vals < 0):.2f}%")
    print(f"[{mode}] mean_fastbss_sdr={fb_sdr_vals.mean():.3f}")
    print(f"[{mode}] median_fastbss_sdr={np.median(fb_sdr_vals):.3f}")
    print(f"[{mode}] p10_fastbss_sdr={np.percentile(fb_sdr_vals, 10):.3f}")
    print(f"[{mode}] p90_fastbss_sdr={np.percentile(fb_sdr_vals, 90):.3f}")
    print(f"[{mode}] fastbss_sdr_failure_rate_lt_0db={100.0 * np.mean(fb_sdr_vals < 0):.2f}%")
    print(f"[{mode}] mean_fastbss_sdr_zm={fb_sdr_zm_vals.mean():.3f}")
    print(f"[{mode}] median_fastbss_sdr_zm={np.median(fb_sdr_zm_vals):.3f}")
    print(f"[{mode}] p10_fastbss_sdr_zm={np.percentile(fb_sdr_zm_vals, 10):.3f}")
    print(f"[{mode}] p90_fastbss_sdr_zm={np.percentile(fb_sdr_zm_vals, 90):.3f}")
    print(f"[{mode}] fastbss_sdr_zm_failure_rate_lt_0db={100.0 * np.mean(fb_sdr_zm_vals < 0):.2f}%")
    print(f"[{mode}] mean_torchmetrics_sdr={tm_sdr_vals.mean():.3f}")
    print(f"[{mode}] median_torchmetrics_sdr={np.median(tm_sdr_vals):.3f}")
    print(f"[{mode}] p10_torchmetrics_sdr={np.percentile(tm_sdr_vals, 10):.3f}")
    print(f"[{mode}] p90_torchmetrics_sdr={np.percentile(tm_sdr_vals, 90):.3f}")
    print(f"[{mode}] torchmetrics_sdr_failure_rate_lt_0db={100.0 * np.mean(tm_sdr_vals < 0):.2f}%")
    print(f"[{mode}] mean_torchmetrics_sdr_zm={tm_sdr_zm_vals.mean():.3f}")
    print(f"[{mode}] median_torchmetrics_sdr_zm={np.median(tm_sdr_zm_vals):.3f}")
    print(f"[{mode}] p10_torchmetrics_sdr_zm={np.percentile(tm_sdr_zm_vals, 10):.3f}")
    print(f"[{mode}] p90_torchmetrics_sdr_zm={np.percentile(tm_sdr_zm_vals, 90):.3f}")
    print(f"[{mode}] torchmetrics_sdr_zm_failure_rate_lt_0db={100.0 * np.mean(tm_sdr_zm_vals < 0):.2f}%")
    print(f"[{mode}] mean_residual_to_target_si_sdr={residual_vals.mean():.3f}")
    print(f"[{mode}] mean_residual_to_target_sdr={sdr_residual_vals.mean():.3f}")
    print(f"[{mode}] mean_residual_to_target_mambavoice_sdr={mv_sdr_residual_vals.mean():.3f}")
    print(f"[{mode}] mean_residual_to_target_torchmetrics_si_sdr={tm_si_sdr_residual_vals.mean():.3f}")
    print(f"[{mode}] mean_residual_to_target_torchmetrics_si_sdr_zm={tm_si_sdr_zm_residual_vals.mean():.3f}")
    print(f"[{mode}] mean_residual_to_target_fastbss_si_sdr={fb_si_sdr_residual_vals.mean():.3f}")
    print(f"[{mode}] mean_residual_to_target_fastbss_si_sdr_zm={fb_si_sdr_zm_residual_vals.mean():.3f}")
    print(f"[{mode}] mean_residual_to_target_fastbss_sdr={fb_sdr_residual_vals.mean():.3f}")
    print(f"[{mode}] mean_residual_to_target_fastbss_sdr_zm={fb_sdr_zm_residual_vals.mean():.3f}")
    print(f"[{mode}] mean_residual_to_target_torchmetrics_sdr={tm_sdr_residual_vals.mean():.3f}")
    print(f"[{mode}] mean_residual_to_target_torchmetrics_sdr_zm={tm_sdr_zm_residual_vals.mean():.3f}")
    print(f"[{mode}] target_left_in_residual_rate={100.0 * np.mean(residual_vals > vals):.2f}%")
    print(f"[{mode}] mean_target_gain={np.mean([r['target_gain'] for r in rows_m]):.3f}")
    print(f"[{mode}] mean_residual_target_gain={np.mean([r['residual_target_gain'] for r in rows_m]):.3f}")
    print(f"[{mode}] mean_pred_interferer_gain_abs={np.mean([r['pred_interferer_gain_abs'] for r in rows_m]):.3f}")
    print(f"[{mode}] mean_pred_target_rms_ratio={np.mean([r['pred_target_rms_ratio'] for r in rows_m]):.3f}")
    print(f"[{mode}] mean_mixture_abs_error={np.mean([r['mixture_abs_error'] for r in rows_m]):.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--num_batches",
        default=None,
        help="Number of validation batches to evaluate. Default/null/all/full evaluates the whole split.",
    )
    ap.add_argument("--num_steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--visual_mode", default="correct", choices=["correct", "zero", "shift", "wrong", "all"])
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--use_ema", action="store_true", help="Load EMA weights from checkpoint if present.")
    ap.add_argument("--no_progress", action="store_true")
    ap.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed for torch/numpy/random. For a fully fixed test manifest also set "
             "data.val.init_kwargs.deterministic=true in the config.",
    )
    args = ap.parse_args()

    seed_everything(args.seed)
    num_batches = _parse_optional_int(args.num_batches)
    cfg = load_config(args.config)

    val_kwargs = cfg.get("data", {}).get("val", {}).get("init_kwargs", {})
    if not val_kwargs.get("deterministic", False):
        print(
            "WARNING: data.val.init_kwargs.deterministic is not set. The dataset "
            "re-draws the interferer, the accompaniment clip and both SNRs on "
            "every pass, so repeat evaluations of the SAME checkpoint will differ "
            "(~0.2-0.4 dB observed). Set deterministic: true for a fixed manifest."
        )
    device = torch.device(cfg.get("training", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    ds = build_dataset(cfg, "val") if cfg.get("data", {}).get("kind") == "vovit" else build_dataset(cfg, "train")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_dict)
    model = load_model_from_checkpoint(args.checkpoint, map_location=device, use_ema=args.use_ema).to(device).eval()

    modes = ["correct", "zero", "shift", "wrong"] if args.visual_mode == "all" else [args.visual_mode]
    rows = []
    total = len(dl)
    if num_batches is not None:
        total = min(total, num_batches)
    iterator = tqdm(dl, total=total, desc="evaluate", disable=args.no_progress)

    needs_donor = "wrong" in modes

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            if num_batches is not None and i >= num_batches:
                break
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            donor = None
            if needs_donor and batch.get("face") is not None:
                # Deterministic donor: the face of a fixed offset into the split,
                # so the "wrong video" condition is reproducible across runs and
                # across checkpoints.
                donor_idx = (i + len(ds) // 2) % len(ds)
                donor_item = ds[donor_idx]
                donor_face = donor_item.get("face") if isinstance(donor_item, dict) else None
                if donor_face is not None:
                    donor = donor_face.unsqueeze(0).expand_as(batch["face"]).contiguous()
            for mode in modes:
                face = _alter_face(batch.get("face"), mode, donor=donor)
                out = model.separate(batch["mixture"], face=face, body=batch.get("body"), num_steps=args.num_steps)
                row = _compute_row(i, mode, out, batch)
                rows.append(row)
                tqdm.write(str(row))

    for mode in modes:
        _print_summary(rows, mode)

    if "correct" in modes:
        correct = [r for r in rows if r["visual_mode"] == "correct"]
        for mode in [m for m in modes if m != "correct"]:
            other = [r for r in rows if r["visual_mode"] == mode]
            if len(correct) == len(other) and correct:
                delta = np.array([c["si_sdr"] - o["si_sdr"] for c, o in zip(correct, other)])
                lo, hi = _bootstrap_ci(delta)
                print(
                    f"[correct-{mode}] si_sdr_delta_mean={delta.mean():.3f} "
                    f"[95% CI {lo:.3f}, {hi:.3f}]  (paired, n={len(delta)})"
                )
                delta_sdr = np.array([c["sdr"] - o["sdr"] for c, o in zip(correct, other)])
                delta_mv_sdr = np.array([c["mambavoice_sdr"] - o["mambavoice_sdr"] for c, o in zip(correct, other)])
                delta_tm_si_sdr = np.array([c["torchmetrics_si_sdr"] - o["torchmetrics_si_sdr"] for c, o in zip(correct, other)])
                delta_tm_si_sdr_zm = np.array([c["torchmetrics_si_sdr_zm"] - o["torchmetrics_si_sdr_zm"] for c, o in zip(correct, other)])
                delta_fb_si_sdr = np.array([c["fastbss_si_sdr"] - o["fastbss_si_sdr"] for c, o in zip(correct, other)])
                delta_fb_si_sdr_zm = np.array([c["fastbss_si_sdr_zm"] - o["fastbss_si_sdr_zm"] for c, o in zip(correct, other)])
                delta_fb_sdr = np.array([c["fastbss_sdr"] - o["fastbss_sdr"] for c, o in zip(correct, other)])
                delta_fb_sdr_zm = np.array([c["fastbss_sdr_zm"] - o["fastbss_sdr_zm"] for c, o in zip(correct, other)])
                delta_tm_sdr = np.array([c["torchmetrics_sdr"] - o["torchmetrics_sdr"] for c, o in zip(correct, other)])
                delta_tm_sdr_zm = np.array([c["torchmetrics_sdr_zm"] - o["torchmetrics_sdr_zm"] for c, o in zip(correct, other)])
                print(f"[correct-{mode}] mean_delta={delta.mean():.3f}")
                print(f"[correct>{mode}] rate={100.0 * np.mean(delta > 0):.2f}%")
                print(f"[correct>{mode}+3dB] rate={100.0 * np.mean(delta > 3):.2f}%")
                print(f"[correct-{mode}] mean_sdr_delta={delta_sdr.mean():.3f}")
                print(f"[correct>{mode}_sdr] rate={100.0 * np.mean(delta_sdr > 0):.2f}%")
                print(f"[correct>{mode}_sdr+3dB] rate={100.0 * np.mean(delta_sdr > 3):.2f}%")
                print(f"[correct-{mode}] mean_mambavoice_sdr_delta={delta_mv_sdr.mean():.3f}")
                print(f"[correct>{mode}_mambavoice_sdr] rate={100.0 * np.mean(delta_mv_sdr > 0):.2f}%")
                print(f"[correct>{mode}_mambavoice_sdr+3dB] rate={100.0 * np.mean(delta_mv_sdr > 3):.2f}%")
                print(f"[correct-{mode}] mean_torchmetrics_si_sdr_delta={delta_tm_si_sdr.mean():.3f}")
                print(f"[correct>{mode}_torchmetrics_si_sdr] rate={100.0 * np.mean(delta_tm_si_sdr > 0):.2f}%")
                print(f"[correct>{mode}_torchmetrics_si_sdr+3dB] rate={100.0 * np.mean(delta_tm_si_sdr > 3):.2f}%")
                print(f"[correct-{mode}] mean_torchmetrics_si_sdr_zm_delta={delta_tm_si_sdr_zm.mean():.3f}")
                print(f"[correct>{mode}_torchmetrics_si_sdr_zm] rate={100.0 * np.mean(delta_tm_si_sdr_zm > 0):.2f}%")
                print(f"[correct>{mode}_torchmetrics_si_sdr_zm+3dB] rate={100.0 * np.mean(delta_tm_si_sdr_zm > 3):.2f}%")
                print(f"[correct-{mode}] mean_fastbss_si_sdr_delta={delta_fb_si_sdr.mean():.3f}")
                print(f"[correct>{mode}_fastbss_si_sdr] rate={100.0 * np.mean(delta_fb_si_sdr > 0):.2f}%")
                print(f"[correct>{mode}_fastbss_si_sdr+3dB] rate={100.0 * np.mean(delta_fb_si_sdr > 3):.2f}%")
                print(f"[correct-{mode}] mean_fastbss_si_sdr_zm_delta={delta_fb_si_sdr_zm.mean():.3f}")
                print(f"[correct>{mode}_fastbss_si_sdr_zm] rate={100.0 * np.mean(delta_fb_si_sdr_zm > 0):.2f}%")
                print(f"[correct>{mode}_fastbss_si_sdr_zm+3dB] rate={100.0 * np.mean(delta_fb_si_sdr_zm > 3):.2f}%")
                print(f"[correct-{mode}] mean_fastbss_sdr_delta={delta_fb_sdr.mean():.3f}")
                print(f"[correct>{mode}_fastbss_sdr] rate={100.0 * np.mean(delta_fb_sdr > 0):.2f}%")
                print(f"[correct>{mode}_fastbss_sdr+3dB] rate={100.0 * np.mean(delta_fb_sdr > 3):.2f}%")
                print(f"[correct-{mode}] mean_fastbss_sdr_zm_delta={delta_fb_sdr_zm.mean():.3f}")
                print(f"[correct>{mode}_fastbss_sdr_zm] rate={100.0 * np.mean(delta_fb_sdr_zm > 0):.2f}%")
                print(f"[correct>{mode}_fastbss_sdr_zm+3dB] rate={100.0 * np.mean(delta_fb_sdr_zm > 3):.2f}%")
                print(f"[correct-{mode}] mean_torchmetrics_sdr_delta={delta_tm_sdr.mean():.3f}")
                print(f"[correct>{mode}_torchmetrics_sdr] rate={100.0 * np.mean(delta_tm_sdr > 0):.2f}%")
                print(f"[correct>{mode}_torchmetrics_sdr+3dB] rate={100.0 * np.mean(delta_tm_sdr > 3):.2f}%")
                print(f"[correct-{mode}] mean_torchmetrics_sdr_zm_delta={delta_tm_sdr_zm.mean():.3f}")
                print(f"[correct>{mode}_torchmetrics_sdr_zm] rate={100.0 * np.mean(delta_tm_sdr_zm > 0):.2f}%")
                print(f"[correct>{mode}_torchmetrics_sdr_zm+3dB] rate={100.0 * np.mean(delta_tm_sdr_zm > 3):.2f}%")

    if args.out_csv:
        path = Path(args.out_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["idx"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
