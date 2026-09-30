#!/usr/bin/env python
"""Evaluate VIST with the metrics of the paper (Secs. 4.2 and 5.3).

For every mixture:
  - BSS-Eval SDR and SIR of the target estimate, with 512-tap distortion
    filters and the target and the rest of the mixture as references, after
    resampling to 16 kHz;
  - SI-SDR of the target estimate;
  - target swapping: the remainder M - S_hat is closer to the target than
    S_hat, in SI-SDR.

Visual interventions (Table 3) are selected with --visual_mode: the correct
landmarks, all-zero landmarks, the landmarks shifted in time by half the clip,
or the landmarks of a singer from another test clip.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import fast_bss_eval
import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vist.data import build_dataset, collate_dict
from vist.trainer import load_config, load_model_from_checkpoint, parse_optional_int, seed_everything

MODES = ("correct", "zero", "shift", "wrong")


def si_sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    scale = np.dot(estimate, reference) / (np.dot(reference, reference) + eps)
    projection = scale * reference
    noise = estimate - projection
    return float(10.0 * np.log10((np.dot(projection, projection) + eps) / (np.dot(noise, noise) + eps)))


def alter_face(face: torch.Tensor, mode: str, donor: torch.Tensor | None = None) -> torch.Tensor:
    """Landmarks [B, T, 2, 68] under one of the visual interventions."""
    if mode == "correct":
        return face
    if mode == "zero":
        return torch.zeros_like(face)
    if mode == "shift":
        return torch.roll(face, shifts=max(1, face.shape[1] // 2), dims=1)
    if mode == "wrong":
        if donor is None or donor.shape != face.shape:
            raise RuntimeError("visual_mode='wrong' needs a donor face of the same shape")
        return donor
    raise ValueError(f"Unknown visual_mode={mode!r}")


def score(mixture: np.ndarray, target: np.ndarray, estimate: np.ndarray, sample_rate: int) -> dict:
    """Paper metrics for one mixture, after resampling to 16 kHz."""
    up, down = 16000, sample_rate
    g = np.gcd(up, down)
    mixture, target, estimate = (resample_poly(x, up // g, down // g) for x in (mixture, target, estimate))
    remainder_ref = mixture - target
    remainder_est = mixture - estimate
    sdr, sir, _sar = fast_bss_eval.bss_eval_sources(
        np.stack([target, remainder_ref]),
        np.stack([estimate, remainder_est]),
        filter_length=512,
        compute_permutation=False,
    )
    si_sdr_estimate = si_sdr(target, estimate)
    si_sdr_remainder = si_sdr(target, remainder_est)
    return {
        "sdr": float(sdr[0]),
        "sir": float(sir[0]),
        "si_sdr": si_sdr_estimate,
        "input_si_sdr": si_sdr(target, mixture),
        "remainder_si_sdr": si_sdr_remainder,
        "swap": float(si_sdr_remainder > si_sdr_estimate),
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean."""
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def summarize(rows: list[dict], modes: list[str]) -> None:
    by_mode = {m: [r for r in rows if r["visual_mode"] == m] for m in modes}
    for mode, rs in by_mode.items():
        if not rs:
            continue
        sdr = np.array([r["sdr"] for r in rs])
        lo, hi = bootstrap_ci(sdr)
        print(
            f"[{mode}] n={len(rs)} "
            f"SDR mean={sdr.mean():.2f} [95% CI {lo:.2f}, {hi:.2f}] median={np.median(sdr):.2f} "
            f"SI-SDR={np.mean([r['si_sdr'] for r in rs]):.2f} "
            f"SIR={np.mean([r['sir'] for r in rs]):.2f} "
            f"swap={100.0 * np.mean([r['swap'] for r in rs]):.1f}%"
        )
    correct = by_mode.get("correct", [])
    for mode in modes:
        other = by_mode.get(mode, [])
        if mode == "correct" or not correct or len(other) != len(correct):
            continue
        delta = np.array([c["sdr"] - o["sdr"] for c, o in zip(correct, other)])
        lo, hi = bootstrap_ci(delta)
        print(f"[correct - {mode}] paired SDR delta={delta.mean():.2f} [95% CI {lo:.2f}, {hi:.2f}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="config whose data.val section holds the evaluation split")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num_steps", type=int, default=1, help="Euler steps (Table 2)")
    ap.add_argument("--visual_mode", default="correct", choices=[*MODES, "all"])
    ap.add_argument("--num_batches", default=None, help="limit the number of mixtures (default: all)")
    ap.add_argument("--use_ema", action="store_true", help="Load EMA weights if the checkpoint stores them separately.")
    ap.add_argument("--out_csv", default=None, help="per-mixture scores")
    ap.add_argument("--save_dir", default=None, help="write mixture/estimate/reference wavs per mixture")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no_progress", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)
    cfg = load_config(args.config)
    if not cfg["data"]["val"].get("init_kwargs", {}).get("deterministic", False):
        print(
            "WARNING: data.val.init_kwargs.deterministic is not set, so the mixtures are "
            "re-drawn on every pass and repeated evaluations will differ."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = build_dataset(cfg, "val")
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_dict)
    model = load_model_from_checkpoint(args.checkpoint, map_location=device, use_ema=args.use_ema).to(device).eval()
    sample_rate = int(cfg["data"]["val"]["init_kwargs"].get("audiorate", 16384))

    modes = list(MODES) if args.visual_mode == "all" else [args.visual_mode]
    limit = parse_optional_int(args.num_batches)
    total = len(dl) if limit is None else min(len(dl), limit)
    rows = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dl, total=total, desc="evaluate", disable=args.no_progress)):
            if i >= total:
                break
            mixture = batch["mixture"].to(device)
            face = batch["face"].to(device)
            donor = None
            if "wrong" in modes:
                # A fixed offset into the split: another singer, reproducible across runs.
                donor = ds[(i + len(ds) // 2) % len(ds)]["face"][None].to(device)
            for mode in modes:
                out = model.separate(mixture, face=alter_face(face, mode, donor), num_steps=args.num_steps)
                m = mixture[0].cpu().numpy()
                s = batch["target"][0].numpy()[: m.shape[-1]]
                s_hat = out["target"][0].cpu().numpy()
                row = {"example": i, "visual_mode": mode, **score(m, s, s_hat, sample_rate)}
                rows.append(row)
                if args.save_dir and mode == "correct":
                    ex_dir = Path(args.save_dir) / f"example_{i:04d}"
                    ex_dir.mkdir(parents=True, exist_ok=True)
                    for name, wav in (
                        ("mixture", m), ("target_estimate", s_hat), ("target_reference", s),
                        ("residual_estimate", m - s_hat), ("residual_reference", m - s),
                    ):
                        sf.write(ex_dir / f"{name}.wav", wav, sample_rate)
                    (ex_dir / "metadata.json").write_text(json.dumps({**row, "num_steps": args.num_steps}, indent=2))

    summarize(rows, modes)
    if args.out_csv:
        path = Path(args.out_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
