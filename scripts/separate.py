#!/usr/bin/env python
"""Separate one mixture with a trained VIST checkpoint.

Input: a waveform file readable by soundfile at 16,384 Hz, and the target
singer's 68 facial landmarks at 25 fps as a .npy array of shape [T, 2, 68].
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from vist.trainer import load_model_from_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mixture", required=True, help="mixture waveform (16,384 Hz)")
    ap.add_argument("--landmarks", required=True, help=".npy facial landmarks [T, 2, 68]")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_steps", type=int, default=1)
    ap.add_argument("--use_ema", action="store_true", help="Load EMA weights if the checkpoint stores them separately.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(args.checkpoint, map_location=device, use_ema=args.use_ema).to(device).eval()

    mixture, sr = sf.read(args.mixture, dtype="float32", always_2d=False)
    if mixture.ndim > 1:
        mixture = mixture.mean(axis=1)
    if sr != 16384:
        raise ValueError(f"expected a 16,384 Hz mixture, got {sr} Hz")
    face = torch.from_numpy(np.load(args.landmarks)).float()[None].to(device)
    out = model.separate(torch.from_numpy(mixture)[None].to(device), face=face, num_steps=args.num_steps)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / "target_estimate.wav", out["target"][0].cpu().numpy(), sr)
    sf.write(out_dir / "residual_estimate.wav", out["residual"][0].cpu().numpy(), sr)
    print(f"wrote {out_dir}/target_estimate.wav and residual_estimate.wav")


if __name__ == "__main__":
    main()
