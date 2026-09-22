#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mambaflow.trainer import load_model_from_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mixture_pt", required=True, help="torch file containing waveform tensor or dict with mixture/face/body")
    ap.add_argument("--out_pt", required=True)
    ap.add_argument("--num_steps", type=int, default=None)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj = torch.load(args.mixture_pt, map_location=device)
    if isinstance(obj, dict):
        mixture = obj["mixture"]
        face = obj.get("face")
        body = obj.get("body")
    else:
        mixture, face, body = obj, None, None
    model = load_model_from_checkpoint(args.checkpoint, map_location=device).to(device).eval()
    out = model.separate(mixture.to(device), face=face.to(device) if torch.is_tensor(face) else None, body=body.to(device) if torch.is_tensor(body) else None, num_steps=args.num_steps)
    Path(args.out_pt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in out.items()}, args.out_pt)
    print(f"saved {args.out_pt}")


if __name__ == "__main__":
    main()
