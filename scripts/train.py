#!/usr/bin/env python
from __future__ import annotations

import argparse
from mambaflow.trainer import load_config, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint file (.pt) to resume from.")
    ap.add_argument("--init_from", type=str, default=None, help="Load model weights only; restart optimizer/scheduler/step.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ckpt = train(cfg, max_steps=args.max_steps, resume_from=args.resume_from, init_from=args.init_from)
    print(f"saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
