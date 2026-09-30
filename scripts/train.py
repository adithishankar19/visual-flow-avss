#!/usr/bin/env python
from __future__ import annotations

import argparse

from vist.trainer import load_config, train


def main():
    ap = argparse.ArgumentParser(description="Train VIST from random initialization.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--resume_from", type=str, default=None, help="Continue the same run from last.pt or best_raw.pt.")
    args = ap.parse_args()
    ckpt = train(load_config(args.config), max_steps=args.max_steps, resume_from=args.resume_from)
    print(f"saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
