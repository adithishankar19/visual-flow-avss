#!/usr/bin/env python
"""Render a machine-specific config from the base experiment YAML.

This keeps the experiment config in version control while moving machine-specific
paths, run directories, and small runtime overrides into the cluster submission
script.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _none_or_value(value: str | None) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "all", "full"}:
        return None
    return text


def _set_if_not_none(mapping: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        mapping[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a runnable Visual-FLOSS config YAML.")
    parser.add_argument("--base-config", required=True, help="Original experiment YAML.")
    parser.add_argument("--out", required=True, help="Path for generated config YAML.")
    parser.add_argument("--run-dir", required=True, help="Directory where checkpoints/results will be written.")

    parser.add_argument("--data-root", default=None, help="Root containing acappella/, musdb_accomp/, audioset_split/.")
    parser.add_argument("--acapella-train", default=None)
    parser.add_argument("--acapella-val", default=None)
    parser.add_argument("--musdb-train", default=None)
    parser.add_argument("--musdb-val", default=None)
    parser.add_argument("--audioset-train", default=None)
    parser.add_argument("--audioset-val", default=None)
    parser.add_argument(
        "--use-test-splits",
        action="store_true",
        help=(
            "Explicitly render Acappella test_unseen / MUSDB test / AudioSet test "
            "into the validation slot. Disabled by default to prevent checkpoint "
            "selection on the held-out test set."
        ),
    )

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--val-num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--val-batches", default=None, help="Integer, or all/null/none for full validation.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-progress", action="store_true", help="Disable tqdm progress bars in log files.")
    args = parser.parse_args()

    base_path = Path(args.base_config)
    with base_path.open("r") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    data_root = Path(args.data_root).expanduser() if args.data_root else None

    def default_path(*parts: str) -> str | None:
        return str(data_root.joinpath(*parts)) if data_root is not None else None

    train_kwargs = cfg.setdefault("data", {}).setdefault("train", {}).setdefault("init_kwargs", {})
    val_kwargs = cfg.setdefault("data", {}).setdefault("val", {}).setdefault("init_kwargs", {})

    default_acapella_val = ("acappella", "splits", "test_unseen") if args.use_test_splits else ("acappella", "splits", "val_seen")
    default_musdb_val = ("musdb_accomp", "test") if args.use_test_splits else ("musdb_accomp", "valid")
    default_audioset_val = ("audioset_split", "test") if args.use_test_splits else ("audioset_split", "val")

    _set_if_not_none(train_kwargs, "data_path", args.acapella_train or default_path("acappella", "splits", "train"))
    _set_if_not_none(val_kwargs, "data_path", args.acapella_val or default_path(*default_acapella_val))
    _set_if_not_none(train_kwargs, "musdb_path", args.musdb_train or default_path("musdb_accomp", "train"))
    _set_if_not_none(val_kwargs, "musdb_path", args.musdb_val or default_path(*default_musdb_val))
    _set_if_not_none(train_kwargs, "audioset_path", args.audioset_train or default_path("audioset_split", "train"))
    _set_if_not_none(val_kwargs, "audioset_path", args.audioset_val or default_path(*default_audioset_val))

    train_cfg = cfg.setdefault("training", {})
    train_cfg["out_dir"] = str(Path(args.run_dir).expanduser())
    train_cfg["device"] = args.device
    _set_if_not_none(train_cfg, "batch_size", args.batch_size)
    _set_if_not_none(train_cfg, "num_workers", args.num_workers)
    _set_if_not_none(train_cfg, "val_batch_size", args.val_batch_size)
    _set_if_not_none(train_cfg, "val_num_workers", args.val_num_workers)
    _set_if_not_none(train_cfg, "epochs", args.epochs)
    _set_if_not_none(train_cfg, "lr", args.lr)
    _set_if_not_none(train_cfg, "val_every", args.val_every)
    if args.val_batches is not None:
        maybe = _none_or_value(args.val_batches)
        train_cfg["val_batches"] = None if maybe is None else int(maybe)
    if args.disable_progress:
        train_cfg["progress"] = False
        train_cfg["val_progress"] = False

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Wrote generated config: {out_path}")
    print("Resolved data paths:")
    print(f"  train data:     {train_kwargs.get('data_path')}")
    print(f"  train musdb:    {train_kwargs.get('musdb_path')}")
    print(f"  train audioset: {train_kwargs.get('audioset_path')}")
    print(f"  val data:       {val_kwargs.get('data_path')}")
    print(f"  val musdb:      {val_kwargs.get('musdb_path')}")
    print(f"  val audioset:   {val_kwargs.get('audioset_path')}")
    print(f"  run dir:        {train_cfg.get('out_dir')}")


if __name__ == "__main__":
    main()
