from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import Dataset


class DictAdapter(Dataset):
    """Turn ``([mixture, face], target)`` items into ``{mixture, face, target}`` dicts."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        if isinstance(item, dict):
            return dict(item)
        (mixture, face), target = item
        return {"mixture": mixture, "face": face, "target": target}


def _load_class(split_cfg: Dict[str, Any]):
    if split_cfg.get("class_path"):
        module_name, _, class_name = str(split_cfg["class_path"]).rpartition(".")
        return getattr(importlib.import_module(module_name), class_name)
    # Pre-release configs point at a file instead of an importable module.
    file_path = Path(split_cfg["file_path"])
    if not file_path.exists() and file_path.name == "dataloaders_acapella_musdb_target_only.py":
        file_path = Path(__file__).with_name("acappella.py")
    spec = importlib.util.spec_from_file_location("_vist_dataset", str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load dataset file {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, split_cfg["class_name"])


def build_dataset(cfg: Dict[str, Any], split: str = "train") -> Dataset:
    """Build ``data.<split>`` from the config (``class_path`` + ``init_kwargs``)."""
    data_cfg = cfg.get("data", {})
    if split not in data_cfg:
        raise KeyError(f"config has no data.{split} section")
    split_cfg = data_cfg[split]
    cls = _load_class(split_cfg)
    return DictAdapter(cls(**dict(split_cfg.get("init_kwargs", {}))))


def collate_dict(batch):
    keys = set().union(*(b.keys() for b in batch))
    out = {}
    for k in keys:
        vals = [b.get(k) for b in batch]
        if vals[0] is None:
            continue
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out
