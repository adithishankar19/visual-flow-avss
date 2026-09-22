from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset

from mambaflow.utils.imports import add_repo_to_path, import_from_dotted_path


class VovitTupleAdapter(Dataset):
    """Wrap MambaVoice/VoViT dataloaders into dict batches for MCFlow.

    Expected raw item patterns:
      (data, target), where data[0]=mixture, data[1]=face, data[2]=body optional
      {'mixture': ..., 'target': ..., 'face': ...}
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        if isinstance(item, dict):
            out = dict(item)
            if "visuals" in out and "face" not in out:
                out["face"] = out["visuals"]
            return out
        if isinstance(item, (tuple, list)) and len(item) == 2:
            data, target = item
            if isinstance(data, dict):
                out = dict(data)
                out["target"] = target
                if "visuals" in out and "face" not in out:
                    out["face"] = out["visuals"]
                return out
            if isinstance(data, (tuple, list)):
                out: Dict[str, Any] = {"mixture": data[0], "target": target}
                if len(data) > 1:
                    out["face"] = data[1]
                if len(data) > 2:
                    out["body"] = data[2]
                return out
            return {"mixture": data, "target": target}
        raise TypeError(f"Unsupported dataset item type: {type(item)!r}")


def build_vovit_dataset(cfg: Dict[str, Any]) -> Dataset:
    """Build a dataloader from your local MambaVoice repo.

    Examples:
      class_path: vovit.display.dataloaders_acapella_ursing.URSingDataset
      init_kwargs: {urs_root: UrSing, audiorate: 16384, is_train: true}

    For files that are not importable as modules, use `file_path` and `class_name`.
    """
    add_repo_to_path(cfg.get("env_repo_var", "MAMBAVOICE_REPO"))
    init_kwargs = dict(cfg.get("init_kwargs", {}))
    if cfg.get("class_path"):
        cls = import_from_dotted_path(cfg["class_path"])
    elif cfg.get("file_path") and cfg.get("class_name"):
        raw_file_path = Path(cfg["file_path"])
        if raw_file_path.is_absolute() or raw_file_path.exists():
            file_path = raw_file_path
        else:
            file_path = Path(os.environ.get(cfg.get("env_repo_var", "MAMBAVOICE_REPO"), ".")) / raw_file_path
        spec = importlib.util.spec_from_file_location("_external_vovit_dataset", str(file_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load dataset file {file_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = getattr(mod, cfg["class_name"])
    else:
        raise ValueError("Dataset config must provide class_path or file_path+class_name")
    return VovitTupleAdapter(cls(**init_kwargs))
