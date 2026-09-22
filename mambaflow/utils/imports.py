from __future__ import annotations

import importlib
import os
import sys
from typing import Any


def add_repo_to_path(env_name: str = "MAMBAVOICE_REPO") -> None:
    repo = os.environ.get(env_name)
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)


def import_from_dotted_path(path: str) -> Any:
    if not isinstance(path, str) or "." not in path:
        raise ValueError(f"Expected dotted import path, got {path!r}")
    module_name, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def get_module_by_path(root: Any, dotted_name: str) -> Any:
    obj = root
    for part in dotted_name.split("."):
        if not hasattr(obj, part):
            raise AttributeError(f"{type(obj).__name__} has no attribute {part!r} while resolving {dotted_name!r}")
        obj = getattr(obj, part)
    return obj
