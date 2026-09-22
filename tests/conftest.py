"""Test-session setup.

Works around a *local environment* defect that has nothing to do with this
codebase: when torch's installed dist-info metadata is malformed,
`importlib.metadata.version("torch")` returns None, `transformers` then does
`version.parse(None)` at import time and raises TypeError.  `torch.optim.AdamW`
imports `torch._dynamo` -> `torch.onnx` -> `transformers`, so constructing an
optimizer fails and no trainer test can run.

`transformers` is not a dependency of this project -- it is only reached
incidentally through torch.onnx's patcher -- so stubbing it changes nothing
about what is under test.  The stub is installed ONLY when the real import is
already broken, so a healthy environment (e.g. the cluster) is left untouched.
"""

from __future__ import annotations

import sys
import types


def _transformers_import_is_broken() -> bool:
    if "transformers" in sys.modules:
        return False
    try:
        import transformers  # noqa: F401
    except Exception:
        return True
    return False


if _transformers_import_is_broken():
    stub = types.ModuleType("transformers")
    stub.__doc__ = "Stub installed by tests/conftest.py; see module docstring."
    sys.modules["transformers"] = stub
