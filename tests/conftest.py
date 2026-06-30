"""Stub heavy ML deps so unit tests run without installing torch/onnxruntime."""

import sys
from types import ModuleType
from unittest.mock import MagicMock


def _make_stub(name: str) -> ModuleType:
    mod = ModuleType(name)
    mod.__spec__ = MagicMock()
    return mod


for _pkg in ["torch", "torchaudio", "transformers", "onnxruntime", "onnxruntime.quantization"]:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _make_stub(_pkg)

# torch sub-modules the server imports
import torch as _t  # noqa: E402  (already stubbed above)

_t.hub = MagicMock()
_t.backends = MagicMock()
_t.backends.mps = MagicMock()
_t.backends.mps.is_available = lambda: False
_t.cuda = MagicMock()
_t.cuda.is_available = lambda: False
