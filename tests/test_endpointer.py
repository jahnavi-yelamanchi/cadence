"""Unit tests for the Endpointer buffer logic (no ONNX model required)."""

import numpy as np
import pytest

from server.endpointer import CHUNK_SAMPLES, SILENCE_THRESHOLD, Endpointer, _softmax


def test_softmax_sums_to_one():
    logits = np.array([1.2, -0.5])
    probs = _softmax(logits)
    assert abs(probs.sum() - 1.0) < 1e-6
    assert (probs >= 0).all()


def test_silence_frame_counting(tmp_path, monkeypatch):
    """Endpointer counts silence frames correctly without hitting ONNX."""
    # Patch _infer so no model file is needed
    called = []

    def fake_infer(self):
        called.append(True)
        return {"label": "turn_end", "confidence": 0.9, "latency_ms": 5.0}

    monkeypatch.setattr(Endpointer, "_infer", fake_infer)

    # Patch __init__ to skip ONNX loading
    original_init = Endpointer.__init__

    def patched_init(self, model_path=None):
        from collections import deque
        self._session = None
        self._buffer = deque(maxlen=100)
        self._silence_frames = 0
        self._last_decision_at = 0.0

    monkeypatch.setattr(Endpointer, "__init__", patched_init)

    ep = Endpointer()

    # Push 7 silent chunks → no trigger yet
    silent_chunk = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
    for _ in range(7):
        result = ep.push(silent_chunk)
        assert result is None

    # 8th silent chunk → trigger
    result = ep.push(silent_chunk)
    assert result is not None
    assert len(called) == 1

    # Non-silent chunk resets silence counter
    loud_chunk = np.ones(CHUNK_SAMPLES, dtype=np.float32) * 0.5
    ep.push(loud_chunk)
    assert ep._silence_frames == 0


def test_chunk_size_assertion(monkeypatch):
    def patched_init(self, model_path=None):
        from collections import deque
        self._session = None
        self._buffer = deque(maxlen=100)
        self._silence_frames = 0
        self._last_decision_at = 0.0

    monkeypatch.setattr(Endpointer, "__init__", patched_init)
    ep = Endpointer()

    with pytest.raises(AssertionError):
        ep.push(np.zeros(100, dtype=np.float32))  # wrong size
