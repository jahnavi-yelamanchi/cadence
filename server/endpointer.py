"""ONNX Runtime inference wrapper with a streaming sliding-window buffer.

The buffer accumulates 20ms PCM chunks from the client WebSocket.
When silence is detected (RMS < threshold for > MIN_SILENCE_FRAMES), the
accumulated 2s window is passed to the Cadence model.
"""

import time
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16000
CHUNK_MS = 20
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)  # 320 samples
WINDOW_SAMPLES = 2 * SAMPLE_RATE  # 32 000 samples (2s)
SILENCE_THRESHOLD = 0.01           # RMS below this → silence
MIN_SILENCE_FRAMES = 8             # 8 * 20ms = 160ms silence before triggering
DEFAULT_MODEL_PATH = Path("model/onnx/cadence.onnx")


class Endpointer:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {model_path}. Run: make export")

        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._buffer: deque[np.ndarray] = deque(
            maxlen=WINDOW_SAMPLES // CHUNK_SAMPLES
        )
        self._silence_frames = 0
        self._last_decision_at = 0.0

    def push(self, chunk: np.ndarray) -> dict | None:
        """Push a 20ms PCM chunk. Returns a decision dict or None.

        Decision dict:
          label:      "turn_end" | "mid_thought"
          confidence: float
          latency_ms: float
        """
        assert len(chunk) == CHUNK_SAMPLES, f"Expected {CHUNK_SAMPLES} samples, got {len(chunk)}"
        self._buffer.append(chunk.astype(np.float32))

        rms = float(np.sqrt(np.mean(chunk**2)))
        is_silent = rms < SILENCE_THRESHOLD

        if is_silent:
            self._silence_frames += 1
        else:
            self._silence_frames = 0

        if self._silence_frames == MIN_SILENCE_FRAMES:
            return self._infer()

        return None

    def _infer(self) -> dict:
        window = np.concatenate(list(self._buffer))[-WINDOW_SAMPLES:]
        if len(window) < WINDOW_SAMPLES:
            window = np.pad(window, (WINDOW_SAMPLES - len(window), 0))

        t0 = time.perf_counter()
        logits = self._session.run(None, {"input_values": window[None, :]})[0][0]
        latency_ms = (time.perf_counter() - t0) * 1000

        probs = _softmax(logits)
        pred = int(np.argmax(probs))
        labels = ["turn_end", "mid_thought"]

        self._last_decision_at = time.perf_counter()
        return {
            "label": labels[pred],
            "confidence": float(probs[pred]),
            "latency_ms": round(latency_ms, 1),
        }

    def reset(self) -> None:
        self._buffer.clear()
        self._silence_frames = 0


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()
