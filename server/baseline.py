"""Silero-VAD baseline wrapper for live side-by-side comparison.

Runs silero-VAD on the same audio chunks as the Cadence endpointer
so the frontend can show both decisions simultaneously.
"""

import numpy as np
import torch

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # silero-VAD expects 512-sample chunks at 16kHz
VAD_THRESHOLD = 0.5


class SileroBaseline:
    def __init__(self):
        self._model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            verbose=False,
        )
        self._model.eval()
        self._speech_frames = 0
        self._silence_frames = 0
        self._in_speech = False

    def push(self, chunk_320: np.ndarray) -> dict | None:
        """Accept a 320-sample (20ms) chunk and return a VAD decision when silence detected.

        silero-VAD needs 512 samples; we buffer two 320-sample chunks → 640,
        then take the first 512 samples. This introduces ~10ms lag which is
        acceptable for a demo baseline.
        """
        if not hasattr(self, "_carry"):
            self._carry = np.array([], dtype=np.float32)

        combined = np.concatenate([self._carry, chunk_320.astype(np.float32)])
        self._carry = combined[CHUNK_SAMPLES:]
        chunk = combined[:CHUNK_SAMPLES]

        if len(chunk) < CHUNK_SAMPLES:
            return None

        tensor = torch.from_numpy(chunk).unsqueeze(0)
        with torch.no_grad():
            prob = self._model(tensor, SAMPLE_RATE).item()

        is_speech = prob > VAD_THRESHOLD

        if is_speech:
            self._speech_frames += 1
            self._silence_frames = 0
            self._in_speech = True
        else:
            self._silence_frames += 1
            if self._silence_frames == 8 and self._in_speech:  # 160ms silence
                self._in_speech = False
                self._speech_frames = 0
                return {"label": "turn_end", "confidence": float(1 - prob)}

        return None

    def reset(self) -> None:
        self._model.reset_states()
        self._speech_frames = 0
        self._silence_frames = 0
        self._in_speech = False
        self._carry = np.array([], dtype=np.float32)
