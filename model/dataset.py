"""PyTorch Dataset for Cadence turn-detection.

Loads 2s audio windows from the labelled parquet files and applies
on-the-fly augmentation during training.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 2 * SAMPLE_RATE  # 32 000 samples
LABEL2ID = {"turn_end": 0, "mid_thought": 1}


class TurnDetectionDataset(Dataset):
    def __init__(
        self,
        parquet_path: str | Path,
        processor: Wav2Vec2Processor,
        augment: bool = False,
    ):
        self.df = pd.read_parquet(parquet_path)
        self.processor = processor
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        audio = self._load_window(row["audio_path"], int(row["pause_start_ms"]))

        if self.augment:
            audio = self._augment(audio)

        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=False,
        )

        return {
            "input_values": inputs.input_values.squeeze(0),
            "label": torch.tensor(LABEL2ID[row["label"]], dtype=torch.long),
        }

    def _load_window(self, audio_path: str, pause_start_ms: int) -> np.ndarray:
        """Load last 2s of the utterance file (speech immediately before the pause).

        pause_start_ms is a meeting-clock timestamp, not a file offset — each
        audio_path is an individual utterance clip, so the pause begins at the
        END of the file. We read the trailing WINDOW_SAMPLES samples.
        """
        info = sf.info(audio_path)
        start_sample = max(0, info.frames - WINDOW_SAMPLES)

        audio, sr = sf.read(audio_path, start=start_sample, dtype="float32")

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample if needed (AMI is 16kHz but guard anyway)
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        # Pad or trim to exact window size
        if len(audio) < WINDOW_SAMPLES:
            audio = np.pad(audio, (WINDOW_SAMPLES - len(audio), 0))
        else:
            audio = audio[:WINDOW_SAMPLES]

        return audio

    def _augment(self, audio: np.ndarray) -> np.ndarray:
        """Lightweight augmentation: gain jitter + additive noise."""
        gain = np.random.uniform(0.8, 1.2)
        audio = audio * gain
        noise_level = np.random.uniform(0.0, 0.005)
        audio = audio + np.random.randn(*audio.shape).astype(np.float32) * noise_level
        return np.clip(audio, -1.0, 1.0)


def collate_fn(batch: list[dict]) -> dict:
    """Pad input_values to the longest sample in the batch."""
    max_len = max(x["input_values"].shape[0] for x in batch)
    padded = torch.zeros(len(batch), max_len)
    for i, x in enumerate(batch):
        L = x["input_values"].shape[0]
        padded[i, :L] = x["input_values"]

    return {
        "input_values": padded,
        "labels": torch.stack([x["label"] for x in batch]),
    }
