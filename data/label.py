"""Auto-label pause events as 'turn_end' or 'mid_thought' from AMI metadata.

Labeling logic (applied per meeting):
  - Sort utterances by begin_time
  - For each adjacent pair (A, B): gap = B.begin_time - A.end_time
  - If gap < MIN_SILENCE_MS: skip (too short, not a real pause)
  - If gap <= RESUME_THRESHOLD_MS AND same speaker continues: mid_thought
  - Otherwise: turn_end
  - Extract the 2s audio window BEFORE the pause onset from A's audio

Output: data/processed/labels.parquet
  columns: audio_path, pause_start_ms, pause_end_ms, gap_ms, label, speaker_id, meeting_id

Run: make data-label
"""

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

MIN_SILENCE_MS = 150
RESUME_THRESHOLD_MS = 2000  # same speaker resumes within 2s → mid_thought
WINDOW_MS = 2000
SAMPLE_RATE = 16000


def load_meeting_audio(audio_path: str) -> np.ndarray:
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio


def process_split(split: str) -> pd.DataFrame:
    meta_path = RAW_DIR / "ami" / split / "metadata.parquet"
    if not meta_path.exists():
        print(f"  [skip] {split} metadata not found — run make data-download first")
        return pd.DataFrame()

    df = pd.read_parquet(meta_path)
    records = []

    for meeting_id, group in tqdm(df.groupby("meeting_id"), desc=f"  {split}"):
        utterances = group.sort_values("begin_time").reset_index(drop=True)

        for i in range(len(utterances) - 1):
            curr = utterances.iloc[i]
            nxt = utterances.iloc[i + 1]

            gap_ms = (nxt["begin_time"] - curr["end_time"]) * 1000
            if gap_ms < MIN_SILENCE_MS:
                continue

            same_speaker = curr["speaker_id"] == nxt["speaker_id"]
            label = (
                "mid_thought"
                if same_speaker and gap_ms <= RESUME_THRESHOLD_MS
                else "turn_end"
            )

            # The model window: 2s of audio ending at the pause onset
            pause_start_ms = int(curr["end_time"] * 1000)
            window_start_ms = max(0, pause_start_ms - WINDOW_MS)

            # Verify the audio segment exists and is long enough
            if not Path(curr["audio_path"]).exists():
                continue
            audio_duration_ms = int(
                sf.info(curr["audio_path"]).duration * 1000
            )
            if audio_duration_ms < WINDOW_MS // 2:
                continue

            records.append(
                {
                    "audio_path": curr["audio_path"],
                    "pause_start_ms": pause_start_ms,
                    "window_start_ms": window_start_ms,
                    "pause_end_ms": int(nxt["begin_time"] * 1000),
                    "gap_ms": round(gap_ms, 1),
                    "label": label,
                    "speaker_id": f"ami_{meeting_id}_{curr['speaker_id']}",
                    "meeting_id": meeting_id,
                    "corpus": "ami",
                    "split": split,
                }
            )

    return pd.DataFrame(records)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dfs = []
    for split in ["train", "validation", "test"]:
        df_split = process_split(split)
        if not df_split.empty:
            dfs.append(df_split)
            print(f"  {split}: {len(df_split):,} pause events")

    if not dfs:
        print("No data processed — run make data-download first")
        return

    df = pd.concat(dfs, ignore_index=True)

    counts = df["label"].value_counts()
    print(f"\nTotal: {len(df):,} events")
    print(f"  turn_end:   {counts.get('turn_end', 0):,}")
    print(f"  mid_thought: {counts.get('mid_thought', 0):,}")
    print(f"  balance: {counts.get('mid_thought', 0) / len(df):.1%} mid_thought")

    out = PROCESSED_DIR / "labels.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
