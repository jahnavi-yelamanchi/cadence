"""Download training corpora via HuggingFace datasets.

Corpus used:
  AMI Meeting Corpus (edinburghcstr/ami, ~100h, Apache 2.0)
  IHM (individual headset mic) — clean per-speaker audio with word-level timestamps.

HF datasets handles caching automatically (~/.cache/huggingface).
Audio arrays + metadata are saved as parquet to data/raw/ami/.

Run: make data-download
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

RAW_DIR = Path("data/raw")
SAMPLE_RATE = 16000


def download_ami(split: str) -> None:
    out_dir = RAW_DIR / "ami" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "metadata.parquet"
    if meta_path.exists():
        print(f"  [skip] {split} metadata already exists")
        return

    print(f"  Loading AMI {split} from HuggingFace (cached after first run)...")
    ds = load_dataset("edinburghcstr/ami", "ihm", split=split, trust_remote_code=True)
    print(f"  {len(ds):,} utterances")

    records = []
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    for row in tqdm(ds, desc=f"  saving {split}"):
        audio_array = np.array(row["audio"]["array"], dtype=np.float32)
        sr = row["audio"]["sampling_rate"]

        # Resample to 16kHz if needed
        if sr != SAMPLE_RATE:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=SAMPLE_RATE)

        # Save audio segment as wav
        uid = f"{row['meeting_id']}_{row['audio_id']}"
        wav_path = audio_dir / f"{uid}.wav"
        if not wav_path.exists():
            sf.write(wav_path, audio_array, SAMPLE_RATE)

        records.append(
            {
                "uid": uid,
                "meeting_id": row["meeting_id"],
                "speaker_id": row["speaker_id"],
                "begin_time": float(row["begin_time"]),
                "end_time": float(row["end_time"]),
                "audio_path": str(wav_path),
                "text": row.get("segment_text", ""),
            }
        )

    df = pd.DataFrame(records).sort_values(["meeting_id", "begin_time"]).reset_index(drop=True)
    df.to_parquet(meta_path, index=False)
    print(f"  Saved {len(df):,} utterances → {meta_path}")


def main(splits: list[str]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("==> AMI Meeting Corpus (via HuggingFace datasets)")
    for split in splits:
        download_ami(split)
    print("\nDownload complete. Next: make data-label")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Cadence training corpora")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        choices=["train", "validation", "test"],
    )
    args = parser.parse_args()
    main(args.splits)
