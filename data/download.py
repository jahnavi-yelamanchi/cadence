"""Download training corpora via HuggingFace datasets.

Corpus used:
  AMI Meeting Corpus (edinburghcstr/ami, ~100h, Apache 2.0)
  IHM (individual headset mic) — clean per-speaker audio with word-level timestamps.

HF datasets handles caching automatically (~/.cache/huggingface).
Audio is written to disk as-is (no re-encoding); metadata saved as parquet.

Run: make data-download
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import Audio, load_dataset
from tqdm import tqdm

RAW_DIR = Path("data/raw")


def download_ami(split: str) -> None:
    out_dir = RAW_DIR / "ami" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "metadata.parquet"
    if meta_path.exists():
        print(f"  [skip] {split} already downloaded")
        return

    print(f"  Loading AMI {split} from HuggingFace...")
    ds = load_dataset("edinburghcstr/ami", "ihm", split=split, trust_remote_code=True)

    # Don't let HF decode audio (avoids torchcodec dependency).
    # We get {"bytes": <raw file bytes>, "path": <filename>} instead.
    ds = ds.cast_column("audio", Audio(decode=False))
    print(f"  {len(ds):,} utterances")

    audio_dir = out_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    records = []

    for row in tqdm(ds, desc=f"  {split}"):
        uid = f"{row['meeting_id']}_{row['audio_id']}"

        # Write raw audio bytes to disk (HF stores them as flac/wav)
        raw = row["audio"]
        ext = Path(raw["path"]).suffix if raw.get("path") else ".flac"
        audio_path = audio_dir / f"{uid}{ext}"
        if not audio_path.exists():
            audio_path.write_bytes(raw["bytes"])

        records.append(
            {
                "uid": uid,
                "meeting_id": row["meeting_id"],
                "speaker_id": row["speaker_id"],
                "begin_time": float(row["begin_time"]),
                "end_time": float(row["end_time"]),
                "audio_path": str(audio_path),
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
