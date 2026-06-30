"""Auto-label pause events as 'turn_end' or 'mid_thought'.

Labeling logic:
  - Find all silence gaps > MIN_SILENCE_MS within a speaker segment
  - If the same speaker resumes within RESUME_THRESHOLD_MS  → mid_thought
  - Otherwise (other speaker speaks or long silence)         → turn_end
  - Extract a 2-second audio window ending at pause onset as the model input

Output: data/processed/labels.parquet with columns:
  audio_path, pause_start_ms, pause_end_ms, label, speaker_id, session_id

Run: make data-label
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
WINDOW_MS = 2000        # audio context window fed to model
MIN_SILENCE_MS = 150    # ignore sub-150ms gaps (breath, hesitation)
RESUME_THRESHOLD_MS = 700  # same speaker resumes → mid_thought


def load_candor_turns(session_dir: Path) -> list[dict]:
    """Load speaker turn annotations from CANDOR JSON transcript."""
    transcript_file = session_dir / "transcript.json"
    if not transcript_file.exists():
        return []

    with transcript_file.open() as f:
        transcript = json.load(f)

    turns = []
    for utt in transcript.get("utterances", []):
        turns.append(
            {
                "speaker": utt["speaker"],
                "start_ms": int(utt["start"] * 1000),
                "end_ms": int(utt["end"] * 1000),
            }
        )
    return sorted(turns, key=lambda t: t["start_ms"])


def find_pauses(turns: list[dict]) -> list[dict]:
    """Find intra-speaker silence events and label each."""
    pauses = []
    for i, turn in enumerate(turns):
        # Look ahead: is there a gap between this turn's end and the next event?
        next_start = turns[i + 1]["start_ms"] if i + 1 < len(turns) else turn["end_ms"] + 5000
        gap_ms = next_start - turn["end_ms"]

        if gap_ms < MIN_SILENCE_MS:
            continue

        next_speaker = turns[i + 1]["speaker"] if i + 1 < len(turns) else None
        same_speaker_resumes = next_speaker == turn["speaker"] and gap_ms <= RESUME_THRESHOLD_MS
        label = "mid_thought" if same_speaker_resumes else "turn_end"

        pauses.append(
            {
                "speaker_id": turn["speaker"],
                "pause_start_ms": turn["end_ms"],
                "pause_end_ms": next_start,
                "gap_ms": gap_ms,
                "label": label,
            }
        )
    return pauses


def extract_audio_window(audio_path: Path, pause_start_ms: int, sample_rate: int = 16000) -> bool:
    """Return True if we can extract a valid 2s window ending at pause_start_ms."""
    try:
        info = sf.info(audio_path)
        duration_ms = int(info.duration * 1000)
        window_start_ms = pause_start_ms - WINDOW_MS
        return window_start_ms >= 0 and pause_start_ms <= duration_ms
    except Exception:
        return False


def process_candor(raw_dir: Path) -> pd.DataFrame:
    candor_dir = raw_dir / "candor"
    if not candor_dir.exists():
        print("CANDOR data not found — run make data-download first")
        return pd.DataFrame()

    records = []
    sessions = sorted(candor_dir.glob("*/"))

    for session in tqdm(sessions, desc="CANDOR sessions"):
        audio_file = session / "audio.wav"
        if not audio_file.exists():
            audio_file = next(session.glob("*.wav"), None)
        if audio_file is None:
            continue

        turns = load_candor_turns(session)
        if not turns:
            continue

        pauses = find_pauses(turns)
        for pause in pauses:
            if extract_audio_window(audio_file, pause["pause_start_ms"]):
                records.append(
                    {
                        "audio_path": str(audio_file),
                        "pause_start_ms": pause["pause_start_ms"],
                        "pause_end_ms": pause["pause_end_ms"],
                        "gap_ms": pause["gap_ms"],
                        "label": pause["label"],
                        "speaker_id": f"candor_{session.name}_{pause['speaker_id']}",
                        "session_id": session.name,
                        "corpus": "candor",
                    }
                )

    return pd.DataFrame(records)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dfs = []
    df_candor = process_candor(RAW_DIR)
    if not df_candor.empty:
        dfs.append(df_candor)
        print(f"CANDOR: {len(df_candor)} pause events")

    if not dfs:
        print("No data processed — download corpora first with: make data-download")
        return

    df = pd.concat(dfs, ignore_index=True)

    label_counts = df["label"].value_counts()
    print(f"\nLabel distribution:\n{label_counts}")
    print(f"Class balance: {label_counts['mid_thought'] / len(df):.1%} mid_thought")

    out_path = PROCESSED_DIR / "labels.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df)} samples → {out_path}")


if __name__ == "__main__":
    main()
