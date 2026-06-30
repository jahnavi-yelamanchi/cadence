"""Speaker-disjoint train / val / test split.

Speaker-disjoint ensures the model generalises to unseen speakers rather than
memorising prosodic patterns from specific individuals.

Splits: 80% train / 10% val / 10% test (by number of unique speakers).

Output:
  data/processed/train.parquet
  data/processed/val.parquet
  data/processed/test.parquet

Run: make data-split
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")


def split(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    speakers = df["speaker_id"].str.split("_").str[:2].str.join("_").unique()
    rng.shuffle(speakers)

    n = len(speakers)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    test_spk = set(speakers[:n_test])
    val_spk = set(speakers[n_test : n_test + n_val])
    train_spk = set(speakers[n_test + n_val :])

    # Extract session prefix for grouping (speaker_id format: corpus_session_spkid)
    session_col = df["session_id"]

    train = df[~session_col.isin(test_spk | val_spk)].copy()
    val = df[session_col.isin(val_spk)].copy()
    test = df[session_col.isin(test_spk)].copy()

    return train, val, test


def main(seed: int = 42) -> None:
    src = PROCESSED_DIR / "labels.parquet"
    if not src.exists():
        print("labels.parquet not found — run make data-label first")
        return

    df = pd.read_parquet(src)
    train, val, test = split(df, seed=seed)

    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        path = PROCESSED_DIR / f"{name}.parquet"
        split_df.to_parquet(path, index=False)
        tc = split_df["label"].value_counts()
        print(
            f"{name:5s}: {len(split_df):5d} samples | "
            f"turn_end {tc.get('turn_end', 0):4d} | mid_thought {tc.get('mid_thought', 0):4d} | "
            f"{len(split_df['session_id'].unique()):3d} sessions"
        )

    print(f"\nSaved splits → {PROCESSED_DIR}/{{train,val,test}}.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.seed)
