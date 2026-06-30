"""Meeting-disjoint train / val / test split.

Splitting by meeting (not speaker) ensures no audio from the same recording
session leaks across splits, which is the strongest leakage-prevention guarantee
available in AMI.

Splits: 80% train / 10% val / 10% test by number of unique meetings.

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
    meetings = df["meeting_id"].unique().copy()
    rng.shuffle(meetings)

    n = len(meetings)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    test_mtg = set(meetings[:n_test])
    val_mtg = set(meetings[n_test : n_test + n_val])

    train = df[~df["meeting_id"].isin(test_mtg | val_mtg)].copy()
    val = df[df["meeting_id"].isin(val_mtg)].copy()
    test = df[df["meeting_id"].isin(test_mtg)].copy()

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
            f"turn_end {tc.get('turn_end', 0):4d} | "
            f"mid_thought {tc.get('mid_thought', 0):4d} | "
            f"{split_df['meeting_id'].nunique():3d} meetings"
        )

    print(f"\nSaved → {PROCESSED_DIR}/{{train,val,test}}.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.seed)
