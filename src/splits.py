"""Group-aware train/validation/test splits and leave-one-source-out folds.

Why groups: near-identical messages (same spam campaign) share a `dup_group`
id from src/data.py. All members of a group must land on the same side of a
split, otherwise the model is graded on near-copies of what it trained on.

Run a summary from the project root:
    python -m src.splits
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

RANDOM_SEED = 42

# 5 folds -> 20% held out. Applied twice: 80/20 for test, then 80/20 of the rest
# for validation, giving roughly 64% train / 16% validation / 20% test.
N_FOLDS = 5


def _stratify_key(df: pd.DataFrame) -> pd.Series:
    """Keep each (source, label) combination represented in every split."""
    return df["source"] + "_" + df["label"].astype(str)


def _first_fold(df: pd.DataFrame, seed: int):
    """Return (kept, held_out) frames from one stratified, group-aware fold."""
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    keep_pos, hold_pos = next(splitter.split(df, _stratify_key(df),
                                             groups=df["dup_group"]))
    return df.iloc[keep_pos], df.iloc[hold_pos]


def train_val_test_split(df: pd.DataFrame, seed: int = RANDOM_SEED):
    """Split into (train, val, test) with no dup_group shared between any two."""
    train_val, test = _first_fold(df, seed)
    train, val = _first_fold(train_val, seed + 1)
    assert_no_group_overlap(train, val, test)
    return train, val, test


def leave_one_source_out(df: pd.DataFrame, held_out: str):
    """Train on every other source, test on `held_out`.

    Training rows whose dup_group also appears in the test source are removed,
    so the test source is not contaminated by copies in the training data.
    Returns (train, test, n_removed_from_train).
    """
    if held_out not in set(df["source"]):
        raise ValueError(f"Unknown source '{held_out}'. Available: "
                         f"{sorted(df['source'].unique())}")
    test = df[df["source"] == held_out]
    train = df[df["source"] != held_out]
    overlapping = train["dup_group"].isin(set(test["dup_group"]))
    removed = int(overlapping.sum())
    train = train[~overlapping]
    assert_no_group_overlap(train, test)
    return train, test, removed


def assert_no_group_overlap(*frames: pd.DataFrame) -> None:
    """Raise if any dup_group appears in more than one of the given frames."""
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            shared = set(frames[i]["dup_group"]) & set(frames[j]["dup_group"])
            if shared:
                raise AssertionError(
                    f"Leakage: {len(shared)} duplicate groups shared between "
                    f"split {i} and split {j}")


def summarize(df: pd.DataFrame) -> None:
    train, val, test = train_val_test_split(df)
    print("=" * 60)
    print(f"RANDOM-SEED {RANDOM_SEED}; GROUP-AWARE TRAIN / VALIDATION / TEST")
    for name, part in (("train", train), ("val", val), ("test", test)):
        print(f"\n{name}: {len(part):,} rows ({len(part) / len(df):.1%})")
        print(pd.crosstab(part["source"], part["label"]).to_string())
    print("\nGroup overlap between splits: none (checked)")

    print("\n" + "=" * 60)
    print("LEAVE-ONE-SOURCE-OUT FOLDS")
    rows = []
    for src in sorted(df["source"].unique()):
        tr, te, removed = leave_one_source_out(df, src)
        rows.append({"held_out": src, "train_rows": len(tr), "test_rows": len(te),
                     "test_label0": int((te["label"] == 0).sum()),
                     "test_label1": int((te["label"] == 1).sum()),
                     "removed_from_train": removed})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nNote: a source with no label-0 rows (nazario) can only measure recall.")


def main() -> int:
    from src.data import build_dataset
    parser = argparse.ArgumentParser(description="Summarize the data splits.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    try:
        df, _ = build_dataset(args.data_dir)
    except (FileNotFoundError, ValueError) as err:
        print(f"Error: {err}")
        return 1
    summarize(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
