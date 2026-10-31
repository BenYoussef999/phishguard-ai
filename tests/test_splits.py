"""Tests for src/splits.py using a synthetic combined dataset."""

import numpy as np
import pandas as pd
import pytest

from src.splits import (assert_no_group_overlap, leave_one_source_out,
                        train_val_test_split)


def make_df(seed: int = 0) -> pd.DataFrame:
    """Synthetic data: 4 sources; some large duplicate groups; nazario is label 1 only."""
    rng = np.random.default_rng(seed)
    rows, gid = [], 0
    layout = {"ceas08": (0, 1), "enron": (0, 1), "spamassassin": (0, 1), "nazario": (1,)}
    for source, labels in layout.items():
        for label in labels:
            for _ in range(60):                 # 60 groups per (source, label)
                size = int(rng.choice([1, 1, 1, 5, 12]))   # some big duplicate groups
                for _ in range(size):
                    rows.append((source, f"s{gid}", f"b{gid}", label, gid))
                gid += 1
    df = pd.DataFrame(rows, columns=["source", "subject", "body", "label", "dup_group"])
    # make one group appear in two sources to test cross-source removal
    df.loc[df["source"] == "enron", "dup_group"] = df.loc[
        df["source"] == "enron", "dup_group"]
    shared = int(df.loc[df["source"] == "ceas08", "dup_group"].iloc[0])
    idx = df.index[df["source"] == "enron"][0]
    df.loc[idx, "dup_group"] = shared
    return df.reset_index(drop=True)


def test_split_has_no_group_overlap_and_covers_all_rows():
    df = make_df()
    train, val, test = train_val_test_split(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert_no_group_overlap(train, val, test)


def test_split_proportions_are_roughly_64_16_20():
    df = make_df()
    train, val, test = train_val_test_split(df)
    assert 0.55 < len(train) / len(df) < 0.72
    assert 0.10 < len(val) / len(df) < 0.22
    assert 0.14 < len(test) / len(df) < 0.26


def test_split_is_reproducible_and_seed_dependent():
    df = make_df()
    a = train_val_test_split(df, seed=1)[2].index
    b = train_val_test_split(df, seed=1)[2].index
    c = train_val_test_split(df, seed=2)[2].index
    assert list(a) == list(b)
    assert list(a) != list(c)


def test_every_source_appears_in_every_split():
    df = make_df()
    for part in train_val_test_split(df):
        assert set(part["source"]) == set(df["source"])


def test_leave_one_source_out_excludes_source_and_shared_groups():
    df = make_df()
    train, test, removed = leave_one_source_out(df, "ceas08")
    assert set(test["source"]) == {"ceas08"}
    assert "ceas08" not in set(train["source"])
    assert removed >= 1                      # the group we planted in enron
    assert not set(train["dup_group"]) & set(test["dup_group"])


def test_leave_one_source_out_rejects_unknown_source():
    with pytest.raises(ValueError, match="Unknown source"):
        leave_one_source_out(make_df(), "nope")


def test_overlap_check_raises_on_leakage():
    df = make_df()
    with pytest.raises(AssertionError, match="Leakage"):
        assert_no_group_overlap(df.iloc[:10], df.iloc[:10])
