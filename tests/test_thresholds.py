"""Tests for src/thresholds.py using synthetic scores and tiny synthetic text."""

import numpy as np
import pandas as pd
import pytest

from src.experiments import VARIANTS
from src.thresholds import (assign_bands, band_report, calibration_table,
                            compute_oof_scores, load_scores, save_scores,
                            threshold_sweep)


def score_df() -> pd.DataFrame:
    # 5 legit emails scoring low, 5 bad emails scoring high, 2 nazario (bad)
    rows = [("ceas08", 0, s) for s in (0.05, 0.1, 0.15, 0.2, 0.35)]
    rows += [("ceas08", 1, s) for s in (0.65, 0.8, 0.9, 0.95, 0.4)]
    rows += [("nazario", 1, s) for s in (0.3, 0.7)]
    return pd.DataFrame(rows, columns=["source", "label", "score"]).assign(dup_group=range(12))


def test_assign_bands_boundaries_and_validation():
    bands = assign_bands([0.1, 0.3, 0.69, 0.7, 0.99], 0.3, 0.7)
    assert list(bands) == ["legit", "suspicious", "suspicious", "bad", "bad"]
    for bad_pair in [(0.7, 0.3), (-0.1, 0.5), (0.2, 1.5)]:
        with pytest.raises(ValueError):
            assign_bands([0.5], *bad_pair)


def test_calibration_table_counts_and_shares():
    df = pd.DataFrame({"score": [0.05, 0.08, 0.95, 0.97, 0.99],
                       "label": [0, 1, 1, 1, 0]})
    table = calibration_table(df, bins=10)
    assert table["n"].sum() == 5
    low = table[table["score_range"] == "0.0-0.1"].iloc[0]
    high = table[table["score_range"] == "0.9-1.0"].iloc[0]
    assert low["n"] == 2 and low["share_actually_bad"] == pytest.approx(0.5)
    assert high["n"] == 3 and high["share_actually_bad"] == pytest.approx(2 / 3)


def test_calibration_table_puts_score_of_exactly_one_in_last_bin():
    table = calibration_table(pd.DataFrame({"score": [1.0, 0.0], "label": [1, 0]}))
    assert table["n"].sum() == 2


def test_sweep_recall_never_increases_with_threshold():
    sweep = threshold_sweep(score_df())
    assert sweep["recall_all_bad"].is_monotonic_decreasing
    assert sweep["false_alarm_rate"].is_monotonic_decreasing
    at_half = sweep[sweep["threshold"] == 0.5].iloc[0]
    # bad scores >= 0.5: 0.65 0.8 0.9 0.95 0.7 -> 5 of 7; legit none
    assert at_half["recall_all_bad"] == pytest.approx(5 / 7)
    assert at_half["false_alarm_rate"] == 0.0
    assert at_half["recall_nazario"] == pytest.approx(0.5)


def test_band_report_counts_are_consistent():
    rep = band_report(score_df(), 0.3, 0.7)
    s, h = rep["summary"], rep["headline"]
    assert s["n"].sum() == 12
    assert h["total_bad"] == 7 and h["total_legit"] == 5
    # bad emails scoring < 0.3: none (0.3 itself is 'suspicious'); legit >= 0.7: none
    assert h["bad_called_legit"] == 0 and h["legit_called_bad"] == 0
    # legit in [0.3, 0.7): only 0.35
    assert h["legit_called_suspicious"] == 1
    assert set(rep["per_source"]["source"]) == {"ceas08", "nazario"}


def test_scores_cache_roundtrip_and_validation(tmp_path):
    path = tmp_path / "scores.csv"
    save_scores(score_df(), path)
    loaded = load_scores(path)
    assert len(loaded) == 12 and list(loaded.columns) == ["source", "label", "dup_group", "score"]
    pd.DataFrame({"x": [1]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="--build"):
        load_scores(tmp_path / "bad.csv")


def test_oof_scores_cover_every_email_once_and_stay_in_range():
    rows, gid = [], 0
    layout = {"ceas08": (0, 1), "enron": (0, 1), "spamassassin": (0, 1), "nazario": (1,)}
    for source, labels in layout.items():
        for label in labels:
            for i in range(20):
                cls = "winprize claimfree urgent" if label else "agenda review project"
                extra = "onlyenron" if source == "enron" else "sharedword"
                rows.append((source, f"s{gid}", f"{cls} {extra} text {i}", label, gid))
                gid += 1
    df = pd.DataFrame(rows, columns=["source", "subject", "body", "label", "dup_group"])
    scores = compute_oof_scores(df, VARIANTS["no_email"], min_df=1, min_source_df=2)
    assert len(scores) == len(df)
    assert scores["score"].between(0, 1).all()
    assert scores.groupby("source").size().to_dict() == df.groupby("source").size().to_dict()
    # bad emails should score higher than legitimate ones on this easy synthetic data
    assert scores[scores["label"] == 1]["score"].mean() > scores[scores["label"] == 0]["score"].mean()
