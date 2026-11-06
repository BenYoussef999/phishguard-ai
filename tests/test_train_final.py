"""Tests for src/train_final.py on small synthetic data."""

import json

import joblib
import pandas as pd
import pytest

from src.preprocessing import make_text
from src.train_final import FINAL_NAME, run_final


def letters(n: int) -> str:
    """Letters-only unique token (digits are normalized away by preprocessing)."""
    out = ""
    while True:
        n, r = divmod(n, 26)
        out = chr(97 + r) + out
        if n == 0:
            return "uniq" + out
        n -= 1


def synthetic_df(per_class: int = 60) -> pd.DataFrame:
    rows, gid = [], 0
    layout = {"ceas08": (0, 1), "enron": (0, 1), "spamassassin": (0, 1), "nazario": (1,)}
    for source, labels in layout.items():
        for label in labels:
            for i in range(per_class):
                cls = "winprize claimfree urgent" if label else "agenda review project"
                rows.append((source, f"s{gid}", f"{cls} sharedword text {i} {letters(gid)}",
                             label, gid))
                gid += 1
    return pd.DataFrame(rows, columns=["source", "subject", "body", "label", "dup_group"])


def run(df, tmp_path, **kw):
    return run_final(df, 0.25, 0.70, tmp_path / "models", tmp_path / "reports",
                     min_df=1, min_source_df=2, **kw)


def test_run_final_saves_model_metadata_and_report(tmp_path):
    df = synthetic_df()
    out = run(df, tmp_path)
    assert out["n_fit"] + out["n_test"] == len(df)

    meta = json.loads(out["meta_path"].read_text(encoding="utf-8"))
    assert meta["thresholds"] == {"t_low": 0.25, "t_high": 0.70}
    assert meta["score_is_probability"] is False
    assert meta["variant"] == "no_email" and meta["limitations"]
    assert set(meta["training_sources"]) == set(df["source"])

    report = json.loads(out["metrics_path"].read_text(encoding="utf-8"))
    assert report["headline"]["total_bad"] > 0
    assert "all" in report["classic_at_threshold_0.5"]

    model = joblib.load(out["model_path"])
    scores = model.predict_proba(make_text(df.head(5)))[:, 1]
    assert ((scores >= 0) & (scores <= 1)).all()


def test_second_run_is_refused_unless_forced(tmp_path):
    df = synthetic_df()
    run(df, tmp_path)
    with pytest.raises(FileExistsError, match="tuning set"):
        run(df, tmp_path)
    run(df, tmp_path, force=True)          # explicit override works


def test_invalid_thresholds_fail_before_anything_is_saved(tmp_path):
    with pytest.raises(ValueError):
        run_final(synthetic_df(), 0.8, 0.2, tmp_path / "models", tmp_path / "reports",
                  min_df=1, min_source_df=2)
    assert not (tmp_path / "models").exists()
    assert not (tmp_path / "reports").exists()


def test_final_model_is_fit_without_the_test_rows(tmp_path):
    from src.splits import train_val_test_split
    df = synthetic_df()
    out = run(df, tmp_path)
    vocab = joblib.load(out["model_path"]).named_steps["tfidf"].vocabulary_
    train, val, test = train_val_test_split(df)      # same deterministic split
    fit = pd.concat([train, val])
    # every email carries a unique word: fit emails must be in the vocabulary,
    # test emails must NOT be (otherwise the test rows leaked into training)
    assert all(letters(g) in vocab for g in fit["dup_group"])
    assert not any(letters(g) in vocab for g in test["dup_group"])
    assert len(test) > 0


def test_model_handles_unseen_and_empty_text(tmp_path):
    out = run(synthetic_df(), tmp_path)
    model = joblib.load(out["model_path"])
    assert len(model.predict(["totally unseen words here", ""])) == 2
