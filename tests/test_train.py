"""Tests for src/train.py on tiny synthetic data (no real dataset needed)."""

import joblib
import pandas as pd

from src.preprocessing import make_text
from src.train import build_pipeline, evaluate_pipeline, fit_pipeline


def tiny_df(n: int = 30) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(("s", f"win prize {i}", "claim free money now urgent verify account",
                     1))
        rows.append(("s", f"meeting {i}", "agenda for the project review next week",
                     0))
    return pd.DataFrame(rows, columns=["source", "subject", "body", "label"])


def test_pipeline_fits_predicts_and_separates_obvious_cases():
    df = tiny_df()
    pipe = fit_pipeline(df, min_df=1)
    proba = pipe.predict_proba(make_text(df.head(2)))
    assert proba.shape == (2, 2)
    m = evaluate_pipeline(pipe, df)
    assert m["fn"] == 0 and m["fp"] == 0


def test_vocabulary_is_learned_from_training_data_only():
    train = tiny_df()
    test = pd.DataFrame([("s", "x", "zzzuniqueword appears only in test", 1)],
                        columns=train.columns)
    pipe = fit_pipeline(train, min_df=1)
    vocab = pipe.named_steps["tfidf"].vocabulary_
    assert "zzzuniqueword" not in vocab
    pipe.predict(make_text(test))      # unseen words must not break prediction


def test_saved_pipeline_reloads_and_predicts_identically(tmp_path):
    df = tiny_df()
    pipe = fit_pipeline(df, min_df=1)
    path = tmp_path / "model.joblib"
    joblib.dump(pipe, path)
    reloaded = joblib.load(path)
    text = make_text(df.head(5))
    assert list(pipe.predict(text)) == list(reloaded.predict(text))


def test_pipeline_handles_empty_and_missing_text():
    pipe = fit_pipeline(tiny_df(), min_df=1)
    assert len(pipe.predict(["\n", "", "<<<>>>"])) == 3


def test_build_pipeline_is_deterministic():
    a, b = build_pipeline(min_df=1), build_pipeline(min_df=1)
    assert a.named_steps["clf"].random_state == b.named_steps["clf"].random_state
