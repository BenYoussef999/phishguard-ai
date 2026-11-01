"""Tests for src/experiments.py on tiny synthetic multi-source data."""

import pandas as pd
import pytest

from src.experiments import (VARIANTS, build_variant_pipeline, make_cleaner,
                             run_variant, source_robust_vocabulary,
                             summarize_variant)
from src.preprocessing import clean_text, clean_text_variant, make_text
from src.train import fit_pipeline


def multi_source_df(n: int = 20) -> pd.DataFrame:
    """4 sources. 'sharedword' occurs in several sources; 'onlyenron' in one."""
    rows, gid = [], 0
    spec = {"ceas08": (0, 1), "enron": (0, 1), "spamassassin": (0, 1), "nazario": (1,)}
    for source, labels in spec.items():
        for label in labels:
            for i in range(n):
                cls = "winprize claimfree urgent" if label else "agenda review project"
                extra = "onlyenron" if source == "enron" else "sharedword"
                rows.append((source, f"subj {gid}", f"{cls} {extra} text {i} {source}x",
                             label, gid))
                gid += 1
    return pd.DataFrame(rows, columns=["source", "subject", "body", "label", "dup_group"])


def test_email_mode_drop_removes_address_without_placeholder():
    assert "emailtoken" in clean_text_variant("write to a@b.com now", "token")
    dropped = clean_text_variant("write to a@b.com now", "drop")
    assert "emailtoken" not in dropped and "a@b" not in dropped
    assert dropped == "write to now"


def test_invalid_email_mode_is_rejected_and_default_is_unchanged():
    with pytest.raises(ValueError):
        clean_text_variant("x", "bogus")
    assert clean_text("a@b.com") == clean_text_variant("a@b.com", "token")


def test_baseline_variant_matches_the_original_training_pipeline():
    df = multi_source_df()
    original = fit_pipeline(df, min_df=1)
    variant = build_variant_pipeline(df, min_df=1, **VARIANTS["baseline"])
    text = make_text(df)
    assert list(original.predict(text)) == list(variant.predict(text))
    assert (original.named_steps["tfidf"].vocabulary_
            == variant.named_steps["tfidf"].vocabulary_)


def test_robust_vocabulary_keeps_shared_terms_and_drops_single_source_terms():
    df = multi_source_df()
    train = df[df["source"] != "nazario"]
    vocab = source_robust_vocabulary(train, make_cleaner("token"), min_source_df=2)
    assert "sharedword" in vocab          # present in ceas08 and spamassassin
    assert "onlyenron" not in vocab       # present in enron only
    assert "winprize" in vocab


def test_robust_vocabulary_uses_only_the_rows_it_is_given():
    df = multi_source_df()
    train = df[df["source"].isin(["ceas08", "spamassassin"])]
    vocab = source_robust_vocabulary(train, make_cleaner("token"), min_source_df=2)
    assert "onlyenron" not in vocab and "enronx" not in vocab


def test_robust_vocabulary_errors_when_nothing_is_shared():
    df = multi_source_df()
    one_source = df[df["source"] == "enron"]
    with pytest.raises(ValueError, match="source-robust"):
        source_robust_vocabulary(one_source, make_cleaner("token"), min_source_df=2)


def test_svm_variant_fits_predicts_and_has_no_probabilities():
    df = multi_source_df()
    pipe = build_variant_pipeline(df, min_df=1, **VARIANTS["svm_baseline"])
    assert set(pipe.predict(make_text(df))) <= {0, 1}
    assert not hasattr(pipe.named_steps["clf"], "predict_proba")


def test_run_variant_covers_every_source_and_flags_recall_only():
    df = multi_source_df()
    res = run_variant(df, "robust_no_email", VARIANTS["robust_no_email"],
                      min_df=1, min_source_df=2, verbose=False)
    assert set(res) == {"ceas08", "enron", "spamassassin", "nazario"}
    assert res["nazario"]["recall_only"] is True
    assert res["ceas08"]["recall_only"] is False


def test_summarize_variant_skips_undefined_values():
    res = {
        "ceas08": {"f1": 0.8, "recall": 0.9, "false_positive_rate": 0.1},
        "enron": {"f1": None, "recall": 0.7, "false_positive_rate": None},
        "nazario": {"recall": 0.5},
    }
    s = summarize_variant(res)
    assert s["nazario_recall"] == 0.5
    assert s["macro_f1"] == pytest.approx(0.8)
    assert s["macro_recall"] == pytest.approx(0.8)
    assert s["macro_fpr"] == pytest.approx(0.1)


def test_every_variant_spec_is_valid_and_svm_no_email_exists():
    for name, spec in VARIANTS.items():
        assert set(spec) == {"email_mode", "robust_vocab", "model"}, name
        assert spec["email_mode"] in ("token", "drop")
        assert spec["model"] in ("logreg", "svm")
    assert VARIANTS["svm_no_email"] == dict(email_mode="drop", robust_vocab=False,
                                            model="svm")
