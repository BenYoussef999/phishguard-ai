"""Tests for src/predict.py. Builds a tiny model on synthetic data in a temp folder."""

import json
import logging
import random
import time

import joblib
import pytest
import sklearn

from src.experiments import VARIANTS, build_variant_pipeline
from src.predict import (BAND_ADVICE, BAND_LABELS, MAX_INPUT_CHARS, InputError,
                         ModelLoadError, band_for_score, load_model, predict_email,
                         validate_text_inputs)
from src.preprocessing import MAX_CHARS
from src.thresholds import assign_bands


def tiny_training_df():
    import pandas as pd
    rows, gid = [], 0
    for source, labels in {"ceas08": (0, 1), "enron": (0, 1), "nazario": (1,)}.items():
        for label in labels:
            for i in range(40):
                cls = ("winprize claimfree urgent verify account" if label
                       else "agenda review project meeting notes")
                rows.append((source, f"s{gid}", f"{cls} sharedword {i}", label, gid))
                gid += 1
    return pd.DataFrame(rows, columns=["source", "subject", "body", "label", "dup_group"])


def write_model(folder, t_low=0.25, t_high=0.70, **meta_overrides):
    pipe = build_variant_pipeline(tiny_training_df(), min_df=1, min_source_df=2,
                                  **VARIANTS["no_email"])
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, folder / "m.joblib")
    meta = {"model": "m", "sklearn_version": sklearn.__version__,
            "thresholds": {"t_low": t_low, "t_high": t_high},
            "score_is_probability": False, "limitations": ["x"]}
    meta.update(meta_overrides)
    (folder / "m_meta.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    folder = tmp_path_factory.mktemp("model")
    write_model(folder)
    return load_model(folder, name="m")


def test_obvious_bad_and_legit_examples_get_the_expected_bands(bundle):
    bad = predict_email(bundle, "winprize claimfree", "urgent verify account winprize")
    ok = predict_email(bundle, "meeting agenda", "project review notes meeting")
    assert bad.band == "bad" and bad.label == BAND_LABELS["bad"]
    assert ok.band == "legit" and ok.label == BAND_LABELS["legit"]
    assert 0.0 <= bad.score <= 1.0 and 0.0 <= ok.score <= 1.0
    assert bad.score > ok.score


def test_subject_only_and_body_only_both_work(bundle):
    assert predict_email(bundle, "meeting agenda", "").band in BAND_LABELS
    assert predict_email(bundle, None, "meeting agenda").band in BAND_LABELS


def test_prediction_is_deterministic(bundle):
    a = predict_email(bundle, "x", "winprize urgent")
    b = predict_email(bundle, "x", "winprize urgent")
    assert a == b


def test_advice_never_claims_safety_or_shows_percentages(bundle):
    assert "not a guarantee" in BAND_ADVICE["legit"]
    for text in BAND_ADVICE.values():
        assert "%" not in text and "100" not in text
    assert "is safe" in BAND_ADVICE["legit"]        # only ever inside "not a guarantee..."
    assert BAND_ADVICE["legit"].index("not a guarantee") < BAND_ADVICE["legit"].index("is safe")


def test_band_for_score_matches_threshold_module_at_boundaries():
    rng = random.Random(0)
    scores = [0.0, 0.25, 0.7, 1.0] + [rng.random() for _ in range(200)]
    expected = list(assign_bands(scores, 0.25, 0.70))
    assert [band_for_score(s, 0.25, 0.70) for s in scores] == expected


# ---------- input handling ----------
def test_empty_and_wrong_type_inputs_are_rejected(bundle):
    for args in [("", ""), ("   ", "\n\t"), (None, None), ("\x00", "\x00")]:
        with pytest.raises(InputError):
            predict_email(bundle, *args)
    for args in [(123, "x"), ("x", ["list"]), (b"bytes", "x")]:
        with pytest.raises(InputError, match="must be text"):
            predict_email(bundle, *args)


def test_oversized_input_is_rejected_not_processed(bundle):
    with pytest.raises(InputError, match="too long"):
        predict_email(bundle, "x", "a" * (MAX_INPUT_CHARS + 1))


def test_long_input_is_truncated_with_a_visible_note(bundle):
    result = predict_email(bundle, "hello", "word " * (MAX_CHARS // 2))
    assert result.truncated is True
    assert any("first" in n for n in result.notes)
    short = predict_email(bundle, "hello", "short body")
    assert short.truncated is False and short.notes == ()


def test_null_bytes_are_removed_and_odd_text_does_not_crash(bundle):
    assert validate_text_inputs("a\x00b", "c") == ("ab", "c")
    for body in ["\U0001F512 \u202e vérifiez", "<script>alert(1)</script>", "a@" * 5000]:
        assert predict_email(bundle, "s", body).band in BAND_LABELS


def test_adversarial_input_is_fast(bundle):
    start = time.perf_counter()
    predict_email(bundle, "s", "a." * 10000)
    assert time.perf_counter() - start < 1.0


def test_message_content_is_never_logged_or_printed(bundle, caplog, capsys):
    secret = "zebra-secret-token-98765"
    with caplog.at_level(logging.DEBUG):
        predict_email(bundle, secret, f"body mentioning {secret}")
    out = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in out.out and secret not in out.err


# ---------- model loading ----------
def test_missing_model_and_metadata_give_helpful_errors(tmp_path):
    with pytest.raises(ModelLoadError, match="train_final"):
        load_model(tmp_path, name="m")
    write_model(tmp_path)
    (tmp_path / "m_meta.json").unlink()
    with pytest.raises(ModelLoadError, match="metadata not found"):
        load_model(tmp_path, name="m")


def test_corrupt_metadata_and_model_files_are_reported(tmp_path):
    write_model(tmp_path)
    (tmp_path / "m_meta.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelLoadError, match="not valid JSON"):
        load_model(tmp_path, name="m")
    write_model(tmp_path)
    (tmp_path / "m.joblib").write_bytes(b"this is not a pickle")
    with pytest.raises(ModelLoadError, match="could not be loaded"):
        load_model(tmp_path, name="m")


@pytest.mark.parametrize("t_low,t_high", [(0.8, 0.2), (-0.1, 0.5), (0.2, 1.5)])
def test_invalid_thresholds_in_metadata_are_refused(tmp_path, t_low, t_high):
    write_model(tmp_path, t_low=t_low, t_high=t_high)
    with pytest.raises(ModelLoadError, match="thresholds"):
        load_model(tmp_path, name="m")


def test_metadata_must_declare_scores_are_not_probabilities(tmp_path):
    write_model(tmp_path, score_is_probability=True)
    with pytest.raises(ModelLoadError, match="probabilities"):
        load_model(tmp_path, name="m")


def test_sklearn_version_mismatch_is_refused_unless_allowed(tmp_path):
    write_model(tmp_path, sklearn_version="0.1.0")
    with pytest.raises(ModelLoadError, match="Retrain"):
        load_model(tmp_path, name="m")
    assert load_model(tmp_path, name="m", allow_version_mismatch=True)


def test_a_non_pipeline_object_is_refused(tmp_path):
    write_model(tmp_path)
    joblib.dump({"not": "a model"}, tmp_path / "m.joblib")
    with pytest.raises(ModelLoadError, match="classification pipeline"):
        load_model(tmp_path, name="m")
