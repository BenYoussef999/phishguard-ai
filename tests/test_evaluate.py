"""Tests for src/evaluate.py with hand-computed expectations."""

import pytest

from src.evaluate import compute_metrics, format_metrics


def test_known_confusion_matrix():
    # truth 0,0,1,1 / pred 0,1,1,0 -> TN=1 FP=1 TP=1 FN=1
    m = compute_metrics([0, 0, 1, 1], [0, 1, 1, 0])
    assert (m["tn"], m["fp"], m["tp"], m["fn"]) == (1, 1, 1, 1)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["false_positive_rate"] == pytest.approx(0.5)
    assert m["recall_only"] is False


def test_recall_only_when_no_legitimate_emails():
    m = compute_metrics([1, 1, 1], [1, 1, 0])
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["fn"] == 1
    assert m["precision"] is None and m["f1"] is None
    assert m["false_positive_rate"] is None
    assert m["recall_only"] is True


def test_no_positive_predictions_gives_undefined_precision():
    m = compute_metrics([0, 1, 1], [0, 0, 0])
    assert m["precision"] is None and m["recall"] == 0.0 and m["f1"] is None


def test_format_metrics_runs_for_both_modes():
    assert "precision" in format_metrics("a", compute_metrics([0, 1], [0, 1]))
    assert "not defined" in format_metrics("b", compute_metrics([1, 1], [1, 0]))
