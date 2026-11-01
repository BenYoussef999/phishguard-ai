"""Tests for the pure helper functions in src/analyze_errors.py."""

import pandas as pd

from src.analyze_errors import (_print_samples, contrastive_terms,
                                histogram_counts, safe, top_terms)
from src.train import fit_pipeline


def tiny_df(n: int = 30) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(("s", f"win prize {i}", "claim free money now urgent", 1))
        rows.append(("s", f"meeting {i}", "agenda for the project review", 0))
    return pd.DataFrame(rows, columns=["source", "subject", "body", "label"])


def test_top_terms_point_in_the_right_direction():
    pipe = fit_pipeline(tiny_df(), min_df=1)
    bad, legit = top_terms(pipe, k=5)
    assert len(bad) == 5 and len(legit) == 5
    assert all(w > 0 for _, w in bad) and all(w < 0 for _, w in legit)
    assert bad[0][0] in {"claim", "free", "money", "urgent", "win", "prize", "now"} \
        or any(t in bad[0][0] for t in ("claim", "free", "money", "urgent", "win"))


def test_histogram_counts_cover_all_values():
    bins = histogram_counts([0.01, 0.05, 0.12, 0.49], 0.0, 0.5, 5)
    assert sum(n for _, _, n in bins) == 4
    assert bins[0][2] == 2 and bins[4][2] == 1


def test_contrastive_terms_finds_the_distinguishing_word():
    a = ["alpha shared words here"] * 25
    b = ["beta shared words here"] * 25
    terms = contrastive_terms(a, b, k=3, min_df=5)
    assert terms[0][0] == "alpha"
    assert terms[0][1] > terms[0][2]


def test_contrastive_terms_needs_enough_emails():
    assert contrastive_terms(["a b"] * 5, ["c d"] * 50) == []


def test_safe_makes_single_line_ascii_and_truncates():
    out = safe("Caf\u00e9\n\ntext\t" + "x" * 500, 20)
    assert "\n" not in out and out.isascii() and len(out) <= 20
    assert safe(None, 10) == ""


def test_samples_print_real_scores_and_order_most_wrong_first(capsys):
    frame = pd.DataFrame({"subject": ["low", "mid", "high"],
                          "body": ["a", "b", "c"],
                          "prob": [0.10, 0.30, 0.45]})
    _print_samples("misses", frame, n_first=1, n_random=0)
    out = capsys.readouterr().out
    assert "score=0.100" in out and "-" not in out.split("score=")[1].split("|")[0]
    _print_samples("false alarms", frame, most_wrong_is_highest=True,
                   n_first=1, n_random=0)
    out = capsys.readouterr().out
    assert "score=0.450" in out
