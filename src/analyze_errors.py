"""Error analysis for the baseline model on one held-out source.

Trains on every OTHER source (leave-one-source-out), then looks at the held-out
source's mistakes: which features the model leans on, how confident its misses
are, which words distinguish missed emails from caught ones, and which
legitimate emails are wrongly flagged.

Usage (from the project root):
    python -m src.analyze_errors                    # held-out source: nazario
    python -m src.analyze_errors --held-out enron
    python -m src.analyze_errors | Tee-Object reports\\error_analysis_nazario.txt

Output contains short snippets of corpus emails (ASCII-sanitized, never
rendered or opened). Read it, but do not click anything you find in it.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.data import build_dataset
from src.preprocessing import clean_text, make_text
from src.splits import RANDOM_SEED, leave_one_source_out
from src.train import fit_pipeline

THRESHOLD = 0.5   # placeholder decision threshold, same as the baseline model


def safe(text, width: int) -> str:
    """Single-line, ASCII-only, truncated text for console output."""
    s = re.sub(r"\s+", " ", "" if text is None else str(text)).strip()
    return s.encode("ascii", errors="replace").decode("ascii")[:width]


def top_terms(pipe, k: int = 25):
    """Return (toward_bad, toward_legit) lists of (term, weight)."""
    weights = pipe.named_steps["clf"].coef_[0]
    names = pipe.named_steps["tfidf"].get_feature_names_out()
    order = np.argsort(weights)
    toward_legit = [(str(names[i]), float(weights[i])) for i in order[:k]]
    toward_bad = [(str(names[i]), float(weights[i])) for i in order[::-1][:k]]
    return toward_bad, toward_legit


def histogram_counts(values, lo: float, hi: float, bins: int = 5):
    """Counts of values per equal-width bin in [lo, hi]. Returns list of (lo, hi, n)."""
    counts, edges = np.histogram(np.asarray(values, dtype=float),
                                 bins=bins, range=(lo, hi))
    return [(float(edges[i]), float(edges[i + 1]), int(counts[i]))
            for i in range(bins)]


def contrastive_terms(texts_a, texts_b, k: int = 15, min_df: int = 10):
    """Words in a larger share of group A's emails than group B's.

    Returns (term, share_in_A, share_in_B), most over-represented first.
    Needs at least 20 emails in each group, otherwise returns [].
    """
    if len(texts_a) < 20 or len(texts_b) < 20:
        return []
    vec = CountVectorizer(preprocessor=clean_text, binary=True, min_df=min_df,
                          stop_words="english")
    vec.fit(list(texts_a) + list(texts_b))
    df_a = np.asarray(vec.transform(texts_a).sum(axis=0)).ravel()
    df_b = np.asarray(vec.transform(texts_b).sum(axis=0)).ravel()
    share_a = (df_a + 1) / (len(texts_a) + 2)
    share_b = (df_b + 1) / (len(texts_b) + 2)
    ratio = share_a / share_b
    names = vec.get_feature_names_out()
    result = []
    for i in np.argsort(-ratio):
        if df_a[i] >= min_df:
            result.append((str(names[i]), float(share_a[i]), float(share_b[i])))
        if len(result) == k:
            break
    return result


def _print_terms(title, terms):
    print(f"\n{title}")
    for term, weight in terms:
        print(f"  {term:<28} {weight:+.3f}")


def _print_contrast(title, rows, label_a, label_b):
    print(f"\n{title}")
    if not rows:
        print("  (not enough emails in one of the groups for this comparison)")
        return
    print(f"  {'word':<22}{label_a:>14}{label_b:>14}")
    for term, a, b in rows:
        print(f"  {term:<22}{a:>13.0%} {b:>13.0%}")


def _print_samples(title, frame, most_wrong_is_highest=False, n_first=5, n_random=5):
    """Print the most confidently wrong emails first, then a random few.

    For missed bad emails the most wrong have the LOWEST score; for false
    alarms they have the HIGHEST. Scores are always printed as they are.
    """
    print(f"\n{title}")
    probs = frame["prob"].to_numpy()
    order = np.argsort(-probs if most_wrong_is_highest else probs)
    shown = [int(i) for i in order[:n_first]]
    rng = np.random.default_rng(RANDOM_SEED)
    rest = [i for i in range(len(frame)) if i not in set(shown)]
    if rest:
        shown += [int(i) for i in rng.choice(rest, size=min(n_random, len(rest)),
                                              replace=False)]
    for i in shown:
        row = frame.iloc[i]
        print(f"  score={probs[i]:.3f} | {safe(row['subject'], 60)}")
        print(f"      {safe(row['body'], 130)}")


def analyze(df: pd.DataFrame, held_out: str) -> None:
    train, test, removed = leave_one_source_out(df, held_out)
    print(f"Held out: {held_out} ({len(test):,} emails). "
          f"Training on {len(train):,} emails from the other sources "
          f"({removed:,} overlapping removed). Threshold = {THRESHOLD}.")
    pipe = fit_pipeline(train)
    probs = pipe.predict_proba(make_text(test))[:, 1]
    pred = (probs >= THRESHOLD).astype(int)
    test = test.assign(prob=probs, pred=pred,
                       body_len=test["body"].str.len())

    fn = test[(test["label"] == 1) & (test["pred"] == 0)]
    tp = test[(test["label"] == 1) & (test["pred"] == 1)]
    fp = test[(test["label"] == 0) & (test["pred"] == 1)]
    tn = test[(test["label"] == 0) & (test["pred"] == 0)]
    print(f"Missed bad emails (FN): {len(fn):,}   caught (TP): {len(tp):,}   "
          f"legit flagged (FP): {len(fp):,}   legit passed (TN): {len(tn):,}")

    print("\n" + "=" * 60)
    print("1. WHAT THE MODEL RELIES ON (this fold's model, weights toward bad / legit)")
    bad, legit = top_terms(pipe)
    _print_terms("Strongest signs of BAD email:", bad)
    _print_terms("Strongest signs of LEGITIMATE email:", legit)
    print("\nLook for source-specific words (project names, companies, list tags) "
          "among the legitimate signs: those are shortcuts, not email behavior.")

    print("\n" + "=" * 60)
    print("2. MISSED BAD EMAILS")
    if len(fn):
        print("How far below the threshold were the misses? (score = model's 0-1 output)")
        for lo, hi, n in histogram_counts(fn["prob"], 0.0, THRESHOLD, 5):
            print(f"  score {lo:.1f}-{hi:.1f}: {n:>6,} ({n / len(fn):.0%})")
        if len(tp):
            print(f"\nMedian body length (characters): missed={fn['body_len'].median():,.0f}"
                  f"   caught={tp['body_len'].median():,.0f}")
            _print_contrast("Words over-represented in MISSED vs CAUGHT emails "
                            "(share of emails containing the word):",
                            contrastive_terms(make_text(fn), make_text(tp)),
                            "missed", "caught")
        _print_samples("Sample misses (5 most confidently wrong, then 5 random):", fn)
    else:
        print("No missed bad emails.")

    print("\n" + "=" * 60)
    print("3. LEGITIMATE EMAILS WRONGLY FLAGGED")
    if len(fp):
        print("How far above the threshold were the false alarms?")
        for lo, hi, n in histogram_counts(fp["prob"], THRESHOLD, 1.0, 5):
            print(f"  score {lo:.1f}-{hi:.1f}: {n:>6,} ({n / len(fp):.0%})")
        _print_contrast("Words over-represented in FALSE ALARMS vs correctly passed "
                        "legitimate emails:",
                        contrastive_terms(make_text(fp), make_text(tn)),
                        "false alarm", "passed")
        _print_samples("Sample false alarms (5 most confidently wrong, then 5 random):",
                       fp, most_wrong_is_highest=True)
    else:
        print("This source has no false alarms (or no legitimate emails).")


def main() -> int:
    parser = argparse.ArgumentParser(description="Error analysis on a held-out source.")
    parser.add_argument("--held-out", default="nazario")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    try:
        df, _ = build_dataset(args.data_dir)
        analyze(df, args.held_out)
    except (FileNotFoundError, ValueError) as err:
        print(f"Error: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
