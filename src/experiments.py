"""Controlled experiments: change ONE thing at a time, compare to the baseline.

Every variant is evaluated with the same leave-one-source-out protocol:
train on the other sources, test on the held-out one. All fitting (including
the source-robust vocabulary) uses the training rows of that fold only.

IMPORTANT HONESTY NOTE: choosing a variant because it scores best on these
folds makes these numbers optimistic for that variant. The 20% test split is
not used here and stays reserved for the final model.

Usage (from the project root):
    python -m src.experiments                                  # 4 logistic-regression variants
    python -m src.experiments --variants all                   # includes the SVM variants
    python -m src.experiments --variants baseline svm_baseline
    python -m src.experiments --quick                          # small smoke run, saves nothing
"""

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data import build_dataset
from src.evaluate import compute_metrics
from src.preprocessing import clean_text_variant, make_text
from src.splits import RANDOM_SEED, leave_one_source_out

VARIANTS = {
    "baseline":            dict(email_mode="token", robust_vocab=False, model="logreg"),
    "no_email":            dict(email_mode="drop",  robust_vocab=False, model="logreg"),
    "robust_vocab":        dict(email_mode="token", robust_vocab=True,  model="logreg"),
    "robust_no_email":     dict(email_mode="drop",  robust_vocab=True,  model="logreg"),
    "svm_baseline":        dict(email_mode="token", robust_vocab=False, model="svm"),
    "svm_robust_no_email": dict(email_mode="drop",  robust_vocab=True,  model="svm"),
}
DEFAULT_VARIANTS = ["baseline", "no_email", "robust_vocab", "robust_no_email"]
LABELED_SOURCES = ["ceas08", "enron", "spamassassin"]   # have both classes


def make_cleaner(email_mode: str):
    """A picklable cleaning function (a partial of a module-level function)."""
    return partial(clean_text_variant, email_mode=email_mode)


def source_robust_vocabulary(train_df: pd.DataFrame, cleaner, min_source_df: int = 3,
                             min_sources: int = 2, max_features: int = 200_000) -> list:
    """Terms that appear in at least `min_source_df` emails of at least
    `min_sources` different training sources.

    Computed from `train_df` only, so the held-out source never influences it.
    """
    counter = CountVectorizer(preprocessor=cleaner, ngram_range=(1, 2),
                              binary=True, min_df=min_source_df)
    X = counter.fit_transform(make_text(train_df)).tocsr()
    terms = counter.get_feature_names_out()
    sources = train_df["source"].to_numpy()
    n_sources_present = np.zeros(X.shape[1], dtype=int)
    for src in np.unique(sources):
        df_in_source = np.asarray(X[sources == src].sum(axis=0)).ravel()
        n_sources_present += (df_in_source >= min_source_df)
    keep = np.flatnonzero(n_sources_present >= min_sources)
    if len(keep) == 0:
        raise ValueError("No term appears in enough training sources; "
                         "cannot build a source-robust vocabulary.")
    if len(keep) > max_features:
        total_df = np.asarray(X.sum(axis=0)).ravel()
        keep = keep[np.argsort(-total_df[keep])[:max_features]]
    return sorted(str(terms[i]) for i in keep)


def build_variant_pipeline(train_df: pd.DataFrame, email_mode: str, robust_vocab: bool,
                           model: str, min_df: int = 5, min_source_df: int = 3,
                           seed: int = RANDOM_SEED) -> Pipeline:
    """Fit one experiment variant on `train_df` only."""
    cleaner = make_cleaner(email_mode)
    if robust_vocab:
        vocabulary = source_robust_vocabulary(train_df, cleaner,
                                              min_source_df=min_source_df)
        tfidf = TfidfVectorizer(preprocessor=cleaner, ngram_range=(1, 2),
                                vocabulary=vocabulary, sublinear_tf=True)
    else:
        tfidf = TfidfVectorizer(preprocessor=cleaner, ngram_range=(1, 2),
                                min_df=min_df, max_features=200_000,
                                sublinear_tf=True)
    if model == "logreg":
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                 random_state=seed)
    elif model == "svm":
        # No predict_proba: fine for comparing decisions, but a chosen SVM
        # would need calibration before we could show probabilities.
        clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000,
                        random_state=seed)
    else:
        raise ValueError(f"Unknown model '{model}'")
    pipe = Pipeline([("tfidf", tfidf), ("clf", clf)])
    pipe.fit(make_text(train_df), train_df["label"])
    return pipe


def run_variant(df: pd.DataFrame, name: str, spec: dict, min_df: int = 5,
                min_source_df: int = 3, verbose: bool = True) -> dict:
    """Leave-one-source-out results for one variant: {source: metrics}."""
    results = {}
    for held_out in sorted(df["source"].unique()):
        train, test, removed = leave_one_source_out(df, held_out)
        t0 = time.time()
        pipe = build_variant_pipeline(train, min_df=min_df,
                                      min_source_df=min_source_df, **spec)
        m = compute_metrics(test["label"], pipe.predict(make_text(test)))
        m["vocab_size"] = len(pipe.named_steps["tfidf"].vocabulary_)
        m["seconds"] = round(time.time() - t0, 1)
        results[held_out] = m
        if verbose:
            if m["recall_only"]:
                detail = f"recall={m['recall']:.4f} missed={m['fn']:,}"
            else:
                detail = (f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
                          f"FN={m['fn']:,} FP={m['fp']:,}")
            print(f"  {name:<20} held out {held_out:<13} {detail}   "
                  f"[vocab {m['vocab_size']:,}, {m['seconds']:.0f}s]", flush=True)
    return results


def summarize_variant(results: dict) -> dict:
    """One row of headline numbers. Undefined values (None) are skipped."""
    labeled = [results[s] for s in LABELED_SOURCES if s in results]

    def mean_of(key):
        vals = [m[key] for m in labeled if m[key] is not None]
        return float(np.mean(vals)) if vals else None

    naz = results.get("nazario")
    return {
        "nazario_recall": naz["recall"] if naz else None,
        "macro_f1": mean_of("f1"),
        "macro_recall": mean_of("recall"),
        "macro_fpr": mean_of("false_positive_rate"),
    }


def subsample_per_source(df: pd.DataFrame, n: int) -> pd.DataFrame:
    parts = [df[df["source"] == src].sample(min(cnt, n), random_state=RANDOM_SEED)
             for src, cnt in df["source"].value_counts().items()]
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run controlled experiments.")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                        help="Names to run, or 'all'. Available: " + ", ".join(VARIANTS))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--quick", action="store_true",
                        help="Use up to 2,000 rows per source; save nothing.")
    args = parser.parse_args()

    names = list(VARIANTS) if args.variants == ["all"] else args.variants
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        print(f"Error: unknown variant(s) {unknown}. Available: {list(VARIANTS)}")
        return 1

    try:
        df, _ = build_dataset(args.data_dir)
    except (FileNotFoundError, ValueError) as err:
        print(f"Error: {err}")
        return 1

    min_df, min_source_df = 5, 3
    if args.quick:
        df = subsample_per_source(df, 2000)
        min_df, min_source_df = 2, 2
        print("QUICK MODE: subsampled data, nothing will be saved.\n")

    all_results, rows = {}, []
    for name in names:
        print(f"\n=== {name}: {VARIANTS[name]}", flush=True)
        results = run_variant(df, name, VARIANTS[name], min_df, min_source_df)
        all_results[name] = results
        rows.append({"variant": name, **summarize_variant(results)})

    table = pd.DataFrame(rows).set_index("variant")
    print("\n" + "=" * 70)
    print("SUMMARY (leave-one-source-out). Compare every row with 'baseline'.")
    print("nazario_recall: share of real phishing caught (higher is better)")
    print("macro_*: average over CEAS_08, Enron, SpamAssassin")
    print("macro_fpr: share of legitimate email wrongly flagged (lower is better)\n")
    print(table.round(4).to_string())
    print("\nA recall gain that comes with a much higher macro_fpr is not an "
          "improvement. Variants were compared on these same folds, so treat the "
          "winner's numbers as optimistic.")

    if not args.quick:
        args.reports_dir.mkdir(parents=True, exist_ok=True)
        out = args.reports_dir / "experiments.json"
        out.write_text(json.dumps(
            {"variants": {n: VARIANTS[n] for n in names},
             "results": all_results,
             "summary": table.reset_index().to_dict(orient="records")},
            indent=2), encoding="utf-8")
        print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
