"""Train and evaluate the baseline: TF-IDF + Logistic Regression.

What this script does:
  1. Builds the combined dataset (src/data.py) and group-aware splits (src/splits.py).
  2. Fits the pipeline on the TRAIN split only, reports validation metrics.
  3. Runs leave-one-source-out: train on the other sources, test on the held-out one.
  4. Saves the fitted pipeline, metadata and metrics (unless --quick).

The 20% TEST split is deliberately not used here. It is reserved for the final
model after model selection.

Usage (from the project root):
    python -m src.train            # full run (can take several minutes)
    python -m src.train --quick    # tiny smoke run, saves nothing
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import build_dataset
from src.evaluate import compute_metrics, format_metrics
from src.preprocessing import clean_text, make_text
from src.splits import RANDOM_SEED, leave_one_source_out, train_val_test_split

MODEL_NAME = "baseline_logreg"
LABEL_MEANING = ("1 = unwanted/malicious (phishing from Nazario; spam from CEAS_08, "
                 "Enron, SpamAssassin), 0 = legitimate")


def build_pipeline(min_df: int = 5, max_features: int = 200_000,
                   seed: int = RANDOM_SEED) -> Pipeline:
    """TF-IDF (1-2 word n-grams) followed by Logistic Regression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,   # also lowercases (done inside clean_text)
            ngram_range=(1, 2),
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=seed)),
    ])


def fit_pipeline(train_df, **kwargs) -> Pipeline:
    """Fit on training rows only, so the vocabulary never sees other data."""
    pipe = build_pipeline(**kwargs)
    pipe.fit(make_text(train_df), train_df["label"])
    return pipe


def evaluate_pipeline(pipe: Pipeline, df) -> dict:
    return compute_metrics(df["label"], pipe.predict(make_text(df)))


def _per_source(pipe, df, title_prefix, results):
    for source in sorted(df["source"].unique()):
        part = df[df["source"] == source]
        m = evaluate_pipeline(pipe, part)
        results[source] = m
        print(format_metrics(f"{title_prefix} / {source}", m))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the baseline model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--quick", action="store_true",
                        help="Use up to 2,000 rows per source; save nothing.")
    args = parser.parse_args()

    try:
        df, _ = build_dataset(args.data_dir)
    except (FileNotFoundError, ValueError) as err:
        print(f"Error: {err}")
        return 1

    min_df = 5
    if args.quick:
        parts = [df[df["source"] == src].sample(min(n, 2000), random_state=RANDOM_SEED)
                 for src, n in df["source"].value_counts().items()]
        df = pd.concat(parts, ignore_index=True)
        min_df = 2
        print("QUICK MODE: subsampled data, nothing will be saved.\n")

    train, val, _test = train_val_test_split(df)   # test split is left untouched
    print(f"train={len(train):,}  val={len(val):,}  (test split held back)\n")

    t0 = time.time()
    pipe = fit_pipeline(train, min_df=min_df)
    print(f"Fitted baseline in {time.time() - t0:.0f}s; "
          f"vocabulary size = {len(pipe.named_steps['tfidf'].vocabulary_):,}\n")

    print("=" * 60)
    print("VALIDATION (group-aware split; same sources as training)")
    results = {"validation": {}, "leave_one_source_out": {}}
    results["validation"]["all"] = evaluate_pipeline(pipe, val)
    print(format_metrics("validation / ALL", results["validation"]["all"]))
    _per_source(pipe, val, "validation", results["validation"])

    print("\n" + "=" * 60)
    print("LEAVE-ONE-SOURCE-OUT (train on the other sources, test on the held-out one)")
    for source in sorted(df["source"].unique()):
        tr, te, removed = leave_one_source_out(df, source)
        t0 = time.time()
        fold_pipe = fit_pipeline(tr, min_df=min_df)
        m = evaluate_pipeline(fold_pipe, te)
        m["removed_from_train"] = removed
        results["leave_one_source_out"][source] = m
        print(format_metrics(f"held out: {source}", m) +
              f"\n  (trained on {len(tr):,} rows in {time.time() - t0:.0f}s; "
              f"{removed:,} overlapping rows removed)")

    if args.quick:
        print("\nQuick run complete; nothing saved.")
        return 0

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.models_dir / f"{MODEL_NAME}.joblib"
    joblib.dump(pipe, model_path)
    meta = {
        "model": MODEL_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "seed": RANDOM_SEED,
        "label_meaning": LABEL_MEANING,
        "n_train": len(train), "n_val": len(val),
        "decision_threshold": 0.5,
        "note": "Threshold is a placeholder; it is not yet tuned or calibrated.",
    }
    (args.models_dir / f"{MODEL_NAME}_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    (args.reports_dir / f"{MODEL_NAME}_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {model_path}\n       {args.models_dir / (MODEL_NAME + '_meta.json')}"
          f"\n       {args.reports_dir / (MODEL_NAME + '_metrics.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
