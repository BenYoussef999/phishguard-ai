"""Train the final model and evaluate it ONCE on the held-back test split.

Config: preprocessing variant 'no_email' + TF-IDF + Logistic Regression, fit on
train + validation. Thresholds for the three risk bands are required arguments,
chosen beforehand from the out-of-source analysis (src/thresholds.py).

The test split is in-distribution (same four sources as training), so these
numbers are optimistic for unfamiliar email. The leave-one-source-out results
remain the honest estimate of generalization.

DO NOT change the model or thresholds based on this output: the test split would
no longer be untouched. The script refuses to run twice unless --force is given.

Usage (from the project root):
    python -m src.train_final --t-low 0.25 --t-high 0.70
"""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from src.data import build_dataset
from src.evaluate import compute_metrics, format_metrics
from src.experiments import VARIANTS, build_variant_pipeline
from src.preprocessing import make_text
from src.splits import RANDOM_SEED, assert_no_group_overlap, train_val_test_split
from src.thresholds import _print_band_report, assign_bands, band_report
from src.train import LABEL_MEANING

FINAL_NAME = "phishguard_model"
LIMITATIONS = [
    "Only one training source (Nazario, about 1,500 emails) contains real phishing; "
    "the other 'bad' sources are spam.",
    "Performance on sources never seen in training is substantially lower than on "
    "familiar sources; see the leave-one-source-out results.",
    "Scores are not calibrated probabilities and must not be displayed as percentages.",
    "Thresholds were chosen on out-of-source scores from the four training corpora; "
    "this final model trains on all four, so its score scale may differ slightly.",
    "Training data is mostly older mailing-list, corporate and spam email; modern "
    "phishing may look different.",
    "A 'likely legitimate' result is not a guarantee that an email is safe.",
]


def _records(frame: pd.DataFrame) -> list:
    """DataFrame -> JSON-safe list of dicts (NaN becomes null)."""
    clean = frame.reset_index().astype(object).where(frame.reset_index().notna(), None)
    return clean.to_dict(orient="records")


def run_final(df: pd.DataFrame, t_low: float, t_high: float, models_dir: Path,
              reports_dir: Path, force: bool = False, min_df: int = 5,
              min_source_df: int = 3) -> dict:
    assign_bands([0.5], t_low, t_high)                     # validate before any work
    metrics_path = reports_dir / f"{FINAL_NAME}_test_metrics.json"
    if metrics_path.exists() and not force:
        raise FileExistsError(
            f"{metrics_path} already exists: the test split has been evaluated. "
            "Re-running after changes would turn it into a tuning set. "
            "Use --force only if you accept that.")

    train, val, test = train_val_test_split(df)
    fit_df = pd.concat([train, val], ignore_index=True)
    assert_no_group_overlap(fit_df, test)

    pipe = build_variant_pipeline(fit_df, min_df=min_df, min_source_df=min_source_df,
                                  **VARIANTS["no_email"])
    scores = pipe.predict_proba(make_text(test))[:, 1]
    scored = pd.DataFrame({"source": test["source"].to_numpy(),
                           "label": test["label"].to_numpy(), "score": scores})

    rep = band_report(scored, t_low, t_high)
    classic = {"all": compute_metrics(scored["label"], scored["score"] >= 0.5)}
    for src, part in scored.groupby("source"):
        classic[src] = compute_metrics(part["label"], part["score"] >= 0.5)

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{FINAL_NAME}.joblib"
    meta_path = models_dir / f"{FINAL_NAME}_meta.json"
    joblib.dump(pipe, model_path)
    meta = {
        "model": FINAL_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "seed": RANDOM_SEED,
        "variant": "no_email",
        "variant_spec": VARIANTS["no_email"],
        "label_meaning": LABEL_MEANING,
        "thresholds": {"t_low": t_low, "t_high": t_high},
        "bands": {"legit": "score < t_low",
                  "suspicious": "t_low <= score < t_high",
                  "bad": "score >= t_high"},
        "score_is_probability": False,
        "training_sources": sorted(df["source"].unique().tolist()),
        "n_fit": int(len(fit_df)),
        "n_test": int(len(test)),
        "limitations": LIMITATIONS,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps({
        "note": "In-distribution test split; optimistic for unfamiliar email.",
        "thresholds": {"t_low": t_low, "t_high": t_high},
        "bands_summary": _records(rep["summary"]),
        "headline": rep["headline"],
        "per_source_bands": _records(rep["per_source"]),
        "classic_at_threshold_0.5": classic,
    }, indent=2), encoding="utf-8")
    return {"report": rep, "classic": classic, "model_path": model_path,
            "meta_path": meta_path, "metrics_path": metrics_path,
            "n_fit": len(fit_df), "n_test": len(test)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train final model, test once.")
    parser.add_argument("--t-low", type=float, required=True)
    parser.add_argument("--t-high", type=float, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--force", action="store_true",
                        help="Allow re-evaluating a test split that was already used.")
    args = parser.parse_args()

    try:
        df, _ = build_dataset(args.data_dir)
        out = run_final(df, args.t_low, args.t_high, args.models_dir,
                        args.reports_dir, force=args.force)
    except (FileNotFoundError, FileExistsError, ValueError) as err:
        print(f"Error: {err}")
        return 1

    print("=" * 70)
    print(f"Final model fit on {out['n_fit']:,} emails; evaluated once on "
          f"{out['n_test']:,} held-back test emails.")
    print("These are IN-DISTRIBUTION results (same four sources as training):")
    print("optimistic for unfamiliar email. Do not tune on this output.")
    _print_band_report(out["report"], args.t_low, args.t_high)
    print("\n" + "=" * 70)
    print("For comparison with earlier runs: classic metrics at a 0.5 threshold")
    print(format_metrics("test / ALL", out["classic"]["all"]))
    for src in sorted(k for k in out["classic"] if k != "all"):
        print(format_metrics(f"test / {src}", out["classic"][src]))
    print(f"\nSaved: {out['model_path']}\n       {out['meta_path']}\n"
          f"       {out['metrics_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
