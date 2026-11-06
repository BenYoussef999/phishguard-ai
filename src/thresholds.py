"""Calibration check and threshold analysis for the chosen model (no_email + LogReg).

Every email is scored by a model that NEVER saw its source (leave-one-source-out),
which approximates how the model behaves on unfamiliar email. Scores are cached
to a CSV with no email text, so threshold experiments are instant.

Steps (from the project root):
    python -m src.thresholds                          # build cache if missing, then
                                                      #   calibration table + threshold sweep
    python -m src.thresholds --build                  # force re-fitting the 4 folds
    python -m src.thresholds --t-low 0.30 --t-high 0.70   # what a chosen pair would do

Labels: 1 = bad (phishing from Nazario, spam elsewhere), 0 = legitimate.
Bands: score < t_low -> "legit"; t_low <= score < t_high -> "suspicious";
       score >= t_high -> "bad".
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import build_dataset
from src.experiments import VARIANTS, build_variant_pipeline
from src.preprocessing import make_text
from src.splits import leave_one_source_out

DEFAULT_CACHE = Path("reports") / "oof_scores_no_email.csv"
CACHE_COLUMNS = ["source", "label", "dup_group", "score"]
BAND_ORDER = ["legit", "suspicious", "bad"]


def compute_oof_scores(df: pd.DataFrame, spec: dict, min_df: int = 5,
                       min_source_df: int = 3, verbose: bool = False) -> pd.DataFrame:
    """Score every email with a model trained on the OTHER sources only."""
    parts = []
    for held_out in sorted(df["source"].unique()):
        train, test, _removed = leave_one_source_out(df, held_out)
        pipe = build_variant_pipeline(train, min_df=min_df,
                                      min_source_df=min_source_df, **spec)
        scores = pipe.predict_proba(make_text(test))[:, 1]
        parts.append(pd.DataFrame({"source": test["source"].to_numpy(),
                                   "label": test["label"].to_numpy(),
                                   "dup_group": test["dup_group"].to_numpy(),
                                   "score": scores}))
        if verbose:
            print(f"  scored {held_out}: {len(test):,} emails", flush=True)
    return pd.concat(parts, ignore_index=True)


def save_scores(scores: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scores[CACHE_COLUMNS].to_csv(path, index=False)


def load_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(CACHE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}; "
                         "re-run with --build.")
    return df


def assign_bands(scores, t_low: float, t_high: float) -> np.ndarray:
    if not (0.0 <= t_low <= t_high <= 1.0):
        raise ValueError(f"Need 0 <= t_low <= t_high <= 1, got {t_low}, {t_high}")
    scores = np.asarray(scores, dtype=float)
    return np.where(scores < t_low, "legit",
                    np.where(scores < t_high, "suspicious", "bad"))


def calibration_table(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Reliability table: does a score of s mean 'bad' about s of the time?"""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(df["score"], edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        part = df[idx == b]
        if len(part) == 0:
            continue
        rows.append({"score_range": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                     "n": int(len(part)),
                     "mean_score": float(part["score"].mean()),
                     "share_actually_bad": float(part["label"].mean())})
    return pd.DataFrame(rows)


def threshold_sweep(df: pd.DataFrame, thresholds=None) -> pd.DataFrame:
    """Single-threshold trade-off table (flag if score >= t)."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    bad = df["label"] == 1
    legit = ~bad
    naz = df["source"] == "nazario"
    rows = []
    for t in thresholds:
        flagged = df["score"] >= t
        tp = int((flagged & bad).sum())
        fp = int((flagged & legit).sum())
        rows.append({
            "threshold": float(t),
            "recall_all_bad": tp / bad.sum() if bad.sum() else np.nan,
            "recall_nazario": ((flagged & naz).sum() / naz.sum()) if naz.any() else np.nan,
            "false_alarm_rate": fp / legit.sum() if legit.sum() else np.nan,
            "precision": tp / (tp + fp) if (tp + fp) else np.nan,
        })
    return pd.DataFrame(rows)


def band_report(df: pd.DataFrame, t_low: float, t_high: float) -> dict:
    """What a (t_low, t_high) pair does to the data."""
    d = df.assign(band=assign_bands(df["score"], t_low, t_high))
    n = len(d)
    summary = (d.groupby("band")["label"].agg(n="size", n_bad="sum")
                 .reindex(BAND_ORDER, fill_value=0))
    summary["share_of_all"] = summary["n"] / n
    summary["share_actually_bad"] = summary["n_bad"] / summary["n"].replace(0, np.nan)

    total_bad = int((d["label"] == 1).sum())
    total_legit = int((d["label"] == 0).sum())
    headline = {
        "bad_called_legit": int(((d["band"] == "legit") & (d["label"] == 1)).sum()),
        "total_bad": total_bad,
        "legit_called_bad": int(((d["band"] == "bad") & (d["label"] == 0)).sum()),
        "legit_called_suspicious": int(((d["band"] == "suspicious")
                                        & (d["label"] == 0)).sum()),
        "total_legit": total_legit,
    }
    per_source = []
    for (src, lab), part in d.groupby(["source", "label"]):
        row = {"source": src, "label": int(lab), "n": int(len(part))}
        for band in BAND_ORDER:
            row[band] = float((part["band"] == band).mean())
        per_source.append(row)
    return {"summary": summary, "headline": headline,
            "per_source": pd.DataFrame(per_source)}


def _print_band_report(rep: dict, t_low: float, t_high: float) -> None:
    h = rep["headline"]
    print(f"\nBANDS for t_low={t_low}, t_high={t_high}")
    print(rep["summary"].round(4).to_string())
    print(f"\nBad emails called 'legit' (missed): {h['bad_called_legit']:,} "
          f"of {h['total_bad']:,} ({h['bad_called_legit'] / h['total_bad']:.1%})")
    print(f"Legitimate emails called 'bad' (hard false alarms): "
          f"{h['legit_called_bad']:,} of {h['total_legit']:,} "
          f"({h['legit_called_bad'] / h['total_legit']:.1%})")
    print(f"Legitimate emails put in 'suspicious': "
          f"{h['legit_called_suspicious']:,} ({h['legit_called_suspicious'] / h['total_legit']:.1%})")
    print("\nPer source and true label (share of that group landing in each band):")
    print(rep["per_source"].round(3).to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibration and threshold analysis.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--build", action="store_true",
                        help="Re-fit the 4 folds and overwrite the score cache.")
    parser.add_argument("--t-low", type=float)
    parser.add_argument("--t-high", type=float)
    args = parser.parse_args()

    if (args.t_low is None) != (args.t_high is None):
        print("Error: give both --t-low and --t-high, or neither.")
        return 1
    if args.t_low is not None:
        try:
            assign_bands([0.5], args.t_low, args.t_high)
        except ValueError as err:
            print(f"Error: {err}")
            return 1

    if args.build or not args.cache.exists():
        try:
            df, _ = build_dataset(args.data_dir)
        except (FileNotFoundError, ValueError) as err:
            print(f"Error: {err}")
            return 1
        print("Scoring every email with a model that never saw its source "
              "(4 fits, a few minutes)...", flush=True)
        scores = compute_oof_scores(df, VARIANTS["no_email"], verbose=True)
        save_scores(scores, args.cache)
        print(f"Saved scores (no email text) to {args.cache}")
    try:
        scores = load_scores(args.cache)
    except (OSError, ValueError) as err:
        print(f"Error: {err}")
        return 1

    base_rate = scores["label"].mean()
    print("=" * 70)
    print(f"{len(scores):,} emails scored out-of-source. Share labeled bad in this "
          f"data: {base_rate:.1%}")
    print("(That share is an artifact of how the datasets were assembled; real "
          "inboxes are mostly legitimate.)")

    print("\n" + "=" * 70)
    print("CALIBRATION: do scores match how often emails are actually bad?")
    print(calibration_table(scores).round(3).to_string(index=False))

    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP (flag an email if score >= threshold)")
    print("recall_all_bad / recall_nazario: share of bad emails caught")
    print("false_alarm_rate: share of legitimate emails flagged")
    print(threshold_sweep(scores).round(3).to_string(index=False))

    if args.t_low is not None:
        _print_band_report(band_report(scores, args.t_low, args.t_high),
                           args.t_low, args.t_high)
    else:
        print("\nNext: pick two thresholds and rerun with --t-low X --t-high Y")
    return 0


if __name__ == "__main__":
    sys.exit(main())
