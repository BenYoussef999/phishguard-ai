"""Deeper read-only inspection of CEAS_08.csv.

Checks: date range, empty text, near-duplicates, sender-domain shortcuts,
the 'urls' column, and a random sample of each class for manual review.

Usage (from the project root):
    python -m src.inspect_deep data\\CEAS_08.csv

Nothing is modified and no URLs are opened.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

SAMPLE_SEED = 42
SAMPLE_PER_CLASS = 5


def safe(text, width: int = 140) -> str:
    """Make text printable on any Windows console: single line, ASCII-only, truncated."""
    s = "" if pd.isna(text) else str(text)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.encode("ascii", errors="replace").decode("ascii")
    return s[:width]


def normalize(text) -> str:
    """Lowercase, drop URLs and digits, keep letters only. Used to spot near-duplicates."""
    s = "" if pd.isna(text) else str(text).lower()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z]+", " ", s)
    return s.strip()


def sender_domain(sender) -> str:
    match = re.search(r"@([A-Za-z0-9.\-]+)", "" if pd.isna(sender) else str(sender))
    return match.group(1).lower() if match else "(none)"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if not args.path.exists():
        print(f"File not found: {args.path}")
        return 1

    df = pd.read_csv(args.path, encoding="utf-8", encoding_errors="replace",
                     low_memory=False)
    n = len(df)

    # ---- Dates -------------------------------------------------------
    print("=" * 60)
    print("DATES")
    dates = pd.to_datetime(df["date"], utc=True, errors="coerce", format="mixed")
    print(f"Unparseable dates: {dates.isna().sum():,} of {n:,}")
    if dates.notna().any():
        print(f"Earliest: {dates.min()}   Latest: {dates.max()}")
        by_month = (pd.crosstab(dates.dt.strftime("%Y-%m"), df["label"])
                    .rename(columns={0: "label0", 1: "label1"}))
        print("Rows per month and label:")
        print(by_month.to_string())

    # ---- Empty text --------------------------------------------------
    print("\n" + "=" * 60)
    print("EMPTY TEXT")
    body_blank = df["body"].fillna("").str.strip().eq("")
    subj_blank = df["subject"].fillna("").str.strip().eq("")
    print(f"Blank bodies:   {body_blank.sum():,}")
    print(f"Blank subjects: {subj_blank.sum():,}")
    print(f"Both blank:     {(body_blank & subj_blank).sum():,}")

    # ---- Length by class ---------------------------------------------
    print("\n" + "=" * 60)
    print("BODY LENGTH (characters) BY LABEL")
    lengths = df["body"].fillna("").str.len()
    print(lengths.groupby(df["label"]).describe().round(0).to_string())

    # ---- Near-duplicates ---------------------------------------------
    print("\n" + "=" * 60)
    print("NEAR-DUPLICATES (body text with URLs, digits, punctuation removed)")
    norm = df["body"].map(normalize)
    nonempty = norm.ne("")
    dup_mask = norm.duplicated(keep=False) & nonempty
    print(f"Rows sharing a normalized body with another row: {dup_mask.sum():,}")
    groups = df[dup_mask].assign(_norm=norm[dup_mask]).groupby("_norm")["label"]
    print(f"Distinct duplicate groups: {groups.ngroups:,}")
    print(f"Groups with conflicting labels: {(groups.nunique() > 1).sum():,}")

    # ---- Sender-domain shortcuts -------------------------------------
    print("\n" + "=" * 60)
    print("SENDER DOMAINS")
    domains = df["sender"].map(sender_domain)
    stats = (df.assign(domain=domains)
               .groupby("domain")["label"]
               .agg(rows="size", share_label1="mean"))
    print(f"Distinct sender domains: {len(stats):,}")
    big = stats[stats["rows"] >= 50].sort_values("rows", ascending=False)
    print("Domains with >= 50 rows (share_label1: 0 = all label 0, 1 = all label 1):")
    print(big.head(15).round(3).to_string())
    pure = big[(big["share_label1"] == 0) | (big["share_label1"] == 1)]
    print(f"Domains with >= 50 rows that are 100% one label: {len(pure):,}")

    # ---- The 'urls' column -------------------------------------------
    print("\n" + "=" * 60)
    print("'urls' COLUMN vs LABEL")
    print(pd.crosstab(df["urls"], df["label"]).to_string())

    # ---- Manual review sample ----------------------------------------
    print("\n" + "=" * 60)
    print(f"RANDOM SAMPLE FOR MANUAL REVIEW (seed={SAMPLE_SEED})")
    for label in (0, 1):
        print(f"\n--- label {label} ---")
        sample = df[df["label"] == label].sample(SAMPLE_PER_CLASS,
                                                 random_state=SAMPLE_SEED)
        for _, row in sample.iterrows():
            print(f"From:    {safe(row['sender'], 70)}")
            print(f"Subject: {safe(row['subject'], 90)}")
            print(f"Body:    {safe(row['body'], 160)}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
