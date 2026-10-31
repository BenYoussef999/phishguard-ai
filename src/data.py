"""Load, clean and combine the phishing-email CSV files.

Output schema (one row per email):
    source     which file the email came from
    subject    str (never missing)
    body       str (never missing)
    label      0 = legitimate, 1 = unwanted/malicious (phishing or spam)
    dup_group  messages with the same normalized text share one group id

Only subject and body are kept: Enron has no sender column, and receiver/date
are dataset artifacts that a model could use as shortcuts.

Run a summary from the project root:
    python -m src.data
    python -m src.data --data-dir data
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

# source name -> file name inside the data directory
SOURCES = {
    "ceas08": "CEAS_08.csv",
    "enron": "Enron.csv",
    "spamassassin": "SpamAssasin.csv",
    "nazario": "Nazario.csv",
}

# Mail-folder metadata records, not real emails (seen in Nazario row 0).
ARTIFACT_SUBJECT = re.compile(r"folder internal data", re.IGNORECASE)

# Below this many normalized characters, near-duplicate matching is unreliable,
# so we fall back to exact (lowercased) text matching.
MIN_NORMALIZED_CHARS = 20


def normalize(text: str) -> str:
    """Lowercase, remove URLs, keep letters only. Used for duplicate matching."""
    s = text.lower()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z]+", " ", s)
    return s.strip()


def _group_key(subject: str, body: str) -> str:
    """Key for near-duplicate grouping.

    Uses the normalized BODY only, because campaigns often vary the subject.
    Very short bodies are unreliable after normalization, so they are matched
    on exact lowercased subject + body instead.
    """
    norm = normalize(body)
    if len(norm) < MIN_NORMALIZED_CHARS:
        norm = "exact:" + (subject + " " + body).lower().strip()
    return hashlib.md5(norm.encode("utf-8", errors="replace")).hexdigest()


def load_source(name: str, path: Path) -> pd.DataFrame:
    """Read one CSV into the common schema. Raises a clear error if columns are missing."""
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace",
                     low_memory=False)
    missing = {"subject", "body", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {sorted(missing)}")
    out = pd.DataFrame({
        "source": name,
        "subject": df["subject"].fillna("").astype(str),
        "body": df["body"].fillna("").astype(str),
        "label": df["label"].astype(int),
    })
    bad = set(out["label"].unique()) - {0, 1}
    if bad:
        raise ValueError(f"{path.name} has unexpected label values: {sorted(bad)}")
    return out


def clean(df: pd.DataFrame):
    """Drop folder-metadata artifacts and rows with no text. Returns (df, counts)."""
    artifact = df["subject"].str.contains(ARTIFACT_SUBJECT)
    blank = (df["subject"].str.strip() == "") & (df["body"].str.strip() == "")
    # A row with a subject but a blank body is kept; one with neither is useless.
    body_blank_only = (df["body"].str.strip() == "") & ~blank
    counts = {"artifacts": int(artifact.sum()),
              "no_text": int((blank & ~artifact).sum()),
              "blank_body_kept": int((body_blank_only & ~artifact).sum())}
    return df[~artifact & ~blank].reset_index(drop=True), counts


def build_dataset(data_dir: Path = Path("data")):
    """Load all sources, clean, add dup_group. Returns (dataframe, per-source report)."""
    frames, report = [], []
    for name, filename in SOURCES.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Download it from the Zenodo record "
                f"(see data/README.md) and place it in {data_dir}/.")
        raw = load_source(name, path)
        cleaned, counts = clean(raw)
        report.append({"source": name, "loaded": len(raw), **counts,
                       "kept": len(cleaned)})
        frames.append(cleaned)
    df = pd.concat(frames, ignore_index=True)
    keys = [_group_key(s, b) for s, b in zip(df["subject"], df["body"])]
    df["dup_group"] = pd.factorize(pd.Series(keys))[0]
    return df, pd.DataFrame(report)


def _safe(text: str, width: int = 90) -> str:
    s = re.sub(r"\s+", " ", text).strip()
    return s.encode("ascii", errors="replace").decode("ascii")[:width]


def summarize(df: pd.DataFrame, report: pd.DataFrame) -> None:
    print("=" * 60)
    print("LOAD AND CLEAN REPORT")
    print(report.to_string(index=False))

    print("\n" + "=" * 60)
    print("ROWS PER SOURCE AND LABEL")
    print(pd.crosstab(df["source"], df["label"]).to_string())

    print("\n" + "=" * 60)
    print("DUPLICATES (normalized body; same group = near-identical message)")
    sizes = df.groupby("dup_group")["source"].transform("size")
    print(f"Rows in a group of 2+ messages: {(sizes > 1).sum():,} of {len(df):,}")
    for src, sub in df.groupby("source"):
        within = sub.duplicated("dup_group", keep=False).sum()
        print(f"  {src:<13} rows duplicated within the source: {within:,}")
    n_src = df.groupby("dup_group")["source"].nunique()
    cross = n_src[n_src > 1].index
    print(f"Groups appearing in MORE THAN ONE source: {len(cross):,} "
          f"({df['dup_group'].isin(cross).sum():,} rows)")
    n_lab = df.groupby("dup_group")["label"].nunique()
    print(f"Groups with conflicting labels: {(n_lab > 1).sum():,}")

    print("\n" + "=" * 60)
    print("LABEL 1 SUBJECTS PER SOURCE (to check what label 1 means; first 3)")
    for src, sub in df[df["label"] == 1].groupby("source"):
        print(f"\n{src}:")
        for subj in sub["subject"].head(3):
            print("   ", _safe(subj))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and summarize the combined dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    try:
        df, report = build_dataset(args.data_dir)
    except (FileNotFoundError, ValueError) as err:
        print(f"Error: {err}")
        return 1
    summarize(df, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
