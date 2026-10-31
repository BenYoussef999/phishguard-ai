"""Read-only inspection of a phishing-email CSV.

Prints structure, label distribution, missing values and duplicate counts.
It does not modify the file and does not open any URLs.

Usage (from the project root):
    python -m src.inspect_data data\\CEAS_08.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def find_column(df: pd.DataFrame, name: str):
    """Return the real column name matching `name` case-insensitively, else None."""
    for col in df.columns:
        if str(col).strip().lower() == name:
            return col
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a phishing email CSV.")
    parser.add_argument("path", type=Path, help="Path to the CSV file")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"File not found: {args.path}")
        return 1

    # Email text can contain odd bytes, so replace undecodable characters
    # instead of crashing.
    df = pd.read_csv(args.path, encoding="utf-8", encoding_errors="replace",
                     low_memory=False)

    print("=" * 60)
    print(f"File: {args.path.name}")
    print(f"Rows: {len(df):,}   Columns: {df.shape[1]}")

    print("\n--- Columns and types ---")
    print(df.dtypes.to_string())

    print("\n--- Missing values per column ---")
    print(df.isna().sum().to_string())

    label_col = find_column(df, "label")
    if label_col is not None:
        print(f"\n--- Label distribution ('{label_col}') ---")
        counts = df[label_col].value_counts(dropna=False)
        print(counts.to_string())
        print((counts / len(df)).round(4).to_string())
    else:
        print("\nNo column named 'label' found. Check the column list above.")

    print("\n--- Duplicates ---")
    print(f"Exact duplicate rows (all columns): {df.duplicated().sum():,}")
    subject_col = find_column(df, "subject")
    body_col = find_column(df, "body")
    if subject_col is not None and body_col is not None:
        text_dups = df.duplicated(subset=[subject_col, body_col]).sum()
        print(f"Duplicate subject+body pairs:       {text_dups:,}")
        if label_col is not None:
            # Same text with different labels = label conflict.
            grouped = df.groupby([subject_col, body_col], dropna=False)[label_col]
            conflicts = (grouped.nunique() > 1).sum()
            print(f"Same text with conflicting labels:  {conflicts:,}")
    else:
        print("Could not find both 'subject' and 'body' columns; "
              "skipping text-duplicate check.")

    print("\n--- Sample rows (text truncated to 80 chars) ---")
    sample = df.head(3).copy()
    for col in sample.columns:
        if pd.api.types.is_string_dtype(sample[col]):
            sample[col] = (sample[col].astype(str)
                           .str.replace(r"\s+", " ", regex=True)
                           .str.slice(0, 80))
    print(sample.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
