"""Tests for src/data.py using small synthetic CSV files."""

import pandas as pd
import pytest

from src.data import build_dataset, clean, normalize, load_source, SOURCES


def _write_all(tmp_path, extra_rows=None):
    """Write tiny CSVs for every source in the real file layout."""
    long_text = "please verify your account password immediately to avoid suspension"
    rows = {
        "ceas08": pd.DataFrame({
            "sender": ["a@x.com", "b@y.com"], "receiver": [None, None],
            "date": ["d", "d"], "subject": ["Hi", "Win now"],
            "body": ["Meeting notes about the quarterly budget review tomorrow",
                     long_text + " http://evil.example/1"],
            "label": [0, 1], "urls": [0, 1]}),
        "enron": pd.DataFrame({
            "subject": ["nom for may", None],
            "body": ["see attached file hplno xls for the daily volumes", "x"],
            "label": [0, 1]}),
        "spamassassin": pd.DataFrame({
            "sender": ["c@z.com"], "receiver": [None], "date": ["d"],
            "subject": ["Re: list"], "body": ["a normal mailing list discussion post here"],
            "label": [0], "urls": [0]}),
        "nazario": pd.DataFrame({
            "sender": ["MAILER-DAEMON@monkey.org", "s@cpanel.com"],
            "receiver": [None, None], "date": ["d", "d"],
            "subject": ["DON'T DELETE THIS MESSAGE -- FOLDER INTERNAL DATA",
                        "Verify Your Account"],
            "body": ["This text is part of the internal format of your mail folder",
                     long_text + " http://evil.example/2"],
            "label": [1, 1], "urls": [1, 1]}),
    }
    for name, frame in rows.items():
        frame.to_csv(tmp_path / SOURCES[name], index=False)


def test_normalize_removes_urls_digits_and_case():
    assert normalize("Visit HTTP://A.com NOW 123!") == "visit now"


def test_build_dataset_schema_and_artifact_removal(tmp_path):
    _write_all(tmp_path)
    df, report = build_dataset(tmp_path)
    assert list(df.columns) == ["source", "subject", "body", "label", "dup_group"]
    assert not df["subject"].isna().any() and not df["body"].isna().any()
    naz = report.set_index("source").loc["nazario"]
    assert naz["artifacts"] == 1 and naz["kept"] == 1


def test_near_duplicates_share_group_across_sources(tmp_path):
    _write_all(tmp_path)
    df, _ = build_dataset(tmp_path)
    phish = df[df["body"].str.contains("verify your account password")]
    # ceas08 and nazario rows differ only by URL and subject -> same group
    assert set(phish["source"]) == {"ceas08", "nazario"}
    assert phish["dup_group"].nunique() == 1


def test_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Zenodo"):
        build_dataset(tmp_path)


def test_missing_column_is_reported(tmp_path):
    pd.DataFrame({"subject": ["a"], "label": [0]}).to_csv(tmp_path / "x.csv", index=False)
    with pytest.raises(ValueError, match="body"):
        load_source("x", tmp_path / "x.csv")


def test_clean_drops_rows_without_any_text():
    df = pd.DataFrame({"source": ["s", "s"], "subject": ["", "ok"],
                       "body": ["", ""], "label": [0, 0]})
    cleaned, counts = clean(df)
    assert len(cleaned) == 1 and counts["no_text"] == 1
