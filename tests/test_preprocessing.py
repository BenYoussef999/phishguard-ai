"""Tests for src/preprocessing.py."""

import math

from src.preprocessing import MAX_CHARS, clean_text, combine_text, make_text


def test_urls_are_replaced_by_placeholder():
    out = clean_text("Click http://evil.example/login?id=1 now or www.bad.example")
    assert "urltoken" in out
    assert "evil" not in out and "bad.example" not in out


def test_email_addresses_are_replaced():
    out = clean_text("Contact user4@gvc.ceas-challenge.cc today")
    assert "emailtoken" in out and "ceas" not in out


def test_digits_and_case_are_normalized():
    assert clean_text("URGENT 2001 Notice 42") == "urgent numtoken notice numtoken"


def test_html_tags_are_removed_but_text_is_kept():
    out = clean_text("<b>Verify</b> your <script>account</script>")
    assert "<" not in out and "verify" in out and "account" in out


def test_none_nan_and_empty_inputs_are_safe():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""
    assert clean_text("") == ""
    assert clean_text("   \n\t ") == ""
    assert clean_text(12345) == "numtoken"


def test_very_long_input_is_truncated():
    assert len(clean_text("a" * 100_000)) <= MAX_CHARS


def test_non_ascii_input_does_not_crash():
    assert isinstance(clean_text("Vérifiez votre compte \u202e \U0001F512"), str)


def test_combine_and_make_text_handle_missing_values():
    assert combine_text("Hi", None) == "Hi\n"
    assert combine_text(None, "Body") == "\nBody"

    class Frame(dict):
        pass
    frame = Frame(subject=["a", None], body=["b", "c"])
    assert make_text(frame) == ["a\nb", "\nc"]
