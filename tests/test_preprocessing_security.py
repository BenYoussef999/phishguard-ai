"""Performance and equivalence tests for the e-mail pattern in src/preprocessing.py."""

import re
import time

import pytest

from src.preprocessing import clean_text

# The previous, unbounded pattern, kept only as a reference for equivalence checks.
_OLD_EMAIL = re.compile(r"\b[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+\b")

ADVERSARIAL = {
    "dots": "a." * 10000,
    "plus": "a+" * 10000,
    "hyphen": "a-" * 10000,
    "at signs": "a@" * 10000,
    "dotted local part then at": "a." * 9000 + "@",
    "angle brackets": "<" * 20000,
    "long url": "http://" + "a" * 19990,
}


@pytest.mark.parametrize("name", list(ADVERSARIAL))
def test_adversarial_input_is_processed_quickly(name):
    start = time.perf_counter()
    out = clean_text(ADVERSARIAL[name])
    elapsed = time.perf_counter() - start
    assert isinstance(out, str)
    assert elapsed < 0.5, f"{name}: {elapsed:.2f}s"


@pytest.mark.parametrize("address", [
    "user4@gvc.ceas-challenge.cc", "jose@monkey.org", "first.last+tag@sub.example.co.uk",
    "c202f8eb239faf8d4b0a5c6a41cde453@gvc.ceas-challenge.cc", "a_b-c@x1.io",
])
def test_normal_addresses_are_replaced_exactly_as_before(address):
    text = f"please write to {address} today"
    assert _OLD_EMAIL.sub(" X ", text) == re.sub(
        r"\b[\w.+\-]{1,64}@[\w\-]{1,63}(?:\.[\w\-]{1,63}){1,10}\b", " X ", text)
    assert "emailtoken" in clean_text(text)
