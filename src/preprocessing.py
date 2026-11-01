"""Text preprocessing shared by training and inference.

Placeholders replace URLs, e-mail addresses and numbers so the model cannot
latch onto source-specific artifacts (a mailing-list domain, a year, a receiver
address). URL structure is analysed separately by explicit rules.

All input is treated as untrusted text: it is only transformed with regular
expressions and never executed, rendered, or fetched.
"""

import re

# Hard cap on characters kept per message (protects memory and runtime).
MAX_CHARS = 20_000

_URL = re.compile(r"(?:https?://|ftp://|www\.)\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+\b")
_HTML_TAG = re.compile(r"<[^>\n]{1,200}>")
_DIGITS = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")


def clean_text(text) -> str:
    """Normalize one message. Never raises on None, NaN, empty or odd input."""
    if text is None or (isinstance(text, float) and text != text):
        return ""
    s = str(text)[:MAX_CHARS]
    s = _URL.sub(" urltoken ", s)
    s = _HTML_TAG.sub(" ", s)
    s = _EMAIL.sub(" emailtoken ", s)
    s = s.lower()
    s = _DIGITS.sub(" numtoken ", s)
    return _SPACES.sub(" ", s).strip()


def combine_text(subject, body) -> str:
    """Join subject and body into the single string the pipeline expects."""
    subject = "" if subject is None else str(subject)
    body = "" if body is None else str(body)
    return f"{subject}\n{body}"


def make_text(df) -> list:
    """Combine the subject and body columns of a DataFrame into model input."""
    return [combine_text(s, b) for s, b in zip(df["subject"], df["body"])]
