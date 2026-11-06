"""Evidence shown next to a prediction. Two kinds, kept strictly separate:

  find_indicators()  Things detected by explicit, hand-written rules in the text
                     (urgent language, credential/payment requests, URL structure,
                     sender details). Static analysis only.
  explain_model()    Words that pushed the MODEL's score up or down. This explains
                     the model, not the email.

Nothing here opens a URL, resolves a domain, or authenticates a sender. Links are
returned defanged (hxxps://example[.]com) so an interface cannot make them
clickable. All input is untrusted text and is only matched with regular
expressions. Nothing is logged or stored.

Try it from the project root:
    python -m src.explain --subject "Verify now" --body "Click http://1.2.3.4/login"
"""

import argparse
import ipaddress
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from src.predict import (InputError, ModelBundle, ModelLoadError, load_model,
                         validate_text_inputs)
from src.preprocessing import MAX_CHARS, combine_text

MAX_URLS_ANALYZED = 50
MAX_SENDER_CHARS = 1_000
MAX_EVIDENCE_ITEMS = 5

NOT_CHECKED = (
    "No link was opened and no domain was looked up, so nothing here says whether a "
    "site is real or dangerous.",
    "The sender was not authenticated. SPF, DKIM and DMARC checks need the full "
    "message headers, which this tool does not read.",
    "Attachments were not examined.",
    "The rules cover English text only and were written by hand. They are not a "
    "complete list of phishing signs.",
)
MODEL_CAVEAT = (
    "These words show what the model weighs, not proof about this email. The model "
    "learned from a few public spam and phishing collections and also leans on "
    "vocabulary typical of those sources, so some terms may reflect where its "
    "training emails came from rather than anything harmful.")

PLACEHOLDERS = {"urltoken": "[link]", "numtoken": "[number]",
                "emailtoken": "[email address]"}

SHORTENERS = frozenset({
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy", "t.ly"})
BRANDS = ("paypal", "apple", "microsoft", "google", "amazon", "netflix", "facebook",
          "instagram", "linkedin", "dropbox", "docusign", "outlook", "office365",
          "adobe", "dhl", "fedex", "usps", "irs", "chase", "wellsfargo",
          "bankofamerica")
_SECOND_LEVEL = {"co", "com", "org", "net", "gov", "edu", "ac"}

# ---------------------------------------------------------------- data classes


class ExplainError(RuntimeError):
    """The model cannot be explained in this way."""


@dataclass(frozen=True)
class Indicator:
    rule_id: str
    category: str            # "language" | "credentials" | "payment" | "url" | "sender"
    title: str
    detail: str
    evidence: tuple          # short, sanitized excerpts; URLs are defanged


@dataclass(frozen=True)
class ModelTerm:
    term: str                # as the model sees it
    display: str             # friendlier text (placeholders spelled out)
    strength: float          # contribution on the model's own scale; NOT a probability


@dataclass(frozen=True)
class Explanation:
    indicators: tuple
    urls_found: int
    toward_bad: tuple
    toward_legit: tuple
    not_checked: tuple
    model_caveat: str
    notes: tuple


# ------------------------------------------------------------------- utilities
def _clip(text, limit: int = 100) -> str:
    s = re.sub(r"[\x00-\x1f\x7f]", " ", "" if text is None else str(text))
    s = " ".join(s.split())
    return s if len(s) <= limit else s[: limit - 3] + "..."


def defang_url(url: str, limit: int = 100) -> str:
    """hxxps://example[.]com/path : readable, but never clickable."""
    s = _clip(url, limit)
    scheme, sep, rest = s.partition("://")
    if not sep:
        scheme, rest = "", s
    else:
        scheme = re.sub(r"(?i)^http", "hxxp", scheme)
        scheme = re.sub(r"(?i)^ftp", "fxp", scheme) + "://"
    host, tail = re.match(r"([^/?#]*)(.*)", rest, re.S).groups()
    return _clip(scheme + host.replace(".", "[.]") + tail, limit)   # defanging adds length


def _registrable(labels):
    """Approximate registrable domain: last 2 labels, or 3 for forms like co.uk."""
    if len(labels) >= 3 and labels[-2] in _SECOND_LEVEL and len(labels[-1]) == 2:
        return labels[-3:]
    return labels[-2:] if len(labels) >= 2 else list(labels)


def _tokens(label: str) -> list:
    return [t for t in re.split(r"[-_\d]+", label.lower()) if t]


def _evidence_list(items) -> tuple:
    seen, out = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
        if len(out) == MAX_EVIDENCE_ITEMS:
            break
    return tuple(out)


# ---------------------------------------------------------------- text rules
_LANGUAGE_RULES = [
    ("urgency", "language", "Urgent or pressuring language",
     "Phrases that push the reader to act quickly. Common in phishing, but also in "
     "legitimate notices.",
     r"\b(?:urgent(?:ly)?|immediate(?:ly)?|act now|right away|as soon as possible|asap"
     r"|within \d+ (?:hours?|days?)|final (?:notice|warning|reminder)"
     r"|last (?:chance|warning)|expires? (?:today|soon|in)|limited time|don'?t delay"
     r"|time[- ]sensitive)\b"),
    ("threat", "language", "Threats about the account or consequences",
     "Wording that warns of lost access or consequences unless you respond.",
     r"\b(?:(?:account|access|mailbox|service)s?\s+(?:will be|has been|have been|is|are"
     r"|was|were)\s+(?:suspended|closed|locked|terminated|disabled|deactivated|blocked"
     r"|restricted|deleted)|legal action|(?:will|may) be (?:suspended|terminated"
     r"|deleted|closed)|permanently (?:closed|deleted|disabled)"
     r"|unauthori[sz]ed (?:access|activity|login|transaction)"
     r"|(?:unusual|suspicious) (?:activity|sign[- ]?in|login))\b"),
    ("credentials", "credentials", "Request for login or personal details",
     "Wording that asks you to confirm, enter or send credentials or identity details.",
     r"\b(?:(?:verify|confirm|validate|update|re-?enter|reset|renew)\s+(?:your\s+)?"
     r"(?:account|password|credentials|identity|login|e-?mail|mailbox|information"
     r"|details|profile)|(?:enter|provide|send|submit)\s+(?:your\s+)?(?:password"
     r"|credentials|login|username|pin|otp)|log\s?in\s+(?:to|here|now)"
     r"|sign\s?in\s+(?:to|here|now)|social security|ssn|date of birth"
     r"|one[- ]time (?:code|password)|security code|verification code)\b"),
    ("payment", "payment", "Request for payment or card details",
     "Wording about paying, transferring money, or sharing card or bank details.",
     r"\b(?:credit card|debit card|card number|cvv|cvc|bank account|routing number"
     r"|iban|billing (?:information|details)|payment (?:details|information|method)"
     r"|update (?:your )?payment|wire transfer|gift cards?|bitcoin|crypto(?:currency)?"
     r"|western union|money transfer|pay(?:ment)? (?:now|immediately|today)"
     r"|overdue (?:invoice|payment|balance)|outstanding (?:invoice|balance))\b"),
    ("mailbox_lure", "language", "Mailbox, quota or delivery notice",
     "Notices about a full mailbox, quarantined or undelivered messages, or a fax. "
     "A common lure, but also used by real mail systems.",
     r"\b(?:mailbox (?:is )?(?:full|quota)|(?:storage|quota) (?:is )?(?:full|exceeded"
     r"|almost full)|(?:pending|undelivered|blocked|quarantined?) (?:messages?|mails?"
     r"|e-?mails?|delivery)|messages? (?:in|on) (?:your )?(?:spam )?quarantine"
     r"|delivery (?:failed|failure)|failed to deliver|new fax|incoming fax"
     r"|fax (?:document|message))\b"),
]
_COMPILED_LANGUAGE = [(rid, cat, title, detail, re.compile(rx, re.IGNORECASE))
                      for rid, cat, title, detail, rx in _LANGUAGE_RULES]


def _language_indicators(flat_text: str) -> list:
    found = []
    for rid, cat, title, detail, rx in _COMPILED_LANGUAGE:
        phrases = [" ".join(m.group(0).lower().split()) for m in rx.finditer(flat_text)]
        if phrases:
            found.append(Indicator(rid, cat, title, detail, _evidence_list(phrases)))
    return found


# ------------------------------------------------------------------ URL rules
# Bracketed IPv6 hosts are matched first, because "]" otherwise ends a link.
_URL_RE = re.compile(
    r"(?:(?:https?|ftp)://\[[0-9a-f:.]{2,45}\]|(?:https?://|ftp://|www\.)[^\s<>\"')\]]+)"
    r"[^\s<>\"')\]]*", re.IGNORECASE)
_ANCHOR_RE = re.compile(
    r"<a\s[^>]{0,500}?href\s*=\s*[\"']([^\"']{1,500})[\"'][^>]{0,500}>(.{0,300}?)</a\s*>",
    re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]{0,200}>")
_SHOWN_HOST_RE = re.compile(r"(?:https?://)?((?:[a-z0-9-]+\.)+[a-z]{2,})", re.IGNORECASE)


def _host_of(url: str):
    """Lower-cased hostname, or None when the URL cannot be parsed."""
    if url.lower().startswith("www."):
        url = "http://" + url
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        return None, None
    return (host.lower(), parts) if host else (None, None)


def _is_numeric_host(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return bool(re.fullmatch(r"\d{8,10}|0x[0-9a-f]+", host))


def _brand_hits(labels) -> set:
    """Brand names that appear in an unusual position of the hostname."""
    reg = _registrable(labels)
    reg_name = reg[0] if len(reg) >= 2 else (reg[0] if reg else "")
    subdomain_labels = labels[: len(labels) - len(reg)]
    hits = set()
    for brand in BRANDS:
        if any(brand in _tokens(lbl) for lbl in subdomain_labels):
            hits.add(brand)                        # e.g. paypal.example.net
        tokens = _tokens(reg_name)
        if brand in tokens and reg_name != brand:
            hits.add(brand)                        # e.g. paypal-support.com
    return hits


def _url_indicators(raw_text: str):
    """Return (indicators, urls_found, notes)."""
    urls = [u.rstrip(".,;:!?") for u in _URL_RE.findall(raw_text)]
    notes, buckets = [], {k: [] for k in
                          ("ip", "userinfo", "idn", "short", "brand", "deep", "href")}
    if len(urls) > MAX_URLS_ANALYZED:
        notes.append(f"Only the first {MAX_URLS_ANALYZED} of {len(urls)} links were analyzed.")
    unparsed = 0
    for url in urls[:MAX_URLS_ANALYZED]:
        host, parts = _host_of(url)
        if host is None:
            unparsed += 1
            continue
        shown = defang_url(url)
        labels = host.split(".")
        if _is_numeric_host(host):
            buckets["ip"].append(shown)
            continue
        if parts.username is not None or "@" in parts.netloc:
            buckets["userinfo"].append(shown)
        if not host.isascii() or any(l.startswith("xn--") for l in labels):
            buckets["idn"].append(shown)
        if host in SHORTENERS or ".".join(labels[-2:]) in SHORTENERS:
            buckets["short"].append(shown)
        if _brand_hits(labels):
            buckets["brand"].append(shown)
        if len(labels) >= 5:
            buckets["deep"].append(shown)
    if unparsed:
        notes.append(f"{unparsed} link(s) could not be parsed and were skipped.")

    for anchor_href, anchor_html in _ANCHOR_RE.findall(raw_text):
        shown_text = _TAG_RE.sub("", anchor_html)
        m = _SHOWN_HOST_RE.search(shown_text)
        href_host, _ = _host_of(anchor_href)
        if not m or not href_host:
            continue
        shown_host = m.group(1).lower()
        if (_registrable(shown_host.split("."))
                != _registrable(href_host.split("."))):
            buckets["href"].append(f"shows {_clip(shown_host, 40).replace('.', '[.]')}, "
                                   f"goes to {defang_url(anchor_href, 60)}")

    spec = [
        ("ip", "url_ip_address", "Link points to a raw IP address",
         "Legitimate sites rarely ask you to log in through a bare IP address."),
        ("userinfo", "url_userinfo", "Link contains a username section before the real host",
         "In a link like name@host, everything before the @ is ignored when connecting; "
         "it can make a link look like it belongs to a familiar name."),
        ("idn", "url_idn", "Link uses an internationalized or look-alike style domain",
         "Domains with xn-- or non-ASCII characters can imitate familiar names."),
        ("short", "url_shortener", "Link uses a known URL shortener",
         "A shortened link hides its real destination until it is opened."),
        ("brand", "url_brand_position", "A brand name appears in an unusual place in a link's host",
         "The name is in a subdomain or joined to other words in the site's main domain "
         "(for example name.example.net). This pattern is common in look-alike links, "
         "but it does not show who owns the site."),
        ("deep", "url_many_subdomains", "Link host has many subdomain levels",
         "Long chains of subdomains can bury the real domain at the end."),
        ("href", "url_text_mismatch", "A link's visible text names a different site than its target",
         "The text you see suggests one site, while the link goes to another."),
    ]
    out = [Indicator(rid, "url", title, detail, _evidence_list(buckets[key]))
           for key, rid, title, detail in spec if buckets[key]]
    return out, len(urls), notes


# --------------------------------------------------------------- sender rules
_ADDRESS_RE = re.compile(r"[^@\s]{1,64}@[A-Za-z0-9.\-]{1,255}\.[A-Za-z]{2,}")


def _sender_indicators(sender) -> list:
    from email.utils import parseaddr
    if sender is None or not str(sender).strip():
        return []
    raw = _clip(sender, MAX_SENDER_CHARS)
    name, addr = parseaddr(raw)
    if not _ADDRESS_RE.fullmatch(addr or ""):
        return [Indicator("sender_malformed", "sender",
                          "Sender is not a valid email address format",
                          "The sender field could not be read as name@domain.",
                          (_clip(raw, 80),))]
    domain = addr.rsplit("@", 1)[1].lower()
    labels = domain.split(".")
    reg = _registrable(labels)
    domain_tokens = {t for lbl in labels for t in _tokens(lbl)}
    found = []
    name_tokens = {t for t in re.split(r"[^a-z]+", name.lower()) if t}
    brands = sorted(b for b in BRANDS if b in name_tokens and b not in domain_tokens)
    if brands:
        found.append(Indicator(
            "sender_brand_mismatch", "sender",
            "Display name mentions a brand the address does not",
            "The name shown to you names a company, but the address domain is different. "
            "This does not verify who actually sent the email.",
            (f"name mentions {', '.join(brands)}; address domain is "
             f"{_clip('.'.join(reg), 60)}",)))
    embedded = re.findall(r"[^@\s<>\"']{1,64}@([A-Za-z0-9.\-]{1,255}\.[A-Za-z]{2,})", name)
    if any(_registrable(e.lower().split(".")) != reg for e in embedded):
        found.append(Indicator(
            "sender_name_address", "sender",
            "Display name contains a different email address",
            "The name shown includes an address whose domain differs from the real sender.",
            (_clip(raw, 80),)))
    if not domain.isascii() or any(l.startswith("xn--") for l in labels):
        found.append(Indicator(
            "sender_idn", "sender", "Sender domain uses an internationalized style",
            "Domains with xn-- or non-ASCII characters can imitate familiar names.",
            (_clip(domain, 80),)))
    return found


# -------------------------------------------------------------- public: rules
def find_indicators(subject, body, sender=None):
    """Rule-based indicators. Returns (indicators, urls_found, notes)."""
    subject, body = validate_text_inputs(subject, body)
    if sender is not None and not isinstance(sender, str):
        raise InputError("Sender must be text.")
    text = combine_text(subject, body)[:MAX_CHARS]
    flat = " ".join(text.split())
    indicators = _language_indicators(flat)
    url_inds, urls_found, notes = _url_indicators(text)
    indicators += url_inds + _sender_indicators(sender)
    return tuple(indicators), urls_found, tuple(notes)


# -------------------------------------------------------------- public: model
@lru_cache(maxsize=2)
def _feature_names(vectorizer):
    return vectorizer.get_feature_names_out()


def _display(term: str) -> str:
    return " ".join(PLACEHOLDERS.get(w, w) for w in term.split())


def explain_model(bundle: ModelBundle, subject, body, k: int = 6):
    """Words in this email that pushed the model's score up or down.

    Returns (toward_bad, toward_legit) as tuples of ModelTerm. Requires a
    TF-IDF + linear model pipeline.
    """
    subject, body = validate_text_inputs(subject, body)
    try:
        vectorizer = bundle.pipeline.named_steps["tfidf"]
        weights = bundle.pipeline.named_steps["clf"].coef_[0]
    except (AttributeError, KeyError) as err:
        raise ExplainError("This model has no per-word weights to explain.") from err
    matrix = vectorizer.transform([combine_text(subject, body)]).tocsr()
    if matrix.nnz == 0:
        return (), ()
    contributions = matrix.data * weights[matrix.indices]
    names = _feature_names(vectorizer)[matrix.indices]
    order = np.argsort(contributions)

    def build(indices):
        return tuple(ModelTerm(str(names[i]), _display(str(names[i])),
                               float(contributions[i])) for i in indices)

    toward_legit = build([i for i in order[:k] if contributions[i] < 0])
    toward_bad = build([i for i in order[::-1][:k] if contributions[i] > 0])
    return toward_bad, toward_legit


def explain_email(bundle: ModelBundle, subject, body, sender=None, k: int = 6) -> Explanation:
    """Both kinds of evidence, kept in separate fields."""
    indicators, urls_found, notes = find_indicators(subject, body, sender)
    toward_bad, toward_legit = explain_model(bundle, subject, body, k=k)
    return Explanation(indicators=indicators, urls_found=urls_found,
                       toward_bad=toward_bad, toward_legit=toward_legit,
                       not_checked=NOT_CHECKED, model_caveat=MODEL_CAVEAT, notes=notes)


def main() -> int:
    parser = argparse.ArgumentParser(description="Explain one email's result.")
    parser.add_argument("--subject", default="")
    parser.add_argument("--body", default="")
    parser.add_argument("--sender", default=None)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    try:
        bundle = load_model(args.models_dir)
        result = explain_email(bundle, args.subject, args.body, args.sender)
    except (ModelLoadError, InputError, ExplainError) as err:
        print(f"Error: {err}")
        return 1

    print("DETECTED BY RULES (hints, not proof):")
    if not result.indicators:
        print("  none")
    for ind in result.indicators:
        print(f"  [{ind.category}] {ind.title}")
        for ev in ind.evidence:
            print(f"      {ev}")
    for note in result.notes:
        print(f"Note: {note}")
    print(f"\nWHAT THE MODEL WEIGHS ({result.model_caveat})")
    print("  toward bad:  " + (", ".join(t.display for t in result.toward_bad) or "none"))
    print("  toward legit:" + " " + (", ".join(t.display for t in result.toward_legit) or "none"))
    print("\nNOT CHECKED:")
    for item in result.not_checked:
        print(f"  - {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
