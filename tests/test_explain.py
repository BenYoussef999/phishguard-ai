"""Tests for src/explain.py."""

import logging
import time

import pandas as pd
import pytest

from src.experiments import VARIANTS, build_variant_pipeline
from src.explain import (NOT_CHECKED, ExplainError, defang_url, explain_email,
                         explain_model, find_indicators)
from src.predict import InputError, ModelBundle


def ids(result):
    return {i.rule_id for i in result[0]}


def body_ids(body, subject="hello", sender=None):
    return ids(find_indicators(subject, body, sender))


# ------------------------------------------------------------ language rules
@pytest.mark.parametrize("text,rule", [
    ("Please respond urgently to this notice", "urgency"),
    ("This is your final notice", "urgency"),
    ("You must act within 24 hours", "urgency"),
    ("Your account will be suspended today", "threat"),
    ("We noticed unusual activity on your profile", "threat"),
    ("Please verify your account to continue", "credentials"),
    ("Enter your password below", "credentials"),
    ("We need your credit card number", "payment"),
    ("Send payment by wire transfer", "payment"),
    ("Your mailbox is full, 3 pending messages", "mailbox_lure"),
    ("You have an incoming fax document", "mailbox_lure"),
])
def test_language_rules_fire_on_matching_phrases(text, rule):
    assert rule in body_ids(text)


@pytest.mark.parametrize("text", [
    "See you at the project meeting tomorrow at noon.",
    "Attached are the notes from our budget review.",
    "Thanks for the update, talk soon.",
])
def test_language_rules_stay_quiet_on_ordinary_text(text):
    assert body_ids(text) == set()


def test_evidence_is_deduplicated_short_and_limited():
    text = " ".join(["URGENT"] * 20 + ["act now", "final notice", "asap", "limited time",
                                       "right away", "time-sensitive"])
    ind = next(i for i in find_indicators("s", text)[0] if i.rule_id == "urgency")
    assert len(ind.evidence) == 5 and len(set(ind.evidence)) == 5
    assert all(len(e) < 40 for e in ind.evidence)


def test_every_indicator_has_title_detail_and_evidence():
    result = find_indicators("Verify now", "Your account will be suspended. "
                             "Click http://1.2.3.4/login", "PayPal <a@gmail.com>")
    assert result[0]
    for ind in result[0]:
        assert ind.title and ind.detail and ind.evidence


# ------------------------------------------------------------------ URL rules
def test_ip_address_links_are_flagged_including_numeric_encodings():
    assert "url_ip_address" in body_ids("go to http://192.168.1.5/login now")
    assert "url_ip_address" in body_ids("go to http://3232235777/login now")
    assert "url_ip_address" in body_ids("go to http://[2001:db8::1]/login now")


def test_userinfo_trick_is_flagged():
    assert "url_userinfo" in body_ids("open http://paypal.com@evil.example/login")


def test_punycode_shortener_and_deep_subdomains_are_flagged():
    assert "url_idn" in body_ids("see http://xn--pple-43d.com/")
    assert "url_shortener" in body_ids("see https://bit.ly/3abcde")
    assert "url_many_subdomains" in body_ids("see http://a.b.c.d.example.com/x")


@pytest.mark.parametrize("url", [
    "http://paypal.com.secure-id.example.net/login",      # brand in a subdomain
    "http://paypal.example.net/login",
    "http://paypal-support.com/help",                      # brand joined to other words
])
def test_brand_in_unusual_position_is_flagged(url):
    assert "url_brand_position" in body_ids(f"visit {url} today")


@pytest.mark.parametrize("url", [
    "https://www.paypal.com/signin", "https://accounts.google.com/login",
    "https://aws.amazon.com/", "https://s3.amazonaws.com/bucket/file",
    "https://www.paypalobjects.com/x.png", "https://www.example.co.uk/page",
    "https://login.microsoftonline.com/", "https://news.ycombinator.com/",
])
def test_ordinary_links_do_not_trigger_url_rules(url):
    assert body_ids(f"read {url} later") == set()


def test_html_anchor_text_mismatch_is_flagged_but_matching_is_not():
    mismatch = '<a href="http://evil.example/x">http://www.paypal.com/login</a>'
    match = '<a href="https://www.paypal.com/login">https://www.paypal.com/login</a>'
    plain = '<a href="https://www.paypal.com/login">Click here</a>'
    assert "url_text_mismatch" in body_ids(mismatch)
    assert "url_text_mismatch" not in body_ids(match)
    assert "url_text_mismatch" not in body_ids(plain)


def test_urls_are_defanged_in_all_evidence():
    result = find_indicators("s", "go http://1.2.3.4/login and https://bit.ly/abc "
                             '<a href="http://evil.example">www.paypal.com</a>')
    evidence = [e for i in result[0] for e in i.evidence]
    assert evidence
    assert not any("http://" in e or "https://" in e for e in evidence)
    assert any("hxxp" in e and "[.]" in e for e in evidence)


def test_defang_url_examples():
    assert defang_url("https://example.com/a.b") == "hxxps://example[.]com/a.b"
    assert defang_url("http://1.2.3.4/x") == "hxxp://1[.]2[.]3[.]4/x"
    assert defang_url("www.example.com") == "www[.]example[.]com"
    assert len(defang_url("http://a.com/" + "x" * 500)) <= 100


def test_malformed_urls_do_not_crash():
    nasty = "http:// http://[::1 http://:80 http://%zz http://a b ftp://[bad www. http://@"
    result = find_indicators("s", nasty)
    assert isinstance(result[0], tuple)


def test_only_the_first_urls_are_analyzed_with_a_note():
    body = " ".join(f"http://192.168.0.{i % 250}/{i}" for i in range(80))
    inds, found, notes = find_indicators("s", body)
    assert found == 80
    assert any("first 50" in n for n in notes)


# ---------------------------------------------------------------- sender rules
def sender_ids(sender):
    return ids(find_indicators("hello", "ordinary body text here", sender))


def test_sender_brand_mismatch_and_embedded_address():
    assert "sender_brand_mismatch" in sender_ids("PayPal Support <help@gmail.com>")
    assert "sender_name_address" in sender_ids('"support@paypal.com" <x@evil.example>')


def test_consistent_or_absent_sender_triggers_nothing():
    assert sender_ids("PayPal <service@paypal.com>") == set()
    assert sender_ids("Alice Martin <alice@example.org>") == set()
    assert sender_ids(None) == set() and sender_ids("") == set() and sender_ids("  ") == set()


def test_malformed_sender_is_reported_without_crashing():
    assert "sender_malformed" in sender_ids("not an email")
    assert "sender_malformed" in sender_ids("<>")
    assert "sender_malformed" in sender_ids("a@b")
    assert sender_ids("x" * 5000) == {"sender_malformed"}


def test_non_text_sender_is_rejected():
    with pytest.raises(InputError):
        find_indicators("s", "b", sender=123)


# --------------------------------------------------------------- model terms
def tiny_bundle():
    rows, gid = [], 0
    for source, labels in {"ceas08": (0, 1), "enron": (0, 1), "nazario": (1,)}.items():
        for label in labels:
            for i in range(40):
                cls = ("winprize claimfree urgent" if label else "agenda review project")
                rows.append((source, f"s{gid}", f"{cls} sharedword {i}", label, gid))
                gid += 1
    df = pd.DataFrame(rows, columns=["source", "subject", "body", "label", "dup_group"])
    pipe = build_variant_pipeline(df, min_df=1, min_source_df=2, **VARIANTS["no_email"])
    return ModelBundle(pipeline=pipe,
                       metadata={"thresholds": {"t_low": 0.25, "t_high": 0.7}})


@pytest.fixture(scope="module")
def bundle():
    return tiny_bundle()


def test_model_terms_point_the_right_way_and_are_limited(bundle):
    bad, legit = explain_model(bundle, "winprize urgent", "claimfree agenda review", k=2)
    assert len(bad) <= 2 and len(legit) <= 2
    assert all(t.strength > 0 for t in bad) and all(t.strength < 0 for t in legit)
    assert {t.term for t in bad} & {"winprize", "claimfree", "urgent"}
    assert {t.term for t in legit} & {"agenda", "review"}
    assert bad == tuple(sorted(bad, key=lambda t: -t.strength))


def test_placeholders_are_spelled_out_for_display(bundle):
    bad, legit = explain_model(bundle, "x", "winprize 12345 http://a.com agenda")
    shown = " ".join(t.display for t in bad + legit)
    assert "numtoken" not in shown and "urltoken" not in shown


def test_text_with_no_known_words_gives_empty_explanation(bundle):
    assert explain_model(bundle, "qqqzzz", "xxxyyy") == ((), ())


def test_model_without_weights_raises_a_clear_error():
    class Fake:
        named_steps = {}
    with pytest.raises(ExplainError, match="per-word weights"):
        explain_model(ModelBundle(pipeline=Fake(), metadata={}), "a", "b")


def test_empty_input_is_rejected_by_both_parts(bundle):
    with pytest.raises(InputError):
        explain_email(bundle, "", "")


# ---------------------------------------------------------- the combined result
def test_explain_email_keeps_rule_evidence_and_model_evidence_separate(bundle):
    exp = explain_email(bundle, "Verify now", "Your account will be suspended. "
                        "Open http://1.2.3.4/login winprize", "PayPal <x@gmail.com>")
    assert {i.rule_id for i in exp.indicators} >= {"threat", "url_ip_address",
                                                   "sender_brand_mismatch"}
    assert all(not hasattr(i, "strength") for i in exp.indicators)
    assert all(not hasattr(t, "rule_id") for t in exp.toward_bad + exp.toward_legit)
    assert exp.not_checked == NOT_CHECKED
    assert "not proof" in exp.model_caveat
    assert any("opened" in n for n in exp.not_checked)
    assert any("authenticated" in n for n in exp.not_checked)


def test_ordinary_email_has_no_indicators_and_still_lists_what_was_not_checked(bundle):
    exp = explain_email(bundle, "Lunch?", "Are we meeting at noon to review the notes?")
    assert exp.indicators == () and exp.not_checked


# ------------------------------------------------------------------- safety
ADVERSARIAL = {
    "dots": "a." * 10000,
    "anchors": '<a href="' * 3000,
    "anchor text": '<a href="http://a.com">' + "x" * 15000,
    "words": "account " * 3000,
    "spaces after account": "account" + " " * 15000 + "will be suspended",
    "many urls": "http://a.b.c.d.e.f.g.h.com/ " * 600,
    "angle": "<" * 20000,
}


@pytest.mark.parametrize("name", list(ADVERSARIAL))
def test_adversarial_input_is_processed_quickly(name, bundle):
    start = time.perf_counter()
    explain_email(bundle, "s", ADVERSARIAL[name], "a" * 900 + "@" + "b" * 300 + ".com")
    assert time.perf_counter() - start < 1.0, name


def test_content_is_never_logged_or_printed(bundle, caplog, capsys):
    secret = "zebra-secret-token-98765"
    with caplog.at_level(logging.DEBUG):
        explain_email(bundle, secret, f"urgent {secret} http://1.2.3.4/x", f"{secret} <a@b.com>")
    out = capsys.readouterr()
    assert secret not in caplog.text and secret not in out.out and secret not in out.err
