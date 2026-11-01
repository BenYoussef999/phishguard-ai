"""Classification metrics for the phishing/spam detector.

Label convention: 1 = unwanted/malicious (positive class), 0 = legitimate.
A false negative is a missed bad email (dangerous); a false positive is a
legitimate email wrongly flagged (costly in a different way).
"""

import numpy as np
from sklearn.metrics import confusion_matrix


def _ratio(numerator: int, denominator: int):
    return float(numerator / denominator) if denominator else None


def compute_metrics(y_true, y_pred) -> dict:
    """Return counts and rates. Rates that are undefined are None, not 0.

    When the data has no legitimate emails (e.g. Nazario), precision, F1 and
    false-positive rate are meaningless, so only recall is reported
    (`recall_only` is True).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = (int(v) for v in
                      confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    n_neg = tn + fp
    recall_only = n_neg == 0

    recall = _ratio(tp, tp + fn)
    precision = None if recall_only else _ratio(tp, tp + fp)
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = float(2 * precision * recall / (precision + recall))

    return {
        "n": int(len(y_true)),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": None if recall_only else _ratio(fp, n_neg),
        "accuracy": _ratio(tp + tn, len(y_true)),
        "recall_only": recall_only,
    }


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def format_metrics(title: str, m: dict) -> str:
    lines = [f"{title}  (n={m['n']:,})"]
    if m["recall_only"]:
        lines.append(f"  recall (caught / total bad): {_fmt(m['recall'])}   "
                     f"caught={m['tp']:,}  missed(FN)={m['fn']:,}")
        lines.append("  precision/F1 not defined: this data has no legitimate emails")
    else:
        lines.append(f"  precision={_fmt(m['precision'])}  recall={_fmt(m['recall'])}  "
                     f"F1={_fmt(m['f1'])}  (accuracy={_fmt(m['accuracy'])})")
        lines.append(f"  confusion: TN={m['tn']:,}  FP={m['fp']:,}  "
                     f"FN={m['fn']:,}  TP={m['tp']:,}")
    return "\n".join(lines)
