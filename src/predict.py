"""Single-email inference with a validated, saved model.

Security rules enforced here:
  * Email text is untrusted. It is only normalized with regular expressions and
    scored by the model. It is never executed, opened, rendered or fetched.
  * This module never logs, prints or stores message content.
  * Model files are pickles: loading one runs code. Only load model files that
    you trained yourself.

The model uses subject + body only. A sender address (if the interface collects
one) is handled by the rule-based checks in src/explain.py, not here.

Try it from the project root:
    python -m src.predict --subject "Verify your account" --body "Click here now"
    python -m src.predict --body-file path\\to\\email.txt
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import sklearn

from src.preprocessing import MAX_CHARS, combine_text

DEFAULT_MODEL_NAME = "phishguard_model"
MAX_INPUT_CHARS = 500_000        # per field; larger input is rejected, not processed

BAND_LABELS = {
    "bad": "Likely Phishing or Spam",
    "suspicious": "Suspicious / Uncertain",
    "legit": "Likely Legitimate",
}
BAND_ADVICE = {
    "bad": ("This email resembles known phishing or spam. Do not click links, open "
            "attachments, or reply with personal information. Verify through a "
            "channel you already trust."),
    "suspicious": ("The model is unsure about this email. Treat links and requests "
                   "for information with caution and verify the sender another way."),
    "legit": ("The model found no strong warning signs. This is not a guarantee that "
              "the email is safe; stay alert to unexpected requests."),
}


class ModelLoadError(RuntimeError):
    """The saved model or its metadata is missing, malformed or incompatible."""


class InputError(ValueError):
    """The submitted email text cannot be analyzed."""


@dataclass(frozen=True)
class ModelBundle:
    pipeline: object
    metadata: dict

    @property
    def t_low(self) -> float:
        return float(self.metadata["thresholds"]["t_low"])

    @property
    def t_high(self) -> float:
        return float(self.metadata["thresholds"]["t_high"])


@dataclass(frozen=True)
class Prediction:
    band: str            # "legit" | "suspicious" | "bad"
    label: str           # display name of the band
    advice: str          # what the reader should do; never claims an email is safe
    score: float         # raw model output in [0, 1]; NOT a probability
    truncated: bool
    notes: tuple


def _major_minor(version: str):
    parts = str(version).split(".")
    return tuple(parts[:2])


def band_for_score(score: float, t_low: float, t_high: float) -> str:
    if score < t_low:
        return "legit"
    if score < t_high:
        return "suspicious"
    return "bad"


def load_model(models_dir: Path = Path("models"), name: str = DEFAULT_MODEL_NAME,
               allow_version_mismatch: bool = False) -> ModelBundle:
    """Load and validate the saved pipeline and its metadata."""
    model_path = Path(models_dir) / f"{name}.joblib"
    meta_path = Path(models_dir) / f"{name}_meta.json"
    howto = ("Train it first with: python -m src.train_final --t-low 0.25 "
             "--t-high 0.70 (see the README).")
    if not model_path.exists():
        raise ModelLoadError(f"Model file not found: {model_path}. {howto}")
    if not meta_path.exists():
        raise ModelLoadError(f"Model metadata not found: {meta_path}. {howto}")

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise ModelLoadError(f"Model metadata {meta_path} is unreadable or not valid "
                             f"JSON ({err.__class__.__name__}). {howto}") from err
    if not isinstance(meta, dict):
        raise ModelLoadError(f"Model metadata {meta_path} has an unexpected format.")

    th = meta.get("thresholds")
    try:
        t_low, t_high = float(th["t_low"]), float(th["t_high"])
    except (TypeError, KeyError, ValueError) as err:
        raise ModelLoadError("Model metadata has no valid thresholds.") from err
    if not (0.0 <= t_low <= t_high <= 1.0):
        raise ModelLoadError(f"Model metadata has invalid thresholds "
                             f"(t_low={t_low}, t_high={t_high}).")
    if meta.get("score_is_probability") is not False:
        raise ModelLoadError("Model metadata does not declare that scores are not "
                             "probabilities; refusing to load it.")

    trained_with = meta.get("sklearn_version")
    if (not allow_version_mismatch
            and _major_minor(trained_with) != _major_minor(sklearn.__version__)):
        raise ModelLoadError(
            f"The model was trained with scikit-learn {trained_with} but "
            f"{sklearn.__version__} is installed. Loading it could give wrong "
            f"results. Retrain the model or install the matching version.")

    try:
        pipeline = joblib.load(model_path)
    except Exception as err:                                 # unreadable pickle
        raise ModelLoadError(f"Model file {model_path} could not be loaded "
                             f"({err.__class__.__name__}). {howto}") from err
    if not hasattr(pipeline, "predict_proba"):
        raise ModelLoadError(f"{model_path} is not a classification pipeline.")
    if list(getattr(pipeline, "classes_", [])) != [0, 1]:
        raise ModelLoadError(f"{model_path} does not have the expected classes [0, 1].")
    return ModelBundle(pipeline=pipeline, metadata=meta)


def validate_text_inputs(subject, body):
    """Return cleaned (subject, body) strings or raise InputError."""
    cleaned = []
    for field, value in (("Subject", subject), ("Body", body)):
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise InputError(f"{field} must be text.")
        if len(value) > MAX_INPUT_CHARS:
            raise InputError(f"{field} is too long ({len(value):,} characters; the "
                             f"limit is {MAX_INPUT_CHARS:,}).")
        cleaned.append(value.replace("\x00", "").strip())
    if not cleaned[0] and not cleaned[1]:
        raise InputError("Enter a subject, a body, or both.")
    return cleaned[0], cleaned[1]


def predict_email(bundle: ModelBundle, subject, body) -> Prediction:
    """Score one email. Raises InputError for unusable input."""
    subject, body = validate_text_inputs(subject, body)
    text = combine_text(subject, body)
    truncated = len(text) > MAX_CHARS
    notes = []
    if truncated:
        notes.append(f"Only the first {MAX_CHARS:,} characters were analyzed.")
    score = float(bundle.pipeline.predict_proba([text])[0, 1])
    band = band_for_score(score, bundle.t_low, bundle.t_high)
    return Prediction(band=band, label=BAND_LABELS[band], advice=BAND_ADVICE[band],
                      score=score, truncated=truncated, notes=tuple(notes))


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify one email.")
    parser.add_argument("--subject", default="")
    parser.add_argument("--body", default="")
    parser.add_argument("--body-file", type=Path,
                        help="Read the body from a UTF-8 text file instead.")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    body = args.body
    if args.body_file is not None:
        try:
            if args.body_file.stat().st_size > 4 * MAX_INPUT_CHARS:
                print("Error: the file is too large to analyze.")
                return 1
            body = args.body_file.read_text(encoding="utf-8", errors="replace")
        except OSError as err:
            print(f"Error: cannot read {args.body_file} ({err.__class__.__name__}).")
            return 1
    try:
        bundle = load_model(args.models_dir)
        result = predict_email(bundle, args.subject, body)
    except (ModelLoadError, InputError) as err:
        print(f"Error: {err}")
        return 1

    print(f"Result: {result.label}")
    print(result.advice)
    for note in result.notes:
        print(f"Note: {note}")
    print(f"(raw model score {result.score:.2f}; this is NOT a probability)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
