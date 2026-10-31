"""Smoke test: confirms core libraries import and a minimal
TF-IDF + Logistic Regression pipeline can fit and predict."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def test_minimal_pipeline_fits_and_predicts():
    texts = [
        "urgent verify your account password now",
        "your account is suspended click here immediately",
        "meeting notes attached for tomorrow",
        "lunch on friday with the team",
    ]
    labels = [1, 1, 0, 0]  # 1 = phishing, 0 = legitimate (toy data)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(random_state=42)),
    ])
    pipeline.fit(texts, labels)

    proba = pipeline.predict_proba(["verify your password now"])
    assert proba.shape == (1, 2)
    assert abs(proba.sum() - 1.0) < 1e-9
