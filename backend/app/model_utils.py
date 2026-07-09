import os
import re
import string
import random
from pathlib import Path
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "fakeguard_model.joblib"


def generate_training_data(n: int = 1800) -> tuple[List[str], List[int]]:
    random.seed(42)
    genuine_templates = [
        "I bought this {product} last week and it arrived on time. The packaging was secure and the quality feels solid.",
        "I use this {product} daily for work and it has been reliable so far. The battery life is better than expected.",
        "This {product} is easy to set up and the instructions were clear. I appreciated the simple design and useful features.",
        "After using the {product} for a few weeks, I can say the build quality is good and the performance is consistent.",
        "I was unsure at first, but the {product} has worked well for me. It feels durable and does what I need.",
    ]
    fake_templates = [
        "This {product} is absolutely amazing and the best thing ever. You need to buy it right now before it sells out!",
        "I cannot believe how perfect this {product} is. It is fantastic, unbelievable, and everyone should own one immediately!",
        "Amazing quality, incredible value, and such a great {product}. Buy it now, you will not regret it!",
        "This {product} is wonderful and flawless. Five stars, best purchase ever, must buy!",
        "Wow, what a stunning {product}. Perfect in every way and insanely good. Buy it now!",
    ]

    products = ["coffee maker", "wireless speaker", "fitness tracker", "water bottle", "desk lamp"]
    adjectives = ["fantastic", "excellent", "reliable", "smooth", "clean"]
    texts: List[str] = []
    labels: List[int] = []

    for _ in range(n // 2):
        product = random.choice(products)
        template = random.choice(genuine_templates)
        review = template.format(product=product)
        if random.random() < 0.15:
            review += " The shipping was delayed but the support team responded quickly."
        texts.append(review)
        labels.append(0)

    for _ in range(n // 2):
        product = random.choice(products)
        template = random.choice(fake_templates)
        review = template.format(product=product)
        review += " " + random.choice(["Amazing!", "Perfect!", "Best ever!", "Must buy!", "Five stars!"])
        review += " " + random.choice(["You will love it.", "I recommend it to everyone.", "Totally worth it."])
        texts.append(review)
        labels.append(1)

    return texts, labels


def extract_linguistic_features(text: str) -> List[float]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    word_count = max(1, len(tokens))
    unique_ratio = len(set(tokens)) / word_count
    punctuation_ratio = sum(1 for ch in text if ch in string.punctuation) / max(1, len(text))
    uppercase_ratio = sum(1 for ch in text if ch.isupper()) / max(1, len(text))
    exclamation_count = text.count("!")
    question_count = text.count("?")
    positive_markers = sum(token in {"good", "great", "excellent", "amazing", "fantastic", "perfect", "love", "worth", "best"} for token in tokens)
    negative_markers = sum(token in {"bad", "poor", "worst", "terrible", "awful", "hate", "broken", "useless"} for token in tokens)
    sentiment_balance = positive_markers - negative_markers
    return [
        float(word_count),
        float(unique_ratio),
        float(punctuation_ratio),
        float(uppercase_ratio),
        float(exclamation_count),
        float(question_count),
        float(sentiment_balance),
    ]


def build_feature_matrix(texts: List[str], vectorizer: TfidfVectorizer | None = None):
    if vectorizer is None:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=4000)
        tfidf = vectorizer.fit_transform(texts)
    else:
        tfidf = vectorizer.transform(texts)

    meta_features = np.array([extract_linguistic_features(text) for text in texts], dtype=float)
    meta_features = csr_matrix(meta_features)
    return vectorizer, hstack([tfidf, meta_features], format="csr")


def train_model() -> Dict[str, Any]:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    texts, labels = generate_training_data(1800)
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

    vectorizer, X_train_matrix = build_feature_matrix(X_train)
    _, X_test_matrix = build_feature_matrix(X_test, vectorizer=vectorizer)

    estimators = [
        ("logreg", LogisticRegression(max_iter=1500, solver="liblinear", class_weight="balanced")),
        ("rf", RandomForestClassifier(n_estimators=220, max_depth=10, random_state=42, class_weight="balanced_subsample")),
    ]
    if XGBClassifier is not None:
        estimators.append(("xgb", XGBClassifier(n_estimators=160, max_depth=4, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8, random_state=42, eval_metric="logloss")))

    from sklearn.ensemble import VotingClassifier

    ensemble = VotingClassifier(estimators=estimators, voting="soft")
    ensemble.fit(X_train_matrix, y_train)

    preds = ensemble.predict(X_test_matrix)
    accuracy = accuracy_score(y_test, preds)

    payload = {
        "vectorizer": vectorizer,
        "classifier": ensemble,
        "accuracy": round(float(accuracy), 3),
    }
    joblib.dump(payload, MODEL_PATH)
    return payload


def load_model() -> Dict[str, Any]:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return train_model()


def predict_text(review: str) -> Dict[str, Any]:
    bundle = load_model()
    vectorizer = bundle["vectorizer"]
    classifier = bundle["classifier"]
    _, features = build_feature_matrix([review], vectorizer=vectorizer)
    probability = float(classifier.predict_proba(features)[0, 1])
    label = "fake" if probability >= 0.5 else "genuine"
    features_summary = {
        "word_count": extract_linguistic_features(review)[0],
        "unique_ratio": round(float(extract_linguistic_features(review)[1]), 3),
        "punctuation_ratio": round(float(extract_linguistic_features(review)[2]), 3),
        "uppercase_ratio": round(float(extract_linguistic_features(review)[3]), 3),
        "exclamation_count": int(extract_linguistic_features(review)[4]),
        "question_count": int(extract_linguistic_features(review)[5]),
        "sentiment_balance": int(extract_linguistic_features(review)[6]),
    }
    return {
        "predicted_label": label,
        "probability_fake": round(probability, 3),
        "confidence": round(max(probability, 1 - probability), 3),
        "features": features_summary,
        "model_accuracy": bundle.get("accuracy", None),
    }
