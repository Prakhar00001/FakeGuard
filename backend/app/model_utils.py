import os
import re
import string
from pathlib import Path
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
MODEL_PATH = BASE_DIR / "artifacts" / "fakeguard_model.joblib"
DATASET_PATH = DATA_DIR / "fake reviews dataset.csv"

FEATURE_NAMES = [
    "word_count", "unique_ratio", "punctuation_ratio", "uppercase_ratio",
    "exclamation_count", "question_count", "sentiment_balance",
    "avg_word_length", "sentence_count", "avg_sentence_length",
    "capital_word_ratio", "repeated_word_ratio", "superlative_count",
    "review_length", "digit_ratio",
]


def load_dataset() -> tuple[pd.DataFrame, List[str], List[int]]:
    """Load the real 40K review dataset from CSV."""
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=["text_"])
    df["label_bin"] = (df["label"] == "CG").astype(int)  # CG=fake=1, OR=genuine=0
    return df, df["text_"].tolist(), df["label_bin"].tolist()


def extract_linguistic_features(text: str) -> List[float]:
    """Extract 15 linguistic and behavioral features from review text."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    raw_tokens = re.findall(r"\b\w+\b", text)
    word_count = max(1, len(tokens))
    unique_ratio = len(set(tokens)) / word_count
    punctuation_ratio = sum(1 for ch in text if ch in string.punctuation) / max(1, len(text))
    uppercase_ratio = sum(1 for ch in text if ch.isupper()) / max(1, len(text))
    exclamation_count = text.count("!")
    question_count = text.count("?")

    positive_words = {"good", "great", "excellent", "amazing", "fantastic", "perfect", "love", "worth", "best", "wonderful", "awesome", "outstanding", "superb", "brilliant"}
    negative_words = {"bad", "poor", "worst", "terrible", "awful", "hate", "broken", "useless", "horrible", "disappointing", "waste", "garbage", "trash"}
    positive_markers = sum(t in positive_words for t in tokens)
    negative_markers = sum(t in negative_words for t in tokens)
    sentiment_balance = positive_markers - negative_markers

    avg_word_length = np.mean([len(t) for t in tokens]) if tokens else 0.0

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = max(1, len(sentences))
    avg_sentence_length = word_count / sentence_count

    capital_word_ratio = sum(1 for t in raw_tokens if t.isupper() and len(t) > 1) / word_count

    from collections import Counter
    token_counts = Counter(tokens)
    repeated_word_ratio = sum(1 for c in token_counts.values() if c > 1) / max(1, len(token_counts))

    superlatives = {"best", "worst", "greatest", "most", "least", "finest", "perfect", "flawless", "ultimate", "unbeatable"}
    superlative_count = sum(t in superlatives for t in tokens)

    review_length = float(len(text))
    digit_ratio = sum(1 for ch in text if ch.isdigit()) / max(1, len(text))

    return [
        float(word_count),
        float(unique_ratio),
        float(punctuation_ratio),
        float(uppercase_ratio),
        float(exclamation_count),
        float(question_count),
        float(sentiment_balance),
        float(avg_word_length),
        float(sentence_count),
        float(avg_sentence_length),
        float(capital_word_ratio),
        float(repeated_word_ratio),
        float(superlative_count),
        float(review_length),
        float(digit_ratio),
    ]


def build_feature_matrix(texts: List[str], vectorizer: TfidfVectorizer | None = None):
    if vectorizer is None:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=10000, sublinear_tf=True)
        tfidf = vectorizer.fit_transform(texts)
    else:
        tfidf = vectorizer.transform(texts)

    meta_features = np.array([extract_linguistic_features(text) for text in texts], dtype=float)
    meta_features = csr_matrix(meta_features)
    return vectorizer, hstack([tfidf, meta_features], format="csr")


def train_model() -> Dict[str, Any]:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df, texts, labels = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vectorizer, X_train_matrix = build_feature_matrix(X_train)
    _, X_test_matrix = build_feature_matrix(X_test, vectorizer=vectorizer)

    estimators = [
        ("logreg", LogisticRegression(max_iter=1500, solver="liblinear", class_weight="balanced")),
        ("rf", RandomForestClassifier(n_estimators=250, max_depth=12, random_state=42, class_weight="balanced_subsample", n_jobs=-1)),
    ]
    if XGBClassifier is not None:
        estimators.append(("xgb", XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.8, random_state=42,
            eval_metric="logloss", n_jobs=-1,
        )))

    ensemble = VotingClassifier(estimators=estimators, voting="soft")
    ensemble.fit(X_train_matrix, y_train)

    preds = ensemble.predict(X_test_matrix)
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)

    payload = {
        "vectorizer": vectorizer,
        "classifier": ensemble,
        "accuracy": round(float(accuracy), 3),
        "f1_score": round(float(f1), 3),
        "precision": round(float(precision), 3),
        "recall": round(float(recall), 3),
        "dataset_size": len(texts),
        "feature_count": len(FEATURE_NAMES),
    }
    joblib.dump(payload, MODEL_PATH)
    return payload


def load_model() -> Dict[str, Any]:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return train_model()


def get_model_stats() -> Dict[str, Any]:
    bundle = load_model()
    return {
        "accuracy": bundle.get("accuracy"),
        "f1_score": bundle.get("f1_score"),
        "precision": bundle.get("precision"),
        "recall": bundle.get("recall"),
        "dataset_size": bundle.get("dataset_size", 40432),
        "feature_count": bundle.get("feature_count", len(FEATURE_NAMES)),
        "linguistic_features": FEATURE_NAMES,
    }


def predict_text(review: str) -> Dict[str, Any]:
    bundle = load_model()
    vectorizer = bundle["vectorizer"]
    classifier = bundle["classifier"]
    _, features = build_feature_matrix([review], vectorizer=vectorizer)
    probability = float(classifier.predict_proba(features)[0, 1])
    label = "fake" if probability >= 0.5 else "genuine"
    raw_features = extract_linguistic_features(review)
    features_summary = {name: round(val, 3) if isinstance(val, float) and val != int(val) else int(val) if val == int(val) else round(val, 3) for name, val in zip(FEATURE_NAMES, raw_features)}
    return {
        "predicted_label": label,
        "probability_fake": round(probability, 3),
        "confidence": round(max(probability, 1 - probability), 3),
        "features": features_summary,
        "model_accuracy": bundle.get("accuracy"),
        "f1_score": bundle.get("f1_score"),
    }
