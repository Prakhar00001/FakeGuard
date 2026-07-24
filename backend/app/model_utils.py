import joblib
import numpy as np
from pathlib import Path
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.sparse import hstack

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

sid = SentimentIntensityAnalyzer()

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "ensemble_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer.pkl"

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as e:
    print(f"Warning: Models not found ({e}). Running in mock mode.")
    model, vectorizer = None, None

def extract_linguistic_features(text: str) -> np.ndarray:
    words = text.split()
    num_words = len(words) if words else 1
    char_count = len(text)
    avg_word_len = sum(len(w) for w in words) / num_words if words else 0
    uppercase_ratio = sum(1 for c in text if c.isupper()) / char_count if char_count > 0 else 0
    exclamation_count = text.count('!')
    question_count = text.count('?')
    digit_count = sum(1 for c in text if c.isdigit())
    
    sent = sid.polarity_scores(text)
    neg, neu, pos, compound = sent['neg'], sent['neu'], sent['pos'], sent['compound']
    ttr = len(set(w.lower() for w in words)) / num_words if num_words > 0 else 0
    burstiness = sum(1 for w in words if len(w) > 8) / num_words if num_words > 0 else 0
    
    features = [
        num_words, char_count, avg_word_len, uppercase_ratio,
        exclamation_count, question_count, digit_count,
        neg, neu, pos, compound, ttr, burstiness, abs(compound),
        1.0 if exclamation_count > 2 else 0.0,
        1.0 if uppercase_ratio > 0.3 else 0.0
    ]
    return np.array(features).reshape(1, -1)

def predict_fake_review(text: str):
    if model is None or vectorizer is None:
        return {
            "prediction": "Fake",
            "confidence": 0.88,
            "probability_fake": 0.88,
            "probability_real": 0.12,
            "explanation": "High frequency of exclamation marks and emotional overstatement."
        }
    
    X_tfidf = vectorizer.transform([text])
    X_features = extract_linguistic_features(text)
    X_combined = hstack([X_tfidf, X_features])
    
    prob = model.predict_proba(X_combined)[0]
    prob_fake, prob_real = float(prob[1]), float(prob[0])
    prediction = "Fake" if prob_fake > 0.5 else "Real"
    confidence = max(prob_fake, prob_real)
    
    explanation = "Pattern indicates deceptive review characteristics." if prob_fake > 0.5 else "Linguistic features appear authentic and balanced."
    
    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "probability_fake": round(prob_fake, 4),
        "probability_real": round(prob_real, 4),
        "explanation": explanation
    }