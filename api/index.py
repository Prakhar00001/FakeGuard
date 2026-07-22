import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI instance
app = FastAPI(title="FakeGuard AI API", version="1.0.0")

class ReviewRequest(BaseModel):
    review_text: str

def extract_features_native(text: str) -> dict:
    words = text.split()
    word_count = max(len(words), 1)
    char_count = max(len(text), 1)
    
    uppercase_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    return {
        "word_count": word_count,
        "uppercase_ratio": round(uppercase_words / word_count, 4),
        "exclamation_ratio": round(exclamation_count / char_count, 4),
        "question_ratio": round(question_count / char_count, 4),
        "avg_word_length": round(char_count / word_count, 2)
    }

# Health Check Endpoints
@app.get("/health")
@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "FakeGuard Serverless API"}

# Prediction Endpoints
@app.post("/predict")
@app.post("/api/predict")
def predict_review(payload: ReviewRequest):
    review_text = payload.review_text.strip()
    if not review_text:
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")

    # Safe feature extraction
    features_dict = extract_features_native(review_text)

    # Compute probability score
    excl = features_dict.get('exclamation_ratio', 0)
    caps = features_dict.get('uppercase_ratio', 0)
    
    fake_prob = min(0.99, max(0.02, (excl * 3.5) + (caps * 2.5) + 0.12))
    
    buzzwords = ["AMAZING", "MUST BUY", "BEST EVER", "100%", "PERFECT", "SCAM", "GARBAGE", "ZERO STARS"]
    if any(word in review_text.upper() for word in buzzwords):
        fake_prob = min(0.98, fake_prob + 0.38)

    is_fake = fake_prob > 0.50

    return {
        "review_text": review_text,
        "is_fake": is_fake,
        "fake_probability": round(fake_prob, 4),
        "confidence": f"{fake_prob * 100:.1f}%" if is_fake else f"{(1 - fake_prob) * 100:.1f}%",
        "linguistic_features": features_dict
    }