import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize FastAPI App
app = FastAPI(
    title="FakeGuard ML API",
    description="Real-time Fake Review Detection API",
    version="1.0.0"
)

# --------------------------------------------------------------------------
# 1. Enable CORS Middleware
# Fixes "No 'Access-Control-Allow-Origin' header" errors from Vercel
# --------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from Vercel frontend or any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],  # Allows all headers
)

# Request Schema
class ReviewRequest(BaseModel):
    review_text: str

# --------------------------------------------------------------------------
# 2. Native Feature Extraction Fallback
# --------------------------------------------------------------------------
def extract_features_native(text: str) -> dict:
    """Fallback native feature extractor."""
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

# --------------------------------------------------------------------------
# 3. API Endpoints
# --------------------------------------------------------------------------
@app.get("/")
@app.get("/health")
@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "FakeGuard API",
        "message": "Backend is running and accepting cross-origin requests!"
    }

@app.post("/predict")
@app.post("/api/predict")
def predict_review(payload: ReviewRequest):
    """Prediction endpoint for analyzing input review text."""
    review_text = payload.review_text.strip()
    
    if not review_text:
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")

    # Try importing feature extractor relative to backend module
    try:
        from backend.feature_engineering import ReviewFeatureExtractor
        extractor = ReviewFeatureExtractor()
        features_dict = extractor.extract_features(review_text)
    except Exception:
        features_dict = extract_features_native(review_text)

    # Compute score
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