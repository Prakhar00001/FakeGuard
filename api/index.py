import os
import sys

# Ensure root paths resolve cleanly on Vercel
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.extend([root_dir, os.path.join(root_dir, "backend")])

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.feature_engineering import ReviewFeatureExtractor

app = FastAPI(title="FakeGuard Unified API", version="1.0.0")

extractor = ReviewFeatureExtractor()

class ReviewRequest(BaseModel):
    review_text: str

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "Unified FakeGuard Vercel API"}

@app.post("/api/predict")
def predict_review(payload: ReviewRequest):
    if not payload.review_text.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")
    
    # Feature extraction (15+ features)
    features_dict = extractor.extract_features(payload.review_text)
    
    fake_probability = round(
        min(0.99, max(0.01, features_dict['exclamation_ratio'] * 2.5 + features_dict['uppercase_ratio'] * 3.0 + 0.12)),
        4
    )
    is_fake = fake_probability > 0.50

    return {
        "review_text": payload.review_text,
        "is_fake": is_fake,
        "fake_probability": fake_probability,
        "confidence": f"{fake_probability * 100:.1f}%" if is_fake else f"{(1 - fake_probability) * 100:.1f}%",
        "linguistic_features": features_dict
    }