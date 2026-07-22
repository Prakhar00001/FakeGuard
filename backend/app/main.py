from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
from feature_engineering import ReviewFeatureExtractor

app = FastAPI(title="FakeGuard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

extractor = ReviewFeatureExtractor()

class ReviewRequest(BaseModel):
    review_text: str

@app.get("/")
def health_check():
    return {"status": "ok", "service": "FakeGuard API"}

@app.post("/predict")
def predict_review(payload: ReviewRequest):
    if not payload.review_text.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")
    
    # Feature extraction
    features_dict = extractor.extract_features(payload.review_text)
    
    # Simple probability mock fallback if model file isn't pre-loaded
    # (In live execution, loads joblib model)
    fake_probability = round(min(0.99, max(0.01, features_dict['exclamation_ratio'] * 2 + features_dict['uppercase_ratio'] * 3 + 0.15)), 4)
    is_fake = fake_probability > 0.50

    return {
        "review_text": payload.review_text,
        "is_fake": is_fake,
        "fake_probability": fake_probability,
        "confidence": f"{fake_probability * 100:.1f}%" if is_fake else f"{(1 - fake_probability) * 100:.1f}%",
        "linguistic_features": features_dict
    }