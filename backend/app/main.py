import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Dynamically resolve import paths for backend subdirectories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([current_dir, parent_dir])

try:
    from app.feature_engineering import ReviewFeatureExtractor
except ModuleNotFoundError:
    from feature_engineering import ReviewFeatureExtractor

app = FastAPI(title="FakeGuard API", version="1.0.0")

# Enable CORS for local testing & Vercel deployment
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
    
    # Extract 15+ linguistic and behavioral features
    features_dict = extractor.extract_features(payload.review_text)
    
    # Calculate review score
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