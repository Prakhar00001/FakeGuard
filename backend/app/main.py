from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import traceback
from .schemas import ReviewRequest, PredictionResponse
from .model_utils import predict_fake_review

app = FastAPI(
    title="FakeGuard API",
    description="AI-powered Fake Review Detection System with Ensemble ML",
    version="1.0.0"
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/predict", response_model=PredictionResponse)
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ReviewRequest):
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Review text cannot be empty.")
        
        result = predict_fake_review(request.text)
        return result
    except Exception as e:
        error_trace = traceback.format_exc()
        print("CRITICAL PREDICTION ERROR:\n", error_trace)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
@app.get("/stats")
def get_stats():
    return {
        "total_reviews_trained": 40000,
        "model_type": "Ensemble (Random Forest + XGBoost + Logistic Regression)",
        "f1_score": 0.88,
        "status": "operational"
    }

@app.get("/api/health")
@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "FakeGuard API"}

# Mount React Static Frontend
frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")