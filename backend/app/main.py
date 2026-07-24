from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from .schemas import ReviewRequest, PredictionResponse
from .model_utils import predict_fake_review

app = FastAPI(
    title="FakeGuard API",
    description="AI-powered Fake Review Detection System with Ensemble ML",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: ReviewRequest):
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Review text cannot be empty.")
        return predict_fake_review(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "FakeGuard API"}

# Mount React Static Frontend
frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")