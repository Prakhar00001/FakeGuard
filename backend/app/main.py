from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .model_utils import predict_text, get_model_stats, train_model

app = FastAPI(title="FakeGuard API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewRequest(BaseModel):
    review: str


class PredictionResponse(BaseModel):
    predicted_label: str
    probability_fake: float
    confidence: float
    features: dict
    model_accuracy: float | None = None
    f1_score: float | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "FakeGuard", "version": "2.0.0"}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: ReviewRequest) -> PredictionResponse:
    return predict_text(req.review)


@app.get("/stats")
def stats() -> dict:
    return get_model_stats()


@app.post("/retrain")
def retrain() -> dict:
    result = train_model()
    return {
        "status": "retrained",
        "accuracy": result["accuracy"],
        "f1_score": result["f1_score"],
        "precision": result["precision"],
        "recall": result["recall"],
        "dataset_size": result["dataset_size"],
    }
