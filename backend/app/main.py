from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .model_utils import predict_text

app = FastAPI(title="FakeGuard API", version="1.0.0")

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


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "FakeGuard"}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: ReviewRequest) -> PredictionResponse:
    return predict_text(req.review)
