from pydantic import BaseModel

class ReviewRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probability_fake: float
    probability_real: float
    explanation: str