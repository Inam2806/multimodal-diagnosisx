from pydantic import BaseModel, Field


class XrayPrediction(BaseModel):
    label: str
    probability: float = Field(ge=0, le=1)
    positive: bool


class XrayDiagnosisResponse(BaseModel):
    model: str
    predictions: list[XrayPrediction]
    disclaimer: str = "Research use only; not a medical diagnosis."
