from pydantic import BaseModel

class PredictRequest(BaseModel):
    clinical_text: str

class PredictResponse(BaseModel):
    icd_prediction: list[str]

class GeneralAdviceResponse(BaseModel):
    advice: str

