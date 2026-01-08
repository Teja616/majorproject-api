from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import PredictRequest, PredictResponse
from app.inference import predict_icd
from app.inference import general_medical_advice
from app.schemas import GeneralAdviceResponse

app = FastAPI(
    title="Medical Coding Assistant v3",
    description="Symptom-based ICD-10 prediction using fine-tuned LLM",
    version="3.0"
)

# CORS (allow frontend calls)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"status": "Medical Coding Assistant v3 is running"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    prediction = predict_icd(request.clinical_text)
    return {"icd_prediction": prediction}

@app.post("/general-advice", response_model=GeneralAdviceResponse)
def general_advice(request: PredictRequest):
    advice = general_medical_advice(request.clinical_text)
    return {"advice": advice}