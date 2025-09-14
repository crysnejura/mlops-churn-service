from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference import predict_label_one

app = FastAPI(title="Churn Prediction API")


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: CustomerData):
    try:
        payload = data.dict()
        result = predict_label_one(payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
