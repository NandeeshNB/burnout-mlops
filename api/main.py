from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle, numpy as np, logging, os
from datetime import datetime

logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

app = FastAPI(
    title="Burnout Risk Prediction API",
    description="Predicts stress/burnout risk in healthcare workers",
    version="1.0.0"
)

# Load model and scaler
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

class HealthcareWorker(BaseModel):
    age: int
    gender: int                  # encoded: 0=Female,1=Male,2=Other
    department: int              # encoded: 0=ER,1=General,...
    weekly_hours: int
    sleep_hours: float
    physical_activity: int
    job_satisfaction: int
    workload_score: int
    support_from_management: int
    years_experience: int
    on_call_frequency: int

class PredictionResponse(BaseModel):
    burnout_risk: int
    risk_label: str
    confidence: float
    timestamp: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "BurnoutRisk", "version": "1.0.0"}

@app.post("/predict", response_model=PredictionResponse)
def predict(worker: HealthcareWorker):
    try:
        features = np.array([[
            worker.age, worker.gender, worker.department,
            worker.weekly_hours, worker.sleep_hours,
            worker.physical_activity, worker.job_satisfaction,
            worker.workload_score, worker.support_from_management,
            worker.years_experience, worker.on_call_frequency
        ]])
        features_scaled = scaler.transform(features)

        prediction  = model.predict(features_scaled)[0]
        confidence  = float(model.predict_proba(features_scaled)[0][prediction])
        risk_label  = "HIGH RISK" if prediction == 1 else "LOW RISK"
        timestamp   = datetime.utcnow().isoformat()

        logging.info(f"Prediction: {risk_label}, Confidence: {confidence:.2f}, "
                     f"Hours: {worker.weekly_hours}, Satisfaction: {worker.job_satisfaction}")

        return PredictionResponse(
            burnout_risk=int(prediction),
            risk_label=risk_label,
            confidence=round(confidence, 3),
            timestamp=timestamp
        )
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    import json
    try:
        with open("reports/metrics.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "Metrics not yet generated"}