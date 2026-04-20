from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_predict_high_risk():
    payload = {
        "age": 38, "gender": 1, "department": 0,
        "weekly_hours": 75, "sleep_hours": 4.0,
        "physical_activity": 0, "job_satisfaction": 2,
        "workload_score": 10, "support_from_management": 1,
        "years_experience": 3, "on_call_frequency": 14
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "burnout_risk" in data
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0

def test_predict_low_risk():
    payload = {
        "age": 30, "gender": 0, "department": 2,
        "weekly_hours": 42, "sleep_hours": 7.5,
        "physical_activity": 5, "job_satisfaction": 8,
        "workload_score": 3, "support_from_management": 9,
        "years_experience": 8, "on_call_frequency": 2
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200