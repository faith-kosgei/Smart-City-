from fastapi import FastAPI
from pydantic import BaseModel
from detection import predict_anomaly  # your existing function

app = FastAPI(title="Anomaly Service")

class AnomalyRequest(BaseModel):
    vehicle_count: int
    avg_speed: float

@app.post("/anomaly")
def detect_anomaly(request: AnomalyRequest):
    result = predict_anomaly(request.vehicle_count, request.avg_speed)
    return {"anomaly": result}
