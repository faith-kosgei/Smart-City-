from fastapi import FastAPI
from pydantic import BaseModel
from detection import predict_anomaly
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Anomaly Service")
Instrumentator().instrument(app).expose(app)

class AnomalyRequest(BaseModel):
    vehicle_count: int
    avg_speed: float

@app.post("/anomaly")
def detect_anomaly(request: AnomalyRequest):
    result = predict_anomaly(request.vehicle_count, request.avg_speed)
    return {"anomaly": result}
