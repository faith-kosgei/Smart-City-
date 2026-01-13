import os
import time
import joblib
from fastapi import FastAPI, Request
import numpy as np
import sys
import pandas as pd
from sqlalchemy import create_engine
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator


ANOMALY_SERVICE_URL = "http://traffic-anomaly:8001/anomaly"

PREDICTION_MODEL_FILE = "/app/model/traffic_model.pkl"
FEATURES_FILE = "/app/model/traffic_features.pkl"
ANOMALY_MODEL_FILE = "/app/model/traffic_anomaly_model.pkl"

DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "traffic")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

# create SQLAlchemy engine
DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DB_URL)

# Wait until the model exists
while not os.path.exists(PREDICTION_MODEL_FILE):
    print(f"Prediction Model file {PREDICTION_MODEL_FILE} not found, waiting 5 seconds...")
    time.sleep(5)
# Wait until the model exists
while not os.path.exists(ANOMALY_MODEL_FILE):
    print(f"Anomaly Model file {ANOMALY_MODEL_FILE} not found, waiting 5 seconds...")
    time.sleep(5)
# Wait until the file exists
while not os.path.exists(FEATURES_FILE):
    print(f"Features file {FEATURES_FILE} not found, waiting 5 seconds...")
    time.sleep(5)

# Load the model once it exists
prediction_model = joblib.load(PREDICTION_MODEL_FILE)
features = joblib.load(FEATURES_FILE)
print("Model loaded successfully!")


app = FastAPI(title="Traffic api")
Instrumentator().instrument(app).expose(app)


class AnomalyRequest(BaseModel):
    vehicle_count: int 
    avg_speed: float

# building the lag features inside the API
def build_features_from_db():
    query = """
    SELECT *
    FROM traffic_data
    ORDER BY timestamp DESC
    LIMIT 4
    """

    df = pd.read_sql(query, engine)

    if len(df) < 4:
        raise ValueError("Not enough data for prediction")

    df = df.sort_values("timestamp")

    latest = {}

    for lag in [1, 2, 3]:
        latest[f"vehicle_count_lag_{lag}"] = df.iloc[-lag]["vehicle_count"]
        latest[f"avg_speed_lag_{lag}"] = df.iloc[-lag]["avg_speed"]

    # encode congestion
    congestion_map = {"low": 0, "medium": 1, "high": 2}
    latest["congestion_level"] = congestion_map.get(
         df.iloc[-1]["congestion_level"], 1
)

    # one-hot weather & road_id
    weather = df.iloc[-1]["weather"]
    road_id = df.iloc[-1]["road_id"]

    latest[f"weather_{weather}"] = 1
    latest[f"road_id_{road_id}"] = 1

    return pd.DataFrame([latest])

def align_features(X):
    for col in features:
        if col not in X.columns:
            X[col] = 0
    return X[features]


@app.get("/")
def root():
    return {"message": "Traffic API is running"}


@app.get("/predict")
def predict_vehicle_count():
    try:
        X = build_features_from_db()
        X = align_features(X)


        prediction = prediction_model.predict(X)[0]

        return {
            "predicted_vehicle_count": float(prediction)
        }

    except Exception as e:
        return {"error": str(e)}



@app.post("/anomaly")
def anomaly(request: AnomalyRequest):
    try:
        response = request.post(
            ANOMALY_SERVICE_URL,
            json = request.dict(),
            timeout=5
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        return{"error": str(e)}
    
   
