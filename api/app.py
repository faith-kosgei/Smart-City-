import os
import time
import joblib
from fastapi import FastAPI, Request
import numpy as np
import sys
import pandas as pd
# from anomaly_services.detection import predict_anomaly

PREDICTION_MODEL_FILE = "/app/model/traffic_model.pkl"
ANOMALY_MODEL_FILE = "/app/model/traffic_anomaly_model.pkl"

# Wait until the model exists
while not os.path.exists(PREDICTION_MODEL_FILE):
    print(f"Prediction Model file {PREDICTION_MODEL_FILE} not found, waiting 5 seconds...")
    time.sleep(5)
# Wait until the model exists
while not os.path.exists(ANOMALY_MODEL_FILE):
    print(f"Anomaly Model file {ANOMALY_MODEL_FILE} not found, waiting 5 seconds...")
    time.sleep(5)

# Load the model once it exists
prediction_model = joblib.load(PREDICTION_MODEL_FILE)
print("Model loaded successfully!")

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Traffic API is running"}

@app.get("/predict")
def predict_traffic(vehicle_count: int, avg_speed: float):   
   return predict_traffic(vehicle_count, avg_speed)

@app.post("/anomaly")
def anomaly(vehicle_count: int, avg_speed: float):
    return predict_anomaly(vehicle_count, avg_speed)
   
