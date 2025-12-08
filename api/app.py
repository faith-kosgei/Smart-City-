import os
import time
import joblib
from fastapi import FastAPI, Request
import numpy as np
import sys
import pandas as pd

# Make anomaly services folder visible
sys.path.append("/app/anomaly_services")
from detection import load_model, predict_anomaly



MODEL_FILE = "/app/model/traffic_model.pkl"

# Wait until the model exists
while not os.path.exists(MODEL_FILE):
    print(f"Model file {MODEL_FILE} not found, waiting 5 seconds...")
    time.sleep(5)

# Load the model once it exists
model = joblib.load(MODEL_FILE)
print("Model loaded successfully!")

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Traffic API is running"}

@app.get("/predict")
def predict():
    # Example prediction, replace with real features   
    sample_features = np.zeros((1, model.coef_.shape[0]))
    prediction = model.predict(sample_features)
    return {"prediction": float(prediction[0])}


@app.post("/anomaly")
async def anomaly_detection(request: Request):
    data = await request.json()
    df = pd.DataFrame([data])
    result = predict_anomaly(df)
    return {
        "input": data,
        "anomaly": bool(result["anomaly"].iloc[0])
    }
