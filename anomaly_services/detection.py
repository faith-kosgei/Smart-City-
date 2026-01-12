import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

# the file path where the model is saved
ANOMALY_MODEL_FILE = "/app/model/traffic_anomaly_model.pkl"

# global model instance
model = None

def load_model():
    """ Loads model once and caches it """

    global model

    if model is None:
        if not os.path.exists(ANOMALY_MODEL_FILE):
            raise FileNotFoundError("The Anomally detection model not Found!. train it first")

        model = joblib.load(ANOMALY_MODEL_FILE)
        print("Anomaly model loaded sucessfully")

    return model


# prediction of anomalies
def predict_anomaly(vehicle_count: int, avg_speed: float):
    model = load_model()

    X = pd.DataFrame([[vehicle_count, avg_speed]])
    pred = model.predict(X)[0]
    score = model.decision_function(X)[0]

    return{
        "is_anomaly": bool(pred == -1),
        "score": float(score)
    }
 





