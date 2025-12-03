from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

MODEL_FILE = "/app/traffic_model.pkl"

# loading the model
model - joblib.load(MODEL_FILE)

app = FastAPI(title="Traffic prediction API")

# defining the input schema
class TrafficInput(BaseModel):
    road_id: str
    weather: str
    congestion_level: str
    avg_speed: float
    accident_flag: int


def preprocess_input(data: TrafficInput):
    df = pd.get_dummies(df, columns=["road_id", "weather"])
    
