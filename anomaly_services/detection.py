import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

# the file path where the model is saved
ANOMALY_MODEL_FILE = "/app/model/traffic_anomaly_model.pkl"

# global model instance
model = None

# training of the anomaly detection model
def train_model(df):
    """ Trains an Isolation Forest model on traffic data.
        df: pandas DatFrame with traffic features
    """

    df = df.copy()

    # only use numeric features for anomaly detection
    features = ["vehicle_count", "avg_speed"]

    # One-hot encode categorical columns if needed
    if "weather" in df.columns:
        df = pd.get_dummies(df, columns=["weather"])
        features += [col for col in df.columns if col.startswith("weather_")]

    if "road_id" in df.columns:
        df = pd.get_dummies(df, columns=["road_id"])
        features += [col for col in df.columns if col.startswith("road_id_")]

    X = df[features]

    # creating Isolation Forest
    model = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42
    )
    model.fit(X)

    # saving the model
    os.makedirs(os.path.dirname(ANOMALY_MODEL_FILE), exist_ok=True)
    joblib.dump(model, ANOMALY_MODEL_FILE)
    print(f"the nomdel is saved to {ANOMALY_MODEL_FILE}")
    return model


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
def predict_anomaly(df):
    """
    Returns anomaly flags for the given traffic data
    df: pandas DataFrame with traffic features
    model: pre-loaded Isolation Forest model
    """

    df = df.copy()  

    # ensure model is loaded
    model = joblib.load(ANOMALY_MODEL_FILE)
    
    
    # Prepare features
    features = ["vehicle_count", "avg_speed"]

    if "weather" in df.columns:
        df = pd.get_dummies(df, columns=["weather"])
        features += [col for col in df.columns if col.startswith("weather_")]

    if "road_id" in df.columns:
        df = pd.get_dummies(df, columns=["road_id"])
        features += [col for col in df.columns if col.startswith("road_id_")]

    # Aligning the columns with training
    X = df[features].reindex(columns=model.feature_names_in, fill_value=0)

    # prediction of anomalies
    df["anomaly"] = model.predict(X) 
     # 1 = normal, -1 = anomaly
    df["anomaly"] = df["anomaly"].map({1: False, -1: True})
    return df
    





