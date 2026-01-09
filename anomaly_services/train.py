import pandas as pd
import os
import joblib
import psycopg2
from sklearn.ensemble import IsolationForest

MODEL_PATH = "/app/model/traffic_anomaly_model.pkl"

# DB_HOST = os.getenv("DB_HOST", "postgres")
# DB_PORT = int(os.getenv("DB_PORT", "5432"))
# DB_NAME = os.getenv("DB_NAME", "traffic")
# DB_USER = os.getenv("DB_USER", "postgres")
# DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

# # create SQLAlchemy engine
# DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# engine = create_engine(DB_URL)

def load_data():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "postgres"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "traffic"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
    )

    df = pd.read_sql(
        "SELECT vehicle_count, avg_speed FROM traffic_data",
        conn,
    )
    conn.close()
    return df


def train_anomaly_model():
    df = load_data()


    model = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42

    )
    model.fit(df)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("anomaly model trained and saved")
    

if __name__ == "__main__":
    train_anomaly_model()

