import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sklearn.metrics import mean_squared_error
import joblib
import time
import os
from xgboost import XGBRegressor


DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "traffic")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

PREDICTION_MODEL_FILE = "/app/model/traffic_model.pkl"


os.makedirs("/app/model", exist_ok=True)
# this shows that it retrains every 30s
SLEEP_INTERVAL = int(os.getenv("SLEEP_INTERVAL", 30)) 


# create SQLAlchemy engine
DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DB_URL)


# fetching data from db
def fetch_data():
    try:
        df = pd.read_sql("SELECT * FROM traffic_data", engine)
        return df
    except Exception as e:
        print(f"Error fetching data:", e)
        return pd.DataFrame()  # This return empty dataframe if error occurs or it fails to fetch the data
   
# data preprocessing for the ML
def preprocess(df):
    df = df.copy()     
    # ensure the timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # sort by time
    df = df.sort_values("timestamp")

    # congestion levels to numeric
    df["congestion_level"] = df["congestion_level"].map({"low": 0, "medium": 1, "high": 2}).fillna(1)

    # use one-hot encode 'weather' and 'road_id' to handle all possible values dynamically
    df = pd.get_dummies(df, columns=["weather", "road_id"])

    return df

 

# feature engineeering
def add_lag_features(df, lags=[1, 2, 3]):
    df = df.copy()

    for lag in lags:
        df[f"vehicle_count_lag_{lag}"] = df["vehicle_count"].shift(lag)
        df[f"avg_speed_lag_{lag}"] = df["avg_speed"].shift(lag)

    return df.dropna()

# training loop
while True:
    try:
        df = fetch_data()

        if len(df) < 10:
            print("Not enough data yet, Waiting ...")
            time.sleep(SLEEP_INTERVAL)
            continue

        df = preprocess(df)

        # feature and target
        X = df.drop(columns=["vehicle_count", "timestamp"])
        y = df["vehicle_count"]

        # train / test split
        X_train, X_test, y_train, y_test = train_series_split(X, y)

        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,

        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Trained model | MSE: {mse:.2f} | Rows used: {len(df)}")

        # make sure model folder exists
        os.makedirs(os.path.dirname(PREDICTION_MODEL_FILE), exist_ok=True)
        # save the model
        joblib.dump(prediction_model, PREDICTION_MODEL_FILE)

        print("prediction model trained successfully and saved")

       
    except KeyboardInterrupt:
        print("Training loop stopped manually.")
        break
    except Exception as e:
        print("Error:", e)
        time.sleep(SLEEP_INTERVAL)

