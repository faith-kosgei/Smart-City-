import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import time
import os



DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "traffic")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

MODEL_FILE = "/app/model/traffic_model.pkl"
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
    # use one-hot encode 'weather' and 'road_id' to handle all possible values dynamically
    df = pd.get_dummies(df, columns=["weather", "road_id"])

    # congestion levels to numeric
    df["congestion_level"] = df["congestion_level"].map({"low": 0, "medium": 1, "high": 2}).fillna(1)

    return df

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # training the model 
        model = LinearRegression()
        model.fit(X_train, y_train)


        # evaluation 
        y_prediction = model.predict(X_test)
        mse = mean_squared_error(y_test, y_prediction)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Trained model | MSE: {mse:.2f} | Rows used: {len(df)}")

        # make sure model folder exists
        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        # save the model
        joblib.dump(model, MODEL_FILE)

        # wait before the next training cycle
        time.sleep(SLEEP_INTERVAL)

    except KeyboardInterrupt:
        print("Training loop stopped manually.")
        break
    except Exception as e:
        print("Error:", e)
        time.sleep(SLEEP_INTERVAL)

