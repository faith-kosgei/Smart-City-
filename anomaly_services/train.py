import pandas as pd
import os
from detection import train_model

# path to the root folder
ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "output.jsonl"

print(f"Loading data from {DATA_FILE}")


# load the data from the jsonl file
df = pd.read_json(DATA_FILE, lines=True)

print("the data loaded successfully")
print(df.head())

# train the anomaly model
train_model(df)

print("the anomaly model training is completed")