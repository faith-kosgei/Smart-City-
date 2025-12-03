import time
import json
import psycopg2
import os
from datetime import datetime, timezone

import pandas as pd 

DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "traffic")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

DATA_FILE = os.getenv("DATA_FILE", "output.jsonl")
SLEEP_INTERVAL = int(os.getenv("SLEEP_INTERVAL", 5))


# coonect to postgreSQL
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS traffic_data(
    timestamp TIMESTAMPTZ,
    road_id TEXT,
    vehicle_count INT,
    avg_speed FLOAT,
    weather TEXT,
    accident_flag INT,
    congestion_level TEXT
);
""")
conn.commit()

print("starting data ingestion .....")

lines_read = 0

while True:
    try:
        with open(DATA_FILE, "r") as f:
            lines = f.readlines()
            new_lines = lines[lines_read:]
            for line in new_lines:
                record = json.loads(line)
                cursor.execute("""
                    INSERT INTO traffic_data (timestamp, road_id, vehicle_count, avg_speed, weather, accident_flag, congestion_level)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    record["timestamp"],
                    record["road_id"],
                    record["vehicle_count"],
                    record["avg_speed"],
                    record["weather"],
                    record["accident_flag"],
                    record["congestion_level"]
                ))
            conn.commit()
            lines_read += len(new_lines)
        time.sleep(SLEEP_INTERVAL)
    except KeyboardInterrupt:
        print("Stopping ingestion...")
        break
    except Exception as e:
        print("Error:", e)
        time.sleep(SLEEP_INTERVAL)