import time
import json
import random
import argparse
from datetime import datetime, timedelta
import numpy as pd
import pandas as pd

class TrafficSimulator:
    def __init__(self, roads=10, interval=5, out_path="output.jsonl"):
        self.roads = roads
        self.interval = interval
        self.out_path = out_path

        # Each road will be given a base traffic pattern
        self.road_profiles = {
            f"road_{i}": {
                "base_volume": random.randint(20, 60),
                "peak_multiplier": random.uniform(1.5, 3.5),
                "weather_sensitivity": random.uniform(0.05, 0.3),
            }
            for i in range(1, roads + 1)

        }
    def _time_of_day_factor(self, timestamp):
        hour = timestamp.hour

        # morning peak hours
        if 7 <= hour <= 9:
            return 2.5
        # evening peak hours
        elif 16 <= hour <= 18:
            return 3.0
        # night hours with low traffic
        elif 0 <= hour <= 5:
            return 0.4
        else:
            return 1.0

    def _simulate_weather(self):
        weather_types = ["clear", "rain", "stormy", "fog"]
        weights = [0.7, 0.2, 0.05, 0.05]
        return random.choices(weather_types, weights)[0]

    def simulate_record(self):
        timestamp = datetime.utcnow()
        weather = self._simulate_weather()

        data = []

        for road_id, profile in self.road_prefiles.items():
            base = profile["base_volume"]
            peak = self._time_of_day_factor(timestamp)
            weather_effect = (
                1.0 - profile["weather_sensitivity"]
                if weather in ["rain", "stormy", "fog"]
                else 1.0
            )
            vehicle_count = np.random.poisson(base * peak * weather_effect)

            avg_speed = max(
                10,
                60 - (vehicle_count / profile["base_volume"]) * 20
                
            )
            # incase of a random accident(which is rare)
            accident_flag = 1 if random.random() < 0.01 else 0

            # in a case where there is congestion (target variable)
            if vehicle_count > (base * peak * 1.2):
                congestion_level = "high"
            elif vehicle_count > (base * peak * 0.8):
                congestion_level = "medium"
            else:
                congestion_level = "low"

            data.append({
                "timestamp": timestamp.isoformat(),
                "road_id": road_id,
                "vehicle_count": int(vehicle_count),
                "avg_speed": round(avg_speed, 2),
                "weather": weather,
                "accident_flag": accident_flag,
                "congestion_level": congestion_level,
            })
        return data
    
    def run(self):
        print(f"Starting the data simulation process every {self.interval}s....\n")
        while True:
            records = self.simulate_record()
            with open(self,out_path, "a") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            
            print(f"[{datetime.utcnow()}]Generated {len(records)} new records")
            time.sleep(self.interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roads", type=int, default=10)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--out", type=str, default="output.jsonl")
    args = parser.parse_args()

    sim = TrafficSimulator(
        roads=args.roads,
        interval=args.interval,
        out_path=args.out,
    )
    sim.run()