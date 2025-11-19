# ingest.py
import os
from dotenv import load_dotenv
import requests
import time
import pymongo
from datetime import datetime, timezone
import argparse

load_dotenv()

OWM_KEY = os.environ.get("OWM_API_KEY")
if not OWM_KEY:
    raise RuntimeError("OWM_API_KEY not set in env")

MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set in env")

client = pymongo.MongoClient(MONGO_URI)
db = client.weather
raw = db.raw_observations

LOCATIONS = [
    {"id": f"sg-{i+1:02d}", "lat": lat, "lon": lon}
    for i, (lat, lon) in enumerate([
        (1.290270, 103.851959), (1.352083, 103.819836), (1.280095, 103.850949),
        (1.300000, 103.800000), (1.320000, 103.830000), (1.340000, 103.750000),
        (1.370000, 103.800000), (1.380000, 103.890000), (1.360000, 103.950000),
        (1.310000, 103.940000), (1.280000, 103.870000), (1.330000, 103.880000),
        (1.360000, 103.870000), (1.300000, 103.820000), (1.320000, 103.840000),
        (1.350000, 103.860000), (1.370000, 103.830000), (1.390000, 103.780000),
        (1.400000, 103.900000), (1.410000, 103.810000), (1.420000, 103.830000),
    ])
]

def fetch_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OWM_KEY}&units=metric"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()

def run_once():
    start = time.time()
    success = 0
    failed = 0
    for loc in LOCATIONS:
        try:
            data = fetch_weather(loc["lat"], loc["lon"])
            doc = {
                "location_id": loc["id"],
                "lat": loc["lat"],
                "lon": loc["lon"],
                "timestamp": datetime.now(timezone.utc),
                "payload": data,
            }
            raw.insert_one(doc)
            success += 1
            print(f"[{loc['id']}] OK at {doc['timestamp']}")
        except Exception as e:
            failed += 1
            print(f"[{loc['id']}] ERROR: {e}")
    elapsed = time.time() - start
    print(f"run_once complete: success={success}, failed={failed}, elapsed={elapsed:.2f}s")
    # return summary for callers
    return {"success": success, "failed": failed, "elapsed": elapsed}

def main_loop():
    while True:
        run_once()
        # keep 60s spacing between batches
        time.sleep(60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one iteration and exit")
    args = parser.parse_args()
    if args.once:
        run_once()
    else:
        main_loop()
