# mongo_to_df.py
from pymongo import MongoClient
import pandas as pd
from datetime import datetime, timezone, timedelta

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "weather"
COL = "raw_observations"

def get_clean_df(days=7, limit=None, mongo_uri=MONGO_URI, db_name=DB_NAME, col=COL):
    """
    Load recent raw_observations from Mongo, flatten payload, normalize columns,
    cast types, handle missing values, and return a cleaned DataFrame.
    - days: how many days back to load (int)
    - limit: optional max rows to load
    """
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
    db = client[db_name]
    cur = db[col].find(
        {"timestamp": {"$gte": datetime.now(timezone.utc) - timedelta(days=days)}}
    ).sort("timestamp", -1)
    if limit:
        cur = cur.limit(limit)

    docs = list(cur)
    if not docs:
        return pd.DataFrame()  # empty

    # Flatten nested JSON (payload.* becomes columns)
    df = pd.json_normalize(docs)
    # Keep / rename fields we care about (add more as needed)
    mapping = {
        "location_id": "location_id",
        "timestamp": "timestamp",
        "payload.main.temp": "temp_c",
        "payload.main.pressure": "pressure_hpa",
        "payload.main.humidity": "humidity_pct",
        "payload.wind.speed": "wind_m_s",
        "payload.wind.deg": "wind_deg",
        "payload.clouds.all": "cloud_pct",
        # rain can be missing; handle keys carefully
        "payload.rain.1h": "rain_1h_mm",
        "payload.snow.1h": "snow_1h_mm",
        "payload.weather": "weather"  # array - keep raw for now
    }
    # Some docs may have slightly different nested keys; use get-style fallback
    for src, dst in mapping.items():
        if src in df.columns:
            df[dst] = df[src]
        else:
            # attempt to extract using json path where possible
            df[dst] = df.get(src, pd.NA)

    # Convert timestamp to pandas datetime and sort ascending
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

    # Numeric casting & sensible missing-value handling
    num_cols = ["temp_c", "pressure_hpa", "humidity_pct", "wind_m_s",
                "wind_deg", "cloud_pct", "rain_1h_mm", "snow_1h_mm"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Replace NaNs for precip with 0 (common in OWM when not raining)
    if "rain_1h_mm" in df.columns:
        df["rain_1h_mm"] = df["rain_1h_mm"].fillna(0.0)
    if "snow_1h_mm" in df.columns:
        df["snow_1h_mm"] = df["snow_1h_mm"].fillna(0.0)

    # Drop duplicates (same location & timestamp)
    df = df.drop_duplicates(subset=["location_id", "timestamp"], keep="last")

    # Optional: basic sanity filters (temperature bounds etc.)
    df = df[(df["temp_c"].isna()) | ((df["temp_c"] > -80) & (df["temp_c"] < 60))]

    # Keep only useful columns
    keep = ["location_id", "timestamp"] + [c for c in num_cols if c in df.columns] + ["weather"]
    keep = [c for c in keep if c in df.columns]
    clean = df[keep].reset_index(drop=True)
    return clean





# Example usage:
#from mongo_to_df import get_clean_df
df = get_clean_df(days=7, limit=None)
df.head()
