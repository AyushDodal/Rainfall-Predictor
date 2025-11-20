# feature_worker.py (fixed)
import os
from pymongo import MongoClient, UpdateOne
import pandas as pd
from datetime import timedelta
import traceback
import time

# Read MONGO_URI from env
MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set in env")

DB = os.environ.get("DB_NAME", "weather")
RAW_COL = "raw_observations"
FEATURES_COL = "features"

client = MongoClient(MONGO_URI)
db = client[DB]
raw = db[RAW_COL]
features_col = db[FEATURES_COL]

# adjust windows (minutes)
WINDOWS = [10, 30, 60]

def compute_features_from_df(df):
    def fe_group(g):
        g = g.sort_values("timestamp").reset_index(drop=True)
        # basic lags
        g["temp_lag_1"] = g["temp_c"].shift(1)
        g["pressure_lag_1"] = g["pressure_hpa"].shift(1)
        g["humidity_lag_1"] = g["humidity_pct"].shift(1)

        # rolling means + std + pressure drop
        for w in WINDOWS:
            n = int(w)
            g[f"temp_roll_{w}"] = g["temp_c"].rolling(window=n, min_periods=1).mean()
            g[f"humidity_roll_{w}"] = g["humidity_pct"].rolling(window=n, min_periods=1).mean()
            g[f"wind_roll_{w}"] = g["wind_m_s"].rolling(window=n, min_periods=1).mean()
            g[f"pressure_drop_{w}"] = g["pressure_hpa"] - g["pressure_hpa"].rolling(window=n, min_periods=1).mean().shift(0)

        g["temp_delta_3"] = g["temp_c"] - g["temp_c"].shift(3)
        g["pressure_delta_3"] = g["pressure_hpa"] - g["pressure_hpa"].shift(3)

        g["hour"] = g["timestamp"].dt.hour
        g["minute"] = g["timestamp"].dt.minute
        g["dayofweek"] = g["timestamp"].dt.dayofweek

        return g

    out = df.groupby("location_id", group_keys=False).apply(fe_group).reset_index(drop=True)
    return out

def load_recent_clean(days=7, limit=None):
    """
    Build a minimal clean DataFrame directly from the `raw` collection (uses MONGO_URI-bound client).
    Avoids importing other modules that may default to localhost.
    """
    q = {}
    if days:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        q["timestamp"] = {"$gte": cutoff.to_pydatetime()}
    docs = list(raw.find(q).sort("timestamp", 1).limit(limit or 100000))
    if not docs:
        return pd.DataFrame()
    rows = []
    for d in docs:
        p = d.get("payload", {}) or {}
        main = p.get("main", {}) or {}
        wind = p.get("wind", {}) or {}
        rain = 0.0
        if isinstance(p.get("rain"), dict):
            rain = p["rain"].get("1h", 0.0) or 0.0
        rows.append({
            "location_id": d.get("location_id"),
            "timestamp": d.get("timestamp"),
            "temp_c": main.get("temp"),
            "pressure_hpa": main.get("pressure"),
            "humidity_pct": main.get("humidity"),
            "wind_m_s": wind.get("speed"),
            "rain_1h_mm": rain
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df

def upsert_features(df_feat):
    ops = []
    for _, r in df_feat.iterrows():
        doc = {
            "location_id": r["location_id"],
            "timestamp": r["timestamp"].to_pydatetime() if hasattr(r["timestamp"], "to_pydatetime") else r["timestamp"],
            "features": r.drop(labels=["location_id","timestamp"]).to_dict()
        }
        ops.append(UpdateOne(
            {"location_id": doc["location_id"], "timestamp": doc["timestamp"]},
            {"$set": doc},
            upsert=True
        ))
    if ops:
        features_col.bulk_write(ops)

def run_feature_worker(days=7, limit=None):
    try:
        df = load_recent_clean(days=days, limit=limit)
        if df.empty:
            print("No data")
            return df
        # ensure timestamp is datetime64 with tz
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        feat_df = compute_features_from_df(df)
        feat_df = feat_df.dropna(subset=["temp_c","pressure_hpa","humidity_pct"], how="any")
        upsert_features(feat_df)
        print("Wrote", len(feat_df), "feature rows")
        return feat_df
    except Exception as e:
        print("Feature worker failed:", e)
        traceback.print_exc()
        return pd.DataFrame()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    if args.once:
        run_feature_worker(days=args.days, limit=args.limit)
    else:
        while True:
            run_feature_worker(days=args.days, limit=args.limit)
            time.sleep(300)
