# feature_worker.py (concise)
from pymongo import MongoClient, UpdateOne
import pandas as pd
from datetime import timedelta, timezone, datetime
import os

MONGO_URI = os.environ["MONGO_URI"]
DB = "weather"
RAW_COL = "raw_observations"
FEATURES_COL = "features"

client = MongoClient(MONGO_URI)
db = client[DB]
raw = db[RAW_COL]
features_col = db[FEATURES_COL]

# adjust windows (minutes)
WINDOWS = [10, 30, 60]

def compute_features_from_df(df):
    """
    df: cleaned DataFrame with columns:
      ['location_id','timestamp','temp_c','pressure_hpa','humidity_pct','wind_m_s','rain_1h_mm']
    returns: DataFrame with feature columns + location_id + timestamp
    """
    def fe_group(g):
        g = g.sort_values("timestamp").reset_index(drop=True)
        # lags
        g["temp_lag_1"] = g["temp_c"].shift(1)
        g["pressure_lag_1"] = g["pressure_hpa"].shift(1)
        g["humidity_lag_1"] = g["humidity_pct"].shift(1)

        # rolling means + std + pressure drop
        for w in WINDOWS:
            n = int(w)  # expects data at 1-min freq; if not, windows are count-based
            g[f"temp_roll_{w}"] = g["temp_c"].rolling(window=n, min_periods=1).mean()
            g[f"humidity_roll_{w}"] = g["humidity_pct"].rolling(window=n, min_periods=1).mean()
            g[f"wind_roll_{w}"] = g["wind_m_s"].rolling(window=n, min_periods=1).mean()
            # pressure change over window (current - mean of previous window)
            g[f"pressure_drop_{w}"] = g["pressure_hpa"] - g["pressure_hpa"].rolling(window=n, min_periods=1).mean().shift(0)

        # deltas over short window
        g["temp_delta_3"] = g["temp_c"] - g["temp_c"].shift(3)
        g["pressure_delta_3"] = g["pressure_hpa"] - g["pressure_hpa"].shift(3)

        # time features
        g["hour"] = g["timestamp"].dt.hour
        g["minute"] = g["timestamp"].dt.minute
        g["dayofweek"] = g["timestamp"].dt.dayofweek

        # keep only latest row per timestamp (features timestamp = current row timestamp)
        return g

    out = df.groupby("location_id", group_keys=False).apply(fe_group).reset_index(drop=True)
    return out

def load_recent_clean(days=7, limit=None):
    """
    Load cleaned data using your existing get_clean_df helper.
    """
    # assume get_clean_df is defined/imported
    from mongo_to_df import get_clean_df
    return get_clean_df(days=days, limit=limit)

def upsert_features(df_feat):
    """
    Upsert features into Mongo features collection.
    Uses (location_id, timestamp) as unique key.
    """
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
    df = load_recent_clean(days=days, limit=limit)
    if df.empty:
        print("No data")
        return df
    # ensure timestamp is datetime64 with tz
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    feat_df = compute_features_from_df(df)
    # optional: drop rows with NaNs in essential features
    feat_df = feat_df.dropna(subset=["temp_c","pressure_hpa","humidity_pct"], how="any")
    # write to mongo
    upsert_features(feat_df)
    print("Wrote", len(feat_df), "feature rows")
    return feat_df

if __name__ == "__main__":
    df_out = run_feature_worker(days=7, limit=None)
    print(df_out.head())
