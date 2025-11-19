# train.py
import os
import argparse
import pickle
from datetime import timedelta

import pandas as pd
import numpy as np
from pymongo import MongoClient
import gridfs
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, brier_score_loss
from xgboost import XGBClassifier

# -----------------------
# Config (env or defaults)
# -----------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "weather")
FEATURES_COL = os.getenv("FEATURES_COL", "features")
RAW_COL = os.getenv("RAW_COL", "raw_observations")
MODELS_COL = os.getenv("MODELS_COL", "models")
MIN_SAMPLES = 200

# -----------------------
# Helpers
# -----------------------
def connect():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return client, db

def fetch_feature_docs(db, days=None):
    q = {}
    if days:
        from datetime import datetime, timezone
        cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=days)
        q["timestamp"] = {"$gte": cutoff}
    docs = list(db[FEATURES_COL].find(q))
    return docs

def compute_label_for_doc(db, location_id, ts):
    """
    Check raw_observations for the next 60 minutes after ts.
    Return 1 if any rain_1h_mm > 0, else 0. If no future data available, return None.
    """
    # ts is a Python datetime (aware)
    end = ts + timedelta(hours=1)
    cursor = db[RAW_COL].find(
        {"location_id": location_id, "timestamp": {"$gt": ts, "$lte": end}, "payload": {"$exists": True}}
    )
    found = False
    for r in cursor:
        found = True
        # try extracting rainfall from payload.rain.1h or payload.rain
        payload = r.get("payload", {})
        rain = 0.0
        if isinstance(payload, dict):
            # nested keys may vary; try common places
            rain = payload.get("rain", {}) if isinstance(payload.get("rain", {}), (int, float)) else payload.get("rain", {}).get("1h", 0.0)
            # if still dict, try 1h
            if isinstance(rain, dict):
                rain = rain.get("1h", 0.0)
        try:
            if float(rain) > 0.0:
                return 1
        except Exception:
            continue
    if not found:
        return None
    return 0

def docs_to_dataframe(docs):
    # docs expected to have: location_id, timestamp, features(dict), optional label_next_hour
    rows = []
    for d in docs:
        feat = d.get("features", {}) if isinstance(d.get("features", {}), dict) else {}
        row = {"location_id": d.get("location_id"), "timestamp": d.get("timestamp")}
        # flatten feature dict
        for k, v in feat.items():
            # avoid nested structures
            if isinstance(v, (list, dict)):
                continue
            row[k] = v
        # label if present
        if "label_next_hour" in d:
            row["label"] = d.get("label_next_hour")
        elif "label" in d:
            row["label"] = d.get("label")
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

def prepare_dataset(db, days=None, min_samples=MIN_SAMPLES):
    docs = fetch_feature_docs(db, days=days)
    if not docs:
        raise RuntimeError("No feature docs found.")
    df = docs_to_dataframe(docs)
    # convert timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Attempt to fill missing labels by looking into raw_observations
    if "label" not in df.columns:
        df["label"] = pd.NA
    missing_label_idx = df[df["label"].isna()].index.tolist()
    if missing_label_idx:
        for i in missing_label_idx:
            loc = df.at[i, "location_id"]
            ts = df.at[i, "timestamp"]
            lbl = compute_label_for_doc(db, loc, ts)
            df.at[i, "label"] = lbl
    # drop rows where label is still missing
    df = df[df["label"].notna()].copy()
    if df.shape[0] < min_samples:
        raise RuntimeError(f"Not enough labeled samples ({df.shape[0]}). Need at least {min_samples}.")
    # select feature columns (numeric)
    exclude = {"location_id", "timestamp", "label"}
    feat_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not feat_cols:
        raise RuntimeError("No numeric features found.")
    X = df[feat_cols].fillna(0.0)
    y = df["label"].astype(int)
    return X, y, feat_cols

def train_and_save(db, X, y, feat_cols, activate=False):
    # time-based split: latest 20% as validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    model = XGBClassifier(n_estimators=200, max_depth=5, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    # metrics
    prob_val = model.predict_proba(X_val)[:,1]
    pred_val = (prob_val >= 0.5).astype(int)
    auc = roc_auc_score(y_val, prob_val)
    prec = precision_score(y_val, pred_val, zero_division=0)
    rec = recall_score(y_val, pred_val, zero_division=0)
    brier = brier_score_loss(y_val, prob_val)
    metrics = {"roc_auc": float(auc), "precision": float(prec), "recall": float(rec), "brier": float(brier)}
    # save to GridFS
    fs = gridfs.GridFS(db)
    blob = pickle.dumps({"model": model, "feature_columns": feat_cols})
    gridfs_id = fs.put(blob, filename=f"model_{pd.Timestamp.utcnow().strftime('%Y%m%d%H%M%S')}.pkl")
    # insert model metadata
    models_coll = db[MODELS_COL]
    model_doc = {
        "model_version": pd.Timestamp.utcnow().strftime("v%Y%m%d%H%M%S"),
        "artifact_gridfs_id": gridfs_id,
        "created_at": pd.Timestamp.utcnow().to_pydatetime(),
        "metrics": metrics,
        "active": bool(activate)
    }
    # if activating, mark other models inactive
    if activate:
        models_coll.update_many({}, {"$set": {"active": False}})
    models_coll.insert_one(model_doc)
    return model_doc










# ----------------------
# Temporary checks
# ----------------------

import pickle
import pandas as pd
from datetime import timedelta
from pymongo import MongoClient
import gridfs
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, brier_score_loss
from sklearn.model_selection import train_test_split  # only for random utilities if needed

# config defaults (adjust to your file's style)
MODELS_COL = "models"
FEATURES_COL = "features"


from datetime import datetime, timezone


def _fetch_feature_docs(db, days=None):
    """
    Load feature docs from MongoDB. Uses a timezone-aware cutoff that works across pandas versions.
    """
    q = {}
    if days:
        # use a timezone-aware Python datetime (works everywhere)
        now_utc = datetime.now(timezone.utc)          # aware datetime in UTC
        cutoff_dt = now_utc - pd.Timedelta(days=days) # still an aware datetime
        # convert to python datetime (already aware) for Mongo query
        q["timestamp"] = {"$gte": cutoff_dt}
    docs = list(db[FEATURES_COL].find(q))
    return docs




def _docs_to_df_with_label(docs):
    rows = []
    for d in docs:
        feat = d.get("features", {}) if isinstance(d.get("features", {}), dict) else {}
        # skip if no label
        label = d.get("label_next_hour", d.get("label", None))
        if label is None:
            continue
        row = {"location_id": d.get("location_id"), "timestamp": d.get("timestamp"), "label": int(label)}
        for k, v in feat.items():
            # flatten only scalar numeric features
            if isinstance(v, (int, float)):
                row[k] = v
            else:
                # try to coerce simple numeric-like strings
                try:
                    row[k] = float(v)
                except Exception:
                    pass
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # ensure timestamp is datetime tz-aware
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df

def train_and_save_2(db,
                     days: int = 14,
                     min_samples: int = 200,
                     activate: bool = False,
                     train_frac: float = 0.70,
                     val_frac: float = 0.15):
    """
    Time-split training:
      - Loads labeled feature docs from `features` collection (last `days` days if set)
      - Sorts by timestamp and splits into train/val/test by time (train_frac, val_frac, test_frac)
      - Trains XGBoost on train, evaluates on val and test (test = honest time-split)
      - Saves model to GridFS and writes model metadata (metrics use TEST metrics)
    Returns model_doc
    """
    # 1) Load docs
    docs = _fetch_feature_docs(db, days=days)
    df = _docs_to_df_with_label(docs)
    if df.empty:
        raise RuntimeError("No labeled feature docs found.")
    # 2) Require minimum samples
    if len(df) < min_samples:
        raise RuntimeError(f"Not enough labeled rows: {len(df)} < {min_samples}")
    # 3) Decide feature columns (numeric columns excluding meta columns)
    exclude = {"location_id", "timestamp", "label"}
    feat_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not feat_cols:
        raise RuntimeError("No numeric feature columns found in features docs.")
    # 4) Sort by time and split by index (time-split)
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df_sorted)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    train_df = df_sorted.iloc[:i_train]
    val_df = df_sorted.iloc[i_train:i_val]
    test_df = df_sorted.iloc[i_val:]
    # 5) Build X/y
    X_train = train_df[feat_cols].fillna(0.0)
    y_train = train_df["label"].astype(int)
    X_val = val_df[feat_cols].fillna(0.0)
    y_val = val_df["label"].astype(int)
    X_test = test_df[feat_cols].fillna(0.0)
    y_test = test_df["label"].astype(int)
    print("Dataset sizes -> train:", X_train.shape, "val:", X_val.shape, "test:", X_test.shape)
    # 6) Train model
    model = XGBClassifier(n_estimators=200, max_depth=5, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    # 7) Validation metrics (optional reporting)
    prob_val = model.predict_proba(X_val)[:, 1]
    pred_val = (prob_val >= 0.5).astype(int)
    metrics_val = {
        "roc_auc": float(roc_auc_score(y_val, prob_val)),
        "precision": float(precision_score(y_val, pred_val, zero_division=0)),
        "recall": float(recall_score(y_val, pred_val, zero_division=0)),
        "brier": float(brier_score_loss(y_val, prob_val))
    }
    print("Validation metrics:", metrics_val)
    # 8) Honest test metrics (time-split test)
    prob_test = model.predict_proba(X_test)[:, 1]
    pred_test = (prob_test >= 0.5).astype(int)
    metrics_test = {
        "roc_auc": float(roc_auc_score(y_test, prob_test)),
        "precision": float(precision_score(y_test, pred_test, zero_division=0)),
        "recall": float(recall_score(y_test, pred_test, zero_division=0)),
        "brier": float(brier_score_loss(y_test, prob_test))
    }
    print("Test (time-split) metrics:", metrics_test)
    # 9) Save model to GridFS and insert metadata (use test metrics)
    fs = gridfs.GridFS(db)
    blob = pickle.dumps({"model": model, "feature_columns": feat_cols})
    gridfs_id = fs.put(blob, filename=f"model_{pd.Timestamp.utcnow().strftime('%Y%m%d%H%M%S')}.pkl")
    models_coll = db[MODELS_COL]
    model_doc = {
        "model_version": pd.Timestamp.utcnow().strftime("v%Y%m%d%H%M%S"),
        "artifact_gridfs_id": gridfs_id,
        "created_at": pd.Timestamp.utcnow().to_pydatetime(),
        "metrics": metrics_test,   # honest test metrics stored as primary metrics
        "validation_metrics": metrics_val,
        "n_samples": int(n),
        "feature_columns": feat_cols,
        "active": bool(activate)
    }
    if activate:
        models_coll.update_many({}, {"$set": {"active": False}})
    models_coll.insert_one(model_doc)
    return model_doc












# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14, help="How many days of features to load (default 14)")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES, help="Minimum labeled samples to proceed")
    parser.add_argument("--activate", action="store_true", help="Activate this model immediately (mark active=true)")
    args = parser.parse_args()

    client, db = connect()
    try:
        X, y, feat_cols = prepare_dataset(db, days=args.days, min_samples=args.min_samples)
    except Exception as e:
        print("ERROR preparing dataset:", e)
        return
    print("Dataset:", X.shape)
    model_doc = train_and_save(db, X, y, feat_cols, activate=args.activate)
    model_doc_2 = train_and_save_2(db, days=14, min_samples=200, activate=True)
    print("Saved model:", model_doc_2["model_version"], "metrics:", model_doc_2["metrics"])

if __name__ == "__main__":
    main()
