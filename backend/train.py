#!/usr/bin/env python3
"""
train.py

Train an XGBoost classifier on labeled feature documents stored in MongoDB (features collection).
Performs a time-based train/val/test split, reports metrics, saves model to GridFS and inserts a model metadata doc.

Usage (example):
    python backend/train.py --days 14 --min-samples 500 --activate

"""

import os
import argparse
import pickle
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from pymongo import MongoClient
import gridfs

from sklearn.metrics import roc_auc_score, precision_score, recall_score, brier_score_loss
from xgboost import XGBClassifier

# -----------------------
# Config (env or defaults)
# -----------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "weather")
FEATURES_COL = os.getenv("FEATURES_COL", "features")
MODELS_COL = os.getenv("MODELS_COL", "models")

DEFAULT_MIN_SAMPLES = 200

# -----------------------
# Helpers
# -----------------------
def connect():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return client, db

def load_feature_docs(db, days: int | None = None):
    """
    Load feature documents from the features collection.
    If days is provided, only load docs with timestamp >= now- days.
    """
    q = {}
    if days is not None:
        now_utc = datetime.now(timezone.utc)
        cutoff = now_utc - pd.Timedelta(days=days)
        q["timestamp"] = {"$gte": cutoff}
    docs = list(db[FEATURES_COL].find(q))
    return docs

def docs_to_dataframe(docs):
    """
    Convert feature documents to a flat DataFrame.
    Expects docs with keys: location_id, timestamp, features (dict), and label_next_hour / label.
    """
    rows = []
    for d in docs:
        feat = d.get("features") or {}
        # prefer explicit label fields
        label = d.get("label_next_hour", d.get("label", None))
        row = {
            "location_id": d.get("location_id"),
            "timestamp": d.get("timestamp"),
            "label": label
        }
        # flatten scalar feature entries
        for k, v in feat.items():
            if isinstance(v, (int, float, bool, str)):
                row[k] = v
            else:
                # skip lists / dicts
                continue
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

def select_numeric_features(df: pd.DataFrame):
    exclude = {"location_id", "timestamp", "label"}
    feat_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return feat_cols

def compute_metrics(y_true, proba):
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, proba))
    }

def train_time_split_and_save(db, days:int, min_samples:int, activate:bool,
                              train_frac:float=0.7, val_frac:float=0.15, xgb_params:dict=None):
    """
    Time-split training pipeline:
      - Load labeled feature docs (optionally last `days`)
      - Build DataFrame, choose numeric features
      - Sort by timestamp and split by time into train/val/test using fractions
      - Train XGBoost on train, evaluate on val and test
      - Save model artifact (pickle) to GridFS and insert metadata into models collection
    Returns the inserted model_doc (dict)
    """
    docs = load_feature_docs(db, days=days)
    if not docs:
        raise RuntimeError("No feature documents found in the features collection.")

    df = docs_to_dataframe(docs)
    if df.empty:
        raise RuntimeError("Converted DataFrame is empty.")

    # ensure timestamp is datetime tz-aware
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # drop rows with missing labels
    df = df[df["label"].notna()].copy()
    if df.empty:
        raise RuntimeError("No labeled rows found (label or label_next_hour required in features docs).")

    n_samples = len(df)
    if n_samples < min_samples:
        raise RuntimeError(f"Not enough labeled samples: {n_samples} < {min_samples}")

    feat_cols = select_numeric_features(df)
    if not feat_cols:
        raise RuntimeError("No numeric feature columns found.")

    # time-based split
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df_sorted)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    train_df = df_sorted.iloc[:i_train]
    val_df = df_sorted.iloc[i_train:i_val]
    test_df = df_sorted.iloc[i_val:]

    X_train = train_df[feat_cols].fillna(0.0)
    y_train = train_df["label"].astype(int)
    X_val = val_df[feat_cols].fillna(0.0)
    y_val = val_df["label"].astype(int)
    X_test = test_df[feat_cols].fillna(0.0)
    y_test = test_df["label"].astype(int)

    print(f"Dataset sizes: total={n}, train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"Using features ({len(feat_cols)}): {feat_cols[:10]}{'...' if len(feat_cols)>10 else ''}")

    # train
    params = xgb_params or {"n_estimators": 200, "max_depth": 5, "use_label_encoder": False, "eval_metric": "logloss"}
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    # validation metrics
    proba_val = model.predict_proba(X_val)[:,1] if len(X_val)>0 else np.array([])
    metrics_val = compute_metrics(y_val, proba_val) if len(proba_val)>0 else {}

    # test metrics (honest)
    proba_test = model.predict_proba(X_test)[:,1] if len(X_test)>0 else np.array([])
    metrics_test = compute_metrics(y_test, proba_test) if len(proba_test)>0 else {}

    print("Validation metrics:", metrics_val)
    print("Test metrics:", metrics_test)

    # save model artifact to GridFS
    fs = gridfs.GridFS(db)
    payload = {"model": model, "feature_columns": feat_cols}
    blob = pickle.dumps(payload)
    gridfs_id = fs.put(blob, filename=f"model_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}.pkl")

    models_coll = db[MODELS_COL]
    model_doc = {
        "model_version": pd.Timestamp.now(tz='UTC').strftime("v%Y%m%d%H%M%S"),
        "artifact_gridfs_id": gridfs_id,
        "created_at": pd.Timestamp.now(tz='UTC').to_pydatetime(),
        "metrics": metrics_test,
        "validation_metrics": metrics_val,
        "n_samples": int(n),
        "feature_columns": feat_cols,
        "active": bool(activate)
    }

    if activate:
        models_coll.update_many({}, {"$set": {"active": False}})
    res = models_coll.insert_one(model_doc)
    model_doc["_id"] = res.inserted_id
    print("Saved model:", model_doc["model_version"], "gridfs_id:", gridfs_id)
    return model_doc

# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14, help="How many days of features to load (default 14). Use 0 or omit to load all.")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES, help="Minimum labeled samples to proceed")
    parser.add_argument("--activate", action="store_true", help="Activate this model immediately (mark active=true)")
    parser.add_argument("--train-frac", type=float, default=0.70, help="Fraction of data used for training (time-split)")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Fraction of data used for validation (time-split)")
    args = parser.parse_args()

    client, db = connect()
    days = args.days if args.days and args.days > 0 else None
    try:
        model_doc = train_time_split_and_save(db, days=days, min_samples=args.min_samples,
                                              activate=args.activate, train_frac=args.train_frac, val_frac=args.val_frac)
        print("Model training complete. Version:", model_doc["model_version"])
        print("Metrics:", model_doc["metrics"])
    except Exception as e:
        print("ERROR during training:", e)

if __name__ == "__main__":
    main()
