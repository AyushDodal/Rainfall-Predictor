#!/usr/bin/env python3
"""
train.py

- Loads labeled feature docs from MongoDB (features collection).
- Performs a time-based train/val/test split (honest time split).
- Trains an XGBoost classifier, computes metrics, saves model artifact to GridFS,
  and inserts a model metadata document in the models collection.
"""

import argparse
import pickle
from datetime import datetime, timezone, timedelta

import pandas as pd
import gridfs
from pymongo import MongoClient
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    brier_score_loss,
)
from xgboost import XGBClassifier

# -----------------------
# CONFIG (env or defaults)
# -----------------------
MONGO_URI = None  # read from env by connect()
DB_NAME = "weather"
FEATURES_COL = "features"
MODELS_COL = "models"

# -----------------------
# Helpers
# -----------------------
def connect(mongo_uri: str = None):
    """Connect to MongoDB and return (client, db)."""
    uri = mongo_uri or MONGO_URI or __import__("os").environ.get("MONGO_URI", "mongodb://localhost:27017")
    client = MongoClient(uri)
    db = client[DB_NAME]
    return client, db


def fetch_labeled_feature_docs(db, days: int = None):
    """
    Fetch feature documents that have a label (label_next_hour or label).
    If days provided, limit to docs with timestamp >= now - days.
    Returns list of docs.
    """
    q = {"$or": [{"label_next_hour": {"$exists": True}}, {"label": {"$exists": True}}]}
    if days:
        now_utc = datetime.now(timezone.utc)
        cutoff = now_utc - timedelta(days=days)
        q["timestamp"] = {"$gte": cutoff}
    docs = list(db[FEATURES_COL].find(q))
    return docs


def docs_to_df_with_label(docs):
    """
    Convert feature docs to DataFrame with columns:
      - location_id, timestamp, label, <feature columns...>
    Coerce numeric-like values where possible.
    """
    rows = []
    for d in docs:
        label = d.get("label_next_hour", d.get("label", None))
        if label is None:
            continue
        features = d.get("features", {}) or {}
        row = {"location_id": d.get("location_id"), "timestamp": d.get("timestamp"), "label": int(label)}
        for k, v in features.items():
            # accept scalar numeric values
            if isinstance(v, (int, float)):
                row[k] = v
            else:
                # try coerce numeric-like strings
                try:
                    row[k] = float(v)
                except Exception:
                    # skip non-numeric values
                    pass
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # ensure timestamp is timezone-aware UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def safe_metrics(y_true, prob_pred):
    """
    Compute metrics safely. If only one class in y_true, roc_auc_score is undefined -> return nan.
    Returns dict with keys roc_auc, precision, recall, brier.
    """
    metrics = {"roc_auc": float("nan"), "precision": 0.0, "recall": 0.0, "brier": float("nan")}
    if len(y_true) == 0:
        return metrics
    try:
        pred = (prob_pred >= 0.5).astype(int)
    except Exception:
        pred = prob_pred >= 0.5
    # precision/recall safe via zero_division=0
    try:
        metrics["precision"] = float(precision_score(y_true, pred, zero_division=0))
    except Exception:
        metrics["precision"] = 0.0
    try:
        metrics["recall"] = float(recall_score(y_true, pred, zero_division=0))
    except Exception:
        metrics["recall"] = 0.0
    try:
        metrics["brier"] = float(brier_score_loss(y_true, prob_pred))
    except Exception:
        metrics["brier"] = float("nan")
    # roc_auc only if both classes present
    unique = pd.Series(y_true).unique()
    if len(unique) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, prob_pred))
        except Exception:
            metrics["roc_auc"] = float("nan")
    return metrics


def time_split(df, feat_cols, train_frac=0.7, val_frac=0.15):
    """
    Time-based split. Returns (X_train,y_train, X_val,y_val, X_test,y_test)
    """
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
    return X_train, y_train, X_val, y_val, X_test, y_test


def save_model_to_gridfs(db, model_obj, feat_cols):
    """Pickle model and feature columns and save to GridFS. Returns gridfs id."""
    fs = gridfs.GridFS(db)
    blob = pickle.dumps({"model": model_obj, "feature_columns": feat_cols})
    fname = f"model_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.pkl"
    gridfs_id = fs.put(blob, filename=fname)
    return gridfs_id


def train_and_save_time_split(db, days=14, min_samples=200, activate=False, train_frac=0.7, val_frac=0.15):
    """
    Time-split training pipeline:
      - Load labeled feature docs (last `days` days if set)
      - Convert to DataFrame, choose numeric feature columns
      - Time-based split, train XGBClassifier, evaluate on val and test (test is honest)
      - Save artifact to GridFS and model metadata to models collection (test metrics)
    Returns model_doc
    """
    docs = fetch_labeled_feature_docs(db, days=days)
    df = docs_to_df_with_label(docs)
    if df.empty:
        raise RuntimeError("No labeled feature docs found.")

    if len(df) < min_samples:
        raise RuntimeError(f"Not enough labeled rows: {len(df)} < {min_samples}")

    # choose numeric feature columns (exclude meta)
    exclude = {"location_id", "timestamp", "label"}
    feat_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not feat_cols:
        raise RuntimeError("No numeric feature columns found.")

    # split
    X_train, y_train, X_val, y_val, X_test, y_test = time_split(df, feat_cols, train_frac=train_frac, val_frac=val_frac)
    print("Dataset sizes: total=%d, train=%s, val=%s, test=%s" % (len(df), X_train.shape, X_val.shape, X_test.shape))

    # train
    model = XGBClassifier(n_estimators=200, max_depth=5, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    # compute metrics
    prob_val = model.predict_proba(X_val)[:, 1] if len(X_val) > 0 else []
    metrics_val = safe_metrics(y_val.tolist() if len(X_val) > 0 else [], prob_val)

    prob_test = model.predict_proba(X_test)[:, 1] if len(X_test) > 0 else []
    metrics_test = safe_metrics(y_test.tolist() if len(X_test) > 0 else [], prob_test)

    print("Validation metrics:", metrics_val)
    print("Test (time-split) metrics:", metrics_test)

    # save model artifact
    gridfs_id = save_model_to_gridfs(db, model, feat_cols)

    # insert model metadata
    models_coll = db[MODELS_COL]
    model_doc = {
        "model_version": datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S"),
        "artifact_gridfs_id": gridfs_id,
        "created_at": datetime.now(timezone.utc),
        "metrics": metrics_test,
        "validation_metrics": metrics_val,
        "n_samples": int(len(df)),
        "feature_columns": feat_cols,
        "active": bool(activate),
    }
    if activate:
        models_coll.update_many({}, {"$set": {"active": False}})
    models_coll.insert_one(model_doc)

    return model_doc


# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Train rainfall model (time-split) and save to GridFS/Mongo.")
    parser.add_argument("--days", type=int, default=14, help="How many days of features to load (default 14).")
    parser.add_argument("--min-samples", type=int, default=200, help="Minimum labeled samples required.")
    parser.add_argument("--activate", action="store_true", help="Mark this model active after saving.")
    parser.add_argument("--mongo-uri", type=str, default=None, help="MongoDB URI (overrides MONGO_URI env).")
    parser.add_argument("--train-frac", type=float, default=0.70, help="Train fraction for time split.")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Validation fraction for time split.")
    args = parser.parse_args()

    client, db = connect(args.mongo_uri)
    try:
        doc = train_and_save_time_split(
            db,
            days=args.days,
            min_samples=args.min_samples,
            activate=args.activate,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
        )
        print("Saved model:", doc["model_version"], "metrics:", doc["metrics"])
    except Exception as e:
        print("ERROR during training:", e)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
