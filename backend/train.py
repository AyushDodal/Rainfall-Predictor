import os
import argparse
import pickle
from datetime import datetime, timezone

import pandas as pd
from pymongo import MongoClient
import gridfs
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    brier_score_loss,
)

# -----------------------
# Config
# -----------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "weather")
FEATURES_COL = os.getenv("FEATURES_COL", "features")
MODELS_COL = os.getenv("MODELS_COL", "models")

DEFAULT_MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "20000"))


# -----------------------
# Mongo helpers
# -----------------------
def connect():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return client, db


def _fetch_feature_docs(db, days=None):
    """
    Load labeled feature docs from MongoDB.
    Uses timezone-aware cutoff so it works in all pandas versions.
    """
    q = {}
    if days:
        now_utc = datetime.now(timezone.utc)
        cutoff_dt = now_utc - pd.Timedelta(days=days)
        q["timestamp"] = {"$gte": cutoff_dt}
    docs = list(db[FEATURES_COL].find(q))
    return docs


def _docs_to_df_with_label(docs):
    """
    Flatten docs into a DataFrame with:
      - location_id, timestamp, label
      - numeric feature columns from `features` dict
    """
    rows = []
    for d in docs:
        feat = d.get("features", {})
        if not isinstance(feat, dict):
            continue

        label = d.get("label_next_hour", d.get("label", None))
        if label is None:
            continue

        row = {
            "location_id": d.get("location_id"),
            "timestamp": d.get("timestamp"),
            "label": int(label),
        }

        for k, v in feat.items():
            # keep only numeric-ish scalars
            if isinstance(v, (int, float)):
                row[k] = v
            else:
                try:
                    row[k] = float(v)
                except Exception:
                    pass

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# -----------------------
# Training
# -----------------------
def train_and_save(
    db,
    days: int | None = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    activate: bool = False,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
):
    """
    Time-split training with class-weighted XGBoost:
      - Load labeled features (optionally last `days` days)
      - Sort by timestamp, split into train/val/test
      - Compute scale_pos_weight = neg/pos on TRAIN SET
      - Train XGBoost with that weight
      - Report validation + test metrics
      - Save model to GridFS and insert metadata
    """
    docs = _fetch_feature_docs(db, days=days)
    df = _docs_to_df_with_label(docs)
    if df.empty:
        raise RuntimeError("No labeled feature docs found.")

    n_rows = len(df)
    if n_rows < min_samples:
        raise RuntimeError(f"Not enough labeled rows: {n_rows} < {min_samples}")

    # choose numeric feature columns
    exclude = {"location_id", "timestamp", "label"}
    feat_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feat_cols:
        raise RuntimeError("No numeric feature columns found in features docs.")

    # time-sorted split
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

    print("Dataset sizes -> total:", n,
          ", train:", X_train.shape,
          ", val:", X_val.shape,
          ", test:", X_test.shape)

    # class balance info (train only)
    pos_train = int((y_train == 1).sum())
    neg_train = int((y_train == 0).sum())
    print(f"Train class counts -> 0: {neg_train}, 1: {pos_train}")

    if pos_train == 0 or neg_train == 0:
        raise RuntimeError(
            f"Training set has a single class (0s={neg_train}, 1s={pos_train}). "
            f"Wait for more positive rain examples before training."
        )

    scale_pos_weight = neg_train / pos_train
    print(f"Using scale_pos_weight = {scale_pos_weight:.2f}")

    # XGBoost with class weight
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        n_jobs=4,
    )
    model.fit(X_train, y_train)

    # ----- validation metrics -----
    prob_val = model.predict_proba(X_val)[:, 1]
    pred_val = (prob_val >= 0.5).astype(int)
    metrics_val = {
        "roc_auc": float(roc_auc_score(y_val, prob_val)) if len(set(y_val)) > 1 else float("nan"),
        "precision": float(precision_score(y_val, pred_val, zero_division=0)),
        "recall": float(recall_score(y_val, pred_val, zero_division=0)),
        "brier": float(brier_score_loss(y_val, prob_val)),
    }
    print("Validation metrics:", metrics_val)

    # ----- test metrics (time-split) -----
    prob_test = model.predict_proba(X_test)[:, 1]
    pred_test = (prob_test >= 0.5).astype(int)
    metrics_test = {
        "roc_auc": float(roc_auc_score(y_test, prob_test)) if len(set(y_test)) > 1 else float("nan"),
        "precision": float(precision_score(y_test, pred_test, zero_division=0)),
        "recall": float(recall_score(y_test, pred_test, zero_division=0)),
        "brier": float(brier_score_loss(y_test, prob_test)),
    }
    print("Test (time-split) metrics:", metrics_test)

    # ----- save model to GridFS -----
    fs = gridfs.GridFS(db)
    blob = pickle.dumps({"model": model, "feature_columns": feat_cols})
    gridfs_id = fs.put(
        blob,
        filename=f"model_{pd.Timestamp.utcnow().strftime('%Y%m%d%H%M%S')}.pkl",
    )

    models_coll = db[MODELS_COL]
    model_doc = {
        "model_version": pd.Timestamp.utcnow().strftime("v%Y%m%d%H%M%S"),
        "artifact_gridfs_id": gridfs_id,
        "created_at": pd.Timestamp.utcnow().to_pydatetime(),
        "metrics": metrics_test,
        "validation_metrics": metrics_val,
        "n_samples": int(n),
        "class_counts": {"train_0": neg_train, "train_1": pos_train},
        "scale_pos_weight": float(scale_pos_weight),
        "feature_columns": feat_cols,
        "active": bool(activate),
    }

    if activate:
        models_coll.update_many({}, {"$set": {"active": False}})
    models_coll.insert_one(model_doc)

    print("Model training complete. Version:", model_doc["model_version"])
    print("Metrics:", model_doc["metrics"])
    return model_doc


# -----------------------
# CLI
# -----------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=None,
                   help="If set, only use the last N days of features.")
    p.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES,
                   help="Minimum labeled rows required to train.")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--activate", action="store_true",
                   help="Mark this model as active after training.")
    args = p.parse_args()

    client, db = connect()
    try:
        train_and_save(
            db,
            days=args.days,
            min_samples=args.min_samples,
            activate=args.activate,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
