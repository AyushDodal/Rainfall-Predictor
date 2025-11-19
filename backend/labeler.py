# labeler.py
import os
from datetime import timedelta
from pymongo import MongoClient, UpdateOne
from datetime import datetime, timezone

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "weather")
FEATURES_COL = os.getenv("FEATURES_COL", "features")
RAW_COL = os.getenv("RAW_COL", "raw_observations")
PRED_COL = os.getenv("PRED_COL", "predictions")

# how far back to search (days) for unlabeled features
DEFAULT_DAYS = 14
BATCH_SIZE = 500

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
features_col = db[FEATURES_COL]
raw_col = db[RAW_COL]
pred_col = db[PRED_COL]

def extract_rain_from_payload(payload):
    """Robustly extract 1h rain amount from OWM-like payloads."""
    if not payload or not isinstance(payload, dict):
        return 0.0
    # try payload.rain (could be dict or number)
    rain = payload.get("rain")
    if isinstance(rain, (int, float)):
        return float(rain)
    if isinstance(rain, dict):
        # common key: "1h"
        val = rain.get("1h")
        if val is None:
            # sometimes total or other keys
            for v in rain.values():
                try:
                    return float(v)
                except Exception:
                    continue
        try:
            return float(val) if val is not None else 0.0
        except Exception:
            return 0.0
    # fallback: sometimes precipitation in other fields
    return 0.0

def compute_label_for_feature(doc):
    """
    doc: feature document with keys 'location_id' and 'timestamp' (datetime)
    returns: 1, 0, or None (if insufficient future data)
    """
    loc = doc.get("location_id")
    ts = doc.get("timestamp")
    if loc is None or ts is None:
        return None
    # ensure tz-aware datetime
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    end = ts + timedelta(hours=1)
    cursor = raw_col.find({"location_id": loc, "timestamp": {"$gt": ts, "$lte": end}})
    found_any = False
    for r in cursor:
        found_any = True
        payload = r.get("payload") or {}
        rain_amt = extract_rain_from_payload(payload)
        if rain_amt is None:
            continue
        try:
            if float(rain_amt) > 0.0:
                return 1
        except Exception:
            continue
    if not found_any:
        return None
    return 0

def main(days=DEFAULT_DAYS, batch_size=BATCH_SIZE, update_predictions=True):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    q = {
        "timestamp": {"$gte": cutoff},
        "$or": [
            {"label_next_hour": {"$exists": False}},
            {"label_next_hour": None}
        ]
    }
    cursor = features_col.find(q).sort("timestamp", 1).limit(10000)
    to_process = []
    for doc in cursor:
        to_process.append(doc)
        if len(to_process) >= batch_size:
            _process_batch(to_process, update_predictions)
            to_process = []
    if to_process:
        _process_batch(to_process, update_predictions)

def _process_batch(docs, update_predictions):
    ops_features = []
    ops_preds = []
    labeled = 0
    skipped = 0
    for d in docs:
        label = compute_label_for_feature(d)
        if label is None:
            skipped += 1
            continue
        labeled += 1
        ts = d["timestamp"]
        loc = d["location_id"]
        # upsert into features collection
        ops_features.append(
            UpdateOne(
                {"_id": d["_id"]},
                {"$set": {"label_next_hour": int(label), "label_generated_at": datetime.now(timezone.utc)}}
            )
        )
        if update_predictions:
            # update matching prediction if exists (match on loc + timestamp)
            ops_preds.append(
                UpdateOne(
                    {"location_id": loc, "timestamp": ts},
                    {"$set": {"label": int(label), "label_generated_at": datetime.now(timezone.utc)}}
                )
            )
    if ops_features:
        features_col.bulk_write(ops_features)
    if ops_preds:
        pred_col.bulk_write(ops_preds)
    print(f"Batch processed: labeled={labeled}, skipped={skipped}, attempted={len(docs)}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=DEFAULT_DAYS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--no-update-predictions", action="store_true", help="Do not update predictions collection")
    args = p.parse_args()
    main(days=args.days, batch_size=args.batch_size, update_predictions=not args.no_update_predictions)
