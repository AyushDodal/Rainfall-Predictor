# labeler.py
import os
from datetime import timedelta, datetime, timezone
from pymongo import MongoClient, UpdateOne
import traceback
import argparse

MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = os.environ.get("DB_NAME", "weather")
FEATURES_COL = "features"
RAW_COL = "raw_observations"
PRED_COL = os.getenv("PRED_COL", "predictions")

DEFAULT_DAYS = 14
BATCH_SIZE = 500

if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set in env")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
features_col = db[FEATURES_COL]
raw_col = db[RAW_COL]
pred_col = db[PRED_COL]

def extract_rain_from_payload(payload):
    if not payload or not isinstance(payload, dict):
        return 0.0
    rain = payload.get("rain")
    if isinstance(rain, (int, float)):
        return float(rain)
    if isinstance(rain, dict):
        val = rain.get("1h")
        if val is None:
            for v in rain.values():
                try:
                    return float(v)
                except Exception:
                    continue
        try:
            return float(val) if val is not None else 0.0
        except Exception:
            return 0.0
    return 0.0

def compute_label_for_feature(doc):
    loc = doc.get("location_id")
    ts = doc.get("timestamp")
    if loc is None or ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    end = ts + timedelta(hours=1)
    cursor = raw_col.find({"location_id": loc, "timestamp": {"$gt": ts, "$lte": end}})
    found_any = False
    for r in cursor:
        found_any = True
        payload = r.get("payload") or {}
        rain_amt = extract_rain_from_payload(payload)
        try:
            if float(rain_amt) > 0.0:
                return 1
        except Exception:
            continue
    if not found_any:
        return None
    return 0

def process_batch(docs, update_predictions):
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
        ops_features.append(
            UpdateOne(
                {"_id": d["_id"]},
                {"$set": {"label_next_hour": int(label), "label_generated_at": datetime.now(timezone.utc)}}
            )
        )
        if update_predictions:
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
    return labeled, skipped

def run_labeler(days=DEFAULT_DAYS, batch_size=BATCH_SIZE, update_predictions=True):
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
    total_labeled = 0
    total_skipped = 0
    for doc in cursor:
        to_process.append(doc)
        if len(to_process) >= batch_size:
            labeled, skipped = process_batch(to_process, update_predictions)
            total_labeled += labeled
            total_skipped += skipped
            to_process = []
    if to_process:
        labeled, skipped = process_batch(to_process, update_predictions)
        total_labeled += labeled
        total_skipped += skipped
    print(f"Labeler complete: total_labeled={total_labeled}, total_skipped={total_skipped}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--no-update-predictions", action="store_true")
    parser.add_argument("--once", action="store_true", help="Run once and exit (default)")
    args = parser.parse_args()

    # labeler is naturally a one-shot job; if you want periodic runs, trigger via scheduler
    run_labeler(days=args.days, batch_size=args.batch_size, update_predictions=not args.no_update_predictions)
