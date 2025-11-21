# serve.py

import os
import pickle
from datetime import datetime, timezone
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
import gridfs

# --------------------
# Config
# --------------------
APP_NAME = "Rainfall Serve"

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "weather")
MODELS_COL = os.getenv("MODELS_COL", "models")
PRED_COL = os.getenv("PRED_COL", "predictions")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
fs = gridfs.GridFS(db)

app = FastAPI(title=APP_NAME)

# simple in-memory cache
_model_cache = {"model": None, "feature_columns": None, "model_version": None}


class BatchFeatures(BaseModel):
    # each item: dict with location_id, timestamp, (optional lat/lon) + feature columns
    items: List[Dict[str, Any]]


# --------------------
# Model loading
# --------------------
def _load_latest_active_model() -> str:
    coll = db[MODELS_COL]
    doc = coll.find_one({"active": True}, sort=[("created_at", -1)])
    if doc is None:
        doc = coll.find_one(sort=[("created_at", -1)])
        if doc is None:
            raise RuntimeError("No model found in models collection.")

    gridfs_id = doc["artifact_gridfs_id"]
    blob = fs.get(gridfs_id).read()
    payload = pickle.loads(blob)

    _model_cache["model"] = payload["model"]
    _model_cache["feature_columns"] = payload["feature_columns"]
    _model_cache["model_version"] = doc.get("model_version", str(doc["_id"]))
    return _model_cache["model_version"]


@app.on_event("startup")
def startup_load_model():
    try:
        v = _load_latest_active_model()
        print(f"[startup] Loaded model: {v}")
    except Exception as e:
        # keep API alive even if no model yet
        print("[startup] Warning: failed to load model:", e)


# --------------------
# Basic routes
# --------------------
@app.get("/")
def root():
    return {"service": APP_NAME, "docs": "/docs"}


@app.get("/health")
def health():
    ok = _model_cache["model"] is not None
    return {"status": "ok" if ok else "no-model",
            "model_version": _model_cache.get("model_version")}


@app.post("/reload")
def reload_model():
    try:
        mv = _load_latest_active_model()
        return {"status": "ok", "model_version": mv}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------
# Scoring endpoint
# --------------------
@app.post("/score")
def score_batch(batch: BatchFeatures):
    if _model_cache["model"] is None:
        raise HTTPException(status_code=500, detail="No model loaded")

    model = _model_cache["model"]
    feat_cols = _model_cache["feature_columns"]
    mv = _model_cache["model_version"]

    rows = batch.items
    X = []
    meta = []

    for r in rows:
        # meta fields we want to keep alongside prediction
        meta.append({
            "location_id": r.get("location_id"),
            "timestamp": r.get("timestamp"),
            "lat": r.get("lat"),
            "lon": r.get("lon"),
        })
        X.append([r.get(c, 0.0) for c in feat_cols])

    try:
        probs = model.predict_proba(X)[:, 1].tolist()
    except Exception:
        preds = model.predict(X)
        probs = [float(p) for p in preds]

    now = datetime.now(timezone.utc)

    docs_to_insert = []
    out = []
    for m, p in zip(meta, probs):
        prob = float(p)
        pred_label = int(prob >= 0.5)

        doc = {
            "location_id": m["location_id"],
            "timestamp": m["timestamp"],   # usually ISO string or datetime
            "lat": m.get("lat"),
            "lon": m.get("lon"),
            "probability": prob,
            "predicted_label": pred_label,
            "pred_prob": prob,             # legacy name, kept for safety
            "model_version": mv,
            "scored_at": now,
        }
        docs_to_insert.append(doc)

        out.append({
            "location_id": m["location_id"],
            "timestamp": m["timestamp"],
            "lat": m.get("lat"),
            "lon": m.get("lon"),
            "probability": prob,
            "predicted_label": pred_label,
            "model_version": mv,
        })

    if docs_to_insert:
        db[PRED_COL].insert_many(docs_to_insert)

    return {"results": out, "model_version": mv}


# --------------------
# Recent predictions for dashboard
# --------------------
@app.get("/predictions/recent")
def recent_predictions(n: int = 100):
    """
    Return up to n most recent prediction docs, newest first.
    Streamlit dashboard uses this endpoint.
    """
    if n <= 0 or n > 2000:
        raise HTTPException(status_code=400, detail="n must be between 1 and 2000")

    coll = db[PRED_COL]
    cursor = coll.find().sort("scored_at", -1).limit(n)

    results = []
    for d in cursor:
        d["_id"] = str(d["_id"])
        ts = d.get("timestamp")
        if isinstance(ts, datetime):
            d["timestamp"] = ts.isoformat()
        sa = d.get("scored_at")
        if isinstance(sa, datetime):
            d["scored_at"] = sa.isoformat()
        results.append(d)

    return results
