# serve.py
import os
import pickle
from datetime import datetime, timezone
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
import gridfs

APP_NAME = "Rainfall Serve"

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "weather")
MODELS_COL = os.getenv("MODELS_COL", "models")
PRED_COL = os.getenv("PRED_COL", "predictions")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
fs = gridfs.GridFS(db)

app = FastAPI(title=APP_NAME)

_model_cache = {"model": None, "feature_columns": None, "model_version": None}


class BatchFeatures(BaseModel):
    items: List[Dict[str, Any]]  # list of dicts w/ location_id, timestamp and feature keys


def _load_latest_active_model():
    doc = db[MODELS_COL].find_one({"active": True}, sort=[("created_at", -1)])
    if doc is None:
        doc = db[MODELS_COL].find_one(sort=[("created_at", -1)])
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
        print(f"Loaded model: {v}")
    except Exception as e:
        # Do not crash — server runs and you can /reload later
        print("Warning: failed to load model on startup:", e)


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
        meta.append({"location_id": r.get("location_id"), "timestamp": r.get("timestamp")})
        X.append([r.get(c, 0.0) for c in feat_cols])

    try:
        probs = model.predict_proba(X)[:, 1].tolist()
    except Exception:
        preds = model.predict(X)
        probs = [float(p) for p in preds]

    now = datetime.now(timezone.utc)
    to_insert = []
    for m, p in zip(meta, probs):
        doc = {
            "location_id": m["location_id"],
            "timestamp": m["timestamp"],
            "pred_prob": float(p),
            "model_version": mv,
            "scored_at": now
        }
        to_insert.append(doc)
    if to_insert:
        db[PRED_COL].insert_many(to_insert)

    out = [{"location_id": m["location_id"], "timestamp": m["timestamp"], "prob": float(p), "model_version": mv}
           for m, p in zip(meta, probs)]

    return {"results": out, "model_version": mv}


@app.post("/reload")
def reload_model():
    try:
        mv = _load_latest_active_model()
        return {"status": "ok", "model_version": mv}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    ok = _model_cache["model"] is not None
    return {"status": "ok" if ok else "no-model", "model_version": _model_cache.get("model_version")}
