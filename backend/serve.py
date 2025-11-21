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

# cache for model + metadata
_model_cache = {"model": None, "feature_columns": None, "model_version": None}


class BatchFeatures(BaseModel):
    # list of dicts with e.g. location_id, timestamp, lat, lon, and feature columns
    items: List[Dict[str, Any]]


def _load_latest_active_model() -> str:
    """Load most recent active model from Mongo/GridFS into memory."""
    coll = db[MODELS_COL]

    doc = coll.find_one({"active": True}, sort=[("created_at", -1)])
    if doc is None:
        doc = coll.find_one(sort=[("created_at", -1)])
        if doc is None:
            raise RuntimeError("No model found in models collection.")

    blob = fs.get(doc["artifact_gridfs_id"]).read()
    payload = pickle.loads(blob)

    _model_cache["model"] = payload["model"]
    _model_cache["feature_columns"] = payload["feature_columns"]
    _model_cache["model_version"] = doc.get("model_version", str(doc["_id"]))
    return _model_cache["model_version"]


@app.on_event("startup")
def startup_load_model():
    try:
        v = _load_latest_active_model()
        print(f"[serve] Loaded model on startup: {v}")
    except Exception as e:
        print("[serve] Warning: failed to load model on startup:", e)


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
        meta.append(
            {
                "location_id": r.get("location_id"),
                "timestamp": r.get("timestamp"),
                "lat": r.get("lat"),
                "lon": r.get("lon"),
            }
        )
        X.append([r.get(c, 0.0) for c in feat_cols])

    try:
        probs = model.predict_proba(X)[:, 1].tolist()
    except Exception:
        preds = model.predict(X)
        probs = [float(p) for p in preds]

    now = datetime.now(timezone.utc)

    docs_to_insert = []
    for m, p in zip(meta, probs):
        ts = m["timestamp"]
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                ts = now

        doc = {
            "location_id": m["location_id"],
            "timestamp": ts,
            "lat": m.get("lat"),
            "lon": m.get("lon"),
            "pred_prob": float(p),
            "model_version": mv,
            "scored_at": now,
        }
        docs_to_insert.append(doc)

    if docs_to_insert:
        db[PRED_COL].insert_many(docs_to_insert)

    out = [
        {
            "location_id": d["location_id"],
            "timestamp": d["timestamp"].astimezone(timezone.utc).isoformat()
            if isinstance(d["timestamp"], datetime)
            else str(d["timestamp"]),
            "lat": d.get("lat"),
            "lon": d.get("lon"),
            "prob": float(d["pred_prob"]),
            "model_version": mv,
        }
        for d in docs_to_insert
    ]

    return {"results": out, "model_version": mv}


@app.get("/predictions/recent")
def recent_predictions(n: int = 100):
    """
    Return up to n most recent prediction docs, newest first.
    Used by the Streamlit dashboard.
    """
    try:
        n = max(1, min(int(n), 2000))
    except Exception:
        n = 100

    try:
        cursor = (
            db[PRED_COL]
            .find({}, projection={"_id": 0})
            .sort("scored_at", -1)
            .limit(n)
        )
        docs = []
        for d in cursor:
            ts = d.get("timestamp")
            if isinstance(ts, datetime):
                ts = ts.astimezone(timezone.utc).isoformat()
            else:
                ts = str(ts)

            prob = float(d.get("pred_prob", 0.0))
            docs.append(
                {
                    "location_id": d.get("location_id"),
                    "lat": d.get("lat"),
                    "lon": d.get("lon"),
                    "timestamp": ts,
                    "probability": prob,
                    "predicted_label": int(prob >= 0.5),
                    "model_version": d.get("model_version"),
                }
            )
        return {"items": docs, "count": len(docs)}
    except Exception as e:
        # log to server console; Streamlit will fall back to sample data
        print("[serve] /predictions/recent failed:", repr(e))
        raise HTTPException(status_code=500, detail="failed to read predictions")


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
    return {
        "status": "ok" if ok else "no-model",
        "model_version": _model_cache.get("model_version"),
    }
