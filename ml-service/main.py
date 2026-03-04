"""
main.py – GraphQL Anomaly Detection ML Service (FastAPI)

Endpoints:
  POST /analyze      – score a feature vector and return anomaly report
  POST /train        – retrain all models from scratch
  POST /compare      – run multi-model comparison and return table
  POST /feedback     – push explicit label feedback (normal/anomaly) for online learning
  GET  /health       – health check
  GET  /metrics      – summary metrics
  GET  /model        – current active model info
"""

from __future__ import annotations

import os
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml.training_data import FEATURE_COLUMNS
from detectors import structural_detector, frequency_detector
from scorer import rule_score, make_report
from ml.online_learner import learner as online_learner

# ── Config ────────────────────────────────────────────────────────────────────
BLOCK_THRESHOLD     = float(os.getenv("BLOCK_THRESHOLD", "0.6"))
MAX_QUERY_DEPTH     = int(os.getenv("MAX_QUERY_DEPTH", "7"))
MAX_ALIAS_COUNT     = int(os.getenv("MAX_ALIAS_COUNT", "10"))
BLOCK_INTROSPECTION = os.getenv("BLOCK_INTROSPECTION", "false").lower() == "true"

CONFIG = {
    "max_query_depth":     MAX_QUERY_DEPTH,
    "max_alias_count":     MAX_ALIAS_COUNT,
    "block_introspection": BLOCK_INTROSPECTION,
}

# ── Metrics ───────────────────────────────────────────────────────────────────
_metrics: dict[str, Any] = {
    "total_analyzed": 0,
    "total_blocked":  0,
    "total_passed":   0,
    "start_time":     time.time(),
}

_last_comparison: list[dict] | None = None

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="GraphQL Anomaly Detection Service",
    description="ML-powered anomaly scoring for GraphQL queries",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class FeatureVector(BaseModel):
    max_depth:           float = Field(0, ge=0)
    total_fields:        float = Field(0, ge=0)
    unique_fields:       float = Field(0, ge=0)
    alias_count:         float = Field(0, ge=0)
    introspection_count: float = Field(0, ge=0)
    fragment_count:      float = Field(0, ge=0)
    estimated_cost:      float = Field(0, ge=0)
    payload_size:        float = Field(0, ge=0)
    field_entropy:       float = Field(0, ge=0)
    nesting_variance:    float = Field(0, ge=0)
    client_ip:           str   = Field("0.0.0.0")
    timestamp:           float = Field(default_factory=time.time)
    query_name:          str   = Field("")


class AnomalyReport(BaseModel):
    ensemble_score:    float
    is_anomaly:        bool
    threshold:         float
    component_scores:  dict
    rule_violations:   dict
    frequency_detail:  dict
    features_received: dict

# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    print("[startup] Pre-loading ML models…")
    # structural_detector loads itself at import time
    print(f"[startup] Active model: {structural_detector.get_model_name()}")
    print(f"[startup] Online learner buffer: {online_learner.status()['buffer_size']} samples")
    print("[startup] Ready.")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _metrics["start_time"], 1),
        "active_model":   structural_detector.get_model_name(),
    }


@app.get("/metrics")
def metrics():
    uptime = time.time() - _metrics["start_time"]
    total  = _metrics["total_analyzed"]
    return {
        **_metrics,
        "uptime_seconds": round(uptime, 1),
        "block_rate":     round(_metrics["total_blocked"] / max(total, 1), 4),
        "config":         CONFIG,
        "block_threshold": BLOCK_THRESHOLD,
        "active_model":   structural_detector.get_model_name(),
        "online_learner": online_learner.status(),
    }


@app.get("/model")
def model_info():
    return {
        "active_model": structural_detector.get_model_name(),
        "last_comparison": _last_comparison,
        "online_learner": online_learner.status(),
    }


@app.post("/analyze", response_model=AnomalyReport)
async def analyze(vector: FeatureVector, request: Request):
    features = vector.model_dump()

    struct_score = structural_detector.score(features)
    client_ip    = vector.client_ip or (request.client.host if request.client else "unknown")
    freq_result  = frequency_detector.record_and_score(client_ip)
    rule_result  = rule_score(features, CONFIG)

    report = make_report(
        features=features,
        structural_score=struct_score,
        freq_result=freq_result,
        rule_result=rule_result,
        threshold=BLOCK_THRESHOLD,
        config=CONFIG,
    )

    _metrics["total_analyzed"] += 1
    if report["is_anomaly"]:
        _metrics["total_blocked"] += 1
    else:
        _metrics["total_passed"] += 1
        # Feed confirmed-normal queries into the online learner buffer
        online_learner.add_normal(features)

    return report


@app.post("/train")
def retrain():
    """Retrain using current best-model pipeline (runs model comparison + saves winner)."""
    import glob
    for f in glob.glob(os.path.join("models", "*.pkl")):
        os.remove(f)
    # Re-run full comparison and save best
    from ml.model_comparison import run_comparison
    global _last_comparison
    results, best_name = run_comparison(save_best=True, verbose=False)
    _last_comparison = results
    structural_detector.reload_models()
    return {
        "status": "models retrained successfully",
        "best_model": best_name,
        "comparison": results,
    }


@app.post("/compare")
def compare_models():
    """Run multi-model comparison and return results table."""
    from ml.model_comparison import run_comparison
    global _last_comparison
    results, best_name = run_comparison(save_best=True, verbose=False)
    _last_comparison = results
    structural_detector.reload_models()
    return {
        "best_model": best_name,
        "results": results,
    }


# ── Feedback & Online Learning ────────────────────────────────────────────────

class FeedbackInput(BaseModel):
    features: dict = Field(default_factory=dict,
                           description="Feature vector (same keys as /analyze input)")
    label:    str  = Field(description="'normal' or 'anomaly'")


@app.post("/feedback")
def feedback(body: FeedbackInput):
    """
    Accept explicit label feedback for online learning.
    Normal samples are buffered; background retraining triggers automatically
    when buffer reaches the threshold.
    Body:  { "features": {...}, "label": "normal" | "anomaly" }
    """
    label = body.label.strip().lower()
    if label not in ("normal", "anomaly"):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="label must be 'normal' or 'anomaly'")

    retrain_triggered = False
    if label == "normal":
        online_learner.add_normal(body.features)
        if online_learner.should_retrain():
            retrain_triggered = online_learner.trigger_retrain_background()

    status = online_learner.status()
    return {
        "status":            "added" if label == "normal" else "noted",
        "label":             label,
        "buffer_size":       status["buffer_size"],
        "since_retrain":     status["since_retrain"],
        "retrain_triggered": retrain_triggered,
    }


@app.get("/learner")
def learner_status():
    """Return online learner buffer statistics."""
    return online_learner.status()
