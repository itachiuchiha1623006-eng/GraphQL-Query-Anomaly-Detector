"""
main.py – GraphQL Anomaly Detection ML Service (FastAPI)

Endpoints:
  POST /analyze      – score a feature vector and return anomaly report
  POST /train        – retrain all models from scratch
  GET  /health       – health check
  GET  /metrics      – summary metrics
"""

from __future__ import annotations

import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml.training_data import FEATURE_COLUMNS
from detectors import structural_detector, frequency_detector
from scorer import rule_score, make_report

# ---------------------------------------------------------------------------
# Config (can be overridden via environment variables)
# ---------------------------------------------------------------------------
BLOCK_THRESHOLD     = float(os.getenv("BLOCK_THRESHOLD", "0.6"))
MAX_QUERY_DEPTH     = int(os.getenv("MAX_QUERY_DEPTH", "7"))
MAX_ALIAS_COUNT     = int(os.getenv("MAX_ALIAS_COUNT", "10"))
BLOCK_INTROSPECTION = os.getenv("BLOCK_INTROSPECTION", "false").lower() == "true"

CONFIG = {
    "max_query_depth":     MAX_QUERY_DEPTH,
    "max_alias_count":     MAX_ALIAS_COUNT,
    "block_introspection": BLOCK_INTROSPECTION,
}

# ---------------------------------------------------------------------------
# Metrics counters (in-memory; no Prometheus dependency)
# ---------------------------------------------------------------------------
_metrics: dict[str, Any] = {
    "total_analyzed": 0,
    "total_blocked":  0,
    "total_passed":   0,
    "start_time":     time.time(),
}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="GraphQL Anomaly Detection Service",
    description="ML-powered anomaly scoring for GraphQL queries",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class FeatureVector(BaseModel):
    """
    Feature vector extracted by the JS middleware from the GraphQL AST.
    All numeric fields default to 0 if not provided.
    """
    max_depth:            float = Field(0, ge=0, description="Max nesting depth")
    total_fields:         float = Field(0, ge=0, description="Total field selections")
    unique_fields:        float = Field(0, ge=0, description="Unique field names")
    alias_count:          float = Field(0, ge=0, description="Number of aliases")
    introspection_count:  float = Field(0, ge=0, description="Introspection field refs")
    fragment_count:       float = Field(0, ge=0, description="Fragment count")
    estimated_cost:       float = Field(0, ge=0, description="Estimated resolver cost")
    payload_size:         float = Field(0, ge=0, description="Query string byte length")
    field_entropy:        float = Field(0, ge=0, description="Shannon entropy of field names")
    nesting_variance:     float = Field(0, ge=0, description="Variance of nesting depths")

    # Metadata – not used in ML models but logged and used for frequency detection
    client_ip:  str = Field("0.0.0.0", description="Client IP address")
    timestamp:  float = Field(default_factory=time.time, description="Unix timestamp")
    query_name: str = Field("", description="Operation name if present")


class AnomalyReport(BaseModel):
    ensemble_score:    float
    is_anomaly:        bool
    threshold:         float
    component_scores:  dict
    rule_violations:   dict
    frequency_detail:  dict
    features_received: dict


# ---------------------------------------------------------------------------
# Startup: pre-load / train models
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    print("[startup] Pre-loading ML models…")
    # This will train them if they don't exist on disk
    structural_detector._get_models()
    print("[startup] Ready.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "uptime_seconds": round(time.time() - _metrics["start_time"], 1)}


@app.get("/metrics")
def metrics():
    uptime = time.time() - _metrics["start_time"]
    total  = _metrics["total_analyzed"]
    return {
        **_metrics,
        "uptime_seconds":   round(uptime, 1),
        "block_rate":       round(_metrics["total_blocked"] / max(total, 1), 4),
        "config":           CONFIG,
        "block_threshold":  BLOCK_THRESHOLD,
    }


@app.post("/analyze", response_model=AnomalyReport)
async def analyze(vector: FeatureVector, request: Request):
    features = vector.model_dump()

    # 1. Structural score (Isolation Forest)
    struct_score = structural_detector.score(features)

    # 2. Frequency score (EWMA)
    client_ip  = vector.client_ip or (request.client.host if request.client else "unknown")
    freq_result = frequency_detector.record_and_score(client_ip)

    # 3. Rule score (deterministic)
    rule_result = rule_score(features, CONFIG)

    # 4. Ensemble report
    report = make_report(
        features=features,
        structural_score=struct_score,
        freq_result=freq_result,
        rule_result=rule_result,
        threshold=BLOCK_THRESHOLD,
        config=CONFIG,
    )

    # Update metrics
    _metrics["total_analyzed"] += 1
    if report["is_anomaly"]:
        _metrics["total_blocked"] += 1
    else:
        _metrics["total_passed"] += 1

    return report


@app.post("/train")
def retrain():
    """Force a retrain of all models (wipes existing .pkl files)."""
    import os, glob
    for f in glob.glob(os.path.join("models", "*.pkl")):
        os.remove(f)
    # Reset singleton
    structural_detector._iso = None
    structural_detector._scaler = None
    structural_detector._get_models()
    return {"status": "models retrained successfully"}
