"""
structural_detector.py  (v2 — multi-model support)
====================================================
Loads the best model selected by model_comparison.py (or falls back to
Isolation Forest). Scores features returning a normalized [0,1] anomaly score.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import joblib

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# ── Load models at import time ────────────────────────────────────────────────

def _load():
    # Prefer best_model (chosen by comparison), fall back to isolation_forest
    for name, mfile, sfile in [
        ('best',      'best_model.pkl',       'best_scaler.pkl'),
        ('isolation', 'isolation_forest.pkl', 'scaler.pkl'),
    ]:
        mpath = os.path.join(MODELS_DIR, mfile)
        spath = os.path.join(MODELS_DIR, sfile)
        if os.path.exists(mpath) and os.path.exists(spath):
            model  = joblib.load(mpath)
            scaler = joblib.load(spath)
            # Read model name if stored
            npath = os.path.join(MODELS_DIR, 'best_model_name.pkl')
            model_name = joblib.load(npath) if os.path.exists(npath) else name
            print(f'[structural_detector] Loaded model: {model_name}')
            return model, scaler, model_name
    return None, None, None

_model, _scaler, _model_name = _load()

FEATURE_ORDER = [
    'max_depth', 'total_fields', 'unique_fields', 'alias_count',
    'introspection_count', 'fragment_count', 'estimated_cost',
    'payload_size', 'field_entropy', 'nesting_variance',
]


def score(features: dict) -> float:
    """
    Returns anomaly score in [0, 1]. Higher = more anomalous.
    Works with both unsupervised (IF/LOF/SVM) and supervised (RF) models.
    """
    global _model, _scaler, _model_name
    if _model is None:
        # Models not trained yet — return neutral score
        return 0.5

    vec = np.array([[features.get(f, 0.0) for f in FEATURE_ORDER]], dtype=float)
    vec_scaled = _scaler.transform(vec)

    # Supervised model (Random Forest) → use predict_proba directly
    if hasattr(_model, 'predict_proba'):
        proba = _model.predict_proba(vec_scaled)[0]
        anomaly_class_idx = list(_model.classes_).index(1) if 1 in _model.classes_ else 1
        return round(float(proba[anomaly_class_idx]), 4)

    # Unsupervised models → convert decision_function to [0,1]
    if hasattr(_model, 'decision_function'):
        raw = float(_model.decision_function(vec_scaled)[0])
        # decision_function: lower = more anomalous (IF uses negative)
        # normalize: clip to [-0.5, 0.5] then map to [0, 1]
        normalized = float(np.clip((-raw + 0.3) / 0.6, 0.0, 1.0))
        return round(normalized, 4)

    if hasattr(_model, 'score_samples'):
        raw = float(_model.score_samples(vec_scaled)[0])
        normalized = float(np.clip((-raw - (-10)) / 15, 0.0, 1.0))
        return round(normalized, 4)

    # Fallback: binary
    pred = _model.predict(vec_scaled)[0]
    return 1.0 if pred == -1 else 0.0


def reload_models():
    """Called by /train endpoint to hot-reload after retraining."""
    global _model, _scaler, _model_name
    _model, _scaler, _model_name = _load()


def get_model_name() -> str:
    return _model_name or 'unknown'
