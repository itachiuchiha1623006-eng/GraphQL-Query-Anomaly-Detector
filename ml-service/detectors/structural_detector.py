"""
structural_detector.py

Isolation Forest –based structural anomaly detector.
Covers: abnormal depth, excessive nesting, field explosion,
        alias abuse, high resolver cost, data exfiltration.

The model is trained (once) on synthetic normal query features.
"""

import os
import numpy as np
from ml.trainer import load_models
from ml.training_data import FEATURE_COLUMNS

# Singleton – load once on module import
_iso, _scaler = None, None


def _get_models():
    global _iso, _scaler
    if _iso is None:
        _iso, _scaler = load_models()
    return _iso, _scaler


def score(features: dict) -> float:
    """
    Returns a structural anomaly score in [0, 1].
    0 = perfectly normal, 1 = highly anomalous.

    Isolation Forest returns:
      +1  → inlier  (normal)
      -1  → outlier (anomaly)
    and a raw decision_function score (higher = more normal).
    We convert this into a [0,1] anomaly score.
    """
    iso, scaler = _get_models()

    # Build feature vector in correct column order
    vec = np.array([[features.get(col, 0) for col in FEATURE_COLUMNS]], dtype=float)
    vec_scaled = scaler.transform(vec)

    # decision_function: negative scores are anomalous; positive are normal.
    # Typical range is roughly [-0.5, 0.5].
    raw = iso.decision_function(vec_scaled)[0]  # scalar

    # Normalize to [0,1] — clamp to [-1, 1] then invert
    clipped = float(np.clip(raw, -1.0, 1.0))
    anomaly_score = (1.0 - clipped) / 2.0  # +1 → 0.0 (normal), -1 → 1.0 (anomalous)
    return round(anomaly_score, 4)


def is_anomaly(features: dict, threshold: float = 0.6) -> bool:
    return score(features) >= threshold
