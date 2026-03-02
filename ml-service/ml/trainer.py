"""
trainer.py
Trains and persists all ML models.

Models saved to ../models/:
  - isolation_forest.pkl   (structural anomaly detector)
  - scaler.pkl             (StandardScaler for IF features)
  - label_encoder.pkl      (unused but kept for pipeline completeness)
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from ml.training_data import get_full_dataset, generate_normal_samples, FEATURE_COLUMNS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def train_isolation_forest():
    """
    Isolation Forest is trained on NORMAL data only (unsupervised / one-class).
    It learns the shape of normal queries and flags anything that looks different.
    contamination=0.05 means we expect ~5% of production traffic to be anomalous.
    """
    print("[trainer] Generating training data…")
    normal_df = generate_normal_samples(n=3000)
    X_normal = normal_df[FEATURE_COLUMNS].values.astype(float)

    print("[trainer] Fitting StandardScaler…")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_normal)

    print("[trainer] Training Isolation Forest…")
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_scaled)

    # Quick sanity: score normal data — majority should be +1 (inlier)
    preds = iso.predict(X_scaled)
    inlier_rate = (preds == 1).mean()
    print(f"[trainer] Isolation Forest inlier rate on training set: {inlier_rate:.2%}")

    # Persist
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(iso, os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    print(f"[trainer] Models saved to {MODELS_DIR}")
    return iso, scaler


def load_models():
    """Load persisted models or train them if they don't exist."""
    iso_path = os.path.join(MODELS_DIR, "isolation_forest.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

    if not os.path.exists(iso_path) or not os.path.exists(scaler_path):
        print("[trainer] No saved models found – training now…")
        return train_isolation_forest()

    iso = joblib.load(iso_path)
    scaler = joblib.load(scaler_path)
    print("[trainer] Loaded pre-trained models from disk.")
    return iso, scaler


if __name__ == "__main__":
    train_isolation_forest()
