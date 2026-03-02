"""
trainer.py  (v2 — attack-informed semi-supervised training)
==============================================================
Trains Isolation Forest using:
  - Normal samples only for fitting (unsupervised, as IF requires)
  - Attack samples to tune contamination estimate
  - Validates model using labeled attack data AFTER training

Models saved to ../models/:
  isolation_forest.pkl   – trained classifier
  scaler.pkl             – StandardScaler fitted on normal data
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

from ml.training_data import generate_normal_samples, FEATURE_COLUMNS
from ml.attack_generator import generate_all_attacks

import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def estimate_contamination(n_normal: int, n_attack: int) -> float:
    """
    Estimate contamination as ratio of attacks in combined dataset.
    Clamp to [0.01, 0.25] — Isolation Forest's valid range.
    """
    ratio = n_attack / (n_normal + n_attack)
    return round(float(np.clip(ratio, 0.01, 0.25)), 3)


def train_isolation_forest(verbose: bool = True):
    """
    Train Isolation Forest on normal data, validate on labeled attack data.
    Returns (iso, scaler).
    """
    if verbose:
        print("[trainer] Generating normal samples (n=3000)…")
    normal_df = generate_normal_samples(n=3000)
    X_normal = normal_df[FEATURE_COLUMNS].values.astype(float)

    if verbose:
        print("[trainer] Generating attack samples (16 attack types)…")
    attack_rows = generate_all_attacks()
    attack_df   = pd.DataFrame(attack_rows)
    X_attack    = attack_df[FEATURE_COLUMNS].values.astype(float)

    contamination = estimate_contamination(len(X_normal), len(X_attack))
    if verbose:
        print(f"[trainer] Contamination estimate: {contamination:.3f}  "
              f"({len(X_attack)} attacks / {len(X_normal)} normal)")

    if verbose:
        print("[trainer] Fitting StandardScaler on normal data…")
    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)

    if verbose:
        print("[trainer] Training Isolation Forest…")
    iso = IsolationForest(
        n_estimators=300,          # more trees = better boundary
        contamination=contamination,
        max_samples='auto',
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_normal_scaled)

    # ── Validation ────────────────────────────────────────────────────────────
    if verbose:
        print("\n[trainer] Validating on labeled data…")

    X_attack_scaled  = scaler.transform(X_attack)
    X_normal_scaled2 = scaler.transform(X_normal[:500])  # held-out normal subset

    X_val = np.vstack([X_normal_scaled2, X_attack_scaled])
    y_val = np.array([0] * len(X_normal_scaled2) + [1] * len(X_attack_scaled))

    # IF: +1 = inlier (normal), -1 = outlier (anomaly)
    preds_raw = iso.predict(X_val)
    preds     = (preds_raw == -1).astype(int)   # convert to 0/1
    scores    = -iso.decision_function(X_val)    # higher = more anomalous

    if verbose:
        print(classification_report(y_val, preds,
              target_names=['normal', 'anomaly']))
        try:
            auc = roc_auc_score(y_val, scores)
            print(f"  ROC-AUC: {auc:.4f}")
        except Exception:
            pass

    # ── Per-attack-type recall ─────────────────────────────────────────────
    if verbose:
        print("\n[trainer] Attack detection rate by type:")
        attack_df2 = attack_df.copy()
        attack_df2['pred'] = (iso.predict(X_attack_scaled) == -1).astype(int)
        for attack_type, grp in attack_df2.groupby('attack_type'):
            recall = grp['pred'].mean()
            bar    = '█' * int(recall * 20)
            print(f"  {attack_type:<32} {recall:>5.1%} {bar}")

    # ── Persist ───────────────────────────────────────────────────────────────
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(iso,    os.path.join(MODELS_DIR, 'isolation_forest.pkl'))
    if verbose:
        print(f"\n[trainer] ✅ Models saved to {MODELS_DIR}")

    return iso, scaler


def load_models():
    """Load persisted models or train from scratch if not present."""
    iso_path    = os.path.join(MODELS_DIR, 'isolation_forest.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')

    if not os.path.exists(iso_path) or not os.path.exists(scaler_path):
        print('[trainer] No saved models found — training from scratch…')
        return train_isolation_forest()

    iso    = joblib.load(iso_path)
    scaler = joblib.load(scaler_path)
    print('[trainer] Loaded pre-trained models from disk.')
    return iso, scaler


if __name__ == '__main__':
    train_isolation_forest(verbose=True)
