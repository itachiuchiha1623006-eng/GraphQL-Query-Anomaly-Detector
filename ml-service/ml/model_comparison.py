"""
model_comparison.py
====================
Trains and evaluates 5 ML models on the labeled GraphQL anomaly dataset.
Outputs a comparison table: precision, recall, F1, ROC-AUC, latency.

Models:
  1. Isolation Forest      (unsupervised — current baseline)
  2. Local Outlier Factor  (density-based, good at boundary cases)
  3. One-Class SVM         (kernel boundary around normal class)
  4. Elliptic Envelope     (Gaussian assumption on normal data)
  5. Random Forest         (supervised — uses labels, highest accuracy)

Usage:
  python ml/model_comparison.py
  python ml/model_comparison.py --save   # also pickles the best model
"""

import os, sys, time, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, roc_auc_score
)

from ml.training_data import generate_blended_normal, generate_normal_samples, FEATURE_COLUMNS
from ml.attack_generator import generate_all_attacks

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Build dataset ─────────────────────────────────────────────────────────────

def build_dataset():
    try:
        normal_df = generate_blended_normal(n_total=3000, corpus_ratio=0.60)
    except Exception as e:
        print(f'[comparison] WARNING: corpus blend failed ({e}) — using pure synthetic.')
        normal_df = generate_normal_samples(n=3000)
    normal_df['label'] = 0

    attack_rows = generate_all_attacks()
    attack_df   = pd.DataFrame(attack_rows)
    attack_df['label'] = 1

    df = pd.concat([normal_df, attack_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# ── Train & evaluate one unsupervised model ───────────────────────────────────

def eval_unsupervised(model, X_train_scaled, X_test_scaled, y_test, name):
    """Fit on normal-only train split, predict on mixed test set."""
    t0 = time.perf_counter()
    model.fit(X_train_scaled)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    raw_preds = model.predict(X_test_scaled)   # +1=normal, -1=anomaly
    infer_time = (time.perf_counter() - t0) / len(y_test) * 1000  # ms/query

    preds = (raw_preds == -1).astype(int)

    # decision score: higher = more anomalous
    if hasattr(model, 'decision_function'):
        scores = -model.decision_function(X_test_scaled)
    elif hasattr(model, 'score_samples'):
        scores = -model.score_samples(X_test_scaled)
    else:
        scores = preds.astype(float)

    try:
        auc = roc_auc_score(y_test, scores)
    except Exception:
        auc = float('nan')

    return {
        'model':      name,
        'type':       'unsupervised',
        'precision':  round(precision_score(y_test, preds, zero_division=0), 4),
        'recall':     round(recall_score(y_test, preds, zero_division=0), 4),
        'f1':         round(f1_score(y_test, preds, zero_division=0), 4),
        'roc_auc':    round(auc, 4),
        'train_ms':   round(train_time * 1000, 1),
        'infer_ms':   round(infer_time, 4),
        '_model_obj': model,
        '_scaler':    None,
    }

# ── Train & evaluate Random Forest (supervised) ───────────────────────────────

def eval_random_forest(X_train, y_train, X_test, y_test, scaler):
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    X_train_sc = scaler.transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    t0 = time.perf_counter()
    rf.fit(X_train_sc, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds  = rf.predict(X_test_sc)
    infer_time = (time.perf_counter() - t0) / len(y_test) * 1000

    scores = rf.predict_proba(X_test_sc)[:, 1]
    auc = roc_auc_score(y_test, scores)

    return {
        'model':      'Random Forest',
        'type':       'supervised',
        'precision':  round(precision_score(y_test, preds, zero_division=0), 4),
        'recall':     round(recall_score(y_test, preds, zero_division=0), 4),
        'f1':         round(f1_score(y_test, preds, zero_division=0), 4),
        'roc_auc':    round(auc, 4),
        'train_ms':   round(train_time * 1000, 1),
        'infer_ms':   round(infer_time, 4),
        '_model_obj': rf,
        '_scaler':    scaler,
    }

# ── Main comparison ───────────────────────────────────────────────────────────

def run_comparison(save_best=False, verbose=True):
    if verbose:
        print('\n[comparison] Building dataset…')
    df = build_dataset()

    X = df[FEATURE_COLUMNS].values.astype(float)
    y = df['label'].values

    # Split: we need a normal-only train set for unsupervised models
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    X_train_normal = X_train_all[y_train_all == 0]

    # Scaler fitted on normal training data only
    scaler = StandardScaler()
    X_train_normal_sc = scaler.fit_transform(X_train_normal)
    X_test_sc = scaler.transform(X_test)

    contamination = float(np.clip(y.mean(), 0.01, 0.25))

    # ── Define unsupervised models ────────────────────────────────────────────
    unsupervised = [
        ('Isolation Forest',    IsolationForest(n_estimators=300, contamination=contamination, random_state=42, n_jobs=-1)),
        ('Local Outlier Factor',LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)),
        ('One-Class SVM',       OneClassSVM(kernel='rbf', nu=contamination, gamma='scale')),
        ('Elliptic Envelope',   EllipticEnvelope(contamination=contamination, random_state=42)),
    ]

    results = []
    for name, model in unsupervised:
        if verbose:
            print(f'[comparison] Training {name}…')
        r = eval_unsupervised(model, X_train_normal_sc, X_test_sc, y_test, name)
        results.append(r)

    if verbose:
        print('[comparison] Training Random Forest (supervised)…')
    rf_result = eval_random_forest(X_train_all, y_train_all, X_test, y_test, scaler)
    results.append(rf_result)

    # ── Print table ───────────────────────────────────────────────────────────
    if verbose:
        _print_table(results)

    # ── Pick best model (highest F1) ──────────────────────────────────────────
    best = max(results, key=lambda r: r['f1'])
    if verbose:
        print(f'\n[comparison] 🏆 Best model: {best["model"]}  (F1={best["f1"]:.4f})')

    if save_best:
        _save_best(best, scaler)

    # Strip private keys before returning
    clean = [{k: v for k, v in r.items() if not k.startswith('_')} for r in results]
    return clean, best['model']


def _print_table(results):
    cols = ['model', 'type', 'precision', 'recall', 'f1', 'roc_auc', 'train_ms', 'infer_ms']
    header = f"{'Model':<24} {'Type':<14} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>7} {'Train ms':>10} {'Infer ms':>10}"
    print('\n' + '='*90)
    print('  MODEL COMPARISON — GraphQL Anomaly Detection')
    print('='*90)
    print('  ' + header)
    print('  ' + '-'*86)
    for r in results:
        bar = '●' * int(r['f1'] * 20)
        print(f"  {r['model']:<24} {r['type']:<14} {r['precision']:>6.4f} {r['recall']:>6.4f} "
              f"{r['f1']:>6.4f} {r['roc_auc']:>7.4f} {r['train_ms']:>10.1f} {r['infer_ms']:>10.4f}  {bar}")
    print('='*90)


def _save_best(best_result, default_scaler):
    model_obj = best_result['_model_obj']
    scaler    = best_result['_scaler'] or default_scaler
    model_name = best_result['model'].lower().replace(' ', '_')

    joblib.dump(model_obj, os.path.join(MODELS_DIR, 'best_model.pkl'))
    joblib.dump(scaler,    os.path.join(MODELS_DIR, 'best_scaler.pkl'))
    joblib.dump(model_name, os.path.join(MODELS_DIR, 'best_model_name.pkl'))

    print(f'[comparison] ✅ Saved best model ({best_result["model"]}) → models/best_model.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true', help='Save best model to models/')
    args = parser.parse_args()
    run_comparison(save_best=args.save, verbose=True)
