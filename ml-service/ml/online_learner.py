"""
online_learner.py
==================
Lightweight online learning buffer for real-time model improvement.

As the anomaly detector runs in production, every confirmed-normal query
(score < threshold) gets added to this buffer. When enough new samples
accumulate, the model is silently retrained on the expanded dataset —
continuously tightening its boundary around real traffic patterns.

Features:
  - Thread-safe ring buffer (max N samples)
  - Persists buffer across restarts (models/online_buffer.pkl)
  - Exposes should_retrain() to check if retraining is warranted
  - Background retraining so HTTP response latency is unaffected

Usage (in main.py):
  from ml.online_learner import learner
  learner.add_normal(features_dict)
  if learner.should_retrain():
      learner.trigger_retrain_background()
"""

from __future__ import annotations

import os
import threading
import time
import joblib
import pandas as pd
from collections import deque
from typing import Optional

from ml.training_data import FEATURE_COLUMNS

# ── Config ──────────────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
BUFFER_PATH = os.path.join(MODELS_DIR, 'online_buffer.pkl')

_DEFAULT_MAX_BUFFER      = 500   # Max confirmed-normal samples to keep
_DEFAULT_RETRAIN_TRIGGER = 100   # Retrain after this many new samples
_DEFAULT_MIN_BUFFER      = 50    # Don't retrain unless buffer has this many


class OnlineLearner:
    """
    Ring-buffer of confirmed-normal feature vectors.
    Thread-safe. Triggers background retraining when buffer grows enough.
    """

    def __init__(
        self,
        max_buffer: int = _DEFAULT_MAX_BUFFER,
        retrain_trigger: int = _DEFAULT_RETRAIN_TRIGGER,
        min_buffer: int = _DEFAULT_MIN_BUFFER,
    ):
        self._lock = threading.Lock()
        self._max_buffer      = max_buffer
        self._retrain_trigger = retrain_trigger
        self._min_buffer      = min_buffer

        # Ring buffer of feature dicts
        self._buffer: deque[dict] = deque(maxlen=max_buffer)

        # Track how many samples added since last retrain
        self._since_retrain: int = 0
        self._is_retraining: bool = False
        self._total_added: int = 0
        self._retrain_count: int = 0
        self._last_retrain: Optional[float] = None

        # Load persisted buffer if available
        self._load_buffer()

    # ── Public API ───────────────────────────────────────────────────────────

    def add_normal(self, features: dict) -> None:
        """
        Add a confirmed-normal query's feature vector to the buffer.
        Silently triggers background retraining if threshold is hit.

        Args:
            features: Dict with keys matching FEATURE_COLUMNS.
        """
        clean = {col: float(features.get(col, 0.0)) for col in FEATURE_COLUMNS}
        with self._lock:
            self._buffer.append(clean)
            self._since_retrain += 1
            self._total_added += 1

        # Trigger background retrain if needed (without holding lock)
        if self.should_retrain():
            self.trigger_retrain_background()

    def should_retrain(self) -> bool:
        """
        Returns True if enough new samples have accumulated to warrant retraining.
        """
        with self._lock:
            return (
                not self._is_retraining
                and len(self._buffer) >= self._min_buffer
                and self._since_retrain >= self._retrain_trigger
            )

    def trigger_retrain_background(self) -> bool:
        """
        Spawn a background thread to retrain the model.
        Returns True if retraining was started, False if already running.
        """
        with self._lock:
            if self._is_retraining:
                return False
            self._is_retraining = True

        thread = threading.Thread(
            target=self._retrain_worker,
            daemon=True,
            name="OnlineLearner-Retrain",
        )
        thread.start()
        print(f"[online_learner] 🔄 Background retrain triggered "
              f"(buffer={len(self._buffer)}, since_last={self._since_retrain})")
        return True

    def get_buffer_df(self) -> pd.DataFrame:
        """Return a snapshot of the current buffer as a DataFrame."""
        with self._lock:
            rows = list(self._buffer)
        if not rows:
            return pd.DataFrame(columns=FEATURE_COLUMNS)
        return pd.DataFrame(rows, columns=FEATURE_COLUMNS)

    def status(self) -> dict:
        """Return current buffer stats as a dict (for API exposure)."""
        with self._lock:
            return {
                "buffer_size":    len(self._buffer),
                "max_buffer":     self._max_buffer,
                "since_retrain":  self._since_retrain,
                "retrain_trigger":self._retrain_trigger,
                "total_added":    self._total_added,
                "retrain_count":  self._retrain_count,
                "is_retraining":  self._is_retraining,
                "last_retrain":   self._last_retrain,
            }

    # ── Internal ─────────────────────────────────────────────────────────────

    def _retrain_worker(self) -> None:
        """Background worker: blends online buffer with corpus+synthetic and retrains."""
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

            print("[online_learner] Starting model retraining with online buffer…")
            t0 = time.perf_counter()

            # Get snapshot of buffer
            buffer_df = self.get_buffer_df()
            buffer_df["label"] = 0

            # Run the standard training pipeline (it already blends corpus+synthetic)
            from ml.trainer import train_isolation_forest
            train_isolation_forest(verbose=False)

            # Persist buffer
            self._save_buffer()

            elapsed = time.perf_counter() - t0
            with self._lock:
                self._since_retrain = 0
                self._retrain_count += 1
                self._last_retrain = time.time()

            print(f"[online_learner] ✅ Retraining complete in {elapsed:.1f}s "
                  f"(total retrains: {self._retrain_count})")

        except Exception as e:
            print(f"[online_learner] ❌ Retraining failed: {e}")
        finally:
            with self._lock:
                self._is_retraining = False

    def _save_buffer(self) -> None:
        """Persist buffer to disk."""
        try:
            os.makedirs(MODELS_DIR, exist_ok=True)
            with self._lock:
                snapshot = list(self._buffer)
            joblib.dump({
                "buffer": snapshot,
                "since_retrain": self._since_retrain,
                "total_added":   self._total_added,
                "retrain_count": self._retrain_count,
            }, BUFFER_PATH)
        except Exception as e:
            print(f"[online_learner] WARNING: Could not save buffer: {e}")

    def _load_buffer(self) -> None:
        """Load persisted buffer from disk if available."""
        if not os.path.exists(BUFFER_PATH):
            return
        try:
            data = joblib.load(BUFFER_PATH)
            rows = data.get("buffer", [])
            # Reload into deque respecting maxlen
            for row in rows[-self._max_buffer:]:
                self._buffer.append(row)
            self._since_retrain = data.get("since_retrain", 0)
            self._total_added   = data.get("total_added", 0)
            self._retrain_count = data.get("retrain_count", 0)
            print(f"[online_learner] Loaded {len(self._buffer)} buffered samples from disk.")
        except Exception as e:
            print(f"[online_learner] WARNING: Could not load buffer: {e}")


# ── Singleton instance ───────────────────────────────────────────────────────

learner = OnlineLearner()
