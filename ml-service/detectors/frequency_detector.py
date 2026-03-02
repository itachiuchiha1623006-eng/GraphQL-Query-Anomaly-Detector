"""
frequency_detector.py

EWMA (Exponentially Weighted Moving Average) query frequency anomaly detector.

Per client IP we maintain a sliding window of request timestamps.
We compute the current request rate and compare it against an
EWMA baseline — a statistically significant spike triggers an anomaly.
"""

import time
import math
from collections import deque
from threading import Lock


class FrequencyDetector:
    """
    Thread-safe per-IP EWMA frequency anomaly detector.

    Parameters
    ----------
    window_seconds : int
        Rolling time window used to count requests.
    alpha : float
        EWMA smoothing factor. Lower = slower adaptation (stricter baseline).
    spike_multiplier : float
        How many times above EWMA baseline counts as anomalous.
    min_samples : int
        Minimum requests before anomaly detection kicks in (warm-up period).
    """

    def __init__(
        self,
        window_seconds: int = 60,
        alpha: float = 0.15,
        spike_multiplier: float = 3.5,
        min_samples: int = 10,
    ):
        self.window = window_seconds
        self.alpha = alpha
        self.spike_multiplier = spike_multiplier
        self.min_samples = min_samples

        # Per-IP state
        self._timestamps: dict[str, deque] = {}
        self._ewma: dict[str, float] = {}          # smoothed rate (req/min)
        self._total_count: dict[str, int] = {}
        self._lock = Lock()

    def _prune_window(self, ip: str, now: float):
        q = self._timestamps[ip]
        cutoff = now - self.window
        while q and q[0] < cutoff:
            q.popleft()

    def record_and_score(self, client_ip: str) -> dict:
        """
        Record a new request for client_ip and return:
          {
            "score": float [0,1],
            "current_rate": float,   # req/min in current window
            "ewma_baseline": float,  # smoothed baseline rate
            "total_requests": int,
          }
        """
        now = time.time()

        with self._lock:
            if client_ip not in self._timestamps:
                self._timestamps[client_ip] = deque()
                self._ewma[client_ip] = 0.0
                self._total_count[client_ip] = 0

            q = self._timestamps[client_ip]
            q.append(now)
            self._prune_window(client_ip, now)
            self._total_count[client_ip] += 1

            count = len(q)
            total = self._total_count[client_ip]

            # Current rate in requests-per-minute
            current_rate = count * (60.0 / self.window)

            # Update EWMA
            old_ewma = self._ewma[client_ip]
            new_ewma = (
                current_rate
                if old_ewma == 0
                else self.alpha * current_rate + (1 - self.alpha) * old_ewma
            )
            self._ewma[client_ip] = new_ewma

            # Not enough data yet → return safe score
            if total < self.min_samples:
                return {
                    "score": 0.0,
                    "current_rate": round(current_rate, 2),
                    "ewma_baseline": round(new_ewma, 2),
                    "total_requests": total,
                }

            # Anomaly score: how many times above baseline?
            baseline = max(new_ewma, 1.0)
            ratio = current_rate / baseline

            # Sigmoid-like mapping: ratio ≥ spike_multiplier → score ≈ 1.0
            # score = clamp((ratio - 1) / (spike_multiplier - 1), 0, 1)
            if ratio <= 1.0:
                anomaly_score = 0.0
            elif ratio >= self.spike_multiplier:
                anomaly_score = 1.0
            else:
                anomaly_score = (ratio - 1.0) / (self.spike_multiplier - 1.0)

            return {
                "score": round(anomaly_score, 4),
                "current_rate": round(current_rate, 2),
                "ewma_baseline": round(new_ewma, 2),
                "total_requests": total,
            }

    def reset_ip(self, client_ip: str):
        """Clear state for an IP (e.g. after a ban expires)."""
        with self._lock:
            self._timestamps.pop(client_ip, None)
            self._ewma.pop(client_ip, None)
            self._total_count.pop(client_ip, None)


# Module-level singleton shared across requests
_detector = FrequencyDetector()


def record_and_score(client_ip: str) -> dict:
    return _detector.record_and_score(client_ip)
