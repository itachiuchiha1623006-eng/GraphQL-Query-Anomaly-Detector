"""
frequency_detector.py  (v2 — bucket-based EWMA)

Root cause of v1 failure:
  Per-request EWMA updates let the baseline *chase* incoming traffic.
  A gradual flood of 80 req in 4s looks normal because baseline rises with it.

Fix:
  Bucket the time axis into 1-second slots.
  EWMA is updated once per second on the HISTORICAL bucket counts.
  Current rate = requests in the LAST bucket.
  Spike = current rate >> EWMA of past N buckets.

This correctly flags both:
  - Sudden bursts  (attacker jumps from 0→100 rpm instantly)
  - Gradual floods (30-second ramp-up causes bucket counts to diverge from ewma)
"""

import time
from collections import defaultdict, deque
from threading import Lock

# ── Config defaults ──────────────────────────────────────────────────────────
BUCKET_SECONDS   = 1      # width of each time bucket (seconds)
HISTORY_BUCKETS  = 30     # how many past buckets to keep for EWMA baseline
EWMA_ALPHA       = 0.20   # EWMA smoothing (lower = slower adaptation = stricter)
SPIKE_MULTIPLIER = 2.5    # current_bucket >= SPIKE_MULTIPLIER * ewma_baseline → anomaly
MIN_HISTORY      = 2      # minimum buckets before detection activates (warm-up)


class FrequencyDetector:
    """
    Thread-safe per-IP bucket-based EWMA frequency anomaly detector.
    """

    def __init__(
        self,
        bucket_seconds: float = BUCKET_SECONDS,
        history_buckets: int  = HISTORY_BUCKETS,
        alpha: float          = EWMA_ALPHA,
        spike_multiplier: float = SPIKE_MULTIPLIER,
        min_history: int      = MIN_HISTORY,
    ):
        self.bucket_sec      = bucket_seconds
        self.history         = history_buckets
        self.alpha           = alpha
        self.spike_mult      = spike_multiplier
        self.min_history     = min_history

        # Per-IP: deque of (bucket_key, count) pairs — bucket_key = int(ts // bucket_sec)
        self._buckets: dict[str, deque]  = {}
        # Per-IP: current EWMA value
        self._ewma:    dict[str, float]  = {}
        self._lock = Lock()

    def _bucket_key(self, ts: float) -> int:
        return int(ts // self.bucket_sec)

    def _current_bucket(self, ip: str, now_key: int) -> int:
        """Count requests in the CURRENT bucket."""
        dq = self._buckets.get(ip, deque())
        return sum(c for k, c in dq if k == now_key)

    def _update_ewma(self, ip: str, now_key: int):
        """
        Compute EWMA over all HISTORICAL buckets (excluding current one).
        EWMA advances once per second; gaps (no traffic) count as 0.
        """
        dq = self._buckets.get(ip, deque())
        if not dq:
            return 0.0

        # Build a dict of historical bucket counts (exclude current bucket)
        hist = {k: c for k, c in dq if k < now_key}
        if not hist:
            return self._ewma.get(ip, 0.0)

        # Fill any missing buckets with 0 (inactivity periods)
        oldest = min(hist)
        newest = max(hist)
        ewma = self._ewma.get(ip, 0.0)

        for key in range(oldest, newest + 1):
            count = hist.get(key, 0)
            if ewma == 0.0 and count > 0:
                ewma = float(count)   # cold start
            else:
                ewma = self.alpha * count + (1 - self.alpha) * ewma

        self._ewma[ip] = ewma
        return ewma

    def record_and_score(self, client_ip: str) -> dict:
        """
        Record this request and return anomaly score.
        """
        now     = time.time()
        now_key = self._bucket_key(now)

        with self._lock:
            if client_ip not in self._buckets:
                self._buckets[client_ip] = deque()
                self._ewma[client_ip]    = 0.0

            dq = self._buckets[client_ip]

            # Prune buckets older than history window
            cutoff = now_key - self.history
            while dq and dq[0][0] < cutoff:
                dq.popleft()

            # Add / increment current bucket
            if dq and dq[-1][0] == now_key:
                k, c = dq.pop()
                dq.append((k, c + 1))
            else:
                dq.append((now_key, 1))

            # Update EWMA on historical buckets
            ewma = self._update_ewma(client_ip, now_key)

            # Current burst = requests in the current 1-second slot
            current_count = self._current_bucket(client_ip, now_key)

            # Total requests seen so far (across all buckets)
            total = sum(c for _, c in dq)
            distinct_buckets = len(set(k for k, _ in dq if k < now_key))

            # Not enough history yet → return safe score
            if distinct_buckets < self.min_history:
                return {
                    "score": 0.0,
                    "current_bucket_count": current_count,
                    "ewma_baseline": round(ewma, 2),
                    "total_requests": total,
                    "warmup": True,
                }

            # ── Anomaly scoring ──────────────────────────────────────────────
            baseline = max(ewma, 0.5)   # floor at 0.5 to avoid division by zero
            ratio    = current_count / baseline

            if ratio <= 1.0:
                anomaly_score = 0.0
            elif ratio >= self.spike_mult:
                anomaly_score = 1.0
            else:
                # Linear interpolation between 1x and spike_mult
                anomaly_score = (ratio - 1.0) / (self.spike_mult - 1.0)

            return {
                "score":               round(anomaly_score, 4),
                "current_bucket_count": current_count,
                "ewma_baseline":        round(ewma, 2),
                "total_requests":       total,
                "warmup":               False,
            }

    def reset_ip(self, client_ip: str):
        with self._lock:
            self._buckets.pop(client_ip, None)
            self._ewma.pop(client_ip, None)


# ── Module-level singleton ────────────────────────────────────────────────────
_detector = FrequencyDetector()


def record_and_score(client_ip: str) -> dict:
    return _detector.record_and_score(client_ip)
