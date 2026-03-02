"""
tests/test_detectors.py
pytest tests for the ML service detectors and scorer.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import time
from detectors import structural_detector, frequency_detector
from scorer import rule_score, ensemble_score



# ── Helpers ────────────────────────────────────────────────────────────────

def normal_features(**overrides):
    base = {
        "max_depth": 3,
        "total_fields": 5,
        "unique_fields": 5,
        "alias_count": 0,
        "introspection_count": 0,
        "fragment_count": 0,
        "estimated_cost": 15,
        "payload_size": 80,
        "field_entropy": 1.5,
        "nesting_variance": 0.3,
    }
    base.update(overrides)
    return base


# ── Structural Detector (Isolation Forest) ─────────────────────────────────

class TestStructuralDetector:
    def test_normal_query_low_score(self):
        score = structural_detector.score(normal_features())
        assert score < 0.65, f"Normal query got unexpectedly high score: {score}"

    def test_deep_query_high_score(self):
        score = structural_detector.score(normal_features(
            max_depth=20, estimated_cost=500, nesting_variance=7.0
        ))
        assert score > 0.50, f"Deep query should score higher: {score}"

    def test_field_explosion_high_score(self):
        score = structural_detector.score(normal_features(
            total_fields=150, unique_fields=140, estimated_cost=400, field_entropy=6.5
        ))
        assert score > 0.50, f"Field explosion should score higher: {score}"

    def test_score_in_range(self):
        score = structural_detector.score(normal_features())
        assert 0.0 <= score <= 1.0


# ── Frequency Detector (EWMA) ───────────────────────────────────────────────

# ── Frequency Detector (bucket-based EWMA) ─────────────────────────────────

class TestFrequencyDetector:
    def test_warmup_returns_zero(self):
        from detectors.frequency_detector import FrequencyDetector
        det = FrequencyDetector(min_history=5)
        # Only 3 distinct-second buckets → still in warmup
        for _ in range(3):
            result = det.record_and_score("192.168.1.100")
            time.sleep(1.01)
        assert result["score"] == 0.0, "Should return 0 during warm-up"
        assert result["warmup"] is True

    def test_normal_traffic_after_warmup_low_score(self):
        from detectors.frequency_detector import FrequencyDetector
        det = FrequencyDetector(min_history=5, spike_multiplier=2.5)
        # Send 1 req/sec for 6 seconds (normal traffic)
        for _ in range(6):
            result = det.record_and_score("10.0.0.2")
            time.sleep(1.01)
        assert result["score"] < 0.5, f"Steady normal traffic should not be anomalous: {result}"

    def test_burst_after_baseline_is_detected(self):
        from detectors.frequency_detector import FrequencyDetector
        import time as t
        det = FrequencyDetector(min_history=5, spike_multiplier=2.5, alpha=0.2)
        # Warm up with slow traffic: 1 req/sec for 6 seconds → establish low baseline
        for _ in range(6):
            det.record_and_score("attacker-ip")
            t.sleep(1.01)
        # Now burst 20 requests in the same 1-second bucket
        last = None
        for _ in range(20):
            last = det.record_and_score("attacker-ip")
        assert last["score"] > 0.0, f"Burst after slow baseline should be anomalous: {last}"
        assert last["warmup"] is False

    def test_different_ips_isolated(self):
        from detectors.frequency_detector import FrequencyDetector
        det = FrequencyDetector(min_history=3)
        for _ in range(3):
            det.record_and_score("1.1.1.1")
            time.sleep(1.01)
        result = det.record_and_score("2.2.2.2")  # brand new IP
        assert result["score"] == 0.0, "New IP should start fresh"



# ── Rule Scorer ─────────────────────────────────────────────────────────────

class TestRuleScorer:
    CFG = {"max_query_depth": 7, "max_alias_count": 10, "block_introspection": True}

    def test_no_violations(self):
        res = rule_score(normal_features(), self.CFG)
        assert res["score"] == 0.0
        assert len(res["violations"]) == 0

    def test_depth_violation(self):
        res = rule_score(normal_features(max_depth=15), self.CFG)
        assert "depth" in res["violations"]
        assert res["score"] > 0.0

    def test_alias_violation(self):
        res = rule_score(normal_features(alias_count=20), self.CFG)
        assert "alias" in res["violations"]

    def test_introspection_violation(self):
        res = rule_score(normal_features(introspection_count=3), self.CFG)
        assert "introspection" in res["violations"]

    def test_all_violations_max_score(self):
        res = rule_score(
            normal_features(max_depth=20, alias_count=50, introspection_count=5),
            self.CFG
        )
        assert res["score"] == 1.0


# ── Ensemble Scorer ──────────────────────────────────────────────────────────

class TestEnsembleScorer:
    def test_all_zero(self):
        assert ensemble_score(0.0, 0.0, 0.0) == 0.0

    def test_all_one(self):
        assert ensemble_score(1.0, 1.0, 1.0) == 1.0

    def test_weighted_average(self):
        # structural=1.0, rest=0 → should be 0.55
        score = ensemble_score(1.0, 0.0, 0.0)
        assert abs(score - 0.55) < 0.01

    def test_in_range(self):
        score = ensemble_score(0.7, 0.3, 0.5)
        assert 0.0 <= score <= 1.0
