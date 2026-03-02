"""
scorer.py
Weighted ensemble scorer combining Isolation Forest + EWMA + three rule checks.
"""

from __future__ import annotations

WEIGHTS = {
    "structural": 0.55,   # Isolation Forest (covers depth, fields, cost, exfiltration)
    "frequency":  0.25,   # EWMA per-IP rate
    "rules":      0.20,   # Rule-based hard checks (depth, alias, introspection)
}


def rule_score(features: dict, config: dict) -> dict:
    """
    Returns a [0,1] rule score and a dict of violated rules.
    Rules are deterministic — no ML involved.
    """
    violations = {}
    max_depth   = features.get("max_depth", 0)
    alias_count = features.get("alias_count", 0)
    intro_count = features.get("introspection_count", 0)

    if max_depth > config.get("max_query_depth", 7):
        violations["depth"] = f"max_depth={max_depth} > {config.get('max_query_depth', 7)}"

    if alias_count > config.get("max_alias_count", 10):
        violations["alias"] = f"alias_count={alias_count} > {config.get('max_alias_count', 10)}"

    if config.get("block_introspection", False) and intro_count > 0:
        violations["introspection"] = f"introspection_count={intro_count}"

    # Partial scoring: each rule violated adds to the score
    score = min(len(violations) / 3.0, 1.0)
    return {"score": round(score, 4), "violations": violations}


def ensemble_score(
    structural: float,
    frequency: float,
    rules: float,
) -> float:
    """
    Weighted average of all three component scores.
    Returns a float in [0, 1].
    """
    score = (
        WEIGHTS["structural"] * structural
        + WEIGHTS["frequency"] * frequency
        + WEIGHTS["rules"]     * rules
    )
    return round(min(score, 1.0), 4)


def make_report(
    features: dict,
    structural_score: float,
    freq_result: dict,
    rule_result: dict,
    threshold: float,
    config: dict,
) -> dict:
    """Build the full JSON report returned to the JS middleware."""
    freq_score = freq_result["score"]
    rule_scr   = rule_result["score"]
    final      = ensemble_score(structural_score, freq_score, rule_scr)

    return {
        "ensemble_score": final,
        "is_anomaly": final >= threshold,
        "threshold": threshold,
        "component_scores": {
            "structural": structural_score,
            "frequency":  freq_score,
            "rules":      rule_scr,
        },
        "rule_violations": rule_result["violations"],
        "frequency_detail": {
            "current_rate_per_min": freq_result["current_rate"],
            "ewma_baseline":        freq_result["ewma_baseline"],
            "total_requests":       freq_result["total_requests"],
        },
        "features_received": features,
    }
