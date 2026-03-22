from __future__ import annotations


def get_pick_tier(score: float) -> str:
    if score >= 72:
        return "ELITE"
    if score >= 66:
        return "PREMIUM"
    if score >= 60:
        return "STRONG"
    if score >= 54:
        return "NORMAL"
    return "PASS"


def recommendation_score(calibrated_prob: float, reliability: float = 1.0) -> float:
    p = min(max(float(calibrated_prob), 1e-6), 1 - 1e-6)
    rel = min(max(float(reliability), 0.0), 1.0)
    edge = abs(p - 0.5) * 2.0
    # Score in [50, 100], shrunk by reliability.
    return 50.0 + (edge * 50.0 * rel)


def fuse_with_pattern_score(base_score: float, pattern_edge: float, pattern_weight: float = 8.0) -> float:
    base = float(base_score)
    edge = float(min(max(pattern_edge, -1.0), 1.0))
    fused = base + (edge * pattern_weight)
    return float(min(max(fused, 50.0), 99.9))


def should_recommend(score: float, min_score: float = 56.0) -> bool:
    return float(score) >= float(min_score)
