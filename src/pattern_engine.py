from __future__ import annotations

from typing import Dict, List


Pattern = Dict[str, object]


def make_pattern(name: str, strength: float, direction: str, reliability: float, reason: str) -> Pattern:
    return {
        "name": name,
        "strength": float(max(0.0, min(1.0, strength))),
        "direction": direction,
        "reliability": float(max(0.0, min(1.0, reliability))),
        "reason": reason,
    }


def aggregate_pattern_edge(patterns: List[Pattern]) -> float:
    edge = 0.0
    for p in patterns:
        strength = float(p.get("strength", 0.0))
        reliability = float(p.get("reliability", 0.0))
        direction = str(p.get("direction", "neutral")).lower()
        signed = 0.0
        if direction == "positive":
            signed = strength
        elif direction == "negative":
            signed = -strength
        edge += signed * reliability
    return float(max(-1.0, min(1.0, edge)))
