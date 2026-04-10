from __future__ import annotations

from pattern_engine import make_pattern


def generate_ligue1_patterns(feature_row: dict) -> list[dict]:
    patterns = []

    draw_pressure = float(feature_row.get("draw_balance_score", feature_row.get("draw_equilibrium_index", 0)))
    if draw_pressure > 0.6:
        patterns.append(make_pattern("draw_equilibrium", 0.05, "negative", 0.6, "Balanced matchup increases draw risk."))

    if "over_environment_score" in feature_row:
        over_env = float(feature_row.get("over_environment_score", 0))
    else:
        # Fallback proxy from available ligue1 pregame features.
        over_env = float(
            (feature_row.get("home_over_25_rate_L10", 0) + feature_row.get("away_over_25_rate_L10", 0)) / 2.0
        )
    if over_env > 0.55:
        patterns.append(make_pattern("over_environment", 0.07, "positive", 0.7, "Both teams show over-scoring context."))

    return patterns


