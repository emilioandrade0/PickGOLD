from __future__ import annotations

from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pattern_engine import make_pattern


def generate_nhl_patterns(feature_row: dict) -> list[dict]:
    patterns = []

    rest_gap = float(feature_row.get("home_rest_days", 0) - feature_row.get("away_rest_days", 0))
    if rest_gap >= 1:
        patterns.append(make_pattern("home_rest_advantage", 0.07, "positive", 0.75, "Home team has rest advantage."))

    shot_gap = float(feature_row.get("diff_shots_for_L10", 0))
    if shot_gap > 2.0:
        patterns.append(make_pattern("shot_volume_edge", 0.06, "positive", 0.65, "Home shot volume trend is stronger."))

    return patterns
