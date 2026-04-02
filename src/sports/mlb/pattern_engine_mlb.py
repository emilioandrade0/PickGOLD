from __future__ import annotations

import sys
from pathlib import Path

# Ensure project `src` root is on sys.path so imports of shared modules work
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pattern_engine import make_pattern


def generate_mlb_patterns(feature_row: dict) -> list[dict]:
    patterns = []

    yrfi_gap = float(feature_row.get("diff_yrfi_rate_L10", 0))
    if yrfi_gap > 0.08:
        patterns.append(make_pattern("yrfi_environment", 0.07, "positive", 0.7, "Recent first-inning scoring trend is elevated."))

    fatigue_gap = float(feature_row.get("diff_fatigue_index", 0))
    if fatigue_gap > 1:
        patterns.append(make_pattern("home_fatigue_edge", 0.06, "positive", 0.7, "Away fatigue profile is worse."))

    return patterns
