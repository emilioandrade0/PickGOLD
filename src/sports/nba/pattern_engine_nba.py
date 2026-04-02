from __future__ import annotations

import sys
from pathlib import Path

# Agregar la raíz de 'src' al sistema para que encuentre pattern_engine
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Ahora ya puedes importar lo demás
from pattern_engine import make_pattern


from pattern_engine import make_pattern


def generate_nba_patterns(feature_row: dict) -> list[dict]:
    patterns = []

    rest_edge = float(feature_row.get("away_rest_days", 0) - feature_row.get("home_rest_days", 0))
    if rest_edge >= 1:
        patterns.append(make_pattern("home_rest_advantage", 0.08, "positive", 0.8, "Home has more rest days."))

    travel_edge = float(feature_row.get("away_tz_diff", 0))
    if travel_edge >= 2:
        patterns.append(make_pattern("away_travel_penalty", 0.06, "positive", 0.75, "Away crosses multiple time zones."))

    return patterns
