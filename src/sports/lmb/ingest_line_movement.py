from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
CACHE_DIR = BASE_DIR / "data" / "lmb" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LINE_MOVEMENT_CACHE = CACHE_DIR / "line_movement.csv"
LINE_MOVEMENT_HISTORY = CACHE_DIR / "line_movement_history.csv"

LINE_MOVEMENT_COLUMNS = [
    "external_game_id",
    "game_id",
    "date",
    "home_team",
    "away_team",
    "home_team_name",
    "away_team_name",
    "snapshot_count",
    "first_snapshot_utc",
    "last_snapshot_utc",
    "open_line",
    "current_line",
    "line_movement",
    "open_total",
    "current_total",
    "total_movement",
    "current_home_moneyline",
    "current_away_moneyline",
    "current_home_spread",
    "current_total_line",
    "bookmakers_count",
    "market_source",
]

LINE_MOVEMENT_HISTORY_COLUMNS = [
    "snapshot_time_utc",
    "external_game_id",
    "game_id",
    "date",
    "commence_time_utc",
    "home_team",
    "away_team",
    "home_team_name",
    "away_team_name",
    "bookmakers_count",
    "home_moneyline",
    "away_moneyline",
    "home_spread",
    "total_line",
]


def _ensure_csv(path: Path, columns: list[str]) -> None:
    if path.exists():
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(columns=columns)

    for col in columns:
        if col not in df.columns:
            df[col] = None
    df = df[columns].copy()
    df.to_csv(path, index=False)


def main() -> None:
    _ensure_csv(LINE_MOVEMENT_CACHE, LINE_MOVEMENT_COLUMNS)
    _ensure_csv(LINE_MOVEMENT_HISTORY, LINE_MOVEMENT_HISTORY_COLUMNS)
    print(f"LMB line movement cache listo: {LINE_MOVEMENT_CACHE}")
    print(f"LMB line movement history lista: {LINE_MOVEMENT_HISTORY}")


if __name__ == "__main__":
    main()
