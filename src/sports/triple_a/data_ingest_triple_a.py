from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import pandas as pd
import requests


SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
RAW_DATA_DIR = BASE_DIR / "data" / "triple_a" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

FILE_PATH_ADVANCED = RAW_DATA_DIR / "triple_a_advanced_history.csv"
FILE_PATH_UPCOMING = RAW_DATA_DIR / "triple_a_upcoming_schedule.csv"

SPORT_ID = 11  # MiLB Triple-A
INITIAL_BACKFILL_DAYS = 240
INCREMENTAL_BACKFILL_DAYS = 5
UPCOMING_DAYS_AHEAD = 14
LOCAL_TZ = ZoneInfo("America/Mexico_City")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PickGOLD/1.0)",
    "Accept": "application/json",
}

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"

OUTPUT_COLUMNS = [
    "game_id",
    "date",
    "time",
    "season",
    "home_team",
    "away_team",
    "home_starting_pitcher",
    "away_starting_pitcher",
    "home_starting_pitcher_id",
    "away_starting_pitcher_id",
    "pitcher_source",
    "pitcher_data_available",
    "home_runs_total",
    "away_runs_total",
    "home_r1",
    "away_r1",
    "home_r2",
    "away_r2",
    "home_r3",
    "away_r3",
    "home_r4",
    "away_r4",
    "home_r5",
    "away_r5",
    "home_runs_f5",
    "away_runs_f5",
    "attendance",
    "odds_details",
    "odds_over_under",
    "odds_data_quality",
    "home_is_favorite",
    "home_hits",
    "away_hits",
    "status_completed",
    "status_state",
    "status_description",
    "status_detail",
    "home_starting_pitcher_ip",
    "away_starting_pitcher_ip",
    "home_starting_pitcher_er",
    "away_starting_pitcher_er",
    "home_starting_pitcher_hits",
    "away_starting_pitcher_hits",
    "home_starting_pitcher_bb",
    "away_starting_pitcher_bb",
    "home_starting_pitcher_k",
    "away_starting_pitcher_k",
    "home_starting_pitcher_hr",
    "away_starting_pitcher_hr",
    "pitcher_stats_available",
    "home_moneyline_odds",
    "away_moneyline_odds",
    "closing_moneyline_odds",
    "closing_spread_odds",
    "closing_total_odds",
    "market_source",
    "competition_id",
]


def _safe_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _load_existing_history() -> pd.DataFrame:
    if not FILE_PATH_ADVANCED.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    try:
        return pd.read_csv(FILE_PATH_ADVANCED, dtype={"game_id": str, "competition_id": str})
    except Exception:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _format_time(iso_utc: str) -> tuple[str, str]:
    try:
        dt = datetime.fromisoformat(str(iso_utc).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(LOCAL_TZ)
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")
    except Exception:
        raw = str(iso_utc or "")
        return raw[:10], raw[11:16]


def _season_from_date(date_str: str) -> int:
    try:
        return int(str(date_str)[:4])
    except Exception:
        return date.today().year


def _status_fields(game: dict) -> tuple[int, str, str, str]:
    status = game.get("status") or {}
    abstract = str(status.get("abstractGameState") or "").strip().lower()
    detailed = str(status.get("detailedState") or "").strip()
    coded = str(status.get("codedGameState") or "").strip().lower()
    completed = 1 if abstract == "final" or coded == "f" else 0
    state = "post" if completed else ("in" if abstract in {"live", "preview"} and detailed.lower().startswith("in progress") else "pre")
    description = detailed or str(status.get("abstractGameState") or "").strip() or ("Final" if completed else "Scheduled")
    detail = description
    return completed, state, description, detail


def _extract_linescore(game: dict) -> dict:
    innings = (((game.get("linescore") or {}).get("innings")) or [])
    hits = (game.get("linescore") or {}).get("teams") or {}

    def _side_runs(side: str, inning_number: int) -> int:
        try:
            inning = innings[inning_number - 1]
            return _safe_int(((inning.get(side) or {}).get("runs")), 0)
        except Exception:
            return 0

    return {
        "home_r1": _side_runs("home", 1),
        "away_r1": _side_runs("away", 1),
        "home_r2": _side_runs("home", 2),
        "away_r2": _side_runs("away", 2),
        "home_r3": _side_runs("home", 3),
        "away_r3": _side_runs("away", 3),
        "home_r4": _side_runs("home", 4),
        "away_r4": _side_runs("away", 4),
        "home_r5": _side_runs("home", 5),
        "away_r5": _side_runs("away", 5),
        "home_hits": _safe_int((((hits.get("home") or {}).get("hits"))), 0),
        "away_hits": _safe_int((((hits.get("away") or {}).get("hits"))), 0),
    }


def _extract_pitcher(team_blob: dict) -> tuple[str, str | None]:
    probable = team_blob.get("probablePitcher") or {}
    name = str(probable.get("fullName") or "").strip()
    pid = probable.get("id")
    return name, (str(pid) if pid is not None else None)


def _base_row_from_game(game: dict) -> dict:
    game_id = str(game.get("gamePk") or "")
    date_str, time_str = _format_time(game.get("gameDate"))
    season = _season_from_date(date_str)
    teams = game.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    home_team = str(((home.get("team") or {}).get("name")) or "").strip()
    away_team = str(((away.get("team") or {}).get("name")) or "").strip()
    home_pitcher, home_pitcher_id = _extract_pitcher(home)
    away_pitcher, away_pitcher_id = _extract_pitcher(away)
    completed, state, description, detail = _status_fields(game)
    line = _extract_linescore(game)

    home_total = _safe_int(home.get("score"), 0)
    away_total = _safe_int(away.get("score"), 0)

    return {
        "game_id": game_id,
        "date": date_str,
        "time": time_str,
        "season": season,
        "home_team": home_team,
        "away_team": away_team,
        "home_starting_pitcher": home_pitcher,
        "away_starting_pitcher": away_pitcher,
        "home_starting_pitcher_id": home_pitcher_id,
        "away_starting_pitcher_id": away_pitcher_id,
        "pitcher_source": "schedule_probable_pitcher" if (home_pitcher or away_pitcher) else "schedule",
        "pitcher_data_available": int(bool(home_pitcher or away_pitcher)),
        "home_runs_total": home_total,
        "away_runs_total": away_total,
        "home_r1": line["home_r1"],
        "away_r1": line["away_r1"],
        "home_r2": line["home_r2"],
        "away_r2": line["away_r2"],
        "home_r3": line["home_r3"],
        "away_r3": line["away_r3"],
        "home_r4": line["home_r4"],
        "away_r4": line["away_r4"],
        "home_r5": line["home_r5"],
        "away_r5": line["away_r5"],
        "home_runs_f5": line["home_r1"] + line["home_r2"] + line["home_r3"] + line["home_r4"] + line["home_r5"],
        "away_runs_f5": line["away_r1"] + line["away_r2"] + line["away_r3"] + line["away_r4"] + line["away_r5"],
        "attendance": _safe_int(game.get("attendance"), 0),
        "odds_details": "No Line",
        "odds_over_under": 0.0,
        "odds_data_quality": "none",
        "home_is_favorite": -1,
        "home_hits": line["home_hits"],
        "away_hits": line["away_hits"],
        "status_completed": completed,
        "status_state": state,
        "status_description": description,
        "status_detail": detail,
        "home_starting_pitcher_ip": None,
        "away_starting_pitcher_ip": None,
        "home_starting_pitcher_er": None,
        "away_starting_pitcher_er": None,
        "home_starting_pitcher_hits": None,
        "away_starting_pitcher_hits": None,
        "home_starting_pitcher_bb": None,
        "away_starting_pitcher_bb": None,
        "home_starting_pitcher_k": None,
        "away_starting_pitcher_k": None,
        "home_starting_pitcher_hr": None,
        "away_starting_pitcher_hr": None,
        "pitcher_stats_available": 0,
        "home_moneyline_odds": None,
        "away_moneyline_odds": None,
        "closing_moneyline_odds": None,
        "closing_spread_odds": None,
        "closing_total_odds": None,
        "market_source": "none",
        "competition_id": game_id,
    }


def _fetch_schedule_chunk(start_date: date, end_date: date) -> list[dict]:
    params = {
        "sportId": SPORT_ID,
        "startDate": start_date.strftime("%Y-%m-%d"),
        "endDate": end_date.strftime("%Y-%m-%d"),
        "hydrate": "linescore,probablePitcher,team,venue,flags,statusFlags",
    }
    resp = requests.get(SCHEDULE_URL, params=params, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    payload = resp.json() or {}
    out = []
    for day in payload.get("dates") or []:
        out.extend(day.get("games") or [])
    return out


def _fetch_range_rows(start_date: date, end_date: date) -> tuple[list[dict], list[dict]]:
    historical_rows: list[dict] = []
    upcoming_rows: list[dict] = []

    cursor = start_date
    while cursor <= end_date:
        chunk_end = min(cursor + timedelta(days=14), end_date)
        games = _fetch_schedule_chunk(cursor, chunk_end)
        for game in games:
            row = _base_row_from_game(game)
            if row["status_completed"] == 1:
                historical_rows.append(row)
            else:
                upcoming_rows.append(row)
        cursor = chunk_end + timedelta(days=1)
    return historical_rows, upcoming_rows


def _merge_history(existing: pd.DataFrame, fresh_rows: list[dict]) -> pd.DataFrame:
    fresh_df = pd.DataFrame(fresh_rows, columns=OUTPUT_COLUMNS)
    if existing.empty:
        merged = fresh_df
    else:
        merged = pd.concat([existing, fresh_df], ignore_index=True)
    if merged.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    merged = merged.drop_duplicates(subset=["game_id"], keep="last")
    merged = merged.sort_values(["date", "time", "game_id"], kind="stable")
    return merged[OUTPUT_COLUMNS].copy()


def _save_upcoming(rows: list[dict]) -> pd.DataFrame:
    upcoming_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if upcoming_df.empty:
        if FILE_PATH_UPCOMING.exists():
            try:
                existing = pd.read_csv(FILE_PATH_UPCOMING, dtype={"game_id": str, "competition_id": str})
                existing.to_csv(FILE_PATH_UPCOMING, index=False)
                return existing
            except Exception:
                pass
        empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
        empty.to_csv(FILE_PATH_UPCOMING, index=False)
        return empty

    upcoming_df = upcoming_df.drop_duplicates(subset=["game_id"], keep="last")
    upcoming_df = upcoming_df.sort_values(["date", "time", "game_id"], kind="stable")
    upcoming_df.to_csv(FILE_PATH_UPCOMING, index=False)
    return upcoming_df


def main() -> None:
    print("Iniciando ingesta Triple-A (MiLB)...")

    existing = _load_existing_history()
    today = date.today()
    if existing.empty:
        history_start = today - timedelta(days=INITIAL_BACKFILL_DAYS)
        print("No existe historico previo. Se hara backfill inicial.")
    else:
        latest_date = pd.to_datetime(existing["date"], errors="coerce").max()
        latest_day = latest_date.date() if pd.notna(latest_date) else (today - timedelta(days=INCREMENTAL_BACKFILL_DAYS))
        history_start = min(latest_day - timedelta(days=INCREMENTAL_BACKFILL_DAYS), today)
        print(f"Historico previo detectado: {len(existing)} filas. Backfill incremental desde {history_start}.")

    history_end = today
    upcoming_end = today + timedelta(days=UPCOMING_DAYS_AHEAD)

    historical_rows, upcoming_rows = _fetch_range_rows(history_start, upcoming_end)
    merged_history = _merge_history(existing, historical_rows)
    merged_history.to_csv(FILE_PATH_ADVANCED, index=False)
    upcoming_df = _save_upcoming(upcoming_rows)

    print("=" * 64)
    print("RESUMEN ACTUALIZACION TRIPLE-A")
    print("=" * 64)
    print(f"Partidos historicos : {len(merged_history)}")
    print(f"Partidos upcoming   : {len(upcoming_df)}")
    print(f"Historico guardado  : {FILE_PATH_ADVANCED}")
    print(f"Upcoming guardado   : {FILE_PATH_UPCOMING}")
    if not upcoming_df.empty:
        grouped = upcoming_df.groupby("date").size().sort_index()
        for d, c in grouped.items():
            print(f"  > {d}: {c} juegos")
    print("=" * 64)


if __name__ == "__main__":
    main()
