from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import pandas as pd
import requests


SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
RAW_DATA_DIR = BASE_DIR / "data" / "lmb" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

FILE_PATH_ADVANCED = RAW_DATA_DIR / "lmb_advanced_history.csv"
FILE_PATH_UPCOMING = RAW_DATA_DIR / "lmb_upcoming_schedule.csv"

def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, str(default))).strip())
    except Exception:
        return float(default)


INITIAL_BACKFILL_DAYS = _env_int("LMB_INITIAL_BACKFILL_DAYS", 180)
# Optional safety cap for very large first-run backfills (0 disables the cap).
INITIAL_FETCH_CAP_DAYS = _env_int("LMB_INITIAL_FETCH_CAP_DAYS", 0)
INCREMENTAL_BACKFILL_DAYS = _env_int("LMB_INCREMENTAL_BACKFILL_DAYS", 7)
UPCOMING_DAYS_AHEAD = _env_int("LMB_UPCOMING_DAYS_AHEAD", 14)
HTTP_TIMEOUT_SECONDS = _env_float("LMB_HTTP_TIMEOUT_SECONDS", 8.0)
LOCAL_TZ = ZoneInfo("America/Mexico_City")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PickGOLD/1.0)",
    "Accept": "application/json",
}

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
SPORT_ID = 23
LEAGUE_ID = 125
SCHEDULE_HYDRATE = "linescore,probablePitcher,team,venue,flags,statusFlags"
SOURCE_LABEL = f"{SCHEDULE_URL} (sportId={SPORT_ID}, leagueId={LEAGUE_ID})"

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


def _format_date_time(game: dict, fallback_day: date) -> tuple[str, str]:
    iso_utc = str(game.get("gameDate") or "").strip()
    if iso_utc:
        try:
            dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(LOCAL_TZ)
            return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")
        except Exception:
            pass

    official_date = str(game.get("officialDate") or "").strip()
    if official_date:
        return official_date, "00:00"

    return fallback_day.strftime("%Y-%m-%d"), "00:00"


def _season_from_date(date_str: str) -> int:
    try:
        return int(str(date_str)[:4])
    except Exception:
        return date.today().year


def _load_existing_history() -> pd.DataFrame:
    if not FILE_PATH_ADVANCED.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    try:
        df = pd.read_csv(FILE_PATH_ADVANCED, dtype={"game_id": str, "competition_id": str})
    except Exception:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[OUTPUT_COLUMNS].copy()


def _ensure_output_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[OUTPUT_COLUMNS].copy()


def _status_fields_from_game(game: dict) -> tuple[int, str, str, str]:
    status = game.get("status") if isinstance(game.get("status"), dict) else {}
    abstract = str(status.get("abstractGameState") or "").strip().lower()
    coded = str(status.get("codedGameState") or "").strip().lower()
    detailed = str(status.get("detailedState") or "").strip()

    completed = 1 if (abstract == "final" or coded == "f") else 0
    if completed:
        state = "post"
    elif abstract == "live" or coded in {"i", "m"} or "progress" in detailed.lower():
        state = "in"
    else:
        state = "pre"

    description = detailed or str(status.get("abstractGameState") or "").strip() or ("Final" if completed else "Scheduled")
    detail = description
    return completed, state, description, detail


def _extract_linescore(game: dict) -> tuple[list[int], list[int], int, int]:
    linescore = game.get("linescore") if isinstance(game.get("linescore"), dict) else {}
    innings = linescore.get("innings") if isinstance(linescore.get("innings"), list) else []
    teams_blob = linescore.get("teams") if isinstance(linescore.get("teams"), dict) else {}

    def _runs_for_side(inning_blob: dict, side: str) -> int:
        side_blob = inning_blob.get(side) if isinstance(inning_blob.get(side), dict) else {}
        return _safe_int(side_blob.get("runs"), 0)

    home_lines = [_runs_for_side(inning_blob, "home") for inning_blob in innings]
    away_lines = [_runs_for_side(inning_blob, "away") for inning_blob in innings]

    home_hits = _safe_int(((teams_blob.get("home") or {}).get("hits")), 0)
    away_hits = _safe_int(((teams_blob.get("away") or {}).get("hits")), 0)
    return home_lines, away_lines, home_hits, away_hits


def _extract_team_name(team_blob: dict) -> str:
    team = team_blob.get("team") if isinstance(team_blob.get("team"), dict) else {}
    return str(team.get("name") or "").strip()


def _extract_probable_pitcher_name(team_blob: dict) -> str:
    pitcher = team_blob.get("probablePitcher") if isinstance(team_blob.get("probablePitcher"), dict) else {}
    return str(pitcher.get("fullName") or pitcher.get("name") or "").strip()


def _extract_probable_pitcher_id(team_blob: dict) -> str | None:
    pitcher = team_blob.get("probablePitcher") if isinstance(team_blob.get("probablePitcher"), dict) else {}
    pid = pitcher.get("id")
    if pid is None:
        return None
    return str(pid)


def _build_row_from_game(game: dict, fallback_day: date) -> dict | None:
    game_id = str(game.get("gamePk") or "").strip()
    if not game_id:
        return None

    teams = game.get("teams") if isinstance(game.get("teams"), dict) else {}
    home = teams.get("home") if isinstance(teams.get("home"), dict) else {}
    away = teams.get("away") if isinstance(teams.get("away"), dict) else {}
    if not isinstance(home, dict) or not isinstance(away, dict):
        return None

    date_str, time_str = _format_date_time(game, fallback_day=fallback_day)
    season = _season_from_date(date_str)

    completed, state, description, detail = _status_fields_from_game(game)

    home_lines, away_lines, home_hits, away_hits = _extract_linescore(game)

    def _line_at(lines: list[int], index: int) -> int:
        if index < 0 or index >= len(lines):
            return 0
        return _safe_int(lines[index], 0)

    home_total = _safe_int(home.get("score"), 0)
    away_total = _safe_int(away.get("score"), 0)
    home_r = [_line_at(home_lines, i) for i in range(5)]
    away_r = [_line_at(away_lines, i) for i in range(5)]
    home_pitcher = _extract_probable_pitcher_name(home)
    away_pitcher = _extract_probable_pitcher_name(away)
    home_pitcher_id = _extract_probable_pitcher_id(home)
    away_pitcher_id = _extract_probable_pitcher_id(away)
    pitcher_available = 1 if (home_pitcher or away_pitcher) else 0
    home_ml = None
    away_ml = None

    return {
        "game_id": game_id,
        "date": date_str,
        "time": time_str,
        "season": season,
        "home_team": _extract_team_name(home),
        "away_team": _extract_team_name(away),
        "home_starting_pitcher": home_pitcher,
        "away_starting_pitcher": away_pitcher,
        "home_starting_pitcher_id": home_pitcher_id,
        "away_starting_pitcher_id": away_pitcher_id,
        "pitcher_source": "statsapi_probable_pitcher" if (home_pitcher or away_pitcher) else "statsapi_schedule",
        "pitcher_data_available": pitcher_available,
        "home_runs_total": home_total,
        "away_runs_total": away_total,
        "home_r1": home_r[0],
        "away_r1": away_r[0],
        "home_r2": home_r[1],
        "away_r2": away_r[1],
        "home_r3": home_r[2],
        "away_r3": away_r[2],
        "home_r4": home_r[3],
        "away_r4": away_r[3],
        "home_r5": home_r[4],
        "away_r5": away_r[4],
        "home_runs_f5": sum(home_r),
        "away_runs_f5": sum(away_r),
        "attendance": _safe_int(game.get("attendance"), 0),
        "odds_details": "No Line",
        "odds_over_under": 0.0,
        "odds_data_quality": "none",
        "home_is_favorite": -1,
        "home_hits": home_hits,
        "away_hits": away_hits,
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
        "home_moneyline_odds": home_ml,
        "away_moneyline_odds": away_ml,
        "closing_moneyline_odds": None,
        "closing_spread_odds": None,
        "closing_total_odds": None,
        "market_source": "none",
        "competition_id": game_id,
    }


def _fetch_schedule_chunk(start_date: date, end_date: date) -> tuple[list[dict], str | None]:
    params = {
        "sportId": SPORT_ID,
        "leagueId": LEAGUE_ID,
        "startDate": start_date.strftime("%Y-%m-%d"),
        "endDate": end_date.strftime("%Y-%m-%d"),
        "hydrate": SCHEDULE_HYDRATE,
    }
    try:
        resp = requests.get(
            SCHEDULE_URL,
            params=params,
            timeout=HTTP_TIMEOUT_SECONDS,
            headers=HEADERS,
        )
        if resp.status_code >= 400:
            return [], None
        payload = resp.json() or {}
        games: list[dict] = []
        for day_blob in payload.get("dates") or []:
            if isinstance(day_blob, dict):
                day_games = day_blob.get("games") or []
                if isinstance(day_games, list):
                    games.extend(day_games)
        return games, SOURCE_LABEL
    except Exception:
        return [], None


def _fetch_range_rows(start_date: date, end_date: date) -> tuple[list[dict], list[dict], set[str]]:
    historical_rows: list[dict] = []
    upcoming_rows: list[dict] = []
    used_sources: set[str] = set()

    cursor = start_date
    while cursor <= end_date:
        chunk_end = min(cursor + timedelta(days=14), end_date)
        games, source = _fetch_schedule_chunk(cursor, chunk_end)
        if source:
            used_sources.add(source)
        for game in games:
            row = _build_row_from_game(game, fallback_day=chunk_end)
            if row is None:
                continue
            if int(row.get("status_completed", 0) or 0) == 1:
                historical_rows.append(row)
            else:
                upcoming_rows.append(row)
        cursor = chunk_end + timedelta(days=1)

    return historical_rows, upcoming_rows, used_sources


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
    return _ensure_output_schema(merged)


def _save_upcoming(rows: list[dict]) -> pd.DataFrame:
    upcoming_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if upcoming_df.empty:
        if FILE_PATH_UPCOMING.exists():
            try:
                existing = pd.read_csv(FILE_PATH_UPCOMING, dtype={"game_id": str, "competition_id": str})
                existing = _ensure_output_schema(existing)
                existing.to_csv(FILE_PATH_UPCOMING, index=False)
                return existing
            except Exception:
                pass
        empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
        empty.to_csv(FILE_PATH_UPCOMING, index=False)
        return empty

    upcoming_df = upcoming_df.drop_duplicates(subset=["game_id"], keep="last")
    upcoming_df = upcoming_df.sort_values(["date", "time", "game_id"], kind="stable")
    upcoming_df = _ensure_output_schema(upcoming_df)
    upcoming_df.to_csv(FILE_PATH_UPCOMING, index=False)
    return upcoming_df


def main() -> None:
    print("Iniciando ingesta LMB (Liga Mexicana de Beisbol)...")

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
    if existing.empty and INITIAL_FETCH_CAP_DAYS > 0:
        capped_start = history_end - timedelta(days=INITIAL_FETCH_CAP_DAYS)
        if history_start < capped_start:
            print(
                "Backfill inicial limitado por corrida para evitar timeouts "
                + f"({INITIAL_FETCH_CAP_DAYS} dias recientes)."
            )
            history_start = capped_start
    if history_start > history_end:
        history_start = history_end

    historical_rows, upcoming_rows, used_sources = _fetch_range_rows(history_start, upcoming_end)
    merged_history = _merge_history(existing, historical_rows)
    merged_history.to_csv(FILE_PATH_ADVANCED, index=False)
    upcoming_df = _save_upcoming(upcoming_rows)

    print("=" * 64)
    print("RESUMEN ACTUALIZACION LMB")
    print("=" * 64)
    print(f"Partidos historicos : {len(merged_history)}")
    print(f"Partidos upcoming   : {len(upcoming_df)}")
    print(f"Historico guardado  : {FILE_PATH_ADVANCED}")
    print(f"Upcoming guardado   : {FILE_PATH_UPCOMING}")
    if used_sources:
        print(f"Endpoints usados    : {', '.join(sorted(used_sources))}")
    else:
        print("Endpoints usados    : ninguno (se conserva cache local)")
    if not upcoming_df.empty:
        grouped = upcoming_df.groupby("date").size().sort_index()
        for d, c in grouped.items():
            print(f"  > {d}: {c} juegos")
    print("=" * 64)


if __name__ == "__main__":
    main()
