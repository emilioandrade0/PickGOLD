# =========================
# IMPORTS
# =========================
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import math
import os
import sys

import numpy as np
import pandas as pd
import requests

# Ensure project `src` root is on sys.path so imports of shared modules work
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# DESPUÉS importamos los módulos locales compartidos
from odds_market_fields import extract_market_odds_fields, odds_data_quality

# =========================
# CONFIG
# =========================
# Use src root as base so data remains under src/data
BASE_DIR = SRC_ROOT

RAW_DATA_DIR = BASE_DIR / "data" / "mlb" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

FILE_PATH_ADVANCED = RAW_DATA_DIR / "mlb_advanced_history.csv"
FILE_PATH_UPCOMING = RAW_DATA_DIR / "mlb_upcoming_schedule.csv"

PITCHER_CACHE_DIR = BASE_DIR / "data" / "mlb" / "cache" / "pitchers"
PITCHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# External / fallback cache (Baseball-Reference / Statcast / Fangraphs cached JSONs)
EXTERNAL_CACHE_DIR = BASE_DIR / "data" / "mlb" / "cache" / "external"
EXTERNAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
USE_PITCHER_CACHE = True
ONLY_ENRICH_MISSING_PITCHERS = True
MAX_ENRICH_ROWS_PER_RUN = None   # prueba rápida: 500
REQUEST_SLEEP_SECONDS = 0.01
SUMMARY_TIMEOUT = 20
REQUEST_RETRIES = 3
BACKOFF_FACTOR = 0.25

SEASONS_TO_FETCH = {
    "2024": ("2024-02-20", "2024-11-05"),
    "2025": ("2025-02-20", "2025-11-05"),
    "2026": ("2026-02-20", "2026-11-05"),
}

TARGET_DATE_LIMIT = datetime.now().strftime("%Y-%m-%d")
BACKFILL_DAYS = 3
UPCOMING_DAYS_AHEAD = 14
RECENT_MARKET_REFRESH_DAYS = 10

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary?event={event_id}"
CORE_ODDS_URL = (
    "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/"
    "events/{event_id}/competitions/{competition_id}/odds"
)


# =========================
# HELPERS
# =========================
def parse_moneyline_favorite(details_text: str, home_abbr: str, away_abbr: str) -> int:
    if not details_text:
        return -1

    txt = str(details_text).strip().upper()
    if txt in {"N/A", "NO LINE", "PK", "PICK", "PICKEM", "PICK'EM"}:
        return -1

    m = re.match(r"^([A-Z]+)\s*([+-]?\d+(?:\.\d+)?)$", txt)
    if not m:
        return -1

    team_code = m.group(1)
    if team_code == home_abbr:
        return 1
    if team_code == away_abbr:
        return 0
    return -1


def parse_over_under(value) -> float:
    try:
        if value in [None, "", "N/A"]:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_str(value, default=""):
    try:
        if value is None:
            return default
        return str(value).strip()
    except Exception:
        return default


def normalize_pitcher_name(name: str) -> str:
    name = safe_str(name, "")
    if not name:
        return ""
    return re.sub(r"\s+", " ", name).strip()


def is_recent_market_refresh_candidate(row: dict, recent_days: int = RECENT_MARKET_REFRESH_DAYS) -> bool:
    date_str = safe_str(row.get("date"))
    if not date_str:
        return False
    try:
        row_dt = datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return False
    cutoff_dt = datetime.strptime(TARGET_DATE_LIMIT, "%Y-%m-%d") - timedelta(days=max(int(recent_days), 0))
    return row_dt >= cutoff_dt


# =========================
# PITCHER EXTRACTION FROM SCOREBOARD
# =========================
def extract_probable_pitchers(comp: dict):
    result = {
        "home_starting_pitcher": "",
        "away_starting_pitcher": "",
        "home_starting_pitcher_id": "",
        "away_starting_pitcher_id": "",
        "pitcher_source": "",
    }

    probables = comp.get("probables", [])
    if isinstance(probables, list) and probables:
        for p in probables:
            if not isinstance(p, dict):
                continue

            athlete = p.get("athlete") or {}
            team_ref = p.get("team") or {}
            home_away = safe_str(p.get("homeAway")).lower()

            pitcher_name = normalize_pitcher_name(
                athlete.get("displayName")
                or athlete.get("shortName")
                or p.get("displayName")
                or p.get("name")
            )
            pitcher_id = safe_str(athlete.get("id") or p.get("id"))

            if home_away == "home":
                result["home_starting_pitcher"] = pitcher_name
                result["home_starting_pitcher_id"] = pitcher_id
                result["pitcher_source"] = "competition_probables"
            elif home_away == "away":
                result["away_starting_pitcher"] = pitcher_name
                result["away_starting_pitcher_id"] = pitcher_id
                result["pitcher_source"] = "competition_probables"

            team_abbr = safe_str(team_ref.get("abbreviation")).upper()
            team_id = safe_str(team_ref.get("id"))
            if not home_away and pitcher_name:
                result.setdefault("_tmp_probables", [])
                result["_tmp_probables"].append(
                    {
                        "team_abbr": team_abbr,
                        "team_id": team_id,
                        "pitcher_name": pitcher_name,
                        "pitcher_id": pitcher_id,
                    }
                )

    return result


def extract_competitor_pitcher(competitor: dict) -> tuple[str, str, str]:
    probables = competitor.get("probables", [])
    if isinstance(probables, list) and probables:
        for p in probables:
            if not isinstance(p, dict):
                continue
            athlete = p.get("athlete") or {}
            pitcher_name = normalize_pitcher_name(
                athlete.get("displayName")
                or athlete.get("shortName")
                or p.get("displayName")
                or p.get("name")
            )
            pitcher_id = safe_str(athlete.get("id") or p.get("id"))
            if pitcher_name:
                return pitcher_name, pitcher_id, "competitor_probables"

    return "", "", ""


def resolve_starting_pitchers(comp: dict, home_data: dict, away_data: dict) -> dict:
    out = extract_probable_pitchers(comp)

    if not out["home_starting_pitcher"]:
        name, pid, source = extract_competitor_pitcher(home_data)
        if name:
            out["home_starting_pitcher"] = name
            out["home_starting_pitcher_id"] = pid
            out["pitcher_source"] = source or out["pitcher_source"]

    if not out["away_starting_pitcher"]:
        name, pid, source = extract_competitor_pitcher(away_data)
        if name:
            out["away_starting_pitcher"] = name
            out["away_starting_pitcher_id"] = pid
            out["pitcher_source"] = source or out["pitcher_source"]

    tmp = out.pop("_tmp_probables", [])
    if tmp:
        home_abbr = safe_str(home_data.get("team", {}).get("abbreviation")).upper()
        away_abbr = safe_str(away_data.get("team", {}).get("abbreviation")).upper()
        home_id = safe_str(home_data.get("team", {}).get("id"))
        away_id = safe_str(away_data.get("team", {}).get("id"))

        for item in tmp:
            if not out["home_starting_pitcher"] and (
                item["team_abbr"] == home_abbr or (home_id and item["team_id"] == home_id)
            ):
                out["home_starting_pitcher"] = item["pitcher_name"]
                out["home_starting_pitcher_id"] = item["pitcher_id"]
                out["pitcher_source"] = out["pitcher_source"] or "competition_probables_team_match"

            if not out["away_starting_pitcher"] and (
                item["team_abbr"] == away_abbr or (away_id and item["team_id"] == away_id)
            ):
                out["away_starting_pitcher"] = item["pitcher_name"]
                out["away_starting_pitcher_id"] = item["pitcher_id"]
                out["pitcher_source"] = out["pitcher_source"] or "competition_probables_team_match"

    home_ok = safe_str(out["home_starting_pitcher"]).strip() != ""
    away_ok = safe_str(out["away_starting_pitcher"]).strip() != ""
    out["pitcher_data_available"] = int(home_ok and away_ok)

    return out


# =========================
# EXISTING DATA / RANGES
# =========================
def load_existing_data() -> pd.DataFrame:
    if not FILE_PATH_ADVANCED.exists():
        return pd.DataFrame()

    try:
        df_existing = pd.read_csv(FILE_PATH_ADVANCED, dtype={"game_id": str})
        if "date" in df_existing.columns:
            df_existing["date"] = df_existing["date"].astype(str)
        if "date_dt" in df_existing.columns:
            df_existing = df_existing.drop(columns=["date_dt"])
        return df_existing
    except Exception as e:
        print(f"⚠️ No se pudo leer el CSV existente. Se reconstruirá desde cero. Error: {e}")
        return pd.DataFrame()


def build_full_ranges(limit_date_dt: datetime):
    ranges = []
    for season, (start_str, end_str) in SEASONS_TO_FETCH.items():
        start_dt = datetime.strptime(start_str, "%Y-%m-%d")
        end_dt = datetime.strptime(end_str, "%Y-%m-%d")

        if end_dt > limit_date_dt:
            end_dt = limit_date_dt
        if start_dt > limit_date_dt:
            continue

        ranges.append((season, start_dt, end_dt))
    return ranges


def determine_fetch_ranges(existing_df: pd.DataFrame):
    limit_date_dt = datetime.strptime(TARGET_DATE_LIMIT, "%Y-%m-%d")

    if existing_df.empty or "date" not in existing_df.columns:
        print("📂 No existe histórico previo. Se hará descarga completa.")
        return build_full_ranges(limit_date_dt)

    try:
        tmp = existing_df.copy()
        tmp["date_dt"] = pd.to_datetime(tmp["date"], errors="coerce")
        max_date = tmp["date_dt"].max()

        if pd.isna(max_date):
            print("⚠️ No se pudo detectar la última fecha del histórico. Se hará descarga completa.")
            return build_full_ranges(limit_date_dt)

        incremental_start = (max_date - timedelta(days=BACKFILL_DAYS)).normalize()
        print(
            f"📌 Última fecha en histórico: {max_date.strftime('%Y-%m-%d')} | "
            f"Revisando incremental desde: {incremental_start.strftime('%Y-%m-%d')}"
        )

        ranges = []
        for season, (start_str, end_str) in SEASONS_TO_FETCH.items():
            start_dt = datetime.strptime(start_str, "%Y-%m-%d")
            end_dt = datetime.strptime(end_str, "%Y-%m-%d")

            if end_dt > limit_date_dt:
                end_dt = limit_date_dt

            if end_dt < incremental_start:
                continue

            season_start = max(start_dt, incremental_start)
            if season_start <= end_dt:
                ranges.append((season, season_start, end_dt))

        return ranges

    except Exception as e:
        print(f"⚠️ Error calculando rango incremental. Se hará descarga completa. Error: {e}")
        return build_full_ranges(limit_date_dt)


# =========================
# PARSING SCOREBOARD EVENT
# =========================
def get_inning_score(linescores, index):
    try:
        return int(linescores[index]["value"]) if len(linescores) > index else 0
    except Exception:
        return 0


def parse_event_to_row(event: dict, season: str | None = None):
    competitions = event.get("competitions", [])
    if not competitions:
        return None

    comp = competitions[0]
    competitors = comp.get("competitors", [])
    if len(competitors) != 2:
        return None

    home_data = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away_data = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if not home_data or not away_data:
        return None

    pitcher_info = resolve_starting_pitchers(comp, home_data, away_data)

    status = event.get("status", {}).get("type", {})
    completed = bool(status.get("completed", False))
    state = str(status.get("state", "") or "")
    description = str(status.get("description", "") or "")
    detail = str(status.get("detail", "") or "")

    h_abbr = home_data["team"]["abbreviation"]
    a_abbr = away_data["team"]["abbreviation"]

    game_id = str(event["id"])

    dt_utc = datetime.strptime(event["date"], "%Y-%m-%dT%H:%MZ")
    game_date = (dt_utc - timedelta(hours=5)).strftime("%Y-%m-%d")
    game_time = (dt_utc - timedelta(hours=5)).strftime("%H:%M")

    attendance = comp.get("attendance", 0)

    odds = comp.get("odds", [{}])
    odds = odds[0] if odds else {}
    odds_details = odds.get("details", "N/A")
    over_under = parse_over_under(odds.get("overUnder", 0))
    market_odds_fields = extract_market_odds_fields(odds)
    home_is_favorite = parse_moneyline_favorite(odds_details, h_abbr, a_abbr)

    h_linescores = home_data.get("linescores", [])
    a_linescores = away_data.get("linescores", [])

    home_r1 = get_inning_score(h_linescores, 0)
    away_r1 = get_inning_score(a_linescores, 0)
    home_r2 = get_inning_score(h_linescores, 1)
    away_r2 = get_inning_score(a_linescores, 1)
    home_r3 = get_inning_score(h_linescores, 2)
    away_r3 = get_inning_score(a_linescores, 2)
    home_r4 = get_inning_score(h_linescores, 3)
    away_r4 = get_inning_score(a_linescores, 3)
    home_r5 = get_inning_score(h_linescores, 4)
    away_r5 = get_inning_score(a_linescores, 4)

    home_runs_f5 = home_r1 + home_r2 + home_r3 + home_r4 + home_r5
    away_runs_f5 = away_r1 + away_r2 + away_r3 + away_r4 + away_r5

    home_score = safe_int(home_data.get("score", 0))
    away_score = safe_int(away_data.get("score", 0))

    home_hits = 0
    away_hits = 0
    if home_data.get("statistics"):
        home_hits = safe_int(home_data.get("statistics", [{}])[0].get("displayValue", 0), 0)
    if away_data.get("statistics"):
        away_hits = safe_int(away_data.get("statistics", [{}])[0].get("displayValue", 0), 0)

    row = {
        "game_id": game_id,
        "competition_id": safe_str(comp.get("id") or game_id),
        "date": game_date,
        "time": game_time,
        "season": season if season is not None else str(dt_utc.year),
        "home_team": h_abbr,
        "away_team": a_abbr,

        "home_starting_pitcher": pitcher_info["home_starting_pitcher"],
        "away_starting_pitcher": pitcher_info["away_starting_pitcher"],
        "home_starting_pitcher_id": pitcher_info["home_starting_pitcher_id"],
        "away_starting_pitcher_id": pitcher_info["away_starting_pitcher_id"],
        "pitcher_source": pitcher_info["pitcher_source"],
        "pitcher_data_available": pitcher_info["pitcher_data_available"],

        "home_runs_total": home_score,
        "away_runs_total": away_score,

        "home_r1": home_r1,
        "away_r1": away_r1,
        "home_r2": home_r2,
        "away_r2": away_r2,
        "home_r3": home_r3,
        "away_r3": away_r3,
        "home_r4": home_r4,
        "away_r4": away_r4,
        "home_r5": home_r5,
        "away_r5": away_r5,

        "home_runs_f5": home_runs_f5,
        "away_runs_f5": away_runs_f5,

        "attendance": attendance,
        "odds_details": odds_details,
        "odds_over_under": over_under,
        **market_odds_fields,
        "odds_data_quality": odds_data_quality(market_odds_fields),
        "home_is_favorite": home_is_favorite,
        "home_hits": home_hits,
        "away_hits": away_hits,

        "status_completed": int(completed),
        "status_state": state,
        "status_description": description,
        "status_detail": detail,
    }

    return row


# =========================
# SCOREBOARD DOWNLOAD
# =========================
def fetch_games_for_ranges(ranges):
    completed_games = []

    for season, start_dt, end_dt in ranges:
        print(f"   > Escaneando temporada {season}: {start_dt.strftime('%Y-%m-%d')} -> {end_dt.strftime('%Y-%m-%d')}")
        chunks = []
        current_dt = start_dt
        while current_dt <= end_dt:
            chunk_end_dt = min(current_dt + timedelta(days=15), end_dt)
            d1 = current_dt.strftime("%Y%m%d")
            d2 = chunk_end_dt.strftime("%Y%m%d")
            chunks.append((d1, d2))
            current_dt = chunk_end_dt + timedelta(days=1)

        def fetch_chunk(d1: str, d2: str):
            url = f"{ESPN_SCOREBOARD_URL}?dates={d1}-{d2}&limit=500"
            for attempt in range(3):
                try:
                    resp = requests.get(url, timeout=20)
                    resp.raise_for_status()
                    data = resp.json() or {}
                    events = data.get("events", [])
                    return (d1, d2, events, None)
                except Exception as e:
                    if attempt == 2:
                        return (d1, d2, [], str(e))
                    time.sleep(0.2 * (attempt + 1))

        max_workers = min(6, max(1, len(chunks)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fetch_chunk, d1, d2) for d1, d2 in chunks]
            for future in as_completed(futures):
                d1, d2, events, error = future.result()
                if error is not None:
                    print(f"⚠️ Error al consultar chunk {d1}-{d2}: {error}")
                    continue

                for event in events:
                    row = parse_event_to_row(event, season=season)
                    if row is None:
                        continue
                    if int(row["status_completed"]) == 1:
                        completed_games.append(row)

        print(f"     -> ✅ Temporada {season} procesada.")

    return pd.DataFrame(completed_games)


def fetch_upcoming_schedule_for_range(start_date: str, days_ahead: int = UPCOMING_DAYS_AHEAD):
    session = requests.Session()
    base_dt = datetime.strptime(start_date, "%Y-%m-%d")
    upcoming_rows = []

    for day_offset in range(days_ahead + 1):
        day_dt = base_dt + timedelta(days=day_offset)
        day_str = day_dt.strftime("%Y-%m-%d")
        url = f"{ESPN_SCOREBOARD_URL}?dates={day_dt.strftime('%Y%m%d')}&limit=500"

        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            data = resp.json() or {}
            events = data.get("events", [])

            day_count = 0
            for event in events:
                row = parse_event_to_row(event, season=None)
                if row is None:
                    continue

                if day_str != start_date and int(row.get("status_completed", 0)) == 1:
                    continue

                upcoming_rows.append(row)
                day_count += 1

            print(f"   📅 Agenda MLB {day_str}: {day_count} juegos")

        except Exception as e:
            print(f"⚠️ Error descargando agenda MLB {day_str}: {e}")

    if upcoming_rows:
        upcoming_rows = enrich_rows_with_historical_pitchers(upcoming_rows, label="agenda MLB")

    df = pd.DataFrame(upcoming_rows)

    if not df.empty:
        df["game_id"] = df["game_id"].astype(str)
        df["date"] = df["date"].astype(str)
        df = df.sort_values(["date", "time", "game_id"]).drop_duplicates(subset=["game_id"], keep="last").reset_index(drop=True)

    return df


# =========================
# SUMMARY / CACHE HELPERS
# =========================
def fetch_game_summary(event_id: str):
    url = SUMMARY_URL.format(event_id=event_id)
    try:
        resp = requests_with_retry(url, timeout=SUMMARY_TIMEOUT)
        if resp is None:
            return None
        return resp.json()
    except Exception:
        return None


def requests_with_retry(url: str, timeout: int = 10, retries: int = None, backoff: float = None):
    r = REQUEST_RETRIES if retries is None else int(retries)
    b = BACKOFF_FACTOR if backoff is None else float(backoff)
    for attempt in range(1, r + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt == r:
                return None
            sleep_t = b * (2 ** (attempt - 1))
            time.sleep(sleep_t)
    return None


def _external_cache_path(game_id: str) -> Path:
    return EXTERNAL_CACHE_DIR / f"{safe_str(game_id)}.json"


def load_external_cache(game_id: str):
    path = _external_cache_path(game_id)
    if not path.exists():
        return None
    try:
        return pd.read_json(path, typ="series").to_dict()
    except Exception:
        return None


def save_external_cache(game_id: str, data: dict):
    path = _external_cache_path(game_id)
    try:
        pd.Series(data).to_json(path, force_ascii=False, indent=2)
    except Exception:
        pass


def fetch_from_mlb_stats_api_if_possible(game_id: str):
    # Best-effort: some game IDs contain numeric gamePk usable with MLB stats API
    try:
        gid_numeric = int(re.sub(r"[^0-9]", "", safe_str(game_id)))
    except Exception:
        return None

    base = f"https://statsapi.mlb.com/api/v1.1/game/{gid_numeric}/boxscore"
    resp = requests_with_retry(base, timeout=10)
    if resp is None:
        return None
    try:
        data = resp.json() or {}
        payload = {"mlb_boxscore": data}
        save_external_cache(game_id, payload)
        return payload
    except Exception:
        return None


def get_pitcher_cache_path(game_id: str) -> Path:
    return PITCHER_CACHE_DIR / f"{safe_str(game_id)}.json"


def load_pitcher_cache(game_id: str):
    if not USE_PITCHER_CACHE:
        return None

    path = get_pitcher_cache_path(game_id)
    if not path.exists():
        return None

    try:
        return pd.read_json(path, typ="series").to_dict()
    except Exception:
        return None


def save_pitcher_cache(game_id: str, data: dict):
    if not USE_PITCHER_CACHE:
        return

    path = get_pitcher_cache_path(game_id)
    try:
        pd.Series(data).to_json(path, force_ascii=False, indent=2)
    except Exception:
        pass


def _extract_competitors_from_summary(summary_json: dict):
    try:
        comps = summary_json.get("header", {}).get("competitions", [])
        if not comps:
            return None, None
        competitors = comps[0].get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        return home, away
    except Exception:
        return None, None


def _extract_market_from_payload(market_payload: dict, home_abbr: str, away_abbr: str, market_source: str):
    details = safe_str(market_payload.get("details"), "N/A")
    over_under = parse_over_under(market_payload.get("overUnder", 0))
    market_fields = extract_market_odds_fields(market_payload)
    home_is_favorite = parse_moneyline_favorite(details, home_abbr, away_abbr)

    if home_is_favorite == -1:
        home_team_odds = market_payload.get("homeTeamOdds") or {}
        away_team_odds = market_payload.get("awayTeamOdds") or {}
        if isinstance(home_team_odds, dict) and isinstance(away_team_odds, dict):
            if home_team_odds.get("favorite") is True or away_team_odds.get("underdog") is True:
                home_is_favorite = 1
            elif away_team_odds.get("favorite") is True or home_team_odds.get("underdog") is True:
                home_is_favorite = 0

    return {
        "odds_details": details or "N/A",
        "odds_over_under": over_under,
        "home_is_favorite": home_is_favorite,
        **market_fields,
        "odds_data_quality": odds_data_quality(market_fields),
        "market_source": market_source or "",
    }


def _extract_competition_id_from_summary(summary_json: dict, default: str = "") -> str:
    try:
        comps = summary_json.get("header", {}).get("competitions", [])
        if comps and isinstance(comps[0], dict):
            return safe_str(comps[0].get("id") or default)
    except Exception:
        pass
    return safe_str(default)


def fetch_core_odds(event_id: str, competition_id: str):
    event_id = safe_str(event_id)
    competition_id = safe_str(competition_id or event_id)
    if not event_id or not competition_id:
        return None

    url = CORE_ODDS_URL.format(event_id=event_id, competition_id=competition_id)
    try:
        resp = requests_with_retry(url, timeout=SUMMARY_TIMEOUT)
        if resp is None:
            return None
        return resp.json()
    except Exception:
        return None


def _extract_market_payload_from_core_odds(core_json: dict):
    if not isinstance(core_json, dict):
        return None

    items = core_json.get("items")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                return item
    return None


def extract_market_from_summary(
    summary_json: dict,
    home_abbr: str,
    away_abbr: str,
    event_id: str = "",
    competition_id: str = "",
):
    out = {
        "odds_details": "N/A",
        "odds_over_under": 0.0,
        "home_is_favorite": -1,
        "odds_data_quality": "fallback",
    }

    if not isinstance(summary_json, dict):
        return out

    pickcenter = summary_json.get("pickcenter")
    market_source = None
    market_payload = None

    if isinstance(pickcenter, list) and pickcenter:
        for item in pickcenter:
            if isinstance(item, dict):
                market_payload = item
                market_source = "summary_pickcenter"
                break

    if market_payload is None:
        odds = summary_json.get("odds")
        if isinstance(odds, list) and odds:
            for item in odds:
                if isinstance(item, dict):
                    market_payload = item
                    market_source = "summary_odds"
                    break

    if isinstance(market_payload, dict):
        out.update(_extract_market_from_payload(market_payload, home_abbr, away_abbr, market_source or ""))

    needs_core_fallback = (
        out.get("odds_data_quality") != "real"
        or safe_float(out.get("odds_over_under"), 0.0) <= 0.0
        or safe_int(out.get("home_is_favorite"), -1) == -1
    )
    if needs_core_fallback:
        competition_id = safe_str(competition_id) or _extract_competition_id_from_summary(summary_json, event_id)
        core_json = fetch_core_odds(event_id=event_id, competition_id=competition_id)
        core_payload = _extract_market_payload_from_core_odds(core_json)
        if isinstance(core_payload, dict):
            core_out = _extract_market_from_payload(core_payload, home_abbr, away_abbr, "core_odds")
            if (
                core_out.get("odds_data_quality") == "real"
                or safe_float(core_out.get("odds_over_under"), 0.0) > safe_float(out.get("odds_over_under"), 0.0)
                or safe_int(core_out.get("home_is_favorite"), -1) != -1
            ):
                out.update(core_out)

    return out


def _extract_pitcher_from_athletes_block(team_block: dict):
    result = {
        "pitcher_name": "",
        "pitcher_id": "",
        "ip": None,
        "er": None,
        "hits_allowed": None,
        "bb_allowed": None,
        "k": None,
        "hr_allowed": None,
        "source": "",
    }

    if not isinstance(team_block, dict):
        return result

    # ESPN MLB summary payloads typically store pitchers under
    # team_block["statistics"][type="pitching"]["athletes"], with the stat names
    # in `keys` and the values in each athlete row's `stats`.
    stat_groups = team_block.get("statistics")
    if isinstance(stat_groups, list):
        for stat_group in stat_groups:
            if not isinstance(stat_group, dict):
                continue
            if safe_str(stat_group.get("type")).lower() != "pitching":
                continue

            keys = [safe_str(k) for k in (stat_group.get("keys") or [])]
            athletes = stat_group.get("athletes") or []
            preferred = []
            fallback = []

            for item in athletes:
                if not isinstance(item, dict):
                    continue

                athlete = item.get("athlete") or {}
                position = item.get("position") or athlete.get("position") or {}
                pos_abbr = safe_str(position.get("abbreviation")).upper()
                is_pitcher = pos_abbr == "P" or bool(item.get("starter"))
                if not is_pitcher:
                    continue

                display_name = normalize_pitcher_name(
                    athlete.get("displayName") or athlete.get("shortName") or athlete.get("name")
                )
                pitcher_id = safe_str(athlete.get("id"))
                if not display_name:
                    continue

                flat = {}
                raw_stats = item.get("stats") or []
                if isinstance(raw_stats, list) and keys:
                    for idx, key in enumerate(keys):
                        if idx < len(raw_stats):
                            flat[key.lower()] = raw_stats[idx]

                candidate = {
                    "pitcher_name": display_name,
                    "pitcher_id": pitcher_id,
                    "ip": flat.get("fullinnings.partinnings") or flat.get("inningspitched") or flat.get("ip"),
                    "er": flat.get("earnedruns") or flat.get("er"),
                    "hits_allowed": flat.get("hitsallowed") or flat.get("hits"),
                    "bb_allowed": flat.get("walks") or flat.get("bb"),
                    "k": flat.get("strikeouts") or flat.get("k"),
                    "hr_allowed": flat.get("homerunsallowed") or flat.get("homeruns") or flat.get("hr"),
                    "source": "summary_statistics_pitching",
                }

                if bool(item.get("starter")):
                    preferred.append(candidate)
                else:
                    fallback.append(candidate)

            if preferred:
                return preferred[0]
            if fallback:
                return fallback[0]

    candidate_lists = []
    for key in ["athletes", "leaders", "probables", "starters"]:
        value = team_block.get(key)
        if isinstance(value, list):
            candidate_lists.append((key, value))

    for source_key, items in candidate_lists:
        for item in items:
            if not isinstance(item, dict):
                continue

            athlete = item.get("athlete") or item.get("player") or item
            position = athlete.get("position", {}) if isinstance(athlete, dict) else {}
            pos_abbr = str(position.get("abbreviation", "")).upper()

            display_name = ""
            pitcher_id = ""

            if isinstance(athlete, dict):
                display_name = athlete.get("displayName") or athlete.get("shortName") or athlete.get("name") or ""
                pitcher_id = athlete.get("id") or ""
            else:
                display_name = str(athlete)

            display_name = normalize_pitcher_name(display_name)
            pitcher_id = safe_str(pitcher_id)

            is_pitcher = pos_abbr == "P"

            if is_pitcher or source_key in {"probables", "starters"}:
                result["pitcher_name"] = display_name
                result["pitcher_id"] = pitcher_id
                result["source"] = f"summary_{source_key}"

                stats = item.get("statistics") or (athlete.get("statistics") if isinstance(athlete, dict) else None)
                if isinstance(stats, list):
                    flat = {}
                    for s in stats:
                        if not isinstance(s, dict):
                            continue
                        for stat_item in s.get("stats", []) or []:
                            name = stat_item.get("name")
                            val = stat_item.get("value")
                            if name is not None:
                                flat[str(name).lower()] = val

                    result["ip"] = flat.get("inningspitched") or flat.get("ip")
                    result["er"] = flat.get("earnedruns") or flat.get("er")
                    result["hits_allowed"] = flat.get("hitsallowed") or flat.get("hits")
                    result["bb_allowed"] = flat.get("walks") or flat.get("bb")
                    result["k"] = flat.get("strikeouts") or flat.get("k")
                    result["hr_allowed"] = flat.get("homerunsallowed") or flat.get("hr")

                return result

    return result


def _extract_pitchers_from_boxscore(summary_json: dict):
    home_result = {
        "pitcher_name": "",
        "pitcher_id": "",
        "ip": None,
        "er": None,
        "hits_allowed": None,
        "bb_allowed": None,
        "k": None,
        "hr_allowed": None,
        "source": "",
    }
    away_result = dict(home_result)

    try:
        boxscore = summary_json.get("boxscore", {})
        players = boxscore.get("players", [])
        if not isinstance(players, list) or len(players) < 2:
            return home_result, away_result

        extracted = []
        for team_block in players:
            extracted.append(_extract_pitcher_from_athletes_block(team_block))

        if len(extracted) >= 2:
            return extracted[0], extracted[1]
    except Exception:
        pass

    return home_result, away_result


def extract_pitchers_from_summary(summary_json: dict, home_abbr: str, away_abbr: str):
    out = {
        "home_starting_pitcher": "",
        "away_starting_pitcher": "",
        "home_starting_pitcher_id": "",
        "away_starting_pitcher_id": "",
        "home_starting_pitcher_ip": np.nan,
        "away_starting_pitcher_ip": np.nan,
        "home_starting_pitcher_er": np.nan,
        "away_starting_pitcher_er": np.nan,
        "home_starting_pitcher_hits": np.nan,
        "away_starting_pitcher_hits": np.nan,
        "home_starting_pitcher_bb": np.nan,
        "away_starting_pitcher_bb": np.nan,
        "home_starting_pitcher_k": np.nan,
        "away_starting_pitcher_k": np.nan,
        "home_starting_pitcher_hr": np.nan,
        "away_starting_pitcher_hr": np.nan,
        "pitcher_source": "",
        "pitcher_data_available": 0,
        "pitcher_stats_available": 0,
    }

    if not isinstance(summary_json, dict):
        return out

    home_comp, away_comp = _extract_competitors_from_summary(summary_json)

    home_name = ""
    home_id = ""
    away_name = ""
    away_id = ""

    if isinstance(home_comp, dict):
        name, pid, _ = extract_competitor_pitcher(home_comp)
        home_name, home_id = name, pid

    if isinstance(away_comp, dict):
        name, pid, _ = extract_competitor_pitcher(away_comp)
        away_name, away_id = name, pid

    home_box, away_box = _extract_pitchers_from_boxscore(summary_json)

    if not home_name and home_box["pitcher_name"]:
        home_name = home_box["pitcher_name"]
        home_id = home_box["pitcher_id"]
    if not away_name and away_box["pitcher_name"]:
        away_name = away_box["pitcher_name"]
        away_id = away_box["pitcher_id"]

    out["home_starting_pitcher"] = normalize_pitcher_name(home_name)
    out["away_starting_pitcher"] = normalize_pitcher_name(away_name)
    out["home_starting_pitcher_id"] = safe_str(home_id)
    out["away_starting_pitcher_id"] = safe_str(away_id)

    for prefix, box in [("home", home_box), ("away", away_box)]:
        out[f"{prefix}_starting_pitcher_ip"] = safe_float(box.get("ip"), default=np.nan)
        out[f"{prefix}_starting_pitcher_er"] = safe_float(box.get("er"), default=np.nan)
        out[f"{prefix}_starting_pitcher_hits"] = safe_float(box.get("hits_allowed"), default=np.nan)
        out[f"{prefix}_starting_pitcher_bb"] = safe_float(box.get("bb_allowed"), default=np.nan)
        out[f"{prefix}_starting_pitcher_k"] = safe_float(box.get("k"), default=np.nan)
        out[f"{prefix}_starting_pitcher_hr"] = safe_float(box.get("hr_allowed"), default=np.nan)

    home_ok = safe_str(out["home_starting_pitcher"]).strip() != ""
    away_ok = safe_str(out["away_starting_pitcher"]).strip() != ""
    out["pitcher_data_available"] = int(home_ok and away_ok)

    stats_cols = [
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
    ]
    out["pitcher_stats_available"] = int(any(pd.notna(out[c]) for c in stats_cols))

    if out["pitcher_stats_available"]:
        out["pitcher_source"] = "summary_boxscore_players"
    elif out["pitcher_data_available"]:
        out["pitcher_source"] = "summary_competitors_or_probables"

    return out


def _extract_pitchers_from_mlb_payload(payload: dict, home_abbr: str, away_abbr: str):
    # Best-effort extraction from cached external payloads (MLB stats / other)
    out = {
        "home_starting_pitcher": "",
        "away_starting_pitcher": "",
        "home_starting_pitcher_id": "",
        "away_starting_pitcher_id": "",
        "home_starting_pitcher_ip": np.nan,
        "away_starting_pitcher_ip": np.nan,
        "home_starting_pitcher_er": np.nan,
        "away_starting_pitcher_er": np.nan,
        "home_starting_pitcher_hits": np.nan,
        "away_starting_pitcher_hits": np.nan,
        "home_starting_pitcher_bb": np.nan,
        "away_starting_pitcher_bb": np.nan,
        "home_starting_pitcher_k": np.nan,
        "away_starting_pitcher_k": np.nan,
        "home_starting_pitcher_hr": np.nan,
        "away_starting_pitcher_hr": np.nan,
        "pitcher_source": "external_payload",
        "pitcher_data_available": 0,
        "pitcher_stats_available": 0,
    }

    if not isinstance(payload, dict):
        return out

    # Check for MLB boxscore structure
    mlb = payload.get("mlb_boxscore") if isinstance(payload.get("mlb_boxscore"), dict) else payload

    # Look for 'probablePitchers' style objects
    def _search_probables(obj):
        found = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k and "probable" in k.lower() and isinstance(v, list):
                    for item in v:
                        name = safe_str(item.get("fullName") or item.get("displayName") or item.get("name"))
                        team = safe_str(item.get("team", {}).get("abbreviation") or item.get("team"))
                        pid = safe_str(item.get("id") or item.get("personId") or item.get("playerId"))
                        if name:
                            found.append((name, pid, team))
        return found

    probables = _search_probables(mlb)
    if probables:
        for name, pid, team in probables:
            if team.upper() == safe_str(home_abbr).upper():
                out["home_starting_pitcher"] = normalize_pitcher_name(name)
                out["home_starting_pitcher_id"] = pid
            elif team.upper() == safe_str(away_abbr).upper():
                out["away_starting_pitcher"] = normalize_pitcher_name(name)
                out["away_starting_pitcher_id"] = pid

    # Fallback: scan for player entries with position P
    if not out["home_starting_pitcher"] or not out["away_starting_pitcher"]:
        def _find_pitchers_in_players(obj):
            found = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, dict):
                        # player blocks sometimes contain 'person' + 'position'
                        person = v.get("person") or v.get("player") or v
                        position = v.get("position") or (v.get("person", {}) or {}).get("position")
                        team = v.get("team") or (v.get("person", {}) or {}).get("team")
                        if isinstance(position, dict) and str(position.get("abbreviation", "")).upper() == "P":
                            name = safe_str((person.get("fullName") or person.get("displayName") or person.get("name")))
                            pid = safe_str(person.get("id") or person.get("personId") or person.get("playerId"))
                            team_abbr = safe_str(team.get("abbreviation") if isinstance(team, dict) else team)
                            if name:
                                found.append((name, pid, team_abbr))
            return found

        players_found = _find_pitchers_in_players(mlb)
        for name, pid, team in players_found:
            if not out["home_starting_pitcher"] and team.upper() == safe_str(home_abbr).upper():
                out["home_starting_pitcher"] = normalize_pitcher_name(name)
                out["home_starting_pitcher_id"] = pid
            if not out["away_starting_pitcher"] and team.upper() == safe_str(away_abbr).upper():
                out["away_starting_pitcher"] = normalize_pitcher_name(name)
                out["away_starting_pitcher_id"] = pid

    home_ok = safe_str(out["home_starting_pitcher"]).strip() != ""
    away_ok = safe_str(out["away_starting_pitcher"]).strip() != ""
    out["pitcher_data_available"] = int(home_ok and away_ok)

    return out


# =========================
# HISTORICAL ENRICHMENT
# =========================
def enrich_rows_with_historical_pitchers(rows: list[dict], label: str = "históricos") -> list[dict]:
    print(f"\n🎯 Enriqueciendo pitchers {label} por game_id (summary endpoint)...")

    enriched = []
    total = len(rows)

    target_rows = []
    skipped_ready = 0

    for row in rows:
        home_name = safe_str(row.get("home_starting_pitcher"))
        away_name = safe_str(row.get("away_starting_pitcher"))

        invalid_tokens = {"", "NAN", "NONE", "NULL", "0"}
        has_home_name = home_name.strip().upper() not in invalid_tokens
        has_away_name = away_name.strip().upper() not in invalid_tokens
        has_both_names = has_home_name and has_away_name

        pitcher_flag_raw = pd.to_numeric(row.get("pitcher_data_available", 0), errors="coerce")
        pitcher_flag = 0 if pd.isna(pitcher_flag_raw) else int(pitcher_flag_raw)
        stats_flag_raw = pd.to_numeric(row.get("pitcher_stats_available", 0), errors="coerce")
        stats_flag = 0 if pd.isna(stats_flag_raw) else int(stats_flag_raw)
        row_completed_raw = pd.to_numeric(row.get("status_completed", 0), errors="coerce")
        row_completed = 0 if pd.isna(row_completed_raw) else int(row_completed_raw)
        market_quality = safe_str(row.get("odds_data_quality")).lower()
        recent_market_candidate = is_recent_market_refresh_candidate(row)

        should_enrich = True
        if ONLY_ENRICH_MISSING_PITCHERS:
            # Completed games need names + pitcher stats + market data. Pregame
            # rows only require starter names + market data.
            pitcher_ready = has_both_names and pitcher_flag == 1
            if row_completed == 1:
                pitcher_ready = pitcher_ready and stats_flag == 1
            market_ready = (market_quality == "real") or (not recent_market_candidate)
            should_enrich = not (pitcher_ready and market_ready)

        if should_enrich:
            target_rows.append(row)
        else:
            skipped_ready += 1
            enriched.append(row)

    if MAX_ENRICH_ROWS_PER_RUN is not None:
        target_rows = target_rows[:MAX_ENRICH_ROWS_PER_RUN]

    pending_total = len(target_rows)

    print(f"   Total rows en dataset     : {total}")
    print(f"   Ya completos / omitidos   : {skipped_ready}")
    print(f"   Pendientes por enriquecer : {pending_total}")

    if pending_total == 0:
        print("   ✅ No hay juegos pendientes de enriquecer.")
        return rows

    found_names = 0
    found_stats = 0
    cache_hits = 0
    api_hits = 0
    api_misses = 0

    start_time = time.time()

    # Parallelize enrichment to speed up network-bound work
    max_workers = min(6, max(2, (pending_total // 10) or 2))

    def _process_single_row(row: dict):
        game_id = safe_str(row.get("game_id"))
        stats = {"cache_hit": 0, "api_hit": 0, "api_miss": 0, "found_name": 0, "found_stats": 0}
        if not game_id:
            return row, stats

        pitcher_info = load_pitcher_cache(game_id)
        if pitcher_info is not None:
            cached_stats_flag = pd.to_numeric(pitcher_info.get("pitcher_stats_available", 0), errors="coerce")
            cached_market_quality = safe_str(pitcher_info.get("odds_data_quality")).lower()
            row_completed = int(pd.to_numeric(row.get("status_completed", 0), errors="coerce") or 0)
            row_market_quality = safe_str(row.get("odds_data_quality")).lower()
            recent_market_candidate = is_recent_market_refresh_candidate(row)
            # Invalidate stale historical cache entries created before the improved
            # ESPN parser if they still don't contain pitcher stats or real market data.
            needs_stats_refresh = row_completed == 1 and (
                pd.isna(cached_stats_flag) or float(cached_stats_flag) < 1.0
            )
            needs_market_refresh = (
                recent_market_candidate
                and row_market_quality != "real"
                and cached_market_quality != "real"
            )
            if needs_stats_refresh or needs_market_refresh:
                pitcher_info = None

        if pitcher_info is not None:
            stats["cache_hit"] = 1
        else:
            # ESPN summary
            summary_json = fetch_game_summary(game_id)
            if summary_json:
                pitcher_info = extract_pitchers_from_summary(
                    summary_json=summary_json,
                    home_abbr=safe_str(row.get("home_team")),
                    away_abbr=safe_str(row.get("away_team")),
                )
                pitcher_info.update(
                    extract_market_from_summary(
                        summary_json=summary_json,
                        home_abbr=safe_str(row.get("home_team")),
                        away_abbr=safe_str(row.get("away_team")),
                        event_id=game_id,
                        competition_id=safe_str(row.get("competition_id") or game_id),
                    )
                )
                save_pitcher_cache(game_id, pitcher_info)
                stats["api_hit"] = 1
            else:
                external = load_external_cache(game_id)
                if external is not None:
                    pitcher_info = _extract_pitchers_from_mlb_payload(
                        external, safe_str(row.get("home_team")), safe_str(row.get("away_team"))
                    )
                    if pitcher_info and pitcher_info.get("pitcher_data_available", 0):
                        save_pitcher_cache(game_id, pitcher_info)
                        stats["api_hit"] = 1
                    else:
                        mlb_payload = fetch_from_mlb_stats_api_if_possible(game_id)
                        if mlb_payload:
                            pitcher_info = _extract_pitchers_from_mlb_payload(
                                mlb_payload, safe_str(row.get("home_team")), safe_str(row.get("away_team"))
                            )
                            if pitcher_info and pitcher_info.get("pitcher_data_available", 0):
                                save_pitcher_cache(game_id, pitcher_info)
                                stats["api_hit"] = 1
                            else:
                                pitcher_info = {}
                                save_pitcher_cache(game_id, pitcher_info)
                                stats["api_miss"] = 1
                        else:
                            pitcher_info = {}
                            save_pitcher_cache(game_id, pitcher_info)
                            stats["api_miss"] = 1

            time.sleep(REQUEST_SLEEP_SECONDS)

        if pitcher_info:
            for k, v in pitcher_info.items():
                is_nan_float = isinstance(v, float) and np.isnan(v)
                if k not in row or (v not in ["", None] and not is_nan_float):
                    row[k] = v

        stats["found_name"] = int(bool(safe_str(row.get("home_starting_pitcher"))) and bool(safe_str(row.get("away_starting_pitcher"))))
        stats_flag = pd.to_numeric(row.get("pitcher_stats_available", 0), errors="coerce")
        stats["found_stats"] = int((not pd.isna(stats_flag)) and float(stats_flag) >= 1.0)
        return row, stats

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_single_row, r) for r in target_rows]
        try:
            from tqdm import tqdm
            futures_iter = tqdm(as_completed(futures), total=len(futures), desc="Pitchers", unit="game")
        except Exception:
            futures_iter = as_completed(futures)

        completed = 0
        for fut in futures_iter:
            try:
                row_out, s = fut.result()
            except Exception:
                # in case of unexpected error, append raw
                row_out = {}
                s = {"cache_hit": 0, "api_hit": 0, "api_miss": 1, "found_name": 0, "found_stats": 0}

            enriched.append(row_out)
            cache_hits += int(s.get("cache_hit", 0))
            api_hits += int(s.get("api_hit", 0))
            api_misses += int(s.get("api_miss", 0))
            found_names += int(s.get("found_name", 0))
            found_stats += int(s.get("found_stats", 0))

            completed += 1
            if completed % 25 == 0 or completed == pending_total:
                elapsed = time.time() - start_time
                speed = completed / elapsed if elapsed > 0 else 0.0
                eta = (pending_total - completed) / speed if speed > 0 else 0.0
                print(
                    f"   > {completed}/{pending_total} ({(completed/pending_total)*100:.1f}%) | "
                    f"{speed:.2f} game/s | ETA {eta/60:.1f} min | "
                    f"cache={cache_hits} | api_ok={api_hits} | api_fail={api_misses} | "
                    f"names={found_names} | stats={found_stats}"
                )

    enriched_game_ids = {safe_str(r.get("game_id")) for r in enriched}
    for row in rows:
        gid = safe_str(row.get("game_id"))
        if gid not in enriched_game_ids:
            enriched.append(row)

    out_df = pd.DataFrame(enriched)
    if not out_df.empty and "game_id" in out_df.columns:
        out_df["game_id"] = out_df["game_id"].astype(str)
        out_df = out_df.drop_duplicates(subset=["game_id"], keep="last").reset_index(drop=True)

    print(f"   Cache hits             : {cache_hits}")
    print(f"   API hits               : {api_hits}")
    print(f"   API misses             : {api_misses}")
    print(f"   Pitcher names found    : {found_names}")
    print(f"   Pitcher stats found    : {found_stats}")

    return out_df.to_dict(orient="records")


# =========================
# MAIN INGEST
# =========================
def extract_advanced_espn_data():
    print("🚀 Iniciando Extractor Avanzado de ESPN MLB...")

    existing_df = load_existing_data()
    previous_count = len(existing_df)

    fetch_ranges = determine_fetch_ranges(existing_df)

    if fetch_ranges:
        new_df = fetch_games_for_ranges(fetch_ranges)
    else:
        print("✅ No hay rangos nuevos por consultar.")
        new_df = pd.DataFrame()

    if existing_df.empty and new_df.empty:
        final_df = pd.DataFrame()
    elif existing_df.empty:
        all_rows = new_df.to_dict(orient="records")
        all_rows = enrich_rows_with_historical_pitchers(all_rows)
        final_df = pd.DataFrame(all_rows)
    elif new_df.empty:
        all_rows = existing_df.to_dict(orient="records")
        all_rows = enrich_rows_with_historical_pitchers(all_rows)
        final_df = pd.DataFrame(all_rows)
    else:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df["game_id"] = combined_df["game_id"].astype(str)
        combined_df = combined_df.drop_duplicates(subset=["game_id"], keep="last").reset_index(drop=True)

        all_rows = combined_df.to_dict(orient="records")
        all_rows = enrich_rows_with_historical_pitchers(all_rows)
        final_df = pd.DataFrame(all_rows)

    if not final_df.empty:
        final_df["game_id"] = final_df["game_id"].astype(str)
        final_df["date"] = final_df["date"].astype(str)

        # limpiar flags por si vienen mezclados desde cache/CSV
        for col in ["pitcher_data_available", "pitcher_stats_available"]:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(final_df[col], errors="coerce")

        before_dedup = len(final_df)
        final_df = final_df.drop_duplicates(subset=["game_id"], keep="last")
        dedup_removed = before_dedup - len(final_df)

        if "date_dt" in final_df.columns:
            final_df = final_df.drop(columns=["date_dt"])

        final_df = final_df.sort_values(["date", "game_id"], ascending=[False, False]).reset_index(drop=True)
        final_df.to_csv(FILE_PATH_ADVANCED, index=False)
    else:
        dedup_removed = 0

    added_count = len(final_df) - previous_count if not final_df.empty else 0

    print("\n📊 RESUMEN DE ACTUALIZACIÓN MLB")
    print(f"   Partidos previos   : {previous_count}")
    print(f"   Filas descargadas  : {len(new_df)}")
    print(f"   Duplicados quitados: {dedup_removed}")
    print(f"   Partidos finales   : {len(final_df)}")
    print(f"   Netos añadidos     : {added_count}")

    print(f"\n💾 Histórico actualizado en: {FILE_PATH_ADVANCED}")

    if not final_df.empty and "pitcher_data_available" in final_df.columns:
        pitcher_flag = pd.to_numeric(final_df["pitcher_data_available"], errors="coerce").fillna(0)
        pitcher_cov = float(pitcher_flag.mean())
        print(f"   Cobertura pitchers : {pitcher_cov:.2%}")

        if "home_starting_pitcher" in final_df.columns:
            unique_home_pitchers = final_df["home_starting_pitcher"].fillna("").astype(str)
            unique_home_pitchers = unique_home_pitchers[
                ~unique_home_pitchers.str.upper().isin(["", "NAN", "NONE", "NULL", "0"])
            ].nunique()
            print(f"   Home starters únicos: {unique_home_pitchers}")

        if "away_starting_pitcher" in final_df.columns:
            unique_away_pitchers = final_df["away_starting_pitcher"].fillna("").astype(str)
            unique_away_pitchers = unique_away_pitchers[
                ~unique_away_pitchers.str.upper().isin(["", "NAN", "NONE", "NULL", "0"])
            ].nunique()
            print(f"   Away starters únicos: {unique_away_pitchers}")

    if not final_df.empty and "pitcher_stats_available" in final_df.columns:
        stats_flag = pd.to_numeric(final_df["pitcher_stats_available"], errors="coerce").fillna(0)
        stats_cov = float(stats_flag.mean())
        print(f"   Cobertura pitcher stats : {stats_cov:.2%}")

    today_str = datetime.now().strftime("%Y-%m-%d")
    df_upcoming = fetch_upcoming_schedule_for_range(today_str)

    if not df_upcoming.empty:
        df_upcoming.to_csv(FILE_PATH_UPCOMING, index=False)
        print(f"🗓️ Agenda rolling guardada en: {FILE_PATH_UPCOMING}")
        print(f"   Juegos totales agenda: {len(df_upcoming)}")
        print(
            f"   Estados: {sorted(df_upcoming['status_state'].dropna().astype(str).unique().tolist())}"
        )
    else:
        print("⚠️ No se encontraron juegos programados para hoy en la agenda MLB.")

    return final_df


if __name__ == "__main__":
    df_advanced = extract_advanced_espn_data()
    if not df_advanced.empty:
        print(df_advanced.head())
