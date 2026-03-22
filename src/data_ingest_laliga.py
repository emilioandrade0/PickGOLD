import time
from datetime import datetime, timedelta
from pathlib import Path
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from odds_market_fields import extract_market_odds_fields, odds_data_quality

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "data" / "laliga" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

FILE_PATH_ADVANCED = RAW_DATA_DIR / "laliga_advanced_history.csv"
FILE_PATH_UPCOMING = RAW_DATA_DIR / "laliga_upcoming_schedule.csv"

SEASONS_TO_FETCH = {
    "2025": ("2025-01-01", "2025-12-31"),
    "2026": ("2026-01-01", "2026-12-31"),
}

TARGET_DATE_LIMIT = datetime.now().strftime("%Y-%m-%d")
BACKFILL_DAYS = 5
LOCAL_UTC_OFFSET_HOURS = 6
UPCOMING_DAYS_AHEAD = 14

ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/esp.1/scoreboard"
)
ESPN_SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/esp.1/summary?event={event_id}"
)


# -----------------------------
# Helpers
# -----------------------------

def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ASCII", "ignore").decode("ASCII")
    return text.strip().upper()


def normalize_team_abbr(team_data: dict):
    abbr = normalize_text(team_data.get("abbreviation") or "")
    display_name = normalize_text(
        team_data.get("displayName") or team_data.get("shortDisplayName") or ""
    )

    alias_by_abbr = {
        "AME": "AME",
        "ATS": "ATS",
        "CAZ": "CAZ",
        "GDL": "GDL",
        "CHI": "GDL",
        "JUA": "JUA",
        "LEO": "LEO",
        "MAZ": "MAZ",
        "MTY": "MTY",
        "NEC": "NEC",
        "PAC": "PAC",
        "PUE": "PUE",
        "QRO": "QRO",
        "SAN": "SAN",
        "SLP": "SLP",
        "TIG": "TIG",
        "TIJ": "TIJ",
        "TOL": "TOL",
        "PUM": "PUM",
    }

    alias_by_name = {
        "AMERICA": "AME",
        "CLUB AMERICA": "AME",
        "ATLAS": "ATS",
        "CRUZ AZUL": "CAZ",
        "GUADALAJARA": "GDL",
        "CHIVAS": "GDL",
        "JUAREZ": "JUA",
        "FC JUAREZ": "JUA",
        "LEON": "LEO",
        "MAZATLAN": "MAZ",
        "MAZATLAN FC": "MAZ",
        "MONTERREY": "MTY",
        "RAYADOS": "MTY",
        "NECAXA": "NEC",
        "PACHUCA": "PAC",
        "PUEBLA": "PUE",
        "QUERETARO": "QRO",
        "SANTOS": "SAN",
        "SANTOS LAGUNA": "SAN",
        "ATLETICO SAN LUIS": "SLP",
        "SAN LUIS": "SLP",
        "TIGRES": "TIG",
        "TIJUANA": "TIJ",
        "XOLOS": "TIJ",
        "TOLUCA": "TOL",
        "PUMAS": "PUM",
        "PUMAS UNAM": "PUM",
        "UNAM": "PUM",
    }

    if abbr in alias_by_abbr:
        return alias_by_abbr[abbr]

    if display_name in alias_by_name:
        return alias_by_name[display_name]

    if abbr:
        return abbr

    return display_name[:3]


def _extract_corners_from_stats(stats_list):
    if not isinstance(stats_list, list):
        return None
    for stat in stats_list:
        name = str((stat or {}).get("name", "") or "").strip().lower()
        if name in {"woncorners", "corners"}:
            val = (stat or {}).get("displayValue")
            parsed = safe_int(val, default=None)
            if parsed is not None:
                return parsed
    return None


def fetch_event_corners(event_id: str):
    try:
        url = ESPN_SUMMARY_URL.format(event_id=event_id)
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        payload = resp.json() or {}
    except Exception:
        return 0, 0

    teams = (payload.get("boxscore") or {}).get("teams") or []
    if len(teams) < 2:
        return 0, 0

    home_corners = 0
    away_corners = 0
    for side in teams:
        team_meta = side.get("team") or {}
        side_key = str(team_meta.get("homeAway", "") or "").lower()
        corners = _extract_corners_from_stats(side.get("statistics") or [])
        if corners is None:
            continue
        if side_key == "home":
            home_corners = corners
        elif side_key == "away":
            away_corners = corners

    return home_corners, away_corners


# -----------------------------
# Existing Data
# -----------------------------

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
        print(f"âš ï¸ No se pudo leer el CSV existente. Se reconstruirÃ¡ desde cero. Error: {e}")
        return pd.DataFrame()


# -----------------------------
# Ranges
# -----------------------------

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
        print("ðŸ“‚ No existe histÃ³rico previo. Se harÃ¡ descarga completa.")
        return build_full_ranges(limit_date_dt)

    try:

        tmp = existing_df.copy()
        tmp["date_dt"] = pd.to_datetime(tmp["date"], errors="coerce")

        max_date = tmp["date_dt"].max()

        if pd.isna(max_date):
            print("âš ï¸ No se pudo detectar la Ãºltima fecha del histÃ³rico.")
            return build_full_ranges(limit_date_dt)

        incremental_start = (max_date - timedelta(days=BACKFILL_DAYS)).normalize()

        print(
            f"ðŸ“Œ Ãšltima fecha en histÃ³rico: {max_date.strftime('%Y-%m-%d')} | "
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

        print(f"âš ï¸ Error calculando rango incremental: {e}")
        return build_full_ranges(limit_date_dt)


# -----------------------------
# Parse Event
# -----------------------------

def parse_event_to_row(event: dict, season: str | None = None):

    if not isinstance(event, dict):
        return None

    competitions = event.get("competitions") or []
    if not competitions:
        return None

    comp = competitions[0] or {}

    competitors = comp.get("competitors") or []
    if len(competitors) < 2:
        return None

    home_data = next((c for c in competitors if (c or {}).get("homeAway") == "home"), None)
    away_data = next((c for c in competitors if (c or {}).get("homeAway") == "away"), None)

    if not home_data or not away_data:
        return None

    event_status = event.get("status") or {}
    comp_status = comp.get("status") or {}

    status_parent = comp_status if comp_status else event_status
    status_type = status_parent.get("type") or {}

    completed = bool(status_type.get("completed", False))
    state = str(status_type.get("state", "") or "")
    description = str(status_parent.get("description", "") or "")
    detail = str(status_parent.get("detail", "") or "")

    event_date = str(event.get("date") or "").strip()
    if not event_date:
        return None

    dt_utc = None

    for fmt in ("%Y-%m-%dT%H:%MZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt_utc = datetime.strptime(event_date, fmt)
            break
        except ValueError:
            pass

    if dt_utc is None:
        return None

    local_dt = dt_utc - timedelta(hours=LOCAL_UTC_OFFSET_HOURS)

    game_date = local_dt.strftime("%Y-%m-%d")
    game_time = local_dt.strftime("%H:%M")

    home_team = normalize_team_abbr(home_data.get("team") or {})
    away_team = normalize_team_abbr(away_data.get("team") or {})

    home_score = safe_int(home_data.get("score"))
    away_score = safe_int(away_data.get("score"))
    home_corners = _extract_corners_from_stats(home_data.get("statistics") or [])
    away_corners = _extract_corners_from_stats(away_data.get("statistics") or [])
    if home_corners is None:
        home_corners = 0
    if away_corners is None:
        away_corners = 0

    venue = comp.get("venue") or {}

    odds_raw = comp.get("odds") or []

    if isinstance(odds_raw, list) and odds_raw:
        odds = odds_raw[0] or {}
    else:
        odds = {}

    odds_details = str(odds.get("details", "N/A"))
    odds_over_under = safe_float(odds.get("overUnder"))
    market_odds_fields = extract_market_odds_fields(odds)

    game_id = str(event.get("id") or "").strip()
    if not game_id:
        return None

    season_value = season if season else str(local_dt.year)

    return {
        "game_id": game_id,
        "date": game_date,
        "time": game_time,
        "season": season_value,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "home_corners": int(home_corners),
        "away_corners": int(away_corners),
        "total_corners": int(home_corners) + int(away_corners),
        "goal_diff": home_score - away_score,
        "total_goals": home_score + away_score,
        "is_draw": int(home_score == away_score),
        "home_win": int(home_score > away_score),
        "away_win": int(away_score > home_score),
        "attendance": safe_int(comp.get("attendance")),
        "venue": str(venue.get("fullName", "")),
        "odds_details": odds_details,
        "odds_over_under": odds_over_under,
        **market_odds_fields,
        "odds_data_quality": odds_data_quality(market_odds_fields),
        "shootout": int(bool(comp.get("shootout"))),
        "status_completed": int(completed),
        "status_state": state,
        "status_description": description,
        "status_detail": detail,
    }


# -----------------------------
# Fetch ESPN
# -----------------------------

def fetch_games_for_ranges(ranges):

    completed_games = []

    for season, start_dt, end_dt in ranges:

        print(
            f"   > Escaneando temporada {season}: "
            f"{start_dt.strftime('%Y-%m-%d')} -> {end_dt.strftime('%Y-%m-%d')}"
        )

        chunks = []
        current_dt = start_dt
        while current_dt <= end_dt:
            chunk_end_dt = min(current_dt + timedelta(days=14), end_dt)
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
                    events = data.get("events") or []
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
                    print(f"âš ï¸ Error al consultar chunk {d1}-{d2}: {error}")
                    continue

                for event in events:
                    try:
                        row = parse_event_to_row(event, season=season)
                        if row and row["status_completed"]:
                            # If scoreboard payload did not include corners, recover from summary endpoint.
                            if int(row.get("total_corners", 0)) == 0:
                                hc, ac = fetch_event_corners(str(row.get("game_id")))
                                row["home_corners"] = int(hc)
                                row["away_corners"] = int(ac)
                                row["total_corners"] = int(hc) + int(ac)
                            completed_games.append(row)
                    except Exception as inner_e:
                        print(f"âš ï¸ Error procesando evento: {inner_e}")

        print("     -> âœ… Temporada procesada.")

    return pd.DataFrame(completed_games)


# -----------------------------
# Upcoming Schedule
# -----------------------------

def fetch_upcoming_schedule_for_range(start_date: str, days_ahead: int = UPCOMING_DAYS_AHEAD):

    session = requests.Session()
    upcoming_rows = []
    base_dt = datetime.strptime(start_date, "%Y-%m-%d")

    for day_offset in range(days_ahead + 1):
        day_dt = base_dt + timedelta(days=day_offset)
        day_str = day_dt.strftime("%Y-%m-%d")
        url = f"{ESPN_SCOREBOARD_URL}?dates={day_dt.strftime('%Y%m%d')}&limit=500"

        try:

            resp = session.get(url, timeout=20)
            resp.raise_for_status()

            data = resp.json() or {}
            events = data.get("events") or []

            day_count = 0
            for event in events:
                row = parse_event_to_row(event, season=None)
                if not row:
                    continue

                # Conservamos todo para hoy; para fechas futuras no agregamos finales.
                if day_str != start_date and int(row.get("status_completed", 0)) == 1:
                    continue

                upcoming_rows.append(row)
                day_count += 1

            print(f"   ðŸ“… Agenda LaLiga EA Sports {day_str}: {day_count} juegos")

        except Exception as e:
            print(f"âš ï¸ Error descargando agenda LaLiga EA Sports {day_str}: {e}")

    df = pd.DataFrame(upcoming_rows)

    if not df.empty:
        df = (
            df.sort_values(["date", "time", "game_id"])
            .drop_duplicates(subset=["game_id"], keep="last")
            .reset_index(drop=True)
        )

    return df


# -----------------------------
# Main Extractor
# -----------------------------

def extract_advanced_espn_data():

    print("ðŸš€ Iniciando Extractor Avanzado de ESPN LaLiga EA Sports...")

    existing_df = load_existing_data()
    previous_count = len(existing_df)

    fetch_ranges = determine_fetch_ranges(existing_df)

    new_df = fetch_games_for_ranges(fetch_ranges) if fetch_ranges else pd.DataFrame()

    if existing_df.empty and new_df.empty:
        final_df = pd.DataFrame()

    elif existing_df.empty:
        final_df = new_df.copy()

    elif new_df.empty:
        final_df = existing_df.copy()

    else:
        final_df = pd.concat([existing_df, new_df], ignore_index=True)

    if not final_df.empty:

        final_df = final_df.drop_duplicates(subset=["game_id"], keep="last")

        final_df = final_df.sort_values(
            ["date", "time", "game_id"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

        final_df.to_csv(FILE_PATH_ADVANCED, index=False)

    print("\nðŸ“Š RESUMEN DE ACTUALIZACIÃ“N LaLiga EA Sports")
    print(f"   Partidos previos   : {previous_count}")
    print(f"   Filas descargadas  : {len(new_df)}")
    print(f"   Partidos finales   : {len(final_df)}")

    today_str = datetime.now().strftime("%Y-%m-%d")

    df_upcoming = fetch_upcoming_schedule_for_range(today_str)

    if not df_upcoming.empty:
        df_upcoming.to_csv(FILE_PATH_UPCOMING, index=False)
        print(f"ðŸ—“ï¸ Agenda rolling guardada en: {FILE_PATH_UPCOMING}")
        print(f"   Juegos totales agenda: {len(df_upcoming)}")
    else:
        print("âš ï¸ No se encontraron juegos en ventana rolling de LaLiga EA Sports.")

    return final_df


# -----------------------------
# Run
# -----------------------------

if __name__ == "__main__":

    df = extract_advanced_espn_data()

    if not df.empty:
        print(df.head())
