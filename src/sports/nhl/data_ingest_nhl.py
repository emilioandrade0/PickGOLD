"""
Data ingest for NHL games from ESPN API.
Mirrors the structure of Liga MX and MLB data pipelines.
Adds goalie enrichment per game via ESPN summary endpoint.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import sys
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
import requests
from odds_market_fields import extract_market_odds_fields, odds_data_quality

BASE_DIR = SRC_ROOT
RAW_DATA_DIR = BASE_DIR / "data" / "nhl" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

FILE_PATH_ADVANCED = RAW_DATA_DIR / "nhl_advanced_history.csv"
FILE_PATH_UPCOMING = RAW_DATA_DIR / "nhl_upcoming_schedule.csv"

SEASONS_TO_FETCH = {
    "2025": ("2025-10-01", "2025-12-31"),
    "2026": ("2026-01-01", "2026-06-30"),
}

TARGET_DATE_LIMIT = datetime.now().strftime("%Y-%m-%d")
BACKFILL_DAYS = 5
LOCAL_UTC_OFFSET_HOURS = -5  # EST

ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
)

# ESPN summary endpoint for per-event enrichment
ESPN_SUMMARY_URLS = [
    "https://site.web.api.espn.com/apis/site/v2/sports/hockey/nhl/summary",
    "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary",
]

REQUEST_TIMEOUT = 12
SUMMARY_MAX_WORKERS = 8
ENABLE_GOALIE_ENRICHMENT = True


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


def safe_bool_to_int(value):
    return 1 if bool(value) else 0


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ASCII", "ignore").decode("ASCII")
    return text.strip().upper()


def normalize_team_abbr(team_data: dict):
    """
    Normalize NHL team abbreviations from ESPN API.
    """
    abbr = normalize_text(team_data.get("abbreviation") or "")

    nhl_teams = {
        "ANA": "ANA", "BOS": "BOS", "BUF": "BUF", "CGY": "CGY",
        "CAR": "CAR", "CHI": "CHI", "COL": "COL", "DAL": "DAL",
        "DET": "DET", "EDM": "EDM", "FLA": "FLA", "LA": "LAK",
        "LAK": "LAK", "MIN": "MIN", "MTL": "MTL", "NJ": "NJD",
        "NJD": "NJD", "NSH": "NSH", "NYI": "NYI", "NYR": "NYR",
        "OTT": "OTT", "PHI": "PHI", "PIT": "PIT", "SJ": "SJS",
        "SJS": "SJS", "STL": "STL", "TB": "TBL", "TBL": "TBL",
        "TOR": "TOR", "VAN": "VAN", "VGK": "VGK", "WASH": "WSH",
        "WSH": "WSH", "WPG": "WPG", "SEA": "SEA", "UTA": "UTA",
        "UTAH": "UTA", "CBJ": "CBJ",
    }

    return nhl_teams.get(abbr, abbr)


def _request_json(url: str, params: dict | None = None, timeout: int = REQUEST_TIMEOUT):
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json() or {}


def fetch_games_by_date_range(start_date: str, end_date: str) -> list:
    """
    Fetch NHL games from ESPN scoreboard API for a date range.
    """
    games = []
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    date_list = []
    while current_date <= end_dt:
        date_list.append(current_date)
        current_date += timedelta(days=1)

    def fetch_one(day_dt: datetime):
        day_label = day_dt.strftime("%Y-%m-%d")
        date_str = day_dt.strftime("%Y%m%d")
        url = f"{ESPN_SCOREBOARD_URL}?dates={date_str}"

        for attempt in range(3):
            try:
                response = requests.get(url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                data = response.json() or {}
                events = data.get("events", [])
                return day_label, events, None
            except Exception as e:
                if attempt == 2:
                    return day_label, [], str(e)
                time.sleep(0.35 * (attempt + 1))

    max_workers = min(6, max(1, len(date_list)))
    errors = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_one, day_dt) for day_dt in date_list]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Scoreboard {start_date} -> {end_date}",
            unit="día",
        ):
            day_label, events, error = future.result()
            if error is None:
                games.extend(events)
            else:
                errors.append((day_label, error))

    if errors:
        print(f"⚠️ Errores en fetch: {len(errors)} días")
        for day_label, error in errors[:10]:
            print(f"   - {day_label}: {error}")
        if len(errors) > 10:
            print(f"   ... y {len(errors) - 10} más")

    print(f"✅ Total juegos descargados en rango: {len(games)}")
    return games


def _build_goalie_candidate(node: dict, team_abbr: str):
    """
    Build a goalie candidate carrying the parent team context.
    """
    if not isinstance(node, dict):
        return None

    athlete = node.get("athlete", {}) if isinstance(node.get("athlete"), dict) else node
    position = athlete.get("position", {}) or node.get("position", {})

    pos_abbr = normalize_text(position.get("abbreviation") or node.get("positionAbbreviation") or "")
    pos_name = normalize_text(position.get("displayName") or position.get("name") or "")

    if pos_abbr != "G" and "GOAL" not in pos_name:
        return None

    name = (
        athlete.get("displayName")
        or athlete.get("shortName")
        or athlete.get("fullName")
        or node.get("displayName")
        or node.get("shortName")
        or node.get("fullName")
        or ""
    ).strip()
    if not name:
        return None

    goalie_id = str(athlete.get("id") or node.get("id") or "").strip()

    status_blob = " ".join(
        str(x)
        for x in [
            athlete.get("status"),
            node.get("status"),
            athlete.get("note"),
            node.get("note"),
            athlete.get("designation"),
            node.get("designation"),
            athlete.get("description"),
            node.get("description"),
            athlete.get("starter"),
            node.get("starter"),
            athlete.get("confirmed"),
            node.get("confirmed"),
        ]
        if x is not None
    ).upper()

    is_confirmed = any(token in status_blob for token in ["CONFIRMED", "STARTING", "STARTER"])
    is_probable = any(token in status_blob for token in ["PROBABLE", "EXPECTED", "LIKELY"])

    is_confirmed = is_confirmed or bool(
        athlete.get("confirmed")
        or node.get("confirmed")
        or athlete.get("isConfirmed")
        or node.get("isConfirmed")
    )

    is_starter = bool(
        athlete.get("starter")
        or node.get("starter")
        or athlete.get("isStarter")
        or node.get("isStarter")
        or is_confirmed
    )

    stats_blob = (
        athlete.get("statistics")
        or node.get("statistics")
        or node.get("stats")
        or []
    )
    stat_count = len(stats_blob) if isinstance(stats_blob, list) else 0

    return {
        "goalie_id": goalie_id,
        "goalie_name": name,
        "team_abbr": normalize_text(team_abbr),
        "is_confirmed": bool(is_confirmed),
        "is_probable": bool(is_probable),
        "is_starter": bool(is_starter),
        "stat_count": stat_count,
    }


def _score_goalie_candidate(candidate: dict) -> int:
    """
    Rank goalie candidates so the most likely starter wins.
    """
    score = 0
    score += 100 if candidate.get("is_confirmed") else 0
    score += 50 if candidate.get("is_starter") else 0
    score += 20 if candidate.get("is_probable") else 0
    score += min(10, int(candidate.get("stat_count", 0)))
    score += 5 if candidate.get("goalie_id") else 0
    return score


def _append_goalie_candidate(candidates: list, node: dict, team_abbr: str, target_team: str):
    cand = _build_goalie_candidate(node, team_abbr)
    if cand and normalize_text(team_abbr) == normalize_text(target_team):
        candidates.append(cand)



def _extract_goalies_from_summary(summary_data: dict, home_team: str, away_team: str):
    """
    Extract goalies from directed summary sections while preserving team context.
    """
    home_candidates = []
    away_candidates = []

    home_norm = normalize_text(home_team)
    away_norm = normalize_text(away_team)

    boxscore = summary_data.get("boxscore") or {}
    players_groups = boxscore.get("players") or []
    if isinstance(players_groups, list):
        for team_group in players_groups:
            if not isinstance(team_group, dict):
                continue
            team_obj = team_group.get("team") or {}
            team_abbr = normalize_team_abbr(team_obj if isinstance(team_obj, dict) else {})

            statistics_groups = team_group.get("statistics") or []
            for stat_group in statistics_groups:
                if not isinstance(stat_group, dict):
                    continue
                group_name = normalize_text(stat_group.get("name") or "")
                if group_name not in {"GOALIES", "GOALTENDING"}:
                    continue

                for ath_node in stat_group.get("athletes", []) or []:
                    if normalize_text(team_abbr) == home_norm:
                        _append_goalie_candidate(home_candidates, ath_node, team_abbr, home_norm)
                    elif normalize_text(team_abbr) == away_norm:
                        _append_goalie_candidate(away_candidates, ath_node, team_abbr, away_norm)

    rosters = summary_data.get("rosters") or []
    if isinstance(rosters, list):
        for roster_group in rosters:
            if not isinstance(roster_group, dict):
                continue
            team_obj = roster_group.get("team") or {}
            team_abbr = normalize_team_abbr(team_obj if isinstance(team_obj, dict) else {})

            roster_block = roster_group.get("roster") or []
            for roster_item in roster_block:
                if isinstance(roster_item, dict) and isinstance(roster_item.get("items"), list):
                    iter_items = roster_item.get("items") or []
                elif isinstance(roster_item, dict) and isinstance(roster_item.get("athletes"), list):
                    iter_items = roster_item.get("athletes") or []
                else:
                    iter_items = [roster_item]

                for subitem in iter_items:
                    if normalize_text(team_abbr) == home_norm:
                        _append_goalie_candidate(home_candidates, subitem, team_abbr, home_norm)
                    elif normalize_text(team_abbr) == away_norm:
                        _append_goalie_candidate(away_candidates, subitem, team_abbr, away_norm)

    home_candidates.sort(key=_score_goalie_candidate, reverse=True)
    away_candidates.sort(key=_score_goalie_candidate, reverse=True)

    home_best = home_candidates[0] if home_candidates else None
    away_best = away_candidates[0] if away_candidates else None

    found_count = int(home_best is not None) + int(away_best is not None)
    confirmed_count = int(bool(home_best and home_best.get("is_confirmed"))) + int(
        bool(away_best and away_best.get("is_confirmed"))
    )

    if confirmed_count == 2:
        quality = "confirmed_both"
    elif found_count == 2:
        quality = "found_both"
    elif found_count == 1:
        quality = "found_one"
    else:
        quality = "missing"

    return {
        "home_goalie_name": (home_best or {}).get("goalie_name", ""),
        "away_goalie_name": (away_best or {}).get("goalie_name", ""),
        "home_goalie_id": (home_best or {}).get("goalie_id", ""),
        "away_goalie_id": (away_best or {}).get("goalie_id", ""),
        "home_goalie_confirmed": safe_bool_to_int((home_best or {}).get("is_confirmed")),
        "away_goalie_confirmed": safe_bool_to_int((away_best or {}).get("is_confirmed")),
        "home_goalie_found": safe_bool_to_int(home_best is not None),
        "away_goalie_found": safe_bool_to_int(away_best is not None),
        "goalie_data_quality": quality,
    }


def fetch_goalie_info_for_event(event_id: str, home_team: str, away_team: str) -> dict:
    """
    Fetch goalie enrichment from ESPN summary endpoint for a single event.
    """
    default_payload = {
        "home_goalie_name": "",
        "away_goalie_name": "",
        "home_goalie_id": "",
        "away_goalie_id": "",
        "home_goalie_confirmed": 0,
        "away_goalie_confirmed": 0,
        "home_goalie_found": 0,
        "away_goalie_found": 0,
        "goalie_data_quality": "missing",
    }

    if not event_id:
        return default_payload

    last_error = None
    for url in ESPN_SUMMARY_URLS:
        for attempt in range(3):
            try:
                data = _request_json(url, params={"event": event_id}, timeout=REQUEST_TIMEOUT)
                enriched = _extract_goalies_from_summary(data, home_team, away_team)
                if enriched.get("goalie_data_quality") != "missing":
                    return enriched
                # even if missing, this is a valid response; keep it in case the other endpoint is equal
                default_payload = enriched
                break
            except Exception as e:
                last_error = e
                if attempt == 2:
                    continue
                time.sleep(0.35 * (attempt + 1))

    if last_error:
        return {
            **default_payload,
            "goalie_data_quality": "error",
        }

    return default_payload


def build_goalie_enrichment_map(events: list[dict]) -> dict:
    """
    Build a map: game_id -> goalie enrichment
    """
    enrichment_map = {}
    tasks = []

    for event in events:
        try:
            competitions = event.get("competitions") or []
            if not competitions:
                continue
            comp = competitions[0] or {}
            competitors = comp.get("competitors") or []

            home_data = next((c for c in competitors if (c or {}).get("homeAway") == "home"), None)
            away_data = next((c for c in competitors if (c or {}).get("homeAway") == "away"), None)
            if not home_data or not away_data:
                continue

            event_id = str(event.get("id") or "").strip()
            if not event_id:
                continue

            home_team = normalize_team_abbr(home_data.get("team") or {})
            away_team = normalize_team_abbr(away_data.get("team") or {})

            tasks.append((event_id, home_team, away_team))
        except Exception:
            continue

    if not tasks:
        return enrichment_map

    print(f"\n🥅 Enriqueciendo goalies para {len(tasks)} juegos...")

    max_workers = min(SUMMARY_MAX_WORKERS, max(1, len(tasks)))
    quality_counts = {}
    error_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(fetch_goalie_info_for_event, event_id, home_team, away_team): event_id
            for event_id, home_team, away_team in tasks
        }

        for future in tqdm(
            as_completed(future_map),
            total=len(future_map),
            desc="Goalie enrichment",
            unit="juego",
        ):
            event_id = future_map[future]
            try:
                payload = future.result()
            except Exception:
                payload = {
                    "home_goalie_name": "",
                    "away_goalie_name": "",
                    "home_goalie_id": "",
                    "away_goalie_id": "",
                    "home_goalie_confirmed": 0,
                    "away_goalie_confirmed": 0,
                    "home_goalie_found": 0,
                    "away_goalie_found": 0,
                    "goalie_data_quality": "error",
                }
                error_count += 1

            enrichment_map[event_id] = payload
            q = payload.get("goalie_data_quality", "missing")
            quality_counts[q] = quality_counts.get(q, 0) + 1

    print(f"✅ Goalie enrichment completado")
    print(f"   Resumen goalies: {quality_counts}")
    if error_count:
        print(f"   Errores internos: {error_count}")

    return enrichment_map


def parse_event_to_row(event: dict, season: str | None = None, goalie_map: dict | None = None):
    """
    Parse NHL event from ESPN API into a row for the dataset.
    """
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

    local_dt = dt_utc + timedelta(hours=LOCAL_UTC_OFFSET_HOURS)

    game_date = local_dt.strftime("%Y-%m-%d")
    game_time = local_dt.strftime("%H:%M")

    home_team = normalize_team_abbr(home_data.get("team") or {})
    away_team = normalize_team_abbr(away_data.get("team") or {})

    home_score = safe_int(home_data.get("score"))
    away_score = safe_int(away_data.get("score"))

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

    goalie_payload = {
        "home_goalie_name": "",
        "away_goalie_name": "",
        "home_goalie_id": "",
        "away_goalie_id": "",
        "home_goalie_confirmed": 0,
        "away_goalie_confirmed": 0,
        "home_goalie_found": 0,
        "away_goalie_found": 0,
        "goalie_data_quality": "missing",
    }

    if goalie_map and game_id in goalie_map:
        goalie_payload = goalie_map[game_id]

    return {
        "game_id": game_id,
        "date": game_date,
        "time": game_time,
        "season": season_value,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "total_goals": home_score + away_score,
        "is_draw": 1 if home_score == away_score else 0,
        "completed": int(completed),
        "venue_name": venue.get("fullName", ""),
        "odds_over_under": odds_over_under,
        "odds_details": odds_details,
        **market_odds_fields,
        "odds_data_quality": odds_data_quality(market_odds_fields),
        **goalie_payload,
    }


def ingest_all_seasons():
    """
    Ingest all seasons of NHL data.
    Separates completed games from upcoming games.
    """
    completed_rows = []
    upcoming_rows = []
    today_str = datetime.now().strftime("%Y-%m-%d")

    for season, (start_date, end_date) in SEASONS_TO_FETCH.items():
        print(f"\n📊 Fetching season {season}...")
        games = fetch_games_by_date_range(start_date, end_date)

        goalie_map = {}
        if ENABLE_GOALIE_ENRICHMENT and games:
            goalie_map = build_goalie_enrichment_map(games)

        for event in tqdm(games, desc=f"Parsing season {season}", unit="juego"):
            row = parse_event_to_row(event, season, goalie_map=goalie_map)
            if row:
                game_date = str(row.get("date", ""))
                is_completed = bool(row.get("completed"))

                # Historical dataset keeps past completed games.
                # Upcoming dataset keeps today+future board so UI can show
                # current-day finals and future picks together.
                if is_completed and game_date < today_str:
                    completed_rows.append(row)
                else:
                    upcoming_rows.append(row)

    if completed_rows:
        df_completed = pd.DataFrame(completed_rows)
        df_completed = df_completed.sort_values(["date", "time", "game_id"]).drop_duplicates(
            subset=["game_id"], keep="last"
        )
        df_completed.to_csv(FILE_PATH_ADVANCED, index=False)
        print(f"\n✅ Historical data guardado: {FILE_PATH_ADVANCED}")
        print(f"   Total de juegos jugados: {len(df_completed)}")
        print(f"   Columnas: {', '.join(df_completed.columns.tolist())}")

    if upcoming_rows:
        df_upcoming = pd.DataFrame(upcoming_rows)
        df_upcoming = df_upcoming.sort_values(["date", "time", "game_id"]).drop_duplicates(
            subset=["game_id"], keep="last"
        )
        df_upcoming.to_csv(FILE_PATH_UPCOMING, index=False)
        print(f"\n✅ Upcoming games guardado: {FILE_PATH_UPCOMING}")
        print(f"   Total de juegos futuros: {len(df_upcoming)}")
    else:
        print(f"\n⚠️ No upcoming games encontrados")

    return completed_rows, upcoming_rows


if __name__ == "__main__":
    print("🏒 NHL Data Ingestion + Goalie Enrichment")
    print("=" * 60)

    try:
        completed, upcoming = ingest_all_seasons()
    except Exception as e:
        print(f"❌ Error: {e}")
