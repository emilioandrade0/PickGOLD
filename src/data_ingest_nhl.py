"""
Data ingest for NHL games from ESPN API.
Mirrors the structure of Liga MX and MLB data pipelines.
"""
import time
from datetime import datetime, timedelta
from pathlib import Path
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from odds_market_fields import extract_market_odds_fields, odds_data_quality

BASE_DIR = Path(__file__).resolve().parent
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
    """
    Normalize NHL team abbreviations from ESPN API.
    """
    abbr = normalize_text(team_data.get("abbreviation") or "")
    
    # Map of common NHL team abbreviations
    nhl_teams = {
        "ANA": "ANA", "BOS": "BOS", "BUF": "BUF", "CGY": "CGY",
        "CAR": "CAR", "CHI": "CHI", "COL": "COL", "DAL": "DAL",
        "DET": "DET", "EDM": "EDM", "FLA": "FLA", "LA": "LAK",
        "LAK": "LAK", "MIN": "MIN", "MTL": "MTL", "NJ": "NJ",
        "NSH": "NSH", "NYI": "NYI", "NYR": "NYR", "OTT": "OTT",
        "PHI": "PHI", "PIT": "PIT", "SJ": "SJ", "STL": "STL",
        "TB": "TBL", "TBL": "TBL", "TOR": "TOR", "VAN": "VAN",
        "VGK": "VGK", "WASH": "WSH", "WSH": "WSH", "WPG": "WPG",
    }
    
    return nhl_teams.get(abbr, abbr)


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
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json() or {}
                events = data.get("events", [])
                return day_label, events, None
            except Exception as e:
                if attempt == 2:
                    return day_label, [], str(e)
                time.sleep(0.2 * (attempt + 1))

    max_workers = min(6, max(1, len(date_list)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_one, day_dt) for day_dt in date_list]
        for future in as_completed(futures):
            day_label, events, error = future.result()
            if error is None:
                games.extend(events)
                print(f"✅ {day_label}: {len(events)} juegos")
            else:
                print(f"❌ Error fetching {day_label}: {error}")
    
    return games


def parse_event_to_row(event: dict, season: str | None = None):
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
        
        for event in games:
            row = parse_event_to_row(event, season)
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
    
    # Save completed games (for training)
    if completed_rows:
        df_completed = pd.DataFrame(completed_rows)
        df_completed.to_csv(FILE_PATH_ADVANCED, index=False)
        print(f"\n✅ Historical data guardado: {FILE_PATH_ADVANCED}")
        print(f"   Total de juegos jugados: {len(df_completed)}")
        print(f"   Columnas: {', '.join(df_completed.columns.tolist())}")
    
    # Save upcoming games (for live predictions)
    if upcoming_rows:
        df_upcoming = pd.DataFrame(upcoming_rows)
        df_upcoming.to_csv(FILE_PATH_UPCOMING, index=False)
        print(f"\n✅ Upcoming games guardado: {FILE_PATH_UPCOMING}")
        print(f"   Total de juegos futuros: {len(df_upcoming)}")
    else:
        print(f"\n⚠️ No upcoming games encontrados")
    
    return completed_rows, upcoming_rows


if __name__ == "__main__":
    print("🏒 NHL Data Ingestion")
    print("=" * 60)
    
    try:
        completed, upcoming = ingest_all_seasons()
    except Exception as e:
        print(f"❌ Error: {e}")
