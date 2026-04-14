import re
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure src root is on sys.path before importing shared modules
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
import requests
import numpy as np
from bs4 import BeautifulSoup
import urllib3
from urllib3.exceptions import InsecureRequestWarning
from requests.adapters import HTTPAdapter

# Silence verify=False warnings
urllib3.disable_warnings(InsecureRequestWarning)

# --- CONFIGURACIÓN ---
BASE_DIR = SRC_ROOT
RAW_DATA_DIR = BASE_DIR / "data" / "wnba" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
FILE_PATH_ADVANCED = RAW_DATA_DIR / "wnba_advanced_history.csv"

SEASONS_TO_FETCH = {
    "2024": ("2024-05-01", "2024-10-31"),
    "2025": ("2025-05-01", "2025-10-31"),
    "2026": ("2026-05-01", "2026-10-31"),
}

TARGET_DATE_LIMIT = datetime.now().strftime("%Y-%m-%d")
BACKFILL_DAYS = 3

ESPN_TO_WNBA = {
    "GS": "GSV",
    "GSV": "GSV",
    "LAS": "LA",
    "LVA": "LV",
    "NYL": "NY",
    "PHX": "PHO",
    "WSH": "WAS",
}

# --- FUNCIONES DE SOPORTE ---
def fetch_wnba_injury_report():
    url = "https://www.espn.com/wnba/injuries"
    print(f"🕵️ Accediendo al reporte de lesiones de ESPN...")
    
    name_to_abbr = {
        "Dream": "ATL",
        "Sky": "CHI",
        "Sun": "CON",
        "Wings": "DAL",
        "Valkyries": "GSV",
        "Fever": "IND",
        "Sparks": "LA",
        "Aces": "LV",
        "Lynx": "MIN",
        "Liberty": "NY",
        "Mercury": "PHO",
        "Storm": "SEA",
        "Mystics": "WAS",
    }

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        team_headers = soup.find_all('div', class_='Table__Title')
        tables = pd.read_html(io.StringIO(response.text))
        
        if not tables or not team_headers: return pd.DataFrame()

        all_data = []
        for i, table in enumerate(tables):
            full_team_name = team_headers[i].text if i < len(team_headers) else "Unknown"
            abbr = "UNK"
            for name, code in name_to_abbr.items():
                if name in full_team_name:
                    abbr = code
                    break
            
            table['team_abbr'] = abbr
            table.columns = [c.upper() for c in table.columns]
            all_data.append(table)
            
        return pd.concat(all_data, ignore_index=True)
    except Exception as e:
        return pd.DataFrame()

def load_existing_data() -> pd.DataFrame:
    if not FILE_PATH_ADVANCED.exists():
        return pd.DataFrame()
    try:
        df_existing = pd.read_csv(FILE_PATH_ADVANCED, dtype={"game_id": str})
        if "date" in df_existing.columns:
            df_existing["date"] = df_existing["date"].astype(str)
        return df_existing
    except:
        return pd.DataFrame()

def determine_fetch_ranges(existing_df: pd.DataFrame):
    limit_date_dt = datetime.strptime(TARGET_DATE_LIMIT, "%Y-%m-%d")

    if existing_df.empty or "date" not in existing_df.columns:
        print("📂 No existe histórico previo. Se hará descarga completa.")
        return _build_full_ranges(limit_date_dt)

    try:
        existing_df["date_dt"] = pd.to_datetime(existing_df["date"], errors="coerce")
        max_date = existing_df["date_dt"].max()

        if pd.isna(max_date):
            return _build_full_ranges(limit_date_dt)

        incremental_start = (max_date - timedelta(days=BACKFILL_DAYS)).normalize()
        print(f"📌 Última fecha en histórico: {max_date.strftime('%Y-%m-%d')} | Revisando desde: {incremental_start.strftime('%Y-%m-%d')}")

        ranges = []
        for season, (start_str, end_str) in SEASONS_TO_FETCH.items():
            start_dt = datetime.strptime(start_str, "%Y-%m-%d")
            end_dt = min(datetime.strptime(end_str, "%Y-%m-%d"), limit_date_dt)
            if end_dt < incremental_start: continue
            season_start = max(start_dt, incremental_start)
            if season_start <= end_dt:
                ranges.append((season, season_start, end_dt))
        return ranges
    except:
        return _build_full_ranges(limit_date_dt)

def _build_full_ranges(limit_date_dt: datetime):
    ranges = []
    for season, (start_str, end_str) in SEASONS_TO_FETCH.items():
        start_dt = datetime.strptime(start_str, "%Y-%m-%d")
        end_dt = min(datetime.strptime(end_str, "%Y-%m-%d"), limit_date_dt)
        if start_dt <= limit_date_dt:
            ranges.append((season, start_dt, end_dt))
    return ranges

def _parse_float(val):
    try:
        if val in ("", None): return None
        return float(val)
    except:
        return None


def _american_to_decimal(val):
    american = _parse_float(val)
    if american is None or american == 0:
        return None
    if american > 0:
        return round((american / 100.0) + 1.0, 4)
    return round((100.0 / abs(american)) + 1.0, 4)


def _build_session():
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _render_progress(processed_days, total_days, processed_games, total_games_seen):
    chunk_pct = int((processed_days / max(total_days, 1)) * 100)
    filled = int((processed_days / max(total_days, 1)) * 30)
    bar = "#" * filled + "-" * (30 - filled)
    sys.stdout.write(
        f"\r   Progreso: [{bar}] {processed_days}/{total_days} días ({chunk_pct}%) | juegos {processed_games}/{total_games_seen}"
    )
    sys.stdout.flush()


def _fetch_summary(headers, game_id):
    summary_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/summary?event={game_id}"
    try:
        session = _build_session()
        sum_resp = session.get(summary_url, headers=headers, timeout=10)
        if sum_resp.status_code == 200:
            return game_id, sum_resp.json()
    except Exception:
        pass
    return game_id, {}

# --- CORE INGESTOR ---
def fetch_games_for_ranges(ranges):
    all_games = []
    session = _build_session()
    headers = {"User-Agent": "Mozilla/5.0"}
    total_days = sum(((end_dt - start_dt).days) + 1 for _, start_dt, end_dt in ranges)
    processed_days = 0
    total_games_seen = 0
    processed_games = 0

    for season, start_dt, end_dt in ranges:
        print(f"\n   > Escaneando temporada {season}: {start_dt.strftime('%Y-%m-%d')} -> {end_dt.strftime('%Y-%m-%d')}")
        current_dt = start_dt

        while current_dt <= end_dt:
            date_str = current_dt.strftime("%Y%m%d")
            url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard?dates={date_str}&limit=100"

            try:
                resp = session.get(url, headers=headers, timeout=20)
                if resp.status_code == 200:
                    data = resp.json()
                    events = data.get("events", [])
                    print(f"     [+] Día {current_dt.strftime('%Y-%m-%d')}: {len(events)} juegos encontrados.")
                    filtered_events = []
                    for event in events:
                        status_data = event.get("status", {}).get("type", {})
                        is_completed = status_data.get("completed", False)
                        dt_utc = datetime.strptime(event["date"], "%Y-%m-%dT%H:%MZ")
                        game_date = (dt_utc - timedelta(hours=5)).strftime("%Y-%m-%d")
                        if not is_completed and game_date != TARGET_DATE_LIMIT:
                            continue
                        filtered_events.append(event)
                    total_games_seen += len(filtered_events)
                    _render_progress(processed_days, total_days, processed_games, total_games_seen)

                    summary_map = {}
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        futures = []
                        for event in filtered_events:
                            game_id = str(event["id"])
                            futures.append(executor.submit(_fetch_summary, headers, game_id))
                        for future in as_completed(futures):
                            game_id, summary_data = future.result()
                            summary_map[game_id] = summary_data

                    for event in filtered_events:
                        comp = (event.get("competitions") or [{}])[0]
                        competitors = comp.get("competitors", [])
                        if len(competitors) != 2: continue

                        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
                        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
                        if not home or not away: continue

                        h_abbr = ESPN_TO_WNBA.get(home["team"]["abbreviation"], home["team"]["abbreviation"])
                        a_abbr = ESPN_TO_WNBA.get(away["team"]["abbreviation"], away["team"]["abbreviation"])
                        game_id = str(event["id"])
                        summary_data = summary_map.get(game_id, {})

                        # Momios de cierre desde el Pickcenter
                        pickcenter = summary_data.get("pickcenter", [])
                        odds_info = pickcenter[0] if pickcenter else {}
                        
                        home_odds_blob = odds_info.get("homeTeamOdds", {})
                        away_odds_blob = odds_info.get("awayTeamOdds", {})
                        
                        spread_line = _parse_float(odds_info.get("spread"))
                        over_under = _parse_float(odds_info.get("overUnder"))
                        home_money = _american_to_decimal(home_odds_blob.get("moneyLine"))
                        away_money = _american_to_decimal(away_odds_blob.get("moneyLine"))

                        home_spread = None
                        if spread_line is not None:
                            home_fav = home_odds_blob.get("favorite")
                            away_fav = away_odds_blob.get("favorite")
                            if home_fav is True or away_fav is False:
                                home_spread = -abs(spread_line)
                            elif away_fav is True or home_fav is False:
                                home_spread = abs(spread_line)
                            else:
                                home_spread = spread_line

                        attendance = comp.get("attendance", 0)
                        
                        def get_quarter(competitor, index):
                            scores = competitor.get("linescores", [])
                            if len(scores) > index: return int(_parse_float(scores[index].get("value")) or 0)
                            return 0

                        spread_parse_success = home_spread is not None
                        spread_is_pickem = spread_parse_success and home_spread == 0.0
                        
                        # Creando fila base del partido
                        game_row = {
                            "game_id": game_id,
                            "date": game_date,
                            "season": season,
                            "home_team": h_abbr,
                            "away_team": a_abbr,
                            "home_pts_total": int(_parse_float(home.get("score")) or 0),
                            "away_pts_total": int(_parse_float(away.get("score")) or 0),
                            "home_q1": get_quarter(home, 0), "home_q2": get_quarter(home, 1),
                            "home_q3": get_quarter(home, 2), "home_q4": get_quarter(home, 3),
                            "away_q1": get_quarter(away, 0), "away_q2": get_quarter(away, 1),
                            "away_q3": get_quarter(away, 2), "away_q4": get_quarter(away, 3),
                            "attendance": attendance,
                            "home_spread": home_spread if home_spread is not None else 0.0,
                            "spread_abs": abs(home_spread) if home_spread is not None else 0.0,
                            "home_is_favorite": 1 if home_spread is not None and home_spread < 0 else 0,
                            "odds_over_under": over_under if over_under is not None else 0.0,
                            "home_moneyline_odds": home_money,
                            "away_moneyline_odds": away_money,
                            "spread_parse_success": bool(spread_parse_success),
                            "spread_is_pickem": bool(spread_is_pickem),
                            "spread_missing": not spread_parse_success,
                            "market_source": "espn_summary_pickcenter" if spread_parse_success else "missing",
                            "odds_data_quality": "real" if spread_parse_success or over_under else "missing",
                        }

                        # Anexando estadísticas avanzadas (Rebotes, FGM, FGA, etc.)
                        boxscore = summary_data.get("boxscore", {})
                        teams_box = boxscore.get("teams", [])
                        for tb in teams_box:
                            t_abbr = ESPN_TO_WNBA.get(tb.get("team", {}).get("abbreviation"), tb.get("team", {}).get("abbreviation"))
                            stats = tb.get("statistics", [])
                            prefix = "home_" if t_abbr == h_abbr else "away_"
                            
                            for s in stats:
                                name = s.get("name")
                                val = _parse_float(s.get("displayValue"))
                                if name == "fieldGoalsMade": game_row[prefix+"fgm"] = val
                                elif name == "fieldGoalsAttempted": game_row[prefix+"fga"] = val
                                elif name == "threePointFieldGoalsMade": game_row[prefix+"3pm"] = val
                                elif name == "freeThrowsAttempted": game_row[prefix+"fta"] = val
                                elif name == "rebounds": game_row[prefix+"reb"] = val
                                elif name == "offensiveRebounds": game_row[prefix+"orb"] = val
                                elif name == "assists": game_row[prefix+"ast"] = val
                                elif name == "turnovers": game_row[prefix+"tov"] = val

                        all_games.append(game_row)
                        processed_games += 1
                        _render_progress(processed_days, total_days, processed_games, total_games_seen)

            except Exception as e:
                pass
            
            processed_days += 1
            _render_progress(processed_days, total_days, processed_games, total_games_seen)
            current_dt = current_dt + timedelta(days=1)
            time.sleep(0.05)

    sys.stdout.write("\n")
    return pd.DataFrame(all_games)

def enrich_today_data(final_df):
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_mask = final_df['date'] == today_str
    
    if 'home_injuries_count' not in final_df.columns: final_df['home_injuries_count'] = 0.0
    if 'away_injuries_count' not in final_df.columns: final_df['away_injuries_count'] = 0.0

    if not today_mask.any(): return final_df

    print(f"🚀 Analizando bajas para {today_mask.sum()} partidos de hoy...")
    df_inj = fetch_wnba_injury_report()
    
    if not df_inj.empty:
        out_players = df_inj[df_inj['STATUS'].str.contains('Out', case=False, na=False)]
        for idx, row in final_df[today_mask].iterrows():
            final_df.at[idx, 'home_injuries_count'] = float(len(out_players[out_players['TEAM_ABBR'] == row['home_team']]))
            final_df.at[idx, 'away_injuries_count'] = float(len(out_players[out_players['TEAM_ABBR'] == row['away_team']]))
            
    final_df['home_injuries_count'] = final_df['home_injuries_count'].fillna(0.0)
    final_df['away_injuries_count'] = final_df['away_injuries_count'].fillna(0.0)
    final_df["date_dt"] = pd.to_datetime(final_df["date"], errors="coerce")
    return final_df

def extract_advanced_espn_data():
    print("🚀 Iniciando Motor Híbrido Definitivo (Scoreboard V2 + Pickcenter)...")
    existing_df = load_existing_data()
    previous_count = len(existing_df)
    fetch_ranges = determine_fetch_ranges(existing_df)

    if not fetch_ranges:
        print("✅ No hay rangos nuevos por consultar.")
        return existing_df

    new_df = fetch_games_for_ranges(fetch_ranges)
    if new_df.empty:
        print("📭 No se descargaron juegos nuevos.")
        return existing_df

    if existing_df.empty:
        final_df = new_df.copy()
    else:
        final_df = pd.concat([existing_df, new_df], ignore_index=True)

    final_df["game_id"] = final_df["game_id"].astype(str)
    final_df["date"] = final_df["date"].astype(str)

    has_market = np.zeros(len(final_df), dtype=bool)
    if "spread_parse_success" in final_df.columns:
        has_market |= final_df["spread_parse_success"].astype(bool).to_numpy()
    if "odds_over_under" in final_df.columns:
        has_market |= (final_df["odds_over_under"].fillna(0) > 0).to_numpy()

    final_df["has_market"] = has_market.astype(int)
    before_dedup = len(final_df)
    
    final_df = final_df.sort_values(["game_id", "has_market"], ascending=[True, True])
    final_df = final_df.drop_duplicates(subset=["game_id"], keep="last")
    final_df = final_df.drop(columns=["has_market"])

    dedup_removed = before_dedup - len(final_df)
    final_df = final_df.sort_values(["date", "game_id"], ascending=[False, False]).reset_index(drop=True)
    
    final_df = enrich_today_data(final_df)
    final_df.to_csv(FILE_PATH_ADVANCED, index=False)

    print("\n📊 RESUMEN DE ACTUALIZACIÓN")
    print(f"   Partidos previos : {previous_count}")
    print(f"   Filas descargadas: {len(new_df)}")
    print(f"   Duplicados limpiados: {dedup_removed}")
    print(f"   Partidos finales únicos: {len(final_df)}")
    print(f"\n💾 Histórico actualizado en: {FILE_PATH_ADVANCED}")
    return final_df

if __name__ == "__main__":
    df_advanced = extract_advanced_espn_data()
