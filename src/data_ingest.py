import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from odds_market_fields import extract_market_odds_fields, odds_data_quality

# --- CONFIGURACIÓN ---
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
FILE_PATH_ADVANCED = RAW_DATA_DIR / "nba_advanced_history.csv"

SEASONS_TO_FETCH = {
    "2023-24": ("2023-10-20", "2024-06-25"),
    "2024-25": ("2024-10-20", "2025-06-25"),
    "2025-26": ("2025-10-20", "2026-06-25"),
}

TARGET_DATE_LIMIT = datetime.now().strftime("%Y-%m-%d")

# Cuántos días hacia atrás volver a revisar para corregir faltantes/cambios recientes
BACKFILL_DAYS = 3

ESPN_TO_NBA = {
    "GS": "GSW",
    "NY": "NYK",
    "SA": "SAS",
    "NO": "NOP",
    "WSH": "WAS",
    "UTAH": "UTA",
    "CHA": "CHA",
    "BKN": "BKN",
}


def parse_home_spread(spread_text: str, home_abbr: str, away_abbr: str) -> float:
    """
    Convierte texto como:
      'ATL -2.5' -> home_spread = -2.5 si ATL es local
      'ORL -2.5' -> home_spread = +2.5 si ORL es visitante
      'PK' -> 0.0
    """
    if not spread_text:
        return 0.0

    txt = str(spread_text).strip().upper()

    if txt in {"N/A", "NO LINE", "PK", "PICK", "PICKEM", "PICK'EM"}:
        return 0.0

    m = re.match(r"^([A-Z]+)\s*([+-]?\d+(?:\.\d+)?)$", txt)
    if not m:
        return 0.0

    team_code = m.group(1)
    line_value = float(m.group(2))

    favorite_line = -abs(line_value)

    if team_code == home_abbr:
        return favorite_line
    elif team_code == away_abbr:
        return abs(favorite_line)

    return 0.0


def parse_over_under(value) -> float:
    try:
        if value in [None, "", "N/A"]:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def load_existing_data() -> pd.DataFrame:
    if not FILE_PATH_ADVANCED.exists():
        return pd.DataFrame()

    try:
        df_existing = pd.read_csv(FILE_PATH_ADVANCED, dtype={"game_id": str})
        if "date" in df_existing.columns:
            df_existing["date"] = df_existing["date"].astype(str)
        return df_existing
    except Exception as e:
        print(f"⚠️ No se pudo leer el CSV existente. Se reconstruirá desde cero. Error: {e}")
        return pd.DataFrame()


def determine_fetch_ranges(existing_df: pd.DataFrame):
    """
    Si ya existe data:
    - toma la última fecha guardada
    - retrocede BACKFILL_DAYS
    - solo consulta desde ahí hacia adelante
    Si no existe data:
    - consulta todo el rango definido en SEASONS_TO_FETCH
    """
    limit_date_dt = datetime.strptime(TARGET_DATE_LIMIT, "%Y-%m-%d")

    if existing_df.empty or "date" not in existing_df.columns:
        print("📂 No existe histórico previo. Se hará descarga completa.")
        return build_full_ranges(limit_date_dt)

    try:
        existing_df["date_dt"] = pd.to_datetime(existing_df["date"], errors="coerce")
        max_date = existing_df["date_dt"].max()

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

            # la temporada ya terminó antes del inicio incremental
            if end_dt < incremental_start:
                continue

            season_start = max(start_dt, incremental_start)
            if season_start <= end_dt:
                ranges.append((season, season_start, end_dt))

        return ranges

    except Exception as e:
        print(f"⚠️ Error calculando rango incremental. Se hará descarga completa. Error: {e}")
        return build_full_ranges(limit_date_dt)


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


def fetch_games_for_ranges(ranges):
    all_games = []
    session = requests.Session()

    for season, start_dt, end_dt in ranges:
        print(f"   > Escaneando temporada {season}: {start_dt.strftime('%Y-%m-%d')} -> {end_dt.strftime('%Y-%m-%d')}")
        current_dt = start_dt

        while current_dt <= end_dt:
            chunk_end_dt = current_dt + timedelta(days=15)
            if chunk_end_dt > end_dt:
                chunk_end_dt = end_dt

            d1 = current_dt.strftime("%Y%m%d")
            d2 = chunk_end_dt.strftime("%Y%m%d")
            url = (
                "https://site.api.espn.com/apis/site/v2/sports/basketball/"
                f"nba/scoreboard?dates={d1}-{d2}&limit=500"
            )

            try:
                resp = session.get(url, timeout=20)
                resp.raise_for_status()
                data = resp.json()
                events = data.get("events", [])

                for event in events:
                    status = event.get("status", {}).get("type", {})
                    if not status.get("completed", False):
                        continue

                    competitions = event.get("competitions", [])
                    if not competitions:
                        continue
                    comp = competitions[0]

                    competitors = comp.get("competitors", [])
                    if len(competitors) != 2:
                        continue

                    home_data = next((c for c in competitors if c.get("homeAway") == "home"), None)
                    away_data = next((c for c in competitors if c.get("homeAway") == "away"), None)
                    if not home_data or not away_data:
                        continue

                    h_abbr = ESPN_TO_NBA.get(
                        home_data["team"]["abbreviation"],
                        home_data["team"]["abbreviation"],
                    )
                    a_abbr = ESPN_TO_NBA.get(
                        away_data["team"]["abbreviation"],
                        away_data["team"]["abbreviation"],
                    )

                    game_id = str(event["id"])

                    dt_utc = datetime.strptime(event["date"], "%Y-%m-%dT%H:%MZ")
                    game_date = (dt_utc - timedelta(hours=5)).strftime("%Y-%m-%d")

                    attendance = comp.get("attendance", 0)

                    odds = comp.get("odds", [{}])
                    odds = odds[0] if odds else {}
                    spread_text = odds.get("details", "N/A")
                    over_under = parse_over_under(odds.get("overUnder", 0))
                    market_odds_fields = extract_market_odds_fields(odds)

                    home_spread = parse_home_spread(spread_text, h_abbr, a_abbr)
                    spread_abs = abs(home_spread)
                    home_is_favorite = int(home_spread < 0)

                    h_linescores = home_data.get("linescores", [])
                    a_linescores = away_data.get("linescores", [])

                    def get_quarter_score(linescores, index):
                        try:
                            return int(linescores[index]["value"]) if len(linescores) > index else 0
                        except Exception:
                            return 0

                    game_row = {
                        "game_id": game_id,
                        "date": game_date,
                        "season": season,
                        "home_team": h_abbr,
                        "away_team": a_abbr,
                        "home_pts_total": int(home_data.get("score", 0)),
                        "away_pts_total": int(away_data.get("score", 0)),
                        "home_q1": get_quarter_score(h_linescores, 0),
                        "home_q2": get_quarter_score(h_linescores, 1),
                        "home_q3": get_quarter_score(h_linescores, 2),
                        "home_q4": get_quarter_score(h_linescores, 3),
                        "away_q1": get_quarter_score(a_linescores, 0),
                        "away_q2": get_quarter_score(a_linescores, 1),
                        "away_q3": get_quarter_score(a_linescores, 2),
                        "away_q4": get_quarter_score(a_linescores, 3),
                        "attendance": attendance,
                        "odds_spread": spread_text,
                        "home_spread": home_spread,
                        "spread_abs": spread_abs,
                        "home_is_favorite": home_is_favorite,
                        "odds_over_under": over_under,
                        **market_odds_fields,
                        "odds_data_quality": odds_data_quality(market_odds_fields),
                        "home_top_scorer_pts": 0,
                        "away_top_scorer_pts": 0,
                    }

                    try:
                        h_leaders = home_data.get("leaders", [])
                        a_leaders = away_data.get("leaders", [])
                        if h_leaders and h_leaders[0].get("leaders"):
                            game_row["home_top_scorer_pts"] = int(h_leaders[0]["leaders"][0]["value"])
                        if a_leaders and a_leaders[0].get("leaders"):
                            game_row["away_top_scorer_pts"] = int(a_leaders[0]["leaders"][0]["value"])
                    except Exception:
                        pass

                    all_games.append(game_row)

            except Exception as e:
                print(f"⚠️ Error al consultar {url}: {e}")

            current_dt = chunk_end_dt + timedelta(days=1)
            time.sleep(0.5)

        print(f"     -> ✅ Temporada {season} procesada.")

    return pd.DataFrame(all_games)


def extract_advanced_espn_data():
    print("🚀 Iniciando Extractor Avanzado de ESPN...")

    existing_df = load_existing_data()
    previous_count = len(existing_df)

    fetch_ranges = determine_fetch_ranges(existing_df)

    if not fetch_ranges:
        print("✅ No hay rangos nuevos por consultar.")
        if not existing_df.empty:
            print(f"💾 Histórico intacto: {len(existing_df)} partidos.")
        return existing_df

    new_df = fetch_games_for_ranges(fetch_ranges)

    if new_df.empty:
        print("📭 No se descargaron juegos nuevos.")
        if not existing_df.empty:
            print(f"💾 Histórico intacto: {len(existing_df)} partidos.")
        return existing_df

    if existing_df.empty:
        final_df = new_df.copy()
    else:
        final_df = pd.concat([existing_df, new_df], ignore_index=True)

    final_df["game_id"] = final_df["game_id"].astype(str)
    final_df["date"] = final_df["date"].astype(str)

    before_dedup = len(final_df)
    final_df = final_df.drop_duplicates(subset=["game_id"], keep="last")
    dedup_removed = before_dedup - len(final_df)

    final_df = final_df.sort_values(["date", "game_id"], ascending=[False, False]).reset_index(drop=True)
    final_df.to_csv(FILE_PATH_ADVANCED, index=False)

    added_count = len(final_df) - previous_count

    print("\n📊 RESUMEN DE ACTUALIZACIÓN")
    print(f"   Partidos previos : {previous_count}")
    print(f"   Filas descargadas: {len(new_df)}")
    print(f"   Duplicados quitados: {dedup_removed}")
    print(f"   Partidos finales : {len(final_df)}")
    print(f"   Netos añadidos   : {added_count}")

    print(f"\n💾 Histórico actualizado en: {FILE_PATH_ADVANCED}")
    return final_df


if __name__ == "__main__":
    df_advanced = extract_advanced_espn_data()
    if not df_advanced.empty:
        print(df_advanced.head())