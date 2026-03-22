import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from odds_market_fields import extract_market_odds_fields, odds_data_quality

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent

RAW_DATA_DIR = BASE_DIR / "data" / "mlb" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

FILE_PATH_ADVANCED = RAW_DATA_DIR / "mlb_advanced_history.csv"
FILE_PATH_UPCOMING = RAW_DATA_DIR / "mlb_upcoming_schedule.csv"

SEASONS_TO_FETCH = {
    "2024": ("2024-02-20", "2024-11-05"),
    "2025": ("2025-02-20", "2025-11-05"),
    "2026": ("2026-02-20", "2026-11-05"),
}

TARGET_DATE_LIMIT = datetime.now().strftime("%Y-%m-%d")
BACKFILL_DAYS = 3
UPCOMING_DAYS_AHEAD = 14

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"


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


def load_existing_data() -> pd.DataFrame:
    if not FILE_PATH_ADVANCED.exists():
        return pd.DataFrame()

    try:
        df_existing = pd.read_csv(FILE_PATH_ADVANCED, dtype={"game_id": str})
        if "date" in df_existing.columns:
            df_existing["date"] = df_existing["date"].astype(str)

        # Limpieza defensiva por si quedó una columna vieja
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

    status = event.get("status", {}).get("type", {})
    completed = bool(status.get("completed", False))
    state = str(status.get("state", "") or "")
    description = str(status.get("description", "") or "")
    detail = str(status.get("detail", "") or "")

    h_abbr = home_data["team"]["abbreviation"]
    a_abbr = away_data["team"]["abbreviation"]

    game_id = str(event["id"])

    # Mantengo la misma convención que ya venías usando
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
        "date": game_date,
        "time": game_time,
        "season": season if season is not None else str(dt_utc.year),
        "home_team": h_abbr,
        "away_team": a_abbr,

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
    """
    Descarga agenda rolling desde start_date hasta start_date + days_ahead.
    Conserva todos los estados para hoy y deja futuros para predicción multi-fecha.
    """
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

                # Hoy mantenemos todo (in/post/final). Fechas futuras normalmente vendran como pendientes.
                if day_str != start_date and int(row.get("status_completed", 0)) == 1:
                    continue

                upcoming_rows.append(row)
                day_count += 1

            print(f"   📅 Agenda MLB {day_str}: {day_count} juegos")

        except Exception as e:
            print(f"⚠️ Error descargando agenda MLB {day_str}: {e}")

    df = pd.DataFrame(upcoming_rows)

    if not df.empty:
        df["game_id"] = df["game_id"].astype(str)
        df["date"] = df["date"].astype(str)
        df = df.sort_values(["date", "time", "game_id"]).drop_duplicates(subset=["game_id"], keep="last").reset_index(drop=True)

    return df


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
        final_df = new_df.copy()
    elif new_df.empty:
        final_df = existing_df.copy()
    else:
        final_df = pd.concat([existing_df, new_df], ignore_index=True)

    if not final_df.empty:
        final_df["game_id"] = final_df["game_id"].astype(str)
        final_df["date"] = final_df["date"].astype(str)

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

    # Agenda rolling (hoy + proximos dias)
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