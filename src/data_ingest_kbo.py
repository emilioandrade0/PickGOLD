import html
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import cloudscraper
import pandas as pd

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent

RAW_DATA_DIR = BASE_DIR / "data" / "kbo" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

FILE_PATH_ADVANCED = RAW_DATA_DIR / "kbo_advanced_history.csv"
FILE_PATH_UPCOMING = RAW_DATA_DIR / "kbo_upcoming_schedule.csv"

SEASONS_TO_FETCH = {
    "2024": ("2024-02-20", "2024-11-05"),
    "2025": ("2025-02-20", "2025-11-05"),
    "2026": ("2026-02-20", "2026-11-05"),
}

TARGET_DATE_LIMIT = datetime.now().strftime("%Y-%m-%d")
BACKFILL_DAYS = 3
UPCOMING_DAYS_AHEAD = 14

DETAIL_MAX_WORKERS = 14
DETAIL_TIMEOUT_SECONDS = 8
DETAIL_RETRIES = 2

MYKBO_BASE_URL = "https://mykbostats.com"

TEAM_SLUG_TO_ABBR = {
    "Doosan": "DOO",
    "Hanwha": "HAN",
    "Kia": "KIA",
    "Kiwoom": "KIW",
    "KT": "KTW",
    "LG": "LG",
    "Lotte": "LOT",
    "NC": "NCD",
    "Samsung": "SAM",
    "SSG": "SSG",
}

TEAM_NAME_TO_ABBR = {
    "Doosan Bears": "DOO",
    "Hanwha Eagles": "HAN",
    "Kia Tigers": "KIA",
    "Kiwoom Heroes": "KIW",
    "KT Wiz": "KTW",
    "LG Twins": "LG",
    "Lotte Giants": "LOT",
    "NC Dinos": "NCD",
    "Samsung Lions": "SAM",
    "SSG Landers": "SSG",
}


# =========================
# HELPERS
# =========================
def create_http_session():
    return cloudscraper.create_scraper(
        browser={
            "browser": "chrome",
            "platform": "windows",
            "mobile": False,
        }
    )


SESSION = create_http_session()
THREAD_LOCAL = threading.local()


def get_thread_http_session():
    sess = getattr(THREAD_LOCAL, "session", None)
    if sess is None:
        sess = create_http_session()
        THREAD_LOCAL.session = sess
    return sess


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def clean_html_text(raw_html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def to_date_yyyy_mm_dd(yyyymmdd: str) -> str:
    return f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def slug_to_team_abbr(slug: str) -> str:
    # MyKBO slugs can occasionally include extra tokens/hyphens; normalize defensively.
    cleaned = re.sub(r"[^A-Za-z]", "", str(slug)).lower()
    if not cleaned:
        return str(slug).upper()

    for team_slug, abbr in TEAM_SLUG_TO_ABBR.items():
        key = re.sub(r"[^A-Za-z]", "", team_slug).lower()
        if cleaned == key or cleaned.startswith(key) or key.startswith(cleaned):
            return abbr

    return str(slug).upper()


def parse_time_token_to_hhmm(token: str) -> str:
    m = re.match(r"^(\d{1,2}):(\d{2})(am|pm)$", token.strip().lower())
    if not m:
        return ""
    hh = int(m.group(1)) % 12
    mm = int(m.group(2))
    if m.group(3) == "pm":
        hh += 12
    return f"{hh:02d}:{mm:02d}"


def print_progress_bar(prefix: str, current: int, total: int, width: int = 28, started_at: float | None = None):
    if total <= 0:
        return
    current = max(0, min(current, total))
    filled = int(width * (current / total))
    bar = "#" * filled + "-" * (width - filled)
    pct = (current / total) * 100

    suffix = ""
    if started_at is not None and current > 0:
        elapsed = max(0.001, time.time() - started_at)
        rate = current / elapsed
        remaining = max(0, total - current)
        eta_sec = int(remaining / rate) if rate > 0 else 0
        eta_min, eta_s = divmod(eta_sec, 60)
        suffix = f" | {rate:4.1f}/s | ETA {eta_min:02d}:{eta_s:02d}"

    end = "\n" if current >= total else "\r"
    print(f"{prefix} [{bar}] {current}/{total} ({pct:5.1f}%){suffix}", end=end, flush=True)


def fetch_url_text(url: str, retries: int = 3, timeout: int = 15, session=None):
    session = session or get_thread_http_session()
    last_exc = None

    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(0.75 * (attempt + 1))

    raise last_exc


def parse_schedule_anchors(html_content: str):
    pattern = re.compile(
        r'<a[^>]+href="(?P<href>/games/(?P<gid>\d+)-(?P<matchup>[^"/]+)-(?P<dt>\d{8}))"[^>]*>(?P<txt>.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )

    rows = []
    for m in pattern.finditer(html_content):
        game_id = str(m.group("gid"))
        matchup = m.group("matchup")
        if "-vs-" not in matchup:
            continue

        away_slug, home_slug = matchup.split("-vs-", 1)
        date_str = to_date_yyyy_mm_dd(m.group("dt"))
        link_text = clean_html_text(m.group("txt"))

        away_abbr = slug_to_team_abbr(away_slug)
        home_abbr = slug_to_team_abbr(home_slug)

        score_match = re.search(r"(\d+)\s*:\s*(\d+)", link_text)
        away_runs = safe_int(score_match.group(1), 0) if score_match else 0
        home_runs = safe_int(score_match.group(2), 0) if score_match else 0

        lower_txt = link_text.lower()
        if "final" in lower_txt:
            status_completed = 1
            status_state = "post"
            status_description = "Final"
        elif "top" in lower_txt or "bot" in lower_txt:
            status_completed = 0
            status_state = "in"
            status_description = "In Progress"
        else:
            status_completed = 0
            status_state = "pre"
            status_description = "Scheduled"

        tm = re.search(r"(\d{1,2}:\d{2}\s*[ap]m)", link_text, flags=re.IGNORECASE)
        game_time = parse_time_token_to_hhmm(tm.group(1).replace(" ", "").lower()) if tm else ""

        rows.append(
            {
                "game_id": game_id,
                "date": date_str,
                "time": game_time,
                "season": date_str[:4],
                "home_team": home_abbr,
                "away_team": away_abbr,
                "home_runs_total": home_runs,
                "away_runs_total": away_runs,
                "home_r1": 0,
                "away_r1": 0,
                "home_r2": 0,
                "away_r2": 0,
                "home_r3": 0,
                "away_r3": 0,
                "home_r4": 0,
                "away_r4": 0,
                "home_r5": 0,
                "away_r5": 0,
                "home_runs_f5": 0,
                "away_runs_f5": 0,
                "attendance": 0,
                "odds_details": "N/A",
                "odds_over_under": 0.0,
                "home_is_favorite": -1,
                "home_hits": 0,
                "away_hits": 0,
                "detail_parsed": 0,
                "status_completed": status_completed,
                "status_state": status_state,
                "status_description": status_description,
                "status_detail": link_text,
                "game_url": f"{MYKBO_BASE_URL}{m.group('href')}",
            }
        )

    dedup = {}
    for row in rows:
        dedup[row["game_id"]] = row
    return list(dedup.values())


def load_existing_data() -> pd.DataFrame:
    if not FILE_PATH_ADVANCED.exists():
        return pd.DataFrame()

    try:
        df_existing = pd.read_csv(FILE_PATH_ADVANCED, dtype={"game_id": str})
        if "date" in df_existing.columns:
            df_existing["date"] = df_existing["date"].astype(str)

        # Limpieza defensiva por si quedÃ³ una columna vieja
        if "date_dt" in df_existing.columns:
            df_existing = df_existing.drop(columns=["date_dt"])

        return df_existing
    except Exception as e:
        print(f"âš ï¸ No se pudo leer el CSV existente. Se reconstruirÃ¡ desde cero. Error: {e}")
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
        print("ðŸ“‚ No existe histÃ³rico previo. Se harÃ¡ descarga completa.")
        return build_full_ranges(limit_date_dt)

    try:
        tmp = existing_df.copy()
        tmp["date_dt"] = pd.to_datetime(tmp["date"], errors="coerce")
        max_date = tmp["date_dt"].max()

        if pd.isna(max_date):
            print("âš ï¸ No se pudo detectar la Ãºltima fecha del histÃ³rico. Se harÃ¡ descarga completa.")
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
        print(f"âš ï¸ Error calculando rango incremental. Se harÃ¡ descarga completa. Error: {e}")
        return build_full_ranges(limit_date_dt)


def parse_linescore_from_game_html(game_html: str):
    text = clean_html_text(game_html)

    header_match = re.search(r"((?:\d+\s+){8,}\d+)\s+R\s+H\s+E\s+B\s+", text)
    if not header_match:
        return None

    innings = [int(x) for x in header_match.group(1).split()]
    num_innings = len(innings)
    value_count = num_innings + 4

    segment = text[header_match.end(): header_match.end() + 1500]
    row_pattern = re.compile(
        rf"([A-Za-z][A-Za-z .'-]+?)\s+((?:\d+|X)(?:\s+(?:\d+|X)){{{value_count - 1}}})"
    )
    rows = row_pattern.findall(segment)
    if len(rows) < 2:
        return None

    away_name = rows[0][0].strip()
    away_values = rows[0][1].split()
    home_name = rows[1][0].strip()
    home_values = rows[1][1].split()

    if len(away_values) < value_count or len(home_values) < value_count:
        return None

    def token_to_int(token: str):
        if str(token).strip().upper() == "X":
            return 0
        return safe_int(token, 0)

    away_innings = [token_to_int(x) for x in away_values[:num_innings]]
    home_innings = [token_to_int(x) for x in home_values[:num_innings]]

    away_runs_total = token_to_int(away_values[num_innings])
    home_runs_total = token_to_int(home_values[num_innings])
    away_hits = token_to_int(away_values[num_innings + 1])
    home_hits = token_to_int(home_values[num_innings + 1])

    return {
        "away_name": away_name,
        "home_name": home_name,
        "away_innings": away_innings,
        "home_innings": home_innings,
        "away_runs_total": away_runs_total,
        "home_runs_total": home_runs_total,
        "away_hits": away_hits,
        "home_hits": home_hits,
    }


def fetch_completed_games_for_ranges(ranges):
    all_schedule_rows = []
    today_str = datetime.now().strftime("%Y-%m-%d")

    for season, start_dt, end_dt in ranges:
        print(f"   > Escaneando temporada {season}: {start_dt.strftime('%Y-%m-%d')} -> {end_dt.strftime('%Y-%m-%d')}")
        season_week_start = (start_dt - timedelta(days=start_dt.weekday())).date()
        week_start = season_week_start
        end_week = (end_dt + timedelta(days=7)).date()
        total_weeks = ((end_week - season_week_start).days // 7) + 1
        scanned_weeks = 0

        while week_start <= end_week:
            week_url = f"{MYKBO_BASE_URL}/schedule/week_of/{week_start.strftime('%Y-%m-%d')}"
            try:
                week_html = fetch_url_text(week_url)
                week_rows = parse_schedule_anchors(week_html)
                for row in week_rows:
                    if row["date"] < start_dt.strftime("%Y-%m-%d") or row["date"] > end_dt.strftime("%Y-%m-%d"):
                        continue
                    all_schedule_rows.append(row)
            except Exception as exc:
                print(f"⚠️ Error consultando semana {week_start}: {exc}")

            scanned_weeks += 1
            print_progress_bar(f"      Temporada {season}", scanned_weeks, total_weeks)
            week_start = week_start + timedelta(days=7)

        print(f"     -> ✅ Temporada {season} schedule procesado.")

    dedup = {}
    for row in all_schedule_rows:
        dedup[row["game_id"]] = row

    candidates = [
        r for r in dedup.values()
        if int(r.get("status_completed", 0)) == 1 and r.get("date", "") <= today_str
    ]

    print(f"   -> Juegos finalizados a enriquecer: {len(candidates)}")

    completed_rows = []

    def fetch_detail(base_row: dict):
        try:
            game_html = fetch_url_text(
                base_row["game_url"],
                retries=DETAIL_RETRIES,
                timeout=DETAIL_TIMEOUT_SECONDS,
            )
            parsed = parse_linescore_from_game_html(game_html)
            if parsed is None:
                return base_row

            away_team = TEAM_NAME_TO_ABBR.get(parsed["away_name"], base_row["away_team"])
            home_team = TEAM_NAME_TO_ABBR.get(parsed["home_name"], base_row["home_team"])
            away_innings = parsed["away_innings"]
            home_innings = parsed["home_innings"]

            out = dict(base_row)
            out["away_team"] = away_team
            out["home_team"] = home_team
            out["away_runs_total"] = parsed["away_runs_total"]
            out["home_runs_total"] = parsed["home_runs_total"]
            out["away_hits"] = parsed["away_hits"]
            out["home_hits"] = parsed["home_hits"]

            out["away_r1"] = away_innings[0] if len(away_innings) > 0 else 0
            out["home_r1"] = home_innings[0] if len(home_innings) > 0 else 0
            out["away_r2"] = away_innings[1] if len(away_innings) > 1 else 0
            out["home_r2"] = home_innings[1] if len(home_innings) > 1 else 0
            out["away_r3"] = away_innings[2] if len(away_innings) > 2 else 0
            out["home_r3"] = home_innings[2] if len(home_innings) > 2 else 0
            out["away_r4"] = away_innings[3] if len(away_innings) > 3 else 0
            out["home_r4"] = home_innings[3] if len(home_innings) > 3 else 0
            out["away_r5"] = away_innings[4] if len(away_innings) > 4 else 0
            out["home_r5"] = home_innings[4] if len(home_innings) > 4 else 0

            out["away_runs_f5"] = out["away_r1"] + out["away_r2"] + out["away_r3"] + out["away_r4"] + out["away_r5"]
            out["home_runs_f5"] = out["home_r1"] + out["home_r2"] + out["home_r3"] + out["home_r4"] + out["home_r5"]
            out["status_state"] = "post"
            out["status_description"] = "Final"
            out["detail_parsed"] = 1
            return out
        except Exception:
            return base_row

    max_workers = min(DETAIL_MAX_WORKERS, max(1, len(candidates)))
    if candidates:
        print(
            f"   -> Descargando detalle de juegos con {max_workers} workers "
            f"(timeout={DETAIL_TIMEOUT_SECONDS}s, retries={DETAIL_RETRIES})..."
        )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_detail, row) for row in candidates]
        total_details = len(futures)
        done_details = 0
        details_started_at = time.time()
        for future in as_completed(futures):
            completed_rows.append(future.result())
            done_details += 1
            print_progress_bar(
                "   Detalle juegos finalizados",
                done_details,
                total_details,
                started_at=details_started_at,
            )

    out_df = pd.DataFrame(completed_rows)
    if not out_df.empty and "detail_parsed" in out_df.columns:
        parsed_ok = int(out_df["detail_parsed"].fillna(0).astype(int).sum())
        print(f"   -> Detalle parseado OK: {parsed_ok}/{len(out_df)}")

    return out_df


def fetch_upcoming_schedule_for_range(start_date: str, days_ahead: int = UPCOMING_DAYS_AHEAD):
    """
    Descarga agenda rolling desde start_date hasta start_date + days_ahead.
    Conserva todos los estados para hoy y deja futuros para predicciÃ³n multi-fecha.
    """
    base_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = base_dt + timedelta(days=days_ahead)
    upcoming_rows = []

    base_week_start = (base_dt - timedelta(days=base_dt.weekday())).date()
    week_start = base_week_start
    end_week = (end_dt + timedelta(days=7)).date()
    total_weeks = ((end_week - base_week_start).days // 7) + 1
    scanned_weeks = 0

    by_day_counter = {}
    while week_start <= end_week:
        week_url = f"{MYKBO_BASE_URL}/schedule/week_of/{week_start.strftime('%Y-%m-%d')}"
        try:
            week_html = fetch_url_text(week_url)
            week_rows = parse_schedule_anchors(week_html)
            for row in week_rows:
                if row["date"] < start_date or row["date"] > end_dt.strftime("%Y-%m-%d"):
                    continue

                if row["date"] != start_date and int(row.get("status_completed", 0)) == 1:
                    continue

                row.pop("game_url", None)
                upcoming_rows.append(row)
                by_day_counter[row["date"]] = by_day_counter.get(row["date"], 0) + 1
        except Exception as exc:
            print(f"⚠️ Error descargando agenda kbo semana {week_start}: {exc}")

        scanned_weeks += 1
        print_progress_bar("   Agenda rolling KBO", scanned_weeks, total_weeks)
        week_start = week_start + timedelta(days=7)

    for date_key in sorted(by_day_counter.keys()):
        print(f"   📆 Agenda kbo {date_key}: {by_day_counter[date_key]} juegos")

    df = pd.DataFrame(upcoming_rows)

    if not df.empty:
        df["game_id"] = df["game_id"].astype(str)
        df["date"] = df["date"].astype(str)
        df = df.sort_values(["date", "time", "game_id"]).drop_duplicates(subset=["game_id"], keep="last").reset_index(drop=True)

    return df


def extract_advanced_espn_data():
    print("🚀 Iniciando Extractor Avanzado de KBO (MyKBO scraping)...")

    existing_df = load_existing_data()
    previous_count = len(existing_df)

    fetch_ranges = determine_fetch_ranges(existing_df)

    if fetch_ranges:
        new_df = fetch_completed_games_for_ranges(fetch_ranges)
    else:
        print("âœ… No hay rangos nuevos por consultar.")
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

    print("\nðŸ“Š RESUMEN DE ACTUALIZACIÃ“N kbo")
    print(f"   Partidos previos   : {previous_count}")
    print(f"   Filas descargadas  : {len(new_df)}")
    print(f"   Duplicados quitados: {dedup_removed}")
    print(f"   Partidos finales   : {len(final_df)}")
    print(f"   Netos aÃ±adidos     : {added_count}")

    print(f"\nðŸ’¾ HistÃ³rico actualizado en: {FILE_PATH_ADVANCED}")

    # Agenda rolling (hoy + proximos dias)
    today_str = datetime.now().strftime("%Y-%m-%d")
    df_upcoming = fetch_upcoming_schedule_for_range(today_str)

    if not df_upcoming.empty:
        df_upcoming.to_csv(FILE_PATH_UPCOMING, index=False)
        print(f"ðŸ—“ï¸ Agenda rolling guardada en: {FILE_PATH_UPCOMING}")
        print(f"   Juegos totales agenda: {len(df_upcoming)}")
        print(
            f"   Estados: {sorted(df_upcoming['status_state'].dropna().astype(str).unique().tolist())}"
        )
    else:
        print("âš ï¸ No se encontraron juegos programados para hoy en la agenda kbo.")

    return final_df


if __name__ == "__main__":
    df_advanced = extract_advanced_espn_data()
    if not df_advanced.empty:
        print(df_advanced.head())
