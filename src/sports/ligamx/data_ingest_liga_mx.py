import time
from datetime import datetime, timedelta
from pathlib import Path
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
import requests
from odds_market_fields import extract_market_odds_fields, odds_data_quality

BASE_DIR = SRC_ROOT
RAW_DATA_DIR = BASE_DIR / "data" / "liga_mx" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

FILE_PATH_ADVANCED = RAW_DATA_DIR / "liga_mx_advanced_history.csv"
FILE_PATH_UPCOMING = RAW_DATA_DIR / "liga_mx_upcoming_schedule.csv"
FILE_PATH_ALT_HT = RAW_DATA_DIR / "liga_mx_ht_alt_source.csv"
FILE_PATH_ODDS_SNAPSHOTS = RAW_DATA_DIR / "liga_mx_odds_snapshots.csv"

SEASONS_TO_FETCH = {
    "2021": ("2021-01-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
    "2026": ("2026-01-01", "2026-12-31"),
}

TARGET_DATE_LIMIT = datetime.now().strftime("%Y-%m-%d")
BACKFILL_DAYS = 5
LOCAL_UTC_OFFSET_HOURS = 6
UPCOMING_DAYS_AHEAD = 14
EXTERNAL_ODDS_FILE = Path(r"C:\Users\andra\Desktop\Historical\data\exports\historical-odds-liga-mx.csv")

CSV_TEAM_NAME_TO_CODE = {
    "AMERICA": "AME",
    "ATLAS": "ATS",
    "ATLETICO SAN LUIS": "ASL",
    "CRUZ AZUL": "CAZ",
    "FC JUAREZ": "JUA",
    "GUADALAJARA": "GDL",
    "LEON": "LEO",
    "MAZATLAN FC": "MAZ",
    "MONTERREY": "MTY",
    "NECAXA": "NEC",
    "PACHUCA": "PAC",
    "PUEBLA": "PUE",
    "PUMAS": "PUM",
    "QUERETARO": "QRO",
    "SANTOS LAGUNA": "SAN",
    "TIGRES": "UANL",
    "TIJUANA": "TIJ",
    "TOLUCA": "TOL",
}

ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/mex.1/scoreboard"
)
ESPN_SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/mex.1/summary?event={event_id}"
)

USE_SYSTEM_PROXY = str(os.getenv("ESPN_USE_SYSTEM_PROXY", "0")).strip().lower() in {
    "1",
    "true",
    "yes",
}


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


def espn_get_json(url: str, timeout: int = 20, retries: int = 2):
    last_error = None
    for attempt in range(retries + 1):
        try:
            with requests.Session() as session:
                # By default we bypass env proxy because some local envs use
                # placeholder proxies (e.g. 127.0.0.1:9) that block ESPN.
                session.trust_env = USE_SYSTEM_PROXY
                resp = session.get(url, timeout=timeout)
                resp.raise_for_status()
                return resp.json() or {}
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(0.25 * (attempt + 1))
    raise last_error


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


def _extract_first_period_score(linescores):
    if not isinstance(linescores, list) or not linescores:
        return None
    first = linescores[0] or {}
    for key in ["value", "displayValue", "score"]:
        parsed = safe_int(first.get(key), default=None)
        if parsed is not None:
            return parsed
    return None


def _extract_ht_from_summary_payload(payload: dict):
    teams = (payload.get("boxscore") or {}).get("teams") or []
    home_ht = None
    away_ht = None

    for side in teams:
        team_meta = side.get("team") or {}
        side_key = str(team_meta.get("homeAway", "") or "").lower()
        ht_val = _extract_first_period_score(side.get("linescores"))
        if ht_val is None:
            for stat in side.get("statistics") or []:
                stat_name = str((stat or {}).get("name", "") or "").strip().lower()
                if stat_name in {"firsthalfgoals", "first_half_goals", "firsthalfscore"}:
                    ht_val = safe_int((stat or {}).get("displayValue"), default=None)
                    break
        if side_key == "home":
            home_ht = ht_val
        elif side_key == "away":
            away_ht = ht_val

    # Fallback for soccer payloads where HT lives in header competitions competitors.
    if home_ht is None or away_ht is None:
        comps = (payload.get("header") or {}).get("competitions") or []
        if comps:
            competitors = (comps[0] or {}).get("competitors") or []
            for side in competitors:
                side_key = str((side or {}).get("homeAway", "") or "").lower()
                ht_val = _extract_first_period_score((side or {}).get("linescores"))
                if side_key == "home" and ht_val is not None and home_ht is None:
                    home_ht = ht_val
                elif side_key == "away" and ht_val is not None and away_ht is None:
                    away_ht = ht_val

    return home_ht, away_ht


def fetch_event_corners(event_id: str):
    try:
        url = ESPN_SUMMARY_URL.format(event_id=event_id)
        payload = espn_get_json(url, timeout=15, retries=1)
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


def fetch_event_halftime(event_id: str):
    try:
        url = ESPN_SUMMARY_URL.format(event_id=event_id)
        payload = espn_get_json(url, timeout=15, retries=1)
    except Exception:
        return None, None

    return _extract_ht_from_summary_payload(payload)


def backfill_halftime_from_alt_source(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not FILE_PATH_ALT_HT.exists():
        return df

    try:
        alt = pd.read_csv(FILE_PATH_ALT_HT, dtype={"game_id": str})
    except Exception as e:
        print(f"⚠️ No se pudo leer fuente alternativa HT: {e}")
        return df

    required = {"home_ht_score", "away_ht_score"}
    if not required.issubset(set(alt.columns)):
        print("⚠️ Fuente alternativa HT inválida: faltan columnas home_ht_score/away_ht_score.")
        return df

    for c in ["game_id", "date", "home_team", "away_team"]:
        if c not in alt.columns:
            alt[c] = None

    alt["game_id"] = alt["game_id"].astype(str).replace("nan", "").str.strip()
    for c in ["date", "home_team", "away_team"]:
        alt[c] = alt[c].astype(str).str.strip()
    alt["home_team"] = alt["home_team"].apply(normalize_text)
    alt["away_team"] = alt["away_team"].apply(normalize_text)
    alt["home_ht_score"] = pd.to_numeric(alt["home_ht_score"], errors="coerce")
    alt["away_ht_score"] = pd.to_numeric(alt["away_ht_score"], errors="coerce")

    by_game_id = {}
    by_triplet = {}
    for _, r in alt.iterrows():
        hht = r.get("home_ht_score")
        aht = r.get("away_ht_score")
        if pd.isna(hht) or pd.isna(aht):
            continue
        gid = str(r.get("game_id") or "").strip()
        if gid:
            by_game_id[gid] = (int(hht), int(aht))
        key = (str(r.get("date") or "").strip(), str(r.get("home_team") or ""), str(r.get("away_team") or ""))
        if key[0] and key[1] and key[2]:
            by_triplet[key] = (int(hht), int(aht))

    if not by_game_id and not by_triplet:
        print("ℹ️ Fuente alternativa HT no contiene registros útiles.")
        return df

    updates = 0
    for idx, row in df.iterrows():
        h = row.get("home_ht_score")
        a = row.get("away_ht_score")
        if pd.notna(h) and pd.notna(a):
            continue

        gid = str(row.get("game_id") or "").strip()
        key = (
            str(row.get("date") or "").strip(),
            normalize_text(str(row.get("home_team") or "")),
            normalize_text(str(row.get("away_team") or "")),
        )

        vals = by_game_id.get(gid) if gid else None
        if vals is None:
            vals = by_triplet.get(key)
        if vals is None:
            continue

        hht, aht = vals
        df.at[idx, "home_ht_score"] = int(hht)
        df.at[idx, "away_ht_score"] = int(aht)
        df.at[idx, "total_ht_goals"] = int(hht) + int(aht)
        updates += 1

    if updates:
        print(f"   ✅ Fuente alternativa HT aplicada en {updates} juegos.")
    else:
        print("   ℹ️ Fuente alternativa HT no matcheó juegos faltantes.")
    return df


def backfill_missing_halftime_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    for col in ["home_ht_score", "away_ht_score", "total_ht_goals"]:
        if col not in df.columns:
            df[col] = None

    missing_mask = df["home_ht_score"].isna() | df["away_ht_score"].isna()
    missing_ids = df.loc[missing_mask, "game_id"].astype(str).tolist()
    if not missing_ids:
        return df

    print(f"   ⏱️  Backfill HT faltante para {len(missing_ids)} juegos...")

    def _fetch_one(gid: str):
        hht, aht = fetch_event_halftime(gid)
        return gid, hht, aht

    updates = {}
    max_workers = min(8, max(1, len(missing_ids)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_fetch_one, gid) for gid in missing_ids]
        for future in as_completed(futures):
            gid, hht, aht = future.result()
            if hht is None or aht is None:
                continue
            updates[gid] = (int(hht), int(aht))

    if not updates:
        print("   ℹ️  No se recuperaron marcadores HT adicionales.")
        return df

    idx_by_gid = {str(g): i for i, g in enumerate(df["game_id"].astype(str).tolist())}
    applied = 0
    for gid, (hht, aht) in updates.items():
        idx = idx_by_gid.get(gid)
        if idx is None:
            continue
        df.at[idx, "home_ht_score"] = hht
        df.at[idx, "away_ht_score"] = aht
        df.at[idx, "total_ht_goals"] = int(hht) + int(aht)
        applied += 1

    print(f"   ✅ Backfill HT aplicado en {applied} juegos.")
    return df


def _odds_columns():
    return [
        "odds_over_under",
        "home_moneyline_odds",
        "away_moneyline_odds",
        "closing_moneyline_odds",
        "closing_spread_odds",
        "closing_total_odds",
    ]


def _has_real_odds(row: pd.Series) -> bool:
    for c in _odds_columns():
        v = pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0]
        if pd.notna(v) and float(v) != 0.0:
            return True
    return False


def _last_nonzero(series: pd.Series):
    for v in reversed(series.tolist()):
        parsed = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        if pd.notna(parsed) and float(parsed) != 0.0:
            return float(parsed)
    return None


def persist_odds_snapshots(df_upcoming: pd.DataFrame):
    if df_upcoming.empty:
        return

    required = ["game_id", "date", "time", "home_team", "away_team", "odds_data_quality"]
    for c in required + _odds_columns():
        if c not in df_upcoming.columns:
            df_upcoming[c] = None

    snap = df_upcoming[required + _odds_columns()].copy()
    snap["has_real_odds"] = snap.apply(_has_real_odds, axis=1)
    snap = snap[snap["has_real_odds"]].drop(columns=["has_real_odds"])
    if snap.empty:
        return

    snap["snapshot_ts"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    if FILE_PATH_ODDS_SNAPSHOTS.exists():
        try:
            old = pd.read_csv(FILE_PATH_ODDS_SNAPSHOTS, dtype={"game_id": str})
        except Exception:
            old = pd.DataFrame()
        snap = pd.concat([old, snap], ignore_index=True)

    snap["game_id"] = snap["game_id"].astype(str).str.strip()
    snap = snap.sort_values(["game_id", "snapshot_ts"]).drop_duplicates(subset=["game_id"], keep="last")
    snap.to_csv(FILE_PATH_ODDS_SNAPSHOTS, index=False)
    print(f"💾 Snapshots de odds guardados: {len(snap)} juegos -> {FILE_PATH_ODDS_SNAPSHOTS}")


def backfill_history_odds_from_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not FILE_PATH_ODDS_SNAPSHOTS.exists():
        return df
    try:
        snap = pd.read_csv(FILE_PATH_ODDS_SNAPSHOTS, dtype={"game_id": str})
    except Exception as e:
        print(f"⚠️ No se pudo leer snapshots de odds: {e}")
        return df

    if snap.empty:
        return df

    for c in ["game_id"] + _odds_columns():
        if c not in snap.columns:
            return df

    snap = snap.sort_values("snapshot_ts").drop_duplicates(subset=["game_id"], keep="last")
    snap = snap.set_index("game_id")

    updated = 0
    for idx, row in df.iterrows():
        gid = str(row.get("game_id") or "").strip()
        if not gid or gid not in snap.index:
            continue
        snap_row = snap.loc[gid]
        row_changed = False
        for c in _odds_columns():
            curr = pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0]
            repl = pd.to_numeric(pd.Series([snap_row.get(c)]), errors="coerce").iloc[0]
            if (pd.isna(curr) or float(curr) == 0.0) and pd.notna(repl) and float(repl) != 0.0:
                df.at[idx, c] = float(repl)
                row_changed = True
        if row_changed:
            df.at[idx, "odds_data_quality"] = "real"
            updated += 1

    if updated:
        print(f"   ✅ Backfill odds desde snapshot aplicado en {updated} juegos.")
    return df


def preserve_odds_within_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "game_id" not in df.columns:
        return df
    odds_cols = _odds_columns()
    for c in odds_cols:
        if c not in df.columns:
            df[c] = None

    by_gid = df.groupby("game_id", dropna=False)
    fill_maps = {c: by_gid[c].apply(_last_nonzero).to_dict() for c in odds_cols}

    repaired = 0
    for idx, row in df.iterrows():
        gid = row.get("game_id")
        if pd.isna(gid):
            continue
        row_changed = False
        for c in odds_cols:
            curr = pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0]
            best = fill_maps[c].get(gid)
            if (pd.isna(curr) or float(curr) == 0.0) and best is not None:
                df.at[idx, c] = float(best)
                row_changed = True
        if row_changed:
            df.at[idx, "odds_data_quality"] = "real"
            repaired += 1
    if repaired:
        print(f"   ✅ Conservación de odds intra-duplicados aplicada en {repaired} filas.")
    return df


# -----------------------------
# External Odds
# -----------------------------

def _normalize_csv_team_name(name: str) -> str:
    norm = normalize_text(name)
    return CSV_TEAM_NAME_TO_CODE.get(norm, norm)


def load_external_historical_odds() -> pd.DataFrame:
    if not EXTERNAL_ODDS_FILE.exists():
        return pd.DataFrame()

    try:
        odds_df = pd.read_csv(EXTERNAL_ODDS_FILE)
    except Exception as exc:
        print(f"[WARN] No se pudo leer CSV de odds Liga MX: {exc}")
        return pd.DataFrame()

    required = {"event_date", "snapshot_date", "home_team", "away_team", "home_price", "draw_price", "away_price"}
    if not required.issubset(set(odds_df.columns)):
        return pd.DataFrame()

    odds_df = odds_df.copy()
    odds_df["event_dt"] = pd.to_datetime(odds_df["event_date"], utc=True, errors="coerce")
    odds_df["snapshot_dt"] = pd.to_datetime(odds_df["snapshot_date"], utc=True, errors="coerce")
    odds_df = odds_df.dropna(subset=["event_dt", "home_team", "away_team"])
    if odds_df.empty:
        return pd.DataFrame()

    try:
        local_events = odds_df["event_dt"].dt.tz_convert("America/Mexico_City")
    except Exception:
        local_events = odds_df["event_dt"]
    odds_df["date"] = local_events.dt.strftime("%Y-%m-%d")
    odds_df["home_team"] = odds_df["home_team"].map(_normalize_csv_team_name)
    odds_df["away_team"] = odds_df["away_team"].map(_normalize_csv_team_name)

    for col in ["home_price", "draw_price", "away_price"]:
        odds_df[col] = pd.to_numeric(odds_df[col], errors="coerce")

    odds_df = odds_df.sort_values(["date", "home_team", "away_team", "snapshot_dt"])
    odds_df = odds_df.groupby(["date", "home_team", "away_team"], as_index=False).tail(1)

    odds_df["closing_moneyline_odds"] = odds_df[["home_price", "away_price"]].min(axis=1)
    odds_df["home_moneyline_odds"] = odds_df["home_price"]
    odds_df["away_moneyline_odds"] = odds_df["away_price"]
    odds_df["odds_data_quality"] = "historical_csv"

    keep_cols = [
        "date", "home_team", "away_team", "bookmaker", "region", "snapshot_date",
        "home_price", "draw_price", "away_price",
        "home_moneyline_odds", "away_moneyline_odds", "closing_moneyline_odds", "odds_data_quality",
    ]
    return odds_df[keep_cols].copy()


def enrich_with_external_odds(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    odds_df = load_external_historical_odds()
    if odds_df.empty:
        return df

    merged = df.copy().merge(
        odds_df, on=["date", "home_team", "away_team"], how="left", suffixes=("", "_csv")
    )

    for col in [
        "bookmaker", "region", "snapshot_date", "home_price", "draw_price", "away_price",
        "home_moneyline_odds", "away_moneyline_odds", "closing_moneyline_odds", "odds_data_quality",
    ]:
        csv_col = f"{col}_csv"
        if csv_col not in merged.columns:
            continue
        if col not in merged.columns:
            merged[col] = None
        merged[col] = merged[col].where(merged[col].notna(), merged[csv_col])
        merged = merged.drop(columns=[csv_col])

    return merged


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
        print(f"⚠️ No se pudo leer el CSV existente. Se reconstruirá desde cero. Error: {e}")
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
        print("📂 No existe histórico previo. Se hará descarga completa.")
        return build_full_ranges(limit_date_dt)

    try:

        tmp = existing_df.copy()
        tmp["date_dt"] = pd.to_datetime(tmp["date"], errors="coerce")

        max_date = tmp["date_dt"].max()

        if pd.isna(max_date):
            print("⚠️ No se pudo detectar la última fecha del histórico.")
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

        print(f"⚠️ Error calculando rango incremental: {e}")
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
    home_ht_score = _extract_first_period_score(home_data.get("linescores"))
    away_ht_score = _extract_first_period_score(away_data.get("linescores"))
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
        "home_ht_score": home_ht_score,
        "away_ht_score": away_ht_score,
        "total_ht_goals": (
            safe_int(home_ht_score, default=0) + safe_int(away_ht_score, default=0)
            if home_ht_score is not None and away_ht_score is not None
            else None
        ),
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
                    data = espn_get_json(url, timeout=20, retries=1)
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
                    print(f"⚠️ Error al consultar chunk {d1}-{d2}: {error}")
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
                            if row.get("home_ht_score") is None or row.get("away_ht_score") is None:
                                hht, aht = fetch_event_halftime(str(row.get("game_id")))
                                row["home_ht_score"] = hht
                                row["away_ht_score"] = aht
                                if hht is not None and aht is not None:
                                    row["total_ht_goals"] = int(hht) + int(aht)
                            completed_games.append(row)
                    except Exception as inner_e:
                        print(f"⚠️ Error procesando evento: {inner_e}")

        print("     -> ✅ Temporada procesada.")

    return pd.DataFrame(completed_games)


# -----------------------------
# Upcoming Schedule
# -----------------------------

def fetch_upcoming_schedule_for_range(start_date: str, days_ahead: int = UPCOMING_DAYS_AHEAD):
    upcoming_rows = []
    base_dt = datetime.strptime(start_date, "%Y-%m-%d")

    for day_offset in range(days_ahead + 1):
        day_dt = base_dt + timedelta(days=day_offset)
        day_str = day_dt.strftime("%Y-%m-%d")
        url = f"{ESPN_SCOREBOARD_URL}?dates={day_dt.strftime('%Y%m%d')}&limit=500"

        try:
            data = espn_get_json(url, timeout=20, retries=1)
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

            print(f"   📅 Agenda Liga MX {day_str}: {day_count} juegos")

        except Exception as e:
            print(f"⚠️ Error descargando agenda Liga MX {day_str}: {e}")

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

    print("🚀 Iniciando Extractor Avanzado de ESPN Liga MX...")

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
        final_df = backfill_halftime_from_alt_source(final_df)
        final_df = backfill_missing_halftime_scores(final_df)
        final_df = preserve_odds_within_duplicates(final_df)

        final_df = final_df.drop_duplicates(subset=["game_id"], keep="last")
        final_df = backfill_history_odds_from_snapshots(final_df)
        final_df = enrich_with_external_odds(final_df)

        final_df = final_df.sort_values(
            ["date", "time", "game_id"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

        final_df.to_csv(FILE_PATH_ADVANCED, index=False)

    print("\n📊 RESUMEN DE ACTUALIZACIÓN LIGA MX")
    print(f"   Partidos previos   : {previous_count}")
    print(f"   Filas descargadas  : {len(new_df)}")
    print(f"   Partidos finales   : {len(final_df)}")

    today_str = datetime.now().strftime("%Y-%m-%d")

    df_upcoming = fetch_upcoming_schedule_for_range(today_str)

    if not df_upcoming.empty:
        df_upcoming = enrich_with_external_odds(df_upcoming)
        df_upcoming.to_csv(FILE_PATH_UPCOMING, index=False)
        persist_odds_snapshots(df_upcoming)
        print(f"🗓️ Agenda rolling guardada en: {FILE_PATH_UPCOMING}")
        print(f"   Juegos totales agenda: {len(df_upcoming)}")
    else:
        print("⚠️ No se encontraron juegos en ventana rolling de Liga MX.")

    return final_df


# -----------------------------
# Run
# -----------------------------

if __name__ == "__main__":

    df = extract_advanced_espn_data()

    if not df.empty:
        print(df.head())
