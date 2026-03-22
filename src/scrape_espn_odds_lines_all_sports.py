from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from odds_market_fields import extract_market_odds_fields

BASE_DIR = Path(__file__).resolve().parent
OVERRIDES_FILE = BASE_DIR / "data" / "odds_provider" / "closing_odds_overrides.csv"
RAW_SCRAPE_FILE = BASE_DIR / "data" / "odds_provider" / "espn_all_lines_raw.csv"

SPORTS_CONFIG = {
    "nba": {
        "scoreboard": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
        "core_sport": "basketball",
        "core_league": "nba",
        "utc_offset_hours": -5,
    },
    "mlb": {
        "scoreboard": "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
        "core_sport": "baseball",
        "core_league": "mlb",
        "utc_offset_hours": -5,
    },
    "nhl": {
        "scoreboard": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard",
        "core_sport": "hockey",
        "core_league": "nhl",
        "utc_offset_hours": -5,
    },
    "liga_mx": {
        "scoreboard": "https://site.api.espn.com/apis/site/v2/sports/soccer/mex.1/scoreboard",
        "core_sport": "soccer",
        "core_league": "mex.1",
        "utc_offset_hours": -6,
    },
    "laliga": {
        "scoreboard": "https://site.api.espn.com/apis/site/v2/sports/soccer/esp.1/scoreboard",
        "core_sport": "soccer",
        "core_league": "esp.1",
        "utc_offset_hours": -6,
    },
    "kbo": {
        "scoreboard": "https://site.api.espn.com/apis/site/v2/sports/baseball/kbo/scoreboard",
        "core_sport": "baseball",
        "core_league": "kbo",
        "utc_offset_hours": 9,
    },
}

ODDS_COLUMNS = [
    "closing_moneyline_odds",
    "home_moneyline_odds",
    "draw_moneyline_odds",
    "away_moneyline_odds",
    "closing_spread_odds",
    "closing_total_odds",
    "closing_q1_odds",
    "closing_f5_odds",
    "closing_home_over_odds",
    "closing_corners_odds",
    "closing_btts_odds",
]

LINE_COLUMNS = ["closing_spread_line", "closing_total_line", "odds_over_under"]


def _safe_float(value: Any):
    try:
        x = float(value)
        if x != x:
            return None
        return x
    except Exception:
        return None


def _fetch_json(url: str):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json() or {}


def _parse_local_date(event_dt: str, utc_offset_hours: int):
    try:
        dt_utc = datetime.strptime(str(event_dt), "%Y-%m-%dT%H:%MZ")
        local_dt = dt_utc + timedelta(hours=int(utc_offset_hours))
        return local_dt.strftime("%Y-%m-%d")
    except Exception:
        return str(event_dt)[:10]


def _core_odds_url(core_sport: str, core_league: str, event_id: str):
    return (
        f"https://sports.core.api.espn.com/v2/sports/{core_sport}/leagues/{core_league}/"
        f"events/{event_id}/competitions/{event_id}/odds?lang=en&region=us&limit=200"
    )


def _best_odds_item(items: list[dict]):
    if not isinstance(items, list) or not items:
        return None

    scored = []
    for item in items:
        if not isinstance(item, dict):
            continue
        fields = extract_market_odds_fields(item)
        has_lines = (_safe_float(item.get("spread")) is not None) or (_safe_float(item.get("overUnder")) is not None)
        score = len(fields) + (1 if has_lines else 0)
        provider = item.get("provider") or {}
        priority = provider.get("priority") if isinstance(provider, dict) else None
        try:
            prio = int(priority)
        except Exception:
            prio = 999
        scored.append((score, -prio, item))

    if not scored:
        return None

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


def _scrape_sport_date(sport: str, cfg: dict, date_str: str):
    rows = []
    day_token = date_str.replace("-", "")
    sb_url = f"{cfg['scoreboard']}?dates={day_token}&limit=500"

    try:
        payload = _fetch_json(sb_url)
    except Exception:
        return rows

    events = payload.get("events") or []
    if not isinstance(events, list):
        return rows

    for ev in events:
        if not isinstance(ev, dict):
            continue
        event_id = str(ev.get("id") or "").strip()
        if not event_id:
            continue

        local_date = _parse_local_date(ev.get("date"), cfg.get("utc_offset_hours", 0))
        core_url = _core_odds_url(cfg["core_sport"], cfg["core_league"], event_id)

        try:
            core_payload = _fetch_json(core_url)
            item = _best_odds_item(core_payload.get("items") or [])
        except Exception:
            item = None

        if not isinstance(item, dict):
            odds_from_event = []
            try:
                comp = (ev.get("competitions") or [{}])[0]
                odds_from_event = comp.get("odds") or []
            except Exception:
                odds_from_event = []
            item = _best_odds_item(odds_from_event)

        if not isinstance(item, dict):
            continue

        fields = extract_market_odds_fields(item)
        spread_line = _safe_float(item.get("spread"))
        total_line = _safe_float(item.get("overUnder"))

        row = {
            "sport": sport,
            "date": local_date,
            "game_id": event_id,
            "closing_spread_line": spread_line,
            "closing_total_line": total_line,
            "odds_over_under": total_line,
            "source_provider": ((item.get("provider") or {}).get("name") if isinstance(item.get("provider"), dict) else "espn_core_odds"),
        }
        row.update(fields)
        rows.append(row)

    return rows


def _load_overrides() -> pd.DataFrame:
    if OVERRIDES_FILE.exists():
        df = pd.read_csv(OVERRIDES_FILE, dtype=str)
    else:
        df = pd.DataFrame(columns=["sport", "date", "game_id"])

    for key in ["sport", "date", "game_id"]:
        if key not in df.columns:
            df[key] = ""
        if key == "sport":
            df[key] = df[key].fillna("").astype(str).str.strip().str.lower()
        else:
            df[key] = df[key].fillna("").astype(str).str.strip()

    for col in [*ODDS_COLUMNS, *LINE_COLUMNS]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "odds_source_provider" not in df.columns:
        df["odds_source_provider"] = ""

    return df


def _merge_scraped_into_overrides(overrides: pd.DataFrame, scraped_rows: list[dict]):
    if not scraped_rows:
        return overrides, 0, 0

    scraped = pd.DataFrame(scraped_rows)
    for key in ["sport", "date", "game_id"]:
        if key == "sport":
            scraped[key] = scraped[key].fillna("").astype(str).str.strip().str.lower()
        else:
            scraped[key] = scraped[key].fillna("").astype(str).str.strip()

    for col in [*ODDS_COLUMNS, *LINE_COLUMNS]:
        if col not in scraped.columns:
            scraped[col] = pd.NA
        scraped[col] = pd.to_numeric(scraped[col], errors="coerce")

    if "source_provider" not in scraped.columns:
        scraped["source_provider"] = "espn_core_odds"

    scraped = scraped.drop_duplicates(subset=["sport", "date", "game_id"], keep="first")
    merged = overrides.merge(
        scraped[["sport", "date", "game_id", *ODDS_COLUMNS, *LINE_COLUMNS, "source_provider"]],
        on=["sport", "date", "game_id"],
        how="left",
        suffixes=("", "__scraped"),
    )

    cells_filled = 0
    rows_touched = 0

    any_row_change = pd.Series(False, index=merged.index)

    for col in [*ODDS_COLUMNS, *LINE_COLUMNS]:
        src = f"{col}__scraped"
        if src not in merged.columns:
            continue
        fill_mask = merged[col].isna() & merged[src].notna()
        cells_filled += int(fill_mask.sum())
        merged.loc[fill_mask, col] = merged.loc[fill_mask, src]
        any_row_change = any_row_change | fill_mask
        merged.drop(columns=[src], inplace=True)

    src_provider_col = "source_provider"
    if src_provider_col in merged.columns:
        provider_fill = any_row_change & merged["odds_source_provider"].astype(str).str.strip().eq("")
        merged.loc[provider_fill, "odds_source_provider"] = merged.loc[provider_fill, src_provider_col].fillna("espn_core_odds")
        merged.drop(columns=[src_provider_col], inplace=True)

    rows_touched = int(any_row_change.sum())

    return merged, rows_touched, cells_filled


def scrape_and_apply(days_ahead: int = 10, max_workers: int = 8):
    start = date.today()
    targets = [start + timedelta(days=i) for i in range(max(1, int(days_ahead)) + 1)]
    target_dates = [d.strftime("%Y-%m-%d") for d in targets]

    jobs = []
    for sport, cfg in SPORTS_CONFIG.items():
        for d in target_dates:
            jobs.append((sport, cfg, d))

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
        futs = [ex.submit(_scrape_sport_date, sport, cfg, d) for sport, cfg, d in jobs]
        for fut in as_completed(futs):
            try:
                rows.extend(fut.result() or [])
            except Exception:
                pass

    raw = pd.DataFrame(rows)
    if raw.empty:
        raw = pd.DataFrame(columns=["sport", "date", "game_id", *ODDS_COLUMNS, *LINE_COLUMNS, "source_provider"])
    RAW_SCRAPE_FILE.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_SCRAPE_FILE, index=False)

    overrides = _load_overrides()
    merged, rows_touched, cells_filled = _merge_scraped_into_overrides(overrides, rows)
    merged = merged.sort_values(["date", "sport", "game_id"], ascending=[True, True, True]).reset_index(drop=True)
    merged.to_csv(OVERRIDES_FILE, index=False)

    print(f"[OK] scraped rows: {len(raw)} -> {RAW_SCRAPE_FILE}")
    print(f"[OK] overrides updated: {OVERRIDES_FILE}")
    print(f"[OK] rows_touched={rows_touched} cells_filled={cells_filled}")


if __name__ == "__main__":
    scrape_and_apply(days_ahead=10, max_workers=8)
