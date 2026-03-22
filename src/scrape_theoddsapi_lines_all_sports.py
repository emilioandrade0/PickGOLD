from __future__ import annotations

import os
import re
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
OVERRIDES_FILE = BASE_DIR / "data" / "odds_provider" / "closing_odds_overrides.csv"
RAW_FILE = BASE_DIR / "data" / "odds_provider" / "theoddsapi_all_lines_raw.csv"

# Sports with ESPN scoreboard game IDs that can be matched directly.
SPORTS_MATCH_CONFIG = {
    "nba": {
        "theodds_candidates": ["basketball_nba"],
        "scoreboard": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
        "utc_offset_hours": -5,
    },
    "mlb": {
        "theodds_candidates": ["baseball_mlb"],
        "scoreboard": "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
        "utc_offset_hours": -5,
    },
    "nhl": {
        "theodds_candidates": ["icehockey_nhl"],
        "scoreboard": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard",
        "utc_offset_hours": -5,
    },
    "liga_mx": {
        "theodds_candidates": ["soccer_mexico_ligamx", "soccer_mexico_liga_mx"],
        "scoreboard": "https://site.api.espn.com/apis/site/v2/sports/soccer/mex.1/scoreboard",
        "utc_offset_hours": -6,
    },
    "laliga": {
        "theodds_candidates": ["soccer_spain_la_liga", "soccer_spain_laliga"],
        "scoreboard": "https://site.api.espn.com/apis/site/v2/sports/soccer/esp.1/scoreboard",
        "utc_offset_hours": -6,
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

PREFERRED_BOOKMAKERS = ["pinnacle", "draftkings", "fanduel", "betmgm", "williamhill_us"]
DEFAULT_REQUEST_TIMEOUT_SECONDS = 12.0


def _request_timeout_seconds() -> float:
    raw = str(os.getenv("THEODDSAPI_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return DEFAULT_REQUEST_TIMEOUT_SECONDS
    try:
        val = float(raw)
        return max(4.0, min(val, 60.0))
    except Exception:
        return DEFAULT_REQUEST_TIMEOUT_SECONDS


def _norm_team(value: str) -> str:
    txt = str(value or "").strip().lower()
    txt = unicodedata.normalize("NFKD", txt)
    txt = txt.encode("ascii", "ignore").decode("ascii")
    txt = re.sub(r"\b(fc|cf|club|deportivo|athletic|atletico|the)\b", " ", txt)
    txt = re.sub(r"[^a-z0-9]+", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _safe_float(value: Any):
    try:
        x = float(value)
        if x != x:
            return None
        return x
    except Exception:
        return None


def _fetch_json(url: str, params: dict):
    r = requests.get(url, params=params, timeout=_request_timeout_seconds())
    r.raise_for_status()
    return r.json() or {}


def _to_local_date(event_dt: str, utc_offset_hours: int):
    try:
        dt_utc = datetime.strptime(str(event_dt), "%Y-%m-%dT%H:%MZ")
        local_dt = dt_utc + timedelta(hours=int(utc_offset_hours))
        return local_dt.strftime("%Y-%m-%d")
    except Exception:
        return str(event_dt)[:10]


def _to_utc_date(commence_time: str):
    try:
        dt = datetime.fromisoformat(str(commence_time).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).date().strftime("%Y-%m-%d")
    except Exception:
        return str(commence_time)[:10]


def _team_aliases(team_obj: dict):
    out = set()
    if not isinstance(team_obj, dict):
        return out

    for key in ["displayName", "shortDisplayName", "name", "abbreviation"]:
        val = team_obj.get(key)
        n = _norm_team(val)
        if n:
            out.add(n)
    return out


def _build_espn_lookup(days_ahead: int = 10):
    start = date.today()
    target_days = [start + timedelta(days=i) for i in range(max(1, int(days_ahead)) + 1)]

    lookup: dict[tuple[str, str, str], str] = {}

    for sport, cfg in SPORTS_MATCH_CONFIG.items():
        base_url = cfg["scoreboard"]
        offset = int(cfg.get("utc_offset_hours", 0))

        for day in target_days:
            day_token = day.strftime("%Y%m%d")
            try:
                payload = _fetch_json(base_url, {"dates": day_token, "limit": 500})
            except Exception:
                continue

            events = payload.get("events") or []
            if not isinstance(events, list):
                continue

            for ev in events:
                if not isinstance(ev, dict):
                    continue
                game_id = str(ev.get("id") or "").strip()
                if not game_id:
                    continue

                game_date = _to_local_date(ev.get("date"), offset)
                comps = ev.get("competitions") or []
                comp = comps[0] if isinstance(comps, list) and comps else {}
                competitors = comp.get("competitors") or []
                if not isinstance(competitors, list) or len(competitors) < 2:
                    continue

                home = next((c for c in competitors if str(c.get("homeAway") or "").lower() == "home"), None)
                away = next((c for c in competitors if str(c.get("homeAway") or "").lower() == "away"), None)
                if not isinstance(home, dict) or not isinstance(away, dict):
                    continue

                home_aliases = _team_aliases(home.get("team") or {})
                away_aliases = _team_aliases(away.get("team") or {})
                if not home_aliases or not away_aliases:
                    continue

                for h in home_aliases:
                    for a in away_aliases:
                        teams_key = "|".join(sorted([h, a]))
                        lookup[(sport, game_date, teams_key)] = game_id

    return lookup


def _resolve_theodds_sport_keys(api_key: str):
    sports = _fetch_json("https://api.the-odds-api.com/v4/sports", {"apiKey": api_key})
    available = {str(x.get("key") or "") for x in sports if isinstance(x, dict)}

    resolved: dict[str, str | None] = {}
    for sport, cfg in SPORTS_MATCH_CONFIG.items():
        selected = None
        for c in cfg.get("theodds_candidates", []):
            if c in available:
                selected = c
                break
        resolved[sport] = selected
    return resolved


def _choose_bookmaker(event_obj: dict):
    books = event_obj.get("bookmakers") or []
    if not isinstance(books, list) or not books:
        return None

    by_key = {str(b.get("key") or ""): b for b in books if isinstance(b, dict)}
    for key in PREFERRED_BOOKMAKERS:
        if key in by_key:
            return by_key[key]

    return books[0] if isinstance(books[0], dict) else None


def _outcome_for_team(outcomes: list[dict], team_name: str):
    target = _norm_team(team_name)
    for o in outcomes:
        name = _norm_team(o.get("name") or "")
        if name and name == target:
            return o
    return None


def _extract_from_market(book: dict, home_team: str, away_team: str):
    markets = book.get("markets") or []
    if not isinstance(markets, list):
        return {}

    out = {}

    h2h = next((m for m in markets if str(m.get("key") or "") == "h2h"), None)
    if isinstance(h2h, dict):
        outcomes = h2h.get("outcomes") or []
        home = _outcome_for_team(outcomes, home_team)
        away = _outcome_for_team(outcomes, away_team)
        draw = next((o for o in outcomes if str(o.get("name") or "").strip().lower() == "draw"), None)
        home_price = _safe_float((home or {}).get("price"))
        away_price = _safe_float((away or {}).get("price"))
        draw_price = _safe_float((draw or {}).get("price"))
        if home_price is not None:
            out["home_moneyline_odds"] = home_price
        if draw_price is not None:
            out["draw_moneyline_odds"] = draw_price
        if away_price is not None:
            out["away_moneyline_odds"] = away_price
        if home_price is not None and away_price is not None:
            out["closing_moneyline_odds"] = home_price if abs(home_price) >= abs(away_price) else away_price
        elif home_price is not None:
            out["closing_moneyline_odds"] = home_price
        elif away_price is not None:
            out["closing_moneyline_odds"] = away_price

    spreads = next((m for m in markets if str(m.get("key") or "") == "spreads"), None)
    if isinstance(spreads, dict):
        outcomes = spreads.get("outcomes") or []
        home = _outcome_for_team(outcomes, home_team)
        away = _outcome_for_team(outcomes, away_team)
        home_price = _safe_float((home or {}).get("price"))
        away_price = _safe_float((away or {}).get("price"))

        if home_price is not None and away_price is not None:
            out["closing_spread_odds"] = (home_price + away_price) / 2.0
        elif home_price is not None:
            out["closing_spread_odds"] = home_price
        elif away_price is not None:
            out["closing_spread_odds"] = away_price

        home_point = _safe_float((home or {}).get("point"))
        away_point = _safe_float((away or {}).get("point"))
        if home_point is not None:
            out["closing_spread_line"] = abs(home_point)
        elif away_point is not None:
            out["closing_spread_line"] = abs(away_point)

    totals = next((m for m in markets if str(m.get("key") or "") == "totals"), None)
    if isinstance(totals, dict):
        outcomes = totals.get("outcomes") or []
        over = next((o for o in outcomes if str(o.get("name") or "").strip().lower() == "over"), None)
        under = next((o for o in outcomes if str(o.get("name") or "").strip().lower() == "under"), None)
        over_price = _safe_float((over or {}).get("price"))
        under_price = _safe_float((under or {}).get("price"))

        if over_price is not None and under_price is not None:
            out["closing_total_odds"] = (over_price + under_price) / 2.0
        elif over_price is not None:
            out["closing_total_odds"] = over_price
        elif under_price is not None:
            out["closing_total_odds"] = under_price

        point = _safe_float((over or {}).get("point"))
        if point is None:
            point = _safe_float((under or {}).get("point"))
        if point is not None:
            out["closing_total_line"] = point
            out["odds_over_under"] = point

    return out


def _collect_rows(api_key: str, days_ahead: int = 10):
    espn_lookup = _build_espn_lookup(days_ahead=days_ahead)
    resolved_sports = _resolve_theodds_sport_keys(api_key)

    start_day = date.today()
    end_day = start_day + timedelta(days=max(1, int(days_ahead)))

    rows = []
    for sport, sport_key in resolved_sports.items():
        if not sport_key:
            continue

        print(f"[TheOddsAPI] collecting sport={sport} key={sport_key}")

        params = {
            "apiKey": api_key,
            "regions": "us,eu",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
            "dateFormat": "iso",
        }

        try:
            events = _fetch_json(f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds", params)
        except Exception:
            continue

        if not isinstance(events, list):
            continue

        for ev in events:
            if not isinstance(ev, dict):
                continue

            d = _to_utc_date(ev.get("commence_time"))
            try:
                dd = datetime.strptime(d, "%Y-%m-%d").date()
            except Exception:
                continue
            if dd < start_day or dd > end_day:
                continue

            home = str(ev.get("home_team") or "").strip()
            away = str(ev.get("away_team") or "").strip()
            if not home or not away:
                continue

            teams_key = "|".join(sorted([_norm_team(home), _norm_team(away)]))
            game_id = espn_lookup.get((sport, d, teams_key))
            if not game_id:
                continue

            book = _choose_bookmaker(ev)
            if not isinstance(book, dict):
                continue

            fields = _extract_from_market(book, home, away)
            if not fields:
                continue

            row = {
                "sport": sport,
                "date": d,
                "game_id": str(game_id),
                "odds_source_provider": f"theoddsapi:{book.get('key') or 'unknown'}",
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


def _merge_rows(overrides: pd.DataFrame, rows: list[dict]):
    if not rows:
        return overrides, 0, 0

    inc = pd.DataFrame(rows)
    for key in ["sport", "date", "game_id"]:
        if key == "sport":
            inc[key] = inc[key].fillna("").astype(str).str.strip().str.lower()
        else:
            inc[key] = inc[key].fillna("").astype(str).str.strip()

    for col in [*ODDS_COLUMNS, *LINE_COLUMNS]:
        if col not in inc.columns:
            inc[col] = pd.NA
        inc[col] = pd.to_numeric(inc[col], errors="coerce")

    if "odds_source_provider" not in inc.columns:
        inc["odds_source_provider"] = "theoddsapi"

    inc = inc.drop_duplicates(subset=["sport", "date", "game_id"], keep="first")

    merged = overrides.merge(
        inc[["sport", "date", "game_id", *ODDS_COLUMNS, *LINE_COLUMNS, "odds_source_provider"]],
        on=["sport", "date", "game_id"],
        how="left",
        suffixes=("", "__src"),
    )

    rows_touched = 0
    cells_filled = 0
    any_row_change = pd.Series(False, index=merged.index)

    for col in [*ODDS_COLUMNS, *LINE_COLUMNS]:
        src = f"{col}__src"
        if src not in merged.columns:
            continue
        fill_mask = merged[col].isna() & merged[src].notna()
        cells_filled += int(fill_mask.sum())
        merged.loc[fill_mask, col] = merged.loc[fill_mask, src]
        any_row_change = any_row_change | fill_mask
        merged.drop(columns=[src], inplace=True)

    src_provider = "odds_source_provider__src"
    if src_provider in merged.columns:
        fill_provider = any_row_change & merged["odds_source_provider"].astype(str).str.strip().eq("")
        merged.loc[fill_provider, "odds_source_provider"] = merged.loc[fill_provider, src_provider].fillna("theoddsapi")
        merged.drop(columns=[src_provider], inplace=True)

    rows_touched = int(any_row_change.sum())
    return merged, rows_touched, cells_filled


def scrape_and_apply(api_key: str | None = None, days_ahead: int = 10):
    key = str(api_key or os.getenv("THEODDSAPI_KEY") or "").strip()
    if not key:
        print("[SKIP] THEODDSAPI_KEY not set")
        return

    days = max(1, min(int(days_ahead), 7))
    print(f"[TheOddsAPI] days_ahead={days} timeout={_request_timeout_seconds()}s")
    rows = _collect_rows(key, days_ahead=days)
    raw = pd.DataFrame(rows)
    if raw.empty:
        raw = pd.DataFrame(columns=["sport", "date", "game_id", *ODDS_COLUMNS, *LINE_COLUMNS, "odds_source_provider"])

    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_FILE, index=False)

    overrides = _load_overrides()
    merged, rows_touched, cells_filled = _merge_rows(overrides, rows)
    merged = merged.sort_values(["date", "sport", "game_id"], ascending=[True, True, True]).reset_index(drop=True)
    merged.to_csv(OVERRIDES_FILE, index=False)

    print(f"[OK] TheOddsAPI rows: {len(raw)} -> {RAW_FILE}")
    print(f"[OK] overrides updated: {OVERRIDES_FILE}")
    print(f"[OK] rows_touched={rows_touched} cells_filled={cells_filled}")


if __name__ == "__main__":
    scrape_and_apply(days_ahead=3)
