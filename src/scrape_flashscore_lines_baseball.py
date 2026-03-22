from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from difflib import SequenceMatcher
import json
import re
import unicodedata

import pandas as pd
import requests

try:
    from playwright.sync_api import sync_playwright
except Exception:  # pragma: no cover - optional dependency at runtime
    sync_playwright = None

BASE_DIR = Path(__file__).resolve().parent
OVERRIDES_FILE = BASE_DIR / "data" / "odds_provider" / "closing_odds_overrides.csv"
RAW_FILE = BASE_DIR / "data" / "odds_provider" / "flashscore_baseball_raw.csv"

PREDICTIONS_DIRS = {
    "mlb": BASE_DIR / "data" / "mlb" / "predictions",
    "kbo": BASE_DIR / "data" / "kbo" / "predictions",
}

KBO_TEAM_ALIASES = {
    "ssg landers": "ssg",
    "kiwoom heroes": "kiw",
    "lotte giants": "lot",
    "doosan bears": "doo",
    "kt wiz": "ktw",
    "lg twins": "lg",
    "nc dinos": "ncd",
    "kia tigers": "kia",
    "hanwha eagles": "han",
    "samsung lions": "sam",
}

KBO_ABBR_ALIASES = {
    "ssg": "ssg",
    "kiw": "kiw",
    "lot": "lot",
    "doo": "doo",
    "ktw": "ktw",
    "lg": "lg",
    "ncd": "ncd",
    "kia": "kia",
    "han": "han",
    "sam": "sam",
}

SPORTS_MATCH_CONFIG = {
    "mlb": {
        "scoreboard": "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
        "utc_offset_hours": -5,
    },
    "kbo": {
        "scoreboard": "https://site.api.espn.com/apis/site/v2/sports/baseball/kbo/scoreboard",
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


def _safe_float(value):
    try:
        x = float(value)
        if x != x:
            return None
        return x
    except Exception:
        return None


def _norm_team(value: str) -> str:
    txt = str(value or "").strip().lower()
    txt = unicodedata.normalize("NFKD", txt)
    txt = txt.encode("ascii", "ignore").decode("ascii")
    txt = re.sub(r"\b(fc|cf|club|deportivo|athletic|atletico|the|de)\b", " ", txt)
    txt = re.sub(r"[^a-z0-9]+", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    if txt in KBO_TEAM_ALIASES:
        return KBO_TEAM_ALIASES[txt]
    if txt in KBO_ABBR_ALIASES:
        return KBO_ABBR_ALIASES[txt]
    return txt


def _extract_teams_from_game_name(game_name: str):
    text = str(game_name or "").strip()
    if not text:
        return None, None

    # Common format: "AWAY @ HOME"
    if "@" in text:
        parts = [p.strip() for p in text.split("@", 1)]
        if len(parts) == 2:
            return parts[1], parts[0]

    # Fallback separators.
    for sep in [" vs ", " v ", "-", "/"]:
        if sep in text.lower():
            raw_parts = re.split(sep, text, maxsplit=1, flags=re.IGNORECASE)
            if len(raw_parts) == 2:
                return raw_parts[0].strip(), raw_parts[1].strip()

    return None, None


def _extract_time_hhmm(raw: str):
    txt = str(raw or "").strip()
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", txt)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    if hh < 0 or hh > 23 or mm < 0 or mm > 59:
        return None
    return hh * 60 + mm


def _to_local_date(event_dt: str, utc_offset_hours: int):
    try:
        dt_utc = datetime.strptime(str(event_dt), "%Y-%m-%dT%H:%MZ")
        local_dt = dt_utc + timedelta(hours=int(utc_offset_hours))
        return local_dt.strftime("%Y-%m-%d")
    except Exception:
        return str(event_dt)[:10]


def _fetch_json(url: str):
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    return r.json() or {}


def _build_espn_lookup(days_ahead: int = 2):
    start = date.today() - timedelta(days=1)
    target_days = [start + timedelta(days=i) for i in range(max(2, int(days_ahead)) + 2)]

    lookup: dict[tuple[str, str, str], str] = {}

    for sport, cfg in SPORTS_MATCH_CONFIG.items():
        base_url = cfg["scoreboard"]
        offset = int(cfg.get("utc_offset_hours", 0))

        for day in target_days:
            day_token = day.strftime("%Y%m%d")
            try:
                payload = _fetch_json(f"{base_url}?dates={day_token}&limit=500")
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

                home_team = str((home.get("team") or {}).get("displayName") or "").strip()
                away_team = str((away.get("team") or {}).get("displayName") or "").strip()
                if not home_team or not away_team:
                    continue

                teams_key = "|".join(sorted([_norm_team(home_team), _norm_team(away_team)]))
                lookup[(sport, game_date, teams_key)] = game_id

    return lookup


def _build_predictions_lookup(days_ahead: int = 2):
    start = date.today() - timedelta(days=1)
    end = date.today() + timedelta(days=max(2, int(days_ahead)) + 2)

    lookup: dict[tuple[str, str, str], str] = {}
    by_date: dict[tuple[str, str], list[dict]] = {}

    for sport, pred_dir in PREDICTIONS_DIRS.items():
        if not pred_dir.exists():
            continue

        for fp in pred_dir.glob("*.json"):
            date_str = fp.stem
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d").date()
            except Exception:
                continue
            if dt < start or dt > end:
                continue

            try:
                with open(fp, "r", encoding="utf-8") as f:
                    rows = json.load(f)
            except Exception:
                continue
            if not isinstance(rows, list):
                continue

            for row in rows:
                if not isinstance(row, dict):
                    continue
                gid = str(row.get("game_id") or "").strip()
                home = str(row.get("home_team") or "").strip()
                away = str(row.get("away_team") or "").strip()
                if not gid or not home or not away:
                    continue

                teams_key = "|".join(sorted([_norm_team(home), _norm_team(away)]))
                lookup[(sport, date_str, teams_key)] = gid
                by_date.setdefault((sport, date_str), []).append(
                    {
                        "game_id": gid,
                        "home_norm": _norm_team(home),
                        "away_norm": _norm_team(away),
                        "time_min": _extract_time_hhmm(row.get("time")),
                    }
                )

                # Add alternate key from compact game_name formats like "LOT @ SSG".
                gm_home, gm_away = _extract_teams_from_game_name(row.get("game_name"))
                if gm_home and gm_away:
                    gm_key = "|".join(sorted([_norm_team(gm_home), _norm_team(gm_away)]))
                    lookup[(sport, date_str, gm_key)] = gid

    return lookup, by_date


def _team_similarity(a: str, b: str) -> float:
    x = str(a or "").strip().lower()
    y = str(b or "").strip().lower()
    if not x or not y:
        return 0.0
    if x == y:
        return 1.0
    if x in y or y in x:
        return 0.95
    return float(SequenceMatcher(None, x, y).ratio())


def _fallback_match_game_id(sport: str, date_str: str, home: str, away: str, event_time_raw: str, pred_by_date: dict):
    if sport != "kbo":
        return None

    candidates = pred_by_date.get((sport, date_str)) or []
    if not candidates:
        return None

    home_norm = _norm_team(home)
    away_norm = _norm_team(away)
    event_time = _extract_time_hhmm(event_time_raw)

    best_gid = None
    best_score = 0.0

    for cand in candidates:
        h = str(cand.get("home_norm") or "")
        a = str(cand.get("away_norm") or "")

        straight = (_team_similarity(home_norm, h) + _team_similarity(away_norm, a)) / 2.0
        swapped = (_team_similarity(home_norm, a) + _team_similarity(away_norm, h)) / 2.0
        score = max(straight, swapped)

        cand_time = cand.get("time_min")
        if event_time is not None and cand_time is not None:
            delta = abs(int(event_time) - int(cand_time))
            if delta <= 45:
                score += 0.03
            elif delta <= 120:
                score += 0.01

        if score > best_score:
            best_score = score
            best_gid = str(cand.get("game_id") or "")

    # Guardrail: only accept very high-confidence fuzzy matches.
    if best_gid and best_score >= 0.93:
        return best_gid
    return None


def _decimal_to_american(decimal_odds: float):
    d = _safe_float(decimal_odds)
    if d is None or d <= 1.0:
        return None
    if d >= 2.0:
        return round((d - 1.0) * 100.0, 0)
    return round(-100.0 / (d - 1.0), 0)


def _resolve_sport_from_league(league_text: str):
    league = str(league_text or "").strip().lower()
    if "mlb" in league:
        return "mlb"
    if "kbo" in league:
        return "kbo"
    return None


def _collect_flashscore_rows():
    if sync_playwright is None:
        print("[SKIP] playwright is not installed")
        return []

    out = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("https://www.flashscore.com.mx/beisbol/", wait_until="networkidle", timeout=120000)
        page.wait_for_timeout(2500)

        # Odds are rendered only in the dedicated odds tab.
        page.locator('div.filters__tab[data-analytics-alias="odds"]').first.click(timeout=15000)
        page.wait_for_timeout(2500)

        all_rows = page.locator('div.event__match[data-event-row="true"]')
        row_count = all_rows.count()
        odds_pattern = re.compile(r"\b\d+\.\d{2}\b")

        for idx in range(row_count):
            row = all_rows.nth(idx)
            home = row.locator(".event__participant--home").first.inner_text().strip()
            away = row.locator(".event__participant--away").first.inner_text().strip()
            if not home or not away:
                continue

            row_text = row.inner_text().replace("\n", " ")
            nums = [float(x) for x in odds_pattern.findall(row_text)]
            if len(nums) < 2:
                continue

            home_dec = nums[-2]
            away_dec = nums[-1]
            if home_dec <= 1.0 or away_dec <= 1.0:
                continue

            href = row.locator("a.eventRowLink").first.get_attribute("href") or ""
            if href and href.startswith("/"):
                href = f"https://www.flashscore.com.mx{href}"

            league_locator = row.locator(
                "xpath=preceding-sibling::div[contains(@class,'headerLeague__wrapper')][1]//span[contains(@class,'headerLeague__title-text')]"
            ).first
            league = league_locator.inner_text().strip() if league_locator.count() else ""

            out.append(
                {
                    "league": league,
                    "home_team": home,
                    "away_team": away,
                    "event_time": row.locator(".event__time").first.inner_text().strip() if row.locator(".event__time").first.count() else "",
                    "home_decimal_odds": home_dec,
                    "away_decimal_odds": away_dec,
                    "source_url": href,
                }
            )

        browser.close()

    return out


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
        inc["odds_source_provider"] = "flashscore"

    keep_cols = ["sport", "date", "game_id", *ODDS_COLUMNS, *LINE_COLUMNS, "odds_source_provider"]
    inc = inc[keep_cols].drop_duplicates(subset=["sport", "date", "game_id"], keep="first")

    merged = overrides.merge(
        inc,
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
        merged.loc[fill_provider, "odds_source_provider"] = merged.loc[fill_provider, src_provider].fillna("flashscore")
        merged.drop(columns=[src_provider], inplace=True)

    rows_touched = int(any_row_change.sum())
    return merged, rows_touched, cells_filled


def scrape_and_apply(days_ahead: int = 2):
    flash_rows = _collect_flashscore_rows()
    espn_lookup = _build_espn_lookup(days_ahead=days_ahead)
    pred_lookup, pred_by_date = _build_predictions_lookup(days_ahead=days_ahead)

    today_str = date.today().strftime("%Y-%m-%d")
    candidate_dates = [
        (date.today() - timedelta(days=1)).strftime("%Y-%m-%d"),
        today_str,
        (date.today() + timedelta(days=1)).strftime("%Y-%m-%d"),
        (date.today() + timedelta(days=2)).strftime("%Y-%m-%d"),
    ]

    out_rows = []
    for r in flash_rows:
        sport = _resolve_sport_from_league(r.get("league"))
        if sport not in {"mlb", "kbo"}:
            continue

        home = str(r.get("home_team") or "").strip()
        away = str(r.get("away_team") or "").strip()
        if not home or not away:
            continue

        teams_key = "|".join(sorted([_norm_team(home), _norm_team(away)]))
        game_id = None
        game_date = None
        for d in candidate_dates:
            gid = espn_lookup.get((sport, d, teams_key))
            if not gid:
                gid = pred_lookup.get((sport, d, teams_key))
            if not gid:
                gid = _fallback_match_game_id(
                    sport,
                    d,
                    home,
                    away,
                    r.get("event_time"),
                    pred_by_date,
                )
            if gid:
                game_id = str(gid)
                game_date = d
                break
        if not game_id or not game_date:
            continue

        home_dec = _safe_float(r.get("home_decimal_odds"))
        away_dec = _safe_float(r.get("away_decimal_odds"))
        home_american = _decimal_to_american(home_dec)
        away_american = _decimal_to_american(away_dec)
        if home_american is None or away_american is None:
            continue

        closing_ml = home_american if abs(home_american) >= abs(away_american) else away_american

        out_rows.append(
            {
                "sport": sport,
                "date": game_date,
                "game_id": game_id,
                "closing_moneyline_odds": closing_ml,
                "home_moneyline_odds": home_american,
                "away_moneyline_odds": away_american,
                "odds_source_provider": "flashscore:moneyline",
            }
        )

    raw = pd.DataFrame(flash_rows)
    if raw.empty:
        raw = pd.DataFrame(columns=["league", "home_team", "away_team", "home_decimal_odds", "away_decimal_odds", "source_url"])
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_FILE, index=False)

    overrides = _load_overrides()
    merged, rows_touched, cells_filled = _merge_rows(overrides, out_rows)
    merged = merged.sort_values(["date", "sport", "game_id"], ascending=[True, True, True]).reset_index(drop=True)
    merged.to_csv(OVERRIDES_FILE, index=False)

    print(f"[OK] Flashscore rows(raw): {len(flash_rows)} -> {RAW_FILE}")
    print(f"[OK] Flashscore rows(mapped): {len(out_rows)}")
    print(f"[OK] overrides updated: {OVERRIDES_FILE}")
    print(f"[OK] rows_touched={rows_touched} cells_filled={cells_filled}")


if __name__ == "__main__":
    scrape_and_apply(days_ahead=2)
