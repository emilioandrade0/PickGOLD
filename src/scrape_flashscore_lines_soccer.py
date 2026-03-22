from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
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
RAW_FILE = BASE_DIR / "data" / "odds_provider" / "flashscore_soccer_raw.csv"

SPORTS_MATCH_CONFIG = {
    "laliga": {
        "scoreboard": "https://site.api.espn.com/apis/site/v2/sports/soccer/esp.1/scoreboard",
        "utc_offset_hours": -6,
    },
    "liga_mx": {
        "scoreboard": "https://site.api.espn.com/apis/site/v2/sports/soccer/mex.1/scoreboard",
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
    txt = re.sub(r"\b(fc|cf|club|deportivo|athletic|atletico|the|de|ac)\b", " ", txt)
    txt = re.sub(r"[^a-z0-9]+", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


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


def _build_espn_lookup(days_ahead: int = 10):
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


def _decimal_to_american(decimal_odds: float):
    d = _safe_float(decimal_odds)
    if d is None or d <= 1.0:
        return None
    if d >= 2.0:
        return round((d - 1.0) * 100.0, 0)
    return round(-100.0 / (d - 1.0), 0)


def _resolve_sport(category_text: str, league_text: str):
    cat = str(category_text or "").strip().lower()
    league = str(league_text or "").strip().lower()

    if "espana" in unicodedata.normalize("NFKD", cat).encode("ascii", "ignore").decode("ascii") and "laliga" in league:
        return "laliga"
    if "mexico" in unicodedata.normalize("NFKD", cat).encode("ascii", "ignore").decode("ascii") and "liga mx" in league:
        return "liga_mx"
    return None


def _collect_flashscore_rows():
    if sync_playwright is None:
        print("[SKIP] playwright is not installed")
        return []

    out = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("https://www.flashscore.com.mx/futbol/", wait_until="networkidle", timeout=120000)
        page.wait_for_timeout(2500)

        # Odds are rendered only in the dedicated odds tab.
        page.locator('div.filters__tab[data-analytics-alias="odds"]').first.click(timeout=15000)
        page.wait_for_timeout(2500)

        all_rows = page.locator('div.event__match[data-event-row="true"]')
        row_count = all_rows.count()
        odds_pattern = re.compile(r"\b\d+\.\d{2}\b")

        for idx in range(row_count):
            row = all_rows.nth(idx)
            home_loc = row.locator(".event__participant--home").first
            away_loc = row.locator(".event__participant--away").first
            if not home_loc.count() or not away_loc.count():
                continue

            home = home_loc.inner_text().strip()
            away = away_loc.inner_text().strip()
            if not home or not away:
                continue

            row_text = row.inner_text().replace("\n", " ")
            nums = [float(x) for x in odds_pattern.findall(row_text)]
            # 1X2 normally appears as 3 values (home/draw/away)
            if len(nums) < 3:
                continue

            home_dec = nums[-3]
            draw_dec = nums[-2]
            away_dec = nums[-1]
            if home_dec <= 1.0 or away_dec <= 1.0:
                continue

            href = row.locator("a.eventRowLink").first.get_attribute("href") or ""
            if href and href.startswith("/"):
                href = f"https://www.flashscore.com.mx{href}"

            header = row.locator("xpath=preceding-sibling::div[contains(@class,'headerLeague__wrapper')][1]").first
            league = ""
            category = ""
            if header.count():
                league_loc = header.locator(".headerLeague__title-text").first
                cat_loc = header.locator(".headerLeague__category-text").first
                league = league_loc.inner_text().strip() if league_loc.count() else ""
                category = cat_loc.inner_text().strip() if cat_loc.count() else ""

            sport = _resolve_sport(category, league)
            if sport not in {"laliga", "liga_mx"}:
                continue

            out.append(
                {
                    "sport": sport,
                    "category": category,
                    "league": league,
                    "home_team": home,
                    "away_team": away,
                    "home_decimal_odds": home_dec,
                    "draw_decimal_odds": draw_dec,
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
        inc["odds_source_provider"] = "flashscore:soccer"

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
        merged.loc[fill_provider, "odds_source_provider"] = merged.loc[fill_provider, src_provider].fillna("flashscore:soccer")
        merged.drop(columns=[src_provider], inplace=True)

    rows_touched = int(any_row_change.sum())
    return merged, rows_touched, cells_filled


def scrape_and_apply(days_ahead: int = 10):
    flash_rows = _collect_flashscore_rows()
    espn_lookup = _build_espn_lookup(days_ahead=days_ahead)

    today = date.today()
    candidate_dates = [
        (today - timedelta(days=1)).strftime("%Y-%m-%d"),
        today.strftime("%Y-%m-%d"),
        (today + timedelta(days=1)).strftime("%Y-%m-%d"),
        (today + timedelta(days=2)).strftime("%Y-%m-%d"),
    ]

    out_rows = []
    for r in flash_rows:
        sport = str(r.get("sport") or "")
        if sport not in {"laliga", "liga_mx"}:
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
            if gid:
                game_id = str(gid)
                game_date = d
                break
        if not game_id or not game_date:
            continue

        home_dec = _safe_float(r.get("home_decimal_odds"))
        draw_dec = _safe_float(r.get("draw_decimal_odds"))
        away_dec = _safe_float(r.get("away_decimal_odds"))
        if home_dec is None or away_dec is None:
            continue

        home_american = _decimal_to_american(home_dec)
        draw_american = _decimal_to_american(draw_dec) if draw_dec is not None else None
        away_american = _decimal_to_american(away_dec)
        if home_american is None or away_american is None:
            continue

        # Keep closing_moneyline_odds aligned with the stronger side (lower decimal).
        closing_ml = home_american if home_dec <= away_dec else away_american

        out_rows.append(
            {
                "sport": sport,
                "date": game_date,
                "game_id": game_id,
                "closing_moneyline_odds": closing_ml,
                "home_moneyline_odds": home_american,
                "draw_moneyline_odds": draw_american,
                "away_moneyline_odds": away_american,
                "odds_source_provider": "flashscore:soccer_1x2",
            }
        )

    raw = pd.DataFrame(flash_rows)
    if raw.empty:
        raw = pd.DataFrame(
            columns=[
                "sport",
                "category",
                "league",
                "home_team",
                "away_team",
                "home_decimal_odds",
                "draw_decimal_odds",
                "away_decimal_odds",
                "source_url",
            ]
        )
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_FILE, index=False)

    overrides = _load_overrides()
    merged, rows_touched, cells_filled = _merge_rows(overrides, out_rows)
    merged = merged.sort_values(["date", "sport", "game_id"], ascending=[True, True, True]).reset_index(drop=True)
    merged.to_csv(OVERRIDES_FILE, index=False)

    print(f"[OK] Flashscore soccer rows(raw): {len(flash_rows)} -> {RAW_FILE}")
    print(f"[OK] Flashscore soccer rows(mapped): {len(out_rows)}")
    print(f"[OK] overrides updated: {OVERRIDES_FILE}")
    print(f"[OK] rows_touched={rows_touched} cells_filled={cells_filled}")


if __name__ == "__main__":
    scrape_and_apply(days_ahead=10)
