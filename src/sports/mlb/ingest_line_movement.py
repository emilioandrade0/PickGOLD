from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import requests

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
CACHE_DIR = BASE_DIR / "data" / "mlb" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RAW_HISTORY_FILE = BASE_DIR / "data" / "mlb" / "raw" / "mlb_advanced_history.csv"
UPCOMING_FILE = BASE_DIR / "data" / "mlb" / "raw" / "mlb_upcoming_schedule.csv"
LINE_MOVEMENT_CACHE = CACHE_DIR / "line_movement.csv"
LINE_MOVEMENT_HISTORY = CACHE_DIR / "line_movement_history.csv"

SPORT_KEY = "baseball_mlb"
ODDS_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"

TEAM_NAME_TO_ABBR = {
    "arizona diamondbacks": "ARI",
    "atlanta braves": "ATL",
    "athletics": "ATH",
    "baltimore orioles": "BAL",
    "boston red sox": "BOS",
    "chicago cubs": "CHC",
    "chicago white sox": "CHW",
    "cincinnati reds": "CIN",
    "cleveland guardians": "CLE",
    "colorado rockies": "COL",
    "detroit tigers": "DET",
    "houston astros": "HOU",
    "kansas city royals": "KC",
    "los angeles angels": "LAA",
    "los angeles dodgers": "LAD",
    "miami marlins": "MIA",
    "milwaukee brewers": "MIL",
    "minnesota twins": "MIN",
    "new york mets": "NYM",
    "new york yankees": "NYY",
    "philadelphia phillies": "PHI",
    "pittsburgh pirates": "PIT",
    "san diego padres": "SD",
    "san francisco giants": "SF",
    "seattle mariners": "SEA",
    "st. louis cardinals": "STL",
    "tampa bay rays": "TB",
    "texas rangers": "TEX",
    "toronto blue jays": "TOR",
    "washington nationals": "WSH",
}


def _normalize_team_name(name: str) -> str:
    return str(name or "").strip().lower()


def _team_to_abbr(name: str) -> str:
    return TEAM_NAME_TO_ABBR.get(_normalize_team_name(name), "")


def _median_or_none(values: Iterable[float]) -> Optional[float]:
    arr = [float(v) for v in values if pd.notna(v)]
    if not arr:
        return None
    return float(pd.Series(arr).median())


def _safe_float(value) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except Exception:
        return None


def _normalize_moneyline_price(value) -> Optional[float]:
    price = _safe_float(value)
    if price is None:
        return None
    if abs(price) >= 100:
        return float(price)
    if 1.01 <= price <= 20.0:
        if price >= 2.0:
            return float(round((price - 1.0) * 100.0))
        return float(round(-100.0 / (price - 1.0)))
    return None


def _american_to_implied_prob(price: float) -> Optional[float]:
    price = _normalize_moneyline_price(price)
    if price is None:
        return None
    if price > 0:
        return 100.0 / (price + 100.0)
    return abs(price) / (abs(price) + 100.0)


def _implied_prob_to_american(prob: float) -> Optional[float]:
    try:
        prob = float(prob)
    except Exception:
        return None
    if prob <= 0.0 or prob >= 1.0:
        return None
    if prob >= 0.5:
        return float(round(-(100.0 * prob) / (1.0 - prob)))
    return float(round((100.0 * (1.0 - prob)) / prob))


def _load_schedule_lookup() -> Dict[tuple[str, str, str], str]:
    frames = []
    for path in [RAW_HISTORY_FILE, UPCOMING_FILE]:
        if path.exists():
            try:
                df = pd.read_csv(path, dtype={"game_id": str}, low_memory=False)
                keep = [c for c in ["game_id", "date", "home_team", "away_team"] if c in df.columns]
                if len(keep) == 4:
                    frames.append(df[keep].copy())
            except Exception:
                pass

    if not frames:
        return {}

    merged = pd.concat(frames, ignore_index=True)
    merged["game_id"] = merged["game_id"].astype(str)
    merged["date"] = merged["date"].astype(str)
    merged["home_team"] = merged["home_team"].astype(str)
    merged["away_team"] = merged["away_team"].astype(str)
    merged = merged.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    return {
        (row["date"], row["home_team"], row["away_team"]): row["game_id"]
        for _, row in merged.iterrows()
    }


def _extract_market_map(bookmaker: dict) -> Dict[str, dict]:
    out = {}
    for market in bookmaker.get("markets", []) or []:
        key = str(market.get("key") or "").strip().lower()
        if key:
            out[key] = market
    return out


def _extract_event_snapshot(event: dict, schedule_lookup: Dict[tuple[str, str, str], str], fetched_at_utc: str) -> Optional[dict]:
    home_name = str(event.get("home_team") or "").strip()
    away_name = str(event.get("away_team") or "").strip()
    home_abbr = _team_to_abbr(home_name)
    away_abbr = _team_to_abbr(away_name)
    if not home_abbr or not away_abbr:
        return None

    commence = pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")
    if pd.isna(commence):
        return None

    # Match the ESPN ingest date convention used elsewhere in MLB.
    game_date = (commence - pd.Timedelta(hours=5)).strftime("%Y-%m-%d")
    espn_game_id = schedule_lookup.get((game_date, home_abbr, away_abbr), "")

    home_probs = []
    away_probs = []
    total_points = []
    home_spreads = []
    bookmaker_count = 0

    for bookmaker in event.get("bookmakers", []) or []:
        market_map = _extract_market_map(bookmaker)
        if not market_map:
            continue
        bookmaker_count += 1

        h2h = market_map.get("h2h") or {}
        for outcome in h2h.get("outcomes", []) or []:
            name = str(outcome.get("name") or "").strip()
            price = _normalize_moneyline_price(outcome.get("price"))
            implied_prob = _american_to_implied_prob(outcome.get("price"))
            if implied_prob is None:
                continue
            if name == home_name:
                home_probs.append(implied_prob)
            elif name == away_name:
                away_probs.append(implied_prob)

        spreads = market_map.get("spreads") or {}
        for outcome in spreads.get("outcomes", []) or []:
            name = str(outcome.get("name") or "").strip()
            point = _safe_float(outcome.get("point"))
            if point is None:
                continue
            if name == home_name:
                home_spreads.append(point)

        totals = market_map.get("totals") or {}
        for outcome in totals.get("outcomes", []) or []:
            point = _safe_float(outcome.get("point"))
            if point is not None:
                total_points.append(point)

    home_prob = _median_or_none(home_probs)
    away_prob = _median_or_none(away_probs)
    home_price = _implied_prob_to_american(home_prob) if home_prob is not None else None
    away_price = _implied_prob_to_american(away_prob) if away_prob is not None else None
    total_line = _median_or_none(total_points)
    home_spread = _median_or_none(home_spreads)

    if home_price is None and away_price is None and total_line is None and home_spread is None:
        return None

    return {
        "snapshot_time_utc": fetched_at_utc,
        "external_game_id": str(event.get("id") or ""),
        "game_id": str(espn_game_id or ""),
        "date": game_date,
        "commence_time_utc": commence.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "home_team": home_abbr,
        "away_team": away_abbr,
        "home_team_name": home_name,
        "away_team_name": away_name,
        "bookmakers_count": bookmaker_count,
        "home_moneyline": home_price,
        "away_moneyline": away_price,
        "home_spread": home_spread,
        "total_line": total_line,
    }


def fetch_the_odds_api(save_path: Path = LINE_MOVEMENT_CACHE, timeout: int = 30) -> Path:
    api_key = os.environ.get("THE_ODDS_API_KEY") or os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise EnvironmentError("Set THE_ODDS_API_KEY or ODDS_API_KEY to use The Odds API")

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    resp = requests.get(ODDS_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json() or []

    fetched_at_utc = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    schedule_lookup = _load_schedule_lookup()

    snapshots = []
    for event in data:
        row = _extract_event_snapshot(event, schedule_lookup, fetched_at_utc)
        if row is not None:
            snapshots.append(row)

    if not snapshots:
        raise RuntimeError("No se pudieron normalizar snapshots de The Odds API")

    snap_df = pd.DataFrame(snapshots)

    if LINE_MOVEMENT_HISTORY.exists():
        hist_df = pd.read_csv(LINE_MOVEMENT_HISTORY, dtype={"game_id": str, "external_game_id": str}, low_memory=False)
        hist_df = pd.concat([hist_df, snap_df], ignore_index=True)
    else:
        hist_df = snap_df.copy()

    hist_df["snapshot_time_utc"] = hist_df["snapshot_time_utc"].astype(str)
    hist_df = hist_df.drop_duplicates(
        subset=["snapshot_time_utc", "external_game_id", "game_id", "home_team", "away_team"],
        keep="last",
    ).sort_values(["date", "home_team", "away_team", "snapshot_time_utc"])
    hist_df.to_csv(LINE_MOVEMENT_HISTORY, index=False)

    agg_rows = []
    group_cols = ["external_game_id", "game_id", "date", "home_team", "away_team", "home_team_name", "away_team_name"]
    for keys, group in hist_df.groupby(group_cols, dropna=False):
        group = group.sort_values("snapshot_time_utc")
        first = group.iloc[0]
        last = group.iloc[-1]

        open_line = _safe_float(first.get("home_moneyline"))
        current_line = _safe_float(last.get("home_moneyline"))
        open_total = _safe_float(first.get("total_line"))
        current_total = _safe_float(last.get("total_line"))

        agg_rows.append(
            {
                "external_game_id": keys[0],
                "game_id": keys[1],
                "date": keys[2],
                "home_team": keys[3],
                "away_team": keys[4],
                "home_team_name": keys[5],
                "away_team_name": keys[6],
                "snapshot_count": int(len(group)),
                "first_snapshot_utc": str(first.get("snapshot_time_utc") or ""),
                "last_snapshot_utc": str(last.get("snapshot_time_utc") or ""),
                "open_line": open_line,
                "current_line": current_line,
                "line_movement": (
                    None if open_line is None or current_line is None else float(current_line - open_line)
                ),
                "open_total": open_total,
                "current_total": current_total,
                "total_movement": (
                    None if open_total is None or current_total is None else float(current_total - open_total)
                ),
                "current_home_moneyline": _safe_float(last.get("home_moneyline")),
                "current_away_moneyline": _safe_float(last.get("away_moneyline")),
                "current_home_spread": _safe_float(last.get("home_spread")),
                "current_total_line": _safe_float(last.get("total_line")),
                "bookmakers_count": int(_safe_float(last.get("bookmakers_count")) or 0),
                "market_source": "theoddsapi_current",
            }
        )

    out_df = pd.DataFrame(agg_rows)
    out_df = out_df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    out_df.to_csv(save_path, index=False)

    matched = int((out_df["game_id"].astype(str).str.strip() != "").sum()) if not out_df.empty else 0
    print(f"Line movement snapshots saved to: {LINE_MOVEMENT_HISTORY}")
    print(f"Line movement aggregate saved to: {save_path}")
    print(f"Eventos normalizados: {len(out_df)} | Con game_id ESPN: {matched}")
    return save_path


def load_line_movement(path: Path = LINE_MOVEMENT_CACHE) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Line movement cache not found: {path}")
    return pd.read_csv(path, low_memory=False)


def main():
    try:
        fetch_the_odds_api()
    except Exception as e:
        print("Failed to fetch from The Odds API:", e)
        print("Current free-plan flow only supports live/current odds snapshots.")
        print("Line movement cache path:", LINE_MOVEMENT_CACHE)


if __name__ == "__main__":
    main()
