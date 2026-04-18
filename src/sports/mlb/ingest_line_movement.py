from __future__ import annotations

import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple

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
HISTORICAL_DAILY_CACHE_DIR = CACHE_DIR / "theodds_historical_daily"
HISTORICAL_DAILY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
HISTORICAL_BACKFILL_PROGRESS_FILE = HISTORICAL_DAILY_CACHE_DIR / "_backfill_progress.json"
RAW_ADVANCED_FILE = BASE_DIR / "data" / "mlb" / "raw" / "mlb_advanced_history.csv"

SPORT_KEY = "baseball_mlb"
ODDS_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"
ODDS_HISTORICAL_URL = f"https://api.the-odds-api.com/v4/historical/sports/{SPORT_KEY}/odds"

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


def _normalize_date_str(value) -> str:
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return ""
    return dt.strftime("%Y-%m-%d")


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


def _build_expected_upcoming_games(days_ahead: int = 2) -> Set[Tuple[str, str, str]]:
    if not UPCOMING_FILE.exists():
        return set()
    try:
        df = pd.read_csv(UPCOMING_FILE, low_memory=False)
    except Exception:
        return set()

    required = {"date", "home_team", "away_team"}
    if not required.issubset(set(df.columns)):
        return set()

    df = df[["date", "home_team", "away_team"]].copy()
    df["date"] = df["date"].apply(_normalize_date_str)
    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()
    df = df[(df["date"] != "") & (df["home_team"] != "") & (df["away_team"] != "")]

    today = pd.Timestamp.now().normalize()
    max_day = today + pd.Timedelta(days=max(int(days_ahead), 0))
    dt = pd.to_datetime(df["date"], errors="coerce")
    window = df[(dt >= today) & (dt <= max_day)]

    return {
        (row["date"], row["home_team"], row["away_team"])
        for _, row in window.drop_duplicates().iterrows()
    }


def _cache_is_fresh_for_upcoming(
    cache_path: Path,
    ttl_minutes: int = 120,
    days_ahead: int = 2,
    min_coverage: float = 1.0,
) -> Tuple[bool, str]:
    path = Path(cache_path)
    if not path.exists():
        return False, "cache_missing"

    required_cols = {
        "date",
        "home_team",
        "away_team",
        "last_snapshot_utc",
        "current_home_moneyline",
        "current_away_moneyline",
    }
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return False, f"cache_read_error:{exc}"

    if df.empty:
        return False, "cache_empty"
    if not required_cols.issubset(set(df.columns)):
        return False, "cache_missing_columns"

    now_utc = pd.Timestamp.now("UTC")
    cutoff = now_utc - pd.Timedelta(minutes=max(int(ttl_minutes), 1))

    work = df[list(required_cols)].copy()
    work["date"] = work["date"].apply(_normalize_date_str)
    work["home_team"] = work["home_team"].astype(str).str.strip()
    work["away_team"] = work["away_team"].astype(str).str.strip()
    work["last_snapshot_utc"] = pd.to_datetime(work["last_snapshot_utc"], utc=True, errors="coerce")
    work["has_odds"] = work["current_home_moneyline"].notna() & work["current_away_moneyline"].notna()
    work["is_fresh"] = work["last_snapshot_utc"] >= cutoff
    work["usable"] = work["has_odds"] & work["is_fresh"]

    expected = _build_expected_upcoming_games(days_ahead=days_ahead)
    if not expected:
        usable_any = bool(work["usable"].any())
        return usable_any, "no_upcoming_schedule_rows" if usable_any else "no_upcoming_and_no_fresh_rows"

    covered = 0
    for key in expected:
        date_key, home_key, away_key = key
        mask = (
            (work["date"] == date_key)
            & (work["home_team"] == home_key)
            & (work["away_team"] == away_key)
            & work["usable"]
        )
        if bool(mask.any()):
            covered += 1

    coverage = covered / max(len(expected), 1)
    ok = coverage >= float(min_coverage)
    reason = f"coverage={covered}/{len(expected)} ({coverage:.1%}) ttl={ttl_minutes}m"
    return ok, reason


def _resolve_api_key() -> str:
    api_key = (
        os.environ.get("THE_ODDS_API_KEY")
        or os.environ.get("ODDS_API_KEY")
        or os.environ.get("THEODDSAPI_KEY")
    )
    if not api_key:
        raise EnvironmentError("Set THE_ODDS_API_KEY, THEODDSAPI_KEY, or ODDS_API_KEY to use The Odds API")
    return api_key


def _resolve_regions() -> str:
    raw = str(os.environ.get("THEODDS_MLB_REGIONS", "us") or "us").strip().lower()
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        tokens = ["us"]
    return ",".join(dict.fromkeys(tokens))


def _resolve_markets() -> str:
    raw = str(os.environ.get("THEODDS_MLB_MARKETS", "h2h,spreads,totals") or "h2h,spreads,totals").strip().lower()
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        tokens = ["h2h", "spreads", "totals"]
    return ",".join(dict.fromkeys(tokens))


def _count_csv_tokens(value: str) -> int:
    return len([token for token in str(value or "").split(",") if token.strip()])


def _estimate_credit_cost(markets: str, regions: str, historical: bool) -> int:
    unit = 10 if historical else 1
    return unit * max(_count_csv_tokens(markets), 1) * max(_count_csv_tokens(regions), 1)


def _safe_int(value) -> Optional[int]:
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f"{path.name}.tmp.{os.getpid()}.{pd.Timestamp.now('UTC').value}"
    try:
        with open(temp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


def _load_backfill_progress(path: Path = HISTORICAL_BACKFILL_PROGRESS_FILE) -> Dict[str, dict]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _write_backfill_progress(progress: Dict[str, dict], path: Path = HISTORICAL_BACKFILL_PROGRESS_FILE) -> None:
    _atomic_write_json(path, progress)


def _load_cached_payload(day_cache_file: Path, allow_redownload_on_cache_error: bool) -> Optional[Any]:
    if not day_cache_file.exists():
        return None
    try:
        with open(day_cache_file, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        if not allow_redownload_on_cache_error:
            raise RuntimeError(
                f"Cache corrupto para {day_cache_file.name}. Se detiene para proteger creditos. "
                "Si deseas re-descargar ese dia, exporta THEODDS_MLB_ALLOW_REDOWNLOAD_ON_CACHE_ERROR=1"
            ) from exc

        stamp = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%SZ")
        corrupt_path = day_cache_file.with_name(f"{day_cache_file.stem}.corrupt.{stamp}.json")
        try:
            os.replace(day_cache_file, corrupt_path)
            print(f"WARNING: Cache corrupto movido a {corrupt_path.name}; se intentara re-descarga de ese dia.")
        except Exception:
            print(f"WARNING: Cache corrupto detectado en {day_cache_file.name}; se intentara re-descarga de ese dia.")
        return None


def _extract_events_payload(payload) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return data
    return []


def _load_missing_dates_from_raw(
    max_dates: int = 30,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> list[str]:
    if not RAW_ADVANCED_FILE.exists():
        return []
    try:
        df = pd.read_csv(RAW_ADVANCED_FILE, low_memory=False)
    except Exception:
        return []

    required = {"date", "home_moneyline_odds", "away_moneyline_odds"}
    if not {"date"}.issubset(set(df.columns)):
        return []
    for col in ["home_moneyline_odds", "away_moneyline_odds"]:
        if col not in df.columns:
            df[col] = pd.NA

    work = df[["date", "home_moneyline_odds", "away_moneyline_odds"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work[work["date"].notna()]

    today = pd.Timestamp.now().normalize()
    work = work[work["date"] <= today]
    if start_date:
        start_dt = pd.to_datetime(start_date, errors="coerce")
        if pd.notna(start_dt):
            work = work[work["date"] >= start_dt.normalize()]
    if end_date:
        end_dt = pd.to_datetime(end_date, errors="coerce")
        if pd.notna(end_dt):
            work = work[work["date"] <= end_dt.normalize()]

    missing_mask = work["home_moneyline_odds"].isna() | work["away_moneyline_odds"].isna()
    missing_dates = sorted(
        work.loc[missing_mask, "date"].dt.strftime("%Y-%m-%d").dropna().unique().tolist(),
        reverse=True,
    )
    return missing_dates[: max(int(max_dates), 1)]


def _write_snapshots_and_aggregate(snap_df: pd.DataFrame, save_path: Path = LINE_MOVEMENT_CACHE) -> Path:
    if snap_df.empty:
        raise RuntimeError("No se pudieron normalizar snapshots de The Odds API")

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

    out_df = pd.DataFrame(agg_rows).sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    out_df.to_csv(save_path, index=False)

    matched = int((out_df["game_id"].astype(str).str.strip() != "").sum()) if not out_df.empty else 0
    print(f"Line movement snapshots saved to: {LINE_MOVEMENT_HISTORY}")
    print(f"Line movement aggregate saved to: {save_path}")
    print(f"Eventos normalizados: {len(out_df)} | Con game_id ESPN: {matched}")
    return save_path


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
    ttl_minutes = int(os.environ.get("THEODDS_MLB_CACHE_TTL_MINUTES", "120"))
    days_ahead = int(os.environ.get("THEODDS_MLB_CHECK_DAYS_AHEAD", "2"))
    min_coverage = float(os.environ.get("THEODDS_MLB_MIN_COVERAGE", "1.0"))
    force_refresh = str(os.environ.get("THEODDS_MLB_FORCE_REFRESH", "0")).strip().lower() in {"1", "true", "yes", "y"}

    if not force_refresh:
        cache_ok, cache_reason = _cache_is_fresh_for_upcoming(
            cache_path=save_path,
            ttl_minutes=ttl_minutes,
            days_ahead=days_ahead,
            min_coverage=min_coverage,
        )
        if cache_ok:
            print(f"Skipping TheOddsAPI request (cache fresh): {cache_reason}")
            print(f"Using existing cache: {save_path}")
            return save_path
        print(f"Cache refresh required: {cache_reason}")

    api_key = _resolve_api_key()
    regions = _resolve_regions()
    markets = _resolve_markets()
    est_cost = _estimate_credit_cost(markets=markets, regions=regions, historical=False)

    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    resp = requests.get(ODDS_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    x_last = _safe_int(resp.headers.get("x-requests-last"))
    if x_last is not None:
        print(f"TheOdds current fetch credits: last_call={x_last} | configured_max={est_cost} | markets={markets} | regions={regions}")
    data = _extract_events_payload(resp.json() or [])

    fetched_at_utc = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    schedule_lookup = _load_schedule_lookup()

    snapshots = []
    for event in data:
        row = _extract_event_snapshot(event, schedule_lookup, fetched_at_utc)
        if row is not None:
            snapshots.append(row)

    return _write_snapshots_and_aggregate(pd.DataFrame(snapshots), save_path=save_path)


def backfill_historical_theodds(
    save_path: Path = LINE_MOVEMENT_CACHE,
    timeout: int = 30,
    max_dates: int = 30,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Path:
    api_key = _resolve_api_key()
    regions = _resolve_regions()
    markets = _resolve_markets()
    allow_redownload_on_cache_error = str(
        os.environ.get("THEODDS_MLB_ALLOW_REDOWNLOAD_ON_CACHE_ERROR", "0")
    ).strip().lower() in {"1", "true", "yes", "y"}
    max_credits_per_run = _safe_int(os.environ.get("THEODDS_MLB_MAX_CREDITS_PER_RUN", "").strip())
    est_cost_per_download = _estimate_credit_cost(markets=markets, regions=regions, historical=True)

    missing_dates = _load_missing_dates_from_raw(max_dates=max_dates, start_date=start_date, end_date=end_date)
    if not missing_dates:
        print("No hay fechas faltantes en raw para backfill histórico.")
        return save_path

    print(
        "Backfill historical config "
        f"markets={markets} regions={regions} "
        f"estimated_max_credits_per_download={est_cost_per_download} "
        f"dates_to_process={len(missing_dates)}"
    )
    if max_credits_per_run is not None and max_credits_per_run > 0:
        print(f"Backfill credit cap for this run: {max_credits_per_run}")

    schedule_lookup = _load_schedule_lookup()
    progress = _load_backfill_progress()
    collected = []
    downloaded_days = 0
    cache_hits = 0
    credits_spent_this_run = 0
    stopped_for_budget = False

    for date_str in missing_dates:
        day_cache_file = HISTORICAL_DAILY_CACHE_DIR / f"odds_{date_str}.json"
        payload = _load_cached_payload(
            day_cache_file=day_cache_file,
            allow_redownload_on_cache_error=allow_redownload_on_cache_error,
        )
        from_cache = payload is not None
        if from_cache:
            cache_hits += 1

        if payload is None:
            if max_credits_per_run is not None and max_credits_per_run > 0:
                projected = credits_spent_this_run + est_cost_per_download
                if projected > max_credits_per_run:
                    print(
                        "Stopping historical backfill before next API call to respect credit cap: "
                        f"spent={credits_spent_this_run} projected_next={projected} cap={max_credits_per_run}"
                    )
                    stopped_for_budget = True
                    break

            params = {
                "apiKey": api_key,
                "regions": regions,
                "markets": markets,
                "oddsFormat": "american",
                "dateFormat": "iso",
                "date": f"{date_str}T12:00:00Z",
            }
            resp = requests.get(ODDS_HISTORICAL_URL, params=params, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json() or {}
            _atomic_write_json(day_cache_file, payload)
            downloaded_days += 1

            last_cost = _safe_int(resp.headers.get("x-requests-last"))
            if last_cost is None:
                last_cost = est_cost_per_download
            credits_spent_this_run += max(last_cost, 0)

            remaining = _safe_int(resp.headers.get("x-requests-remaining"))
            if remaining is not None:
                print(
                    f"Historical call date={date_str} credits_last={last_cost} "
                    f"credits_remaining={remaining}"
                )

        events = _extract_events_payload(payload)
        normalized_rows = 0
        if not events:
            progress[date_str] = {
                "updated_at_utc": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source": "cache" if from_cache else "download",
                "events_in_payload": 0,
                "normalized_rows": 0,
                "cache_file": day_cache_file.name,
                "credits_spent_this_run": credits_spent_this_run,
            }
            _write_backfill_progress(progress)
            continue

        fetched_at_utc = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        for event in events:
            row = _extract_event_snapshot(event, schedule_lookup, fetched_at_utc)
            if row is not None:
                collected.append(row)
                normalized_rows += 1

        progress[date_str] = {
            "updated_at_utc": fetched_at_utc,
            "source": "cache" if from_cache else "download",
            "events_in_payload": int(len(events)),
            "normalized_rows": int(normalized_rows),
            "cache_file": day_cache_file.name,
            "credits_spent_this_run": credits_spent_this_run,
        }
        _write_backfill_progress(progress)

    print(
        f"Backfill histórico: fechas={len(missing_dates)} | descargas={downloaded_days} | "
        f"cache_dias={cache_hits} | creditos_run={credits_spent_this_run}"
    )
    if stopped_for_budget:
        print("Backfill histórico detenido por limite de creditos configurado.")
    print(f"Backfill progress file: {HISTORICAL_BACKFILL_PROGRESS_FILE}")
    if not collected:
        print("Backfill histórico sin snapshots normalizables para las fechas solicitadas.")
        return save_path
    return _write_snapshots_and_aggregate(pd.DataFrame(collected), save_path=save_path)


def load_line_movement(path: Path = LINE_MOVEMENT_CACHE) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Line movement cache not found: {path}")
    return pd.read_csv(path, low_memory=False)


def main():
    try:
        historical_mode = str(os.environ.get("THEODDS_MLB_HISTORICAL_BACKFILL", "0")).strip().lower() in {"1", "true", "yes", "y"}
        if historical_mode:
            max_dates = int(os.environ.get("THEODDS_MLB_HIST_MAX_DATES", "30"))
            start_date = os.environ.get("THEODDS_MLB_HIST_START_DATE")
            end_date = os.environ.get("THEODDS_MLB_HIST_END_DATE")
            backfill_historical_theodds(
                max_dates=max_dates,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            fetch_the_odds_api()
    except Exception as e:
        print("Failed to fetch from The Odds API:", e)
        print("Verifica API key, plan/permisos de endpoint histórico y límites de crédito.")
        print("Line movement cache path:", LINE_MOVEMENT_CACHE)


if __name__ == "__main__":
    main()
