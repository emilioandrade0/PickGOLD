import json
import re
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

import sys
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
RAW_DATA_DIR = BASE_DIR / "data" / "ncaa_baseball" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

FILE_PATH_ADVANCED = RAW_DATA_DIR / "ncaa_baseball_advanced_history.csv"
FILE_PATH_UPCOMING = RAW_DATA_DIR / "ncaa_baseball_upcoming_schedule.csv"

INITIAL_BACKFILL_DAYS = 21
INCREMENTAL_BACKFILL_DAYS = 2
UPCOMING_DAYS_AHEAD = 14
DETAIL_ENRICH_DAYS = 7

DEFAULT_CONTESTS_DATA_URL = (
    "https://sdataprod.ncaa.com?meta=GetContests_web&extensions="
    "{\"persistedQuery\":{\"version\":1,\"sha256Hash\":\"6b26e5cda954c1302873c52835bfd223e169e2068b12511e92b3ef29fac779c2\"}}"
)
SCORE_SUMMARY_META = "NCAA_GetGamecenterScoringSummaryById_web"
SCORE_SUMMARY_SHA = "7f86673d4875cd18102b7fa598e2bc5da3f49d05a1c15b1add0e2367ee890198"
BOXSCORE_META = "NCAA_GetGamecenterBoxscoreBaseballById_web"
BOXSCORE_SHA = "5e92118b2f424040aa96067aba6d34e882165aaf02e9e73cb9d69317066c6ae8"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.ncaa.com/",
}


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def _normalize_team_code(team: dict) -> str:
    for key in ["name6Char", "nameShort", "seoname", "teamId"]:
        raw = str(team.get(key, "") or "").strip()
        if raw:
            code = "".join(ch for ch in raw.upper() if ch.isalnum())
            if code:
                return code[:8]
    return "NCAA"


def _parse_drupal_scoreboard_settings(target_date: datetime):
    date_path = target_date.strftime("%Y/%m/%d")
    urls = [
        f"https://www.ncaa.com/scoreboard/baseball/d1/{date_path}/all-conf",
        "https://www.ncaa.com/scoreboard/baseball/d1",
    ]

    for url in urls:
        try:
            html = requests.get(url, headers=HEADERS, timeout=30).text
            m = re.search(
                r'<script[^>]*data-drupal-selector="drupal-settings-json"[^>]*>(.*?)</script>',
                html,
                re.S,
            )
            if not m:
                continue
            settings = json.loads(m.group(1))
            scoreboard = settings.get("scoreboard", {})
            contests_url = scoreboard.get("contestsDataUrl")
            if contests_url:
                return {
                    "contests_data_url": contests_url,
                    "sport_code": scoreboard.get("sportCode", "MBA"),
                    "division": str(scoreboard.get("division", "d1")),
                    "season_year": _safe_int(scoreboard.get("seasonYear"), 0),
                }
        except Exception:
            continue

    return {
        "contests_data_url": DEFAULT_CONTESTS_DATA_URL,
        "sport_code": "MBA",
        "division": "d1",
        "season_year": 0,
    }


def _division_to_int(division: str) -> int:
    mapping = {"nc": 0, "d1": 1, "d2": 2, "d3": 3, "d4": 4, "fbs": 11, "fcs": 12}
    return mapping.get(str(division).lower(), 1)


def _infer_season_year(game_date: datetime, default_season_year: int) -> int:
    if default_season_year > 0:
        return default_season_year
    return game_date.year if game_date.month >= 8 else game_date.year - 1


def _build_query_url(base_url: str, query_name: str, variables: dict) -> str:
    query = urlencode(
        {
            "queryName": query_name,
            "variables": json.dumps(variables, separators=(",", ":")),
        }
    )
    sep = "&" if "?" in base_url else "?"
    return f"{base_url}{sep}{query}"


def _fetch_contests_for_date(cfg: dict, game_date: datetime):
    variables = {
        "sportCode": cfg["sport_code"],
        "division": _division_to_int(cfg["division"]),
        "seasonYear": _infer_season_year(game_date, cfg["season_year"]),
        "month": game_date.month,
        "contestDate": game_date.strftime("%m/%d/%Y"),
        "week": None,
    }

    url = _build_query_url(cfg["contests_data_url"], "GetContests_web", variables)
    resp = requests.get(url, headers=HEADERS, timeout=40)
    resp.raise_for_status()
    payload = resp.json() or {}
    return payload.get("data", {}).get("contests", [])


def _fetch_stats_ncaa_org_boxscore(contest_id: int):
    # Controlled attempt to use stats.ncaa.org first; many environments return 403.
    try:
        url = f"https://stats.ncaa.org/contests/{contest_id}/box_score"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        tables = pd.read_html(resp.text)
        if not tables:
            return None
        return {"tables": len(tables)}
    except Exception:
        return None


def _fetch_gql(meta: str, sha: str, contest_id: int):
    base = (
        f"https://sdataprod.ncaa.com?meta={meta}&extensions="
        f"{{\"persistedQuery\":{{\"version\":1,\"sha256Hash\":\"{sha}\"}}}}"
    )
    url = _build_query_url(base, meta, {"contestId": int(contest_id)})
    resp = requests.get(url, headers=HEADERS, timeout=25)
    if resp.status_code != 200:
        return None
    try:
        return resp.json()
    except Exception:
        return None


def _extract_hits_from_boxscore(box_payload: dict):
    box = (box_payload or {}).get("data", {}).get("boxscore", {})
    teams = box.get("teams") or []
    team_box = box.get("teamBoxscore") or []

    team_id_to_side = {}
    for t in teams:
        tid = str(t.get("teamId", ""))
        if not tid:
            continue
        team_id_to_side[tid] = "home" if bool(t.get("isHome")) else "away"

    hits = {"home": 0, "away": 0}
    for tb in team_box:
        tid = str(tb.get("teamId", ""))
        side = team_id_to_side.get(tid)
        if not side:
            continue
        batter_totals = (tb.get("teamStats") or {}).get("batterTotals") or {}
        hits[side] = _safe_int(batter_totals.get("hits"), 0)

    return hits["home"], hits["away"]


def _inning_number(title: str):
    m = re.search(r"(\d+)", str(title or ""))
    if not m:
        return None
    return _safe_int(m.group(1), None)


def _extract_runs_by_inning(summary_payload: dict):
    periods = (summary_payload or {}).get("data", {}).get("scoringSummary", {}).get("periods", []) or []

    inning_runs = {}
    prev_home = 0
    prev_away = 0

    for period in periods:
        inn = _inning_number(period.get("title"))
        if inn is None:
            continue

        events = period.get("summary") or []
        if events:
            last = events[-1]
            cur_home = _safe_int(last.get("homeScore"), prev_home)
            cur_away = _safe_int(last.get("visitScore"), prev_away)
        else:
            cur_home = prev_home
            cur_away = prev_away

        inning_runs[inn] = (
            max(cur_home - prev_home, 0),
            max(cur_away - prev_away, 0),
        )

        prev_home = cur_home
        prev_away = cur_away

    return inning_runs


def _enrich_completed_contest(contest_id: int):
    _ = _fetch_stats_ncaa_org_boxscore(contest_id)
    box_payload = _fetch_gql(BOXSCORE_META, BOXSCORE_SHA, contest_id)
    summary_payload = _fetch_gql(SCORE_SUMMARY_META, SCORE_SUMMARY_SHA, contest_id)

    home_hits, away_hits = _extract_hits_from_boxscore(box_payload or {})
    innings = _extract_runs_by_inning(summary_payload or {})

    def _inn(side: str, n: int):
        pair = innings.get(n, (0, 0))
        return pair[0] if side == "home" else pair[1]

    home_r1 = _inn("home", 1)
    away_r1 = _inn("away", 1)
    home_r2 = _inn("home", 2)
    away_r2 = _inn("away", 2)
    home_r3 = _inn("home", 3)
    away_r3 = _inn("away", 3)
    home_r4 = _inn("home", 4)
    away_r4 = _inn("away", 4)
    home_r5 = _inn("home", 5)
    away_r5 = _inn("away", 5)

    return {
        "home_hits": home_hits,
        "away_hits": away_hits,
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
        "home_runs_f5": home_r1 + home_r2 + home_r3 + home_r4 + home_r5,
        "away_runs_f5": away_r1 + away_r2 + away_r3 + away_r4 + away_r5,
    }


def _parse_contest(contest: dict, enrich: bool):
    teams = contest.get("teams") or []
    if len(teams) < 2:
        return None

    home = next((t for t in teams if bool(t.get("isHome"))), None)
    away = next((t for t in teams if not bool(t.get("isHome"))), None)
    if not home or not away:
        return None

    contest_id = _safe_int(contest.get("contestId"), 0)
    if contest_id <= 0:
        return None

    date_text = str(contest.get("startDate") or "")
    try:
        game_dt = datetime.strptime(date_text, "%m/%d/%Y")
    except Exception:
        return None

    game_date = game_dt.strftime("%Y-%m-%d")
    start_time = str(contest.get("startTime") or "").strip()

    state = str(contest.get("gameState") or "").upper()
    status_display = str(contest.get("statusCodeDisplay") or "").lower()
    completed = state == "F" or status_display in {"final", "post"}

    home_score = _safe_int(home.get("score"), 0)
    away_score = _safe_int(away.get("score"), 0)

    row = {
        "game_id": str(contest_id),
        "date": game_date,
        "time": start_time,
        "season": str(_infer_season_year(game_dt, 0)),
        "home_team": _normalize_team_code(home),
        "away_team": _normalize_team_code(away),
        "home_runs_total": home_score,
        "away_runs_total": away_score,
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
        "odds_details": "No Line",
        "odds_over_under": 0.0,
        "home_is_favorite": -1,
        "home_hits": 0,
        "away_hits": 0,
        "status_completed": int(completed),
        "status_state": "post" if completed else ("in" if state == "I" else "pre"),
        "status_description": "Final" if completed else ("In Progress" if state == "I" else "Scheduled"),
        "status_detail": str(contest.get("finalMessage") or "").strip(),
    }

    if completed and enrich:
        try:
            extra = _enrich_completed_contest(contest_id)
            row.update(extra)
        except Exception:
            pass

    return row


def _load_existing(path: Path):
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, dtype={"game_id": str})
        if "date" in df.columns:
            df["date"] = df["date"].astype(str)
        return df
    except Exception:
        return pd.DataFrame()


def _daterange(start_dt: datetime, end_dt: datetime):
    cur = start_dt
    while cur <= end_dt:
        yield cur
        cur += timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(description="Ingest NCAA Baseball")
    parser.add_argument(
        "--enrich-details",
        action="store_true",
        help="Activa enriquecimiento game-level (mas lento) via endpoints internos",
    )
    args = parser.parse_args()

    print("⚾ Ingest NCAA Baseball (ncaa.com + sdataprod)...")

    today = datetime.now().date()
    cfg = _parse_drupal_scoreboard_settings(datetime.now())

    existing = _load_existing(FILE_PATH_ADVANCED)
    if existing.empty:
        start_hist = today - timedelta(days=INITIAL_BACKFILL_DAYS)
        print(f"   > Inicial: {start_hist} -> {today}")
    else:
        existing_dates = pd.to_datetime(existing["date"], errors="coerce").dropna()
        max_date = existing_dates.max().date() if not existing_dates.empty else (today - timedelta(days=1))
        start_hist = max_date - timedelta(days=INCREMENTAL_BACKFILL_DAYS)
        print(f"   > Incremental desde: {start_hist} (max previo: {max_date})")

    fetched_completed = []
    detail_cutoff = today - timedelta(days=DETAIL_ENRICH_DAYS)

    total_days = (today - start_hist).days + 1
    day_index = 0

    for dt in _daterange(datetime.combine(start_hist, datetime.min.time()), datetime.combine(today, datetime.min.time())):
        day_index += 1
        try:
            contests = _fetch_contests_for_date(cfg, dt)
        except Exception as ex:
            print(f"⚠️ {dt.date()} error contests: {ex}")
            continue

        completed_today = 0

        for c in contests:
            state = str(c.get("gameState") or "").upper()
            status_display = str(c.get("statusCodeDisplay") or "").lower()
            completed = state == "F" or status_display in {"final", "post"}
            if not completed:
                continue

            enrich = args.enrich_details and (dt.date() >= detail_cutoff)
            row = _parse_contest(c, enrich=enrich)
            if row:
                fetched_completed.append(row)
                completed_today += 1

        print(
            f"   > {dt.date()} [{day_index}/{total_days}] contests={len(contests)} completed_added={completed_today}"
        )

    all_completed = pd.concat([existing, pd.DataFrame(fetched_completed)], ignore_index=True)
    if not all_completed.empty:
        all_completed = all_completed.drop_duplicates(subset=["game_id"], keep="last")
        all_completed = all_completed.sort_values(["date", "game_id"]).reset_index(drop=True)
        all_completed.to_csv(FILE_PATH_ADVANCED, index=False)
        print(f"✅ Histórico NCAA Baseball: {FILE_PATH_ADVANCED} ({len(all_completed)} juegos)")
    else:
        print("⚠️ Sin juegos completados NCAA Baseball.")

    upcoming_rows = []
    end_upcoming = today + timedelta(days=UPCOMING_DAYS_AHEAD)
    upcoming_days = (end_upcoming - today).days + 1
    upcoming_idx = 0
    for dt in _daterange(datetime.combine(today, datetime.min.time()), datetime.combine(end_upcoming, datetime.min.time())):
        upcoming_idx += 1
        try:
            contests = _fetch_contests_for_date(cfg, dt)
        except Exception:
            continue

        for c in contests:
            row = _parse_contest(c, enrich=False)
            if row:
                upcoming_rows.append(row)

        print(f"   > upcoming {dt.date()} [{upcoming_idx}/{upcoming_days}] contests={len(contests)}")

    if upcoming_rows:
        up = pd.DataFrame(upcoming_rows).drop_duplicates(subset=["game_id"], keep="last")
        up = up.sort_values(["date", "time", "game_id"]).reset_index(drop=True)
        up.to_csv(FILE_PATH_UPCOMING, index=False)
        print(f"✅ Agenda NCAA Baseball: {FILE_PATH_UPCOMING} ({len(up)} juegos)")
    else:
        print("⚠️ Sin agenda NCAA Baseball en ventana upcoming.")


if __name__ == "__main__":
    main()
