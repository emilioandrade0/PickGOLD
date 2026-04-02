from __future__ import annotations

from datetime import datetime
from io import StringIO
from pathlib import Path
import hashlib
import os
import re
import sys

import numpy as np
import pandas as pd
import requests


SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
RAW_DATA_DIR = BASE_DIR / "data" / "tennis" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_HISTORY_FILE = RAW_DATA_DIR / "tennis_advanced_history.csv"
RAW_UPCOMING_FILE = RAW_DATA_DIR / "tennis_upcoming_schedule.csv"

API_KEY = os.getenv("THE_ODDS_API_KEY", "2c7887cad91583f20215b7d590c617e4").strip()
SPORTS_URL = "https://api.the-odds-api.com/v4/sports/"
SPORT_URL_TEMPLATE = "https://api.the-odds-api.com/v4/sports/{sport_key}/{resource}/"
SACKMANN_RAW_TEMPLATE = "https://raw.githubusercontent.com/JeffSackmann/{repo}/master/{prefix}_matches_{year}.csv"

HISTORY_COLUMNS = [
    "match_id",
    "date",
    "time",
    "season",
    "tour",
    "tournament",
    "surface",
    "round",
    "player_a",
    "player_b",
    "winner",
    "player_a_sets",
    "player_b_sets",
    "player_a_games",
    "player_b_games",
    "player_a_rank",
    "player_b_rank",
    "player_a_odds",
    "player_b_odds",
    "status_completed",
    "status_state",
    "status_description",
    "status_detail",
]

UPCOMING_COLUMNS = [
    "match_id",
    "date",
    "time",
    "season",
    "tour",
    "tournament",
    "surface",
    "round",
    "player_a",
    "player_b",
    "player_a_rank",
    "player_b_rank",
    "player_a_odds",
    "player_b_odds",
    "home_team",
    "away_team",
    "home_moneyline_odds",
    "away_moneyline_odds",
    "status_completed",
    "status_state",
    "status_description",
    "status_detail",
]


def _request_json(url: str, params: dict | None = None) -> list[dict]:
    response = requests.get(
        url,
        params=params or {},
        timeout=30,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; PickGOLD/1.0)",
            "Accept": "application/json",
        },
    )
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, list) else []


def _season_code(dt: datetime) -> str:
    return str(dt.year)


def _clean_text(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        repaired = text.encode("latin-1").decode("utf-8")
    except Exception:
        repaired = text
    return repaired.strip()


def _tour_and_tournament(title: str) -> tuple[str, str]:
    text = _clean_text(title)
    if text.upper().startswith("ATP "):
        return "ATP", text.replace("ATP ", "", 1).strip()
    if text.upper().startswith("WTA "):
        return "WTA", text.replace("WTA ", "", 1).strip()
    return "TENNIS", text


def _american_to_decimal(value) -> float | None:
    try:
        n = float(value)
    except Exception:
        return None
    if n == 0:
        return None
    if n > 0:
        return round((n / 100.0) + 1.0, 4)
    return round((100.0 / abs(n)) + 1.0, 4)


def _hash_prefers_player_a(match_id: str) -> bool:
    digest = hashlib.md5(match_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2 == 0


def _parse_score_sets(score_text: str) -> tuple[int | None, int | None]:
    text = _clean_text(score_text).upper()
    if not text or any(token in text for token in ["W/O", "RET", "DEF", "ABN", "UNP", "LIVE", "SUSP"]):
        return None, None
    winner_sets = 0
    loser_sets = 0
    for token in text.split():
        if not re.match(r"^\\d+-\\d+", token):
            continue
        left, right = token.split("-", 1)
        left = re.sub(r"\\D.*$", "", left)
        right = re.sub(r"\\D.*$", "", right)
        if not left.isdigit() or not right.isdigit():
            continue
        if int(left) > int(right):
            winner_sets += 1
        elif int(right) > int(left):
            loser_sets += 1
    if winner_sets == 0 and loser_sets == 0:
        return None, None
    return winner_sets, loser_sets


def _safe_float(value) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if np.isnan(out):
        return None
    return out


def _extract_consensus_odds(event: dict) -> tuple[float | None, float | None]:
    player_a = _clean_text(event.get("home_team"))
    player_b = _clean_text(event.get("away_team"))
    odds_a = []
    odds_b = []

    for bookmaker in event.get("bookmakers") or []:
        for market in bookmaker.get("markets") or []:
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes") or []:
                name = str(outcome.get("name") or "").strip()
                dec = _american_to_decimal(outcome.get("price"))
                if dec is None:
                    continue
                if name == player_a:
                    odds_a.append(dec)
                elif name == player_b:
                    odds_b.append(dec)

    def _median_or_none(values: list[float]) -> float | None:
        if not values:
            return None
        values = sorted(values)
        mid = len(values) // 2
        if len(values) % 2 == 1:
            return round(values[mid], 4)
        return round((values[mid - 1] + values[mid]) / 2.0, 4)

    return _median_or_none(odds_a), _median_or_none(odds_b)


def _fetch_sackmann_year(repo: str, prefix: str, year: int) -> pd.DataFrame:
    url = SACKMANN_RAW_TEMPLATE.format(repo=repo, prefix=prefix, year=year)
    response = requests.get(
        url,
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0 (compatible; PickGOLD/1.0)"},
    )
    if response.status_code == 404:
        return pd.DataFrame()
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def _rows_from_sackmann_frame(df: pd.DataFrame, tour: str) -> list[dict]:
    rows: list[dict] = []
    if df.empty:
        return rows

    for _, row in df.iterrows():
        winner = _clean_text(row.get("winner_name"))
        loser = _clean_text(row.get("loser_name"))
        if not winner or not loser:
            continue

        tourney_date = str(row.get("tourney_date") or "").strip()
        try:
            dt = datetime.strptime(tourney_date, "%Y%m%d")
            date_str = dt.strftime("%Y-%m-%d")
        except Exception:
            date_str = ""
            dt = datetime.now()

        match_num = str(row.get("match_num") or "").strip()
        tourney_id = _clean_text(row.get("tourney_id"))
        match_id = f"{tour}-{tourney_id}-{match_num}" if tourney_id or match_num else f"{tour}-{winner}-{loser}-{date_str}"

        prefer_winner_first = _hash_prefers_player_a(match_id)
        player_a = winner if prefer_winner_first else loser
        player_b = loser if prefer_winner_first else winner
        winner_sets, loser_sets = _parse_score_sets(row.get("score"))
        if prefer_winner_first:
            sets_a, sets_b = winner_sets, loser_sets
            rank_a = _safe_float(row.get("winner_rank"))
            rank_b = _safe_float(row.get("loser_rank"))
        else:
            sets_a, sets_b = loser_sets, winner_sets
            rank_a = _safe_float(row.get("loser_rank"))
            rank_b = _safe_float(row.get("winner_rank"))

        rows.append(
            {
                "match_id": match_id,
                "date": date_str,
                "time": "",
                "season": _season_code(dt),
                "tour": tour,
                "tournament": _clean_text(row.get("tourney_name")),
                "surface": _clean_text(row.get("surface")) or "UNKNOWN",
                "round": _clean_text(row.get("round")) or "MAIN",
                "player_a": player_a,
                "player_b": player_b,
                "winner": winner,
                "player_a_sets": sets_a,
                "player_b_sets": sets_b,
                "player_a_games": None,
                "player_b_games": None,
                "player_a_rank": rank_a,
                "player_b_rank": rank_b,
                "player_a_odds": None,
                "player_b_odds": None,
                "status_completed": 1,
                "status_state": "post",
                "status_description": "Final",
                "status_detail": "FINAL",
            }
        )
    return rows


def _fetch_sackmann_backfill() -> pd.DataFrame:
    current_year = datetime.now().year
    years = list(range(current_year - 3, current_year))
    all_rows: list[dict] = []
    sources = [("ATP", "tennis_atp", "atp"), ("WTA", "tennis_wta", "wta")]

    for tour, repo, prefix in sources:
        found_years = 0
        for year in reversed(years):
            try:
                frame = _fetch_sackmann_year(repo, prefix, year)
            except Exception:
                continue
            if frame.empty:
                continue
            found_years += 1
            all_rows.extend(_rows_from_sackmann_frame(frame, tour))
        print(f"  > Backfill {tour}: {found_years} temporadas cargadas")

    if not all_rows:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    out = pd.DataFrame(all_rows)
    out = out[HISTORY_COLUMNS].copy()
    out = out.drop_duplicates(subset=["match_id"], keep="last")
    out = out.sort_values(["date", "time", "match_id"], kind="stable")
    return out


def _winner_from_scores(event: dict) -> tuple[str, int | None, int | None]:
    scores = event.get("scores") or []
    score_map = {}
    for item in scores:
        try:
            score_map[_clean_text(item.get("name"))] = int(float(item.get("score") or 0))
        except Exception:
            continue

    player_a = _clean_text(event.get("home_team"))
    player_b = _clean_text(event.get("away_team"))
    sets_a = score_map.get(player_a)
    sets_b = score_map.get(player_b)

    winner = ""
    if sets_a is not None and sets_b is not None:
        if sets_a > sets_b:
            winner = player_a
        elif sets_b > sets_a:
            winner = player_b
    return winner, sets_a, sets_b


def _row_from_event(event: dict, odds_map: dict[str, tuple[float | None, float | None]]) -> dict:
    commence_dt = pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")
    date_str = commence_dt.strftime("%Y-%m-%d") if pd.notna(commence_dt) else ""
    time_str = commence_dt.strftime("%H:%M") if pd.notna(commence_dt) else ""
    tour, tournament = _tour_and_tournament(event.get("sport_title"))
    player_a = _clean_text(event.get("home_team"))
    player_b = _clean_text(event.get("away_team"))
    player_a_odds, player_b_odds = odds_map.get(str(event.get("id") or ""), (None, None))
    winner, sets_a, sets_b = _winner_from_scores(event)
    completed = bool(event.get("completed"))
    season_dt = commence_dt.to_pydatetime() if pd.notna(commence_dt) else datetime.now()

    return {
        "match_id": str(event.get("id") or ""),
        "date": date_str,
        "time": time_str,
        "season": _season_code(season_dt),
        "tour": tour,
        "tournament": tournament,
        "surface": "UNKNOWN",
        "round": "MAIN",
        "player_a": player_a,
        "player_b": player_b,
        "home_team": player_a,
        "away_team": player_b,
        "winner": winner,
        "player_a_sets": sets_a,
        "player_b_sets": sets_b,
        "player_a_games": None,
        "player_b_games": None,
        "player_a_rank": None,
        "player_b_rank": None,
        "player_a_odds": player_a_odds,
        "player_b_odds": player_b_odds,
        "home_moneyline_odds": player_a_odds,
        "away_moneyline_odds": player_b_odds,
        "status_completed": 1 if completed else 0,
        "status_state": "post" if completed else "pre",
        "status_description": "Final" if completed else "Scheduled",
        "status_detail": "FINAL" if completed else "Scheduled",
    }


def _fetch_active_tennis_sports() -> list[dict]:
    sports = _request_json(SPORTS_URL, params={"apiKey": API_KEY})
    out = []
    for item in sports:
        key = str(item.get("key") or "")
        if not key.startswith("tennis_"):
            continue
        if bool(item.get("has_outrights")):
            continue
        if not bool(item.get("active", False)):
            continue
        out.append(item)
    return out


def _fetch_tennis_events() -> list[dict]:
    all_rows = []
    active_sports = _fetch_active_tennis_sports()
    print(f"Tennis ingest: torneos activos detectados = {len(active_sports)}")

    for sport in active_sports:
        sport_key = str(sport.get("key") or "")
        sport_title = str(sport.get("title") or sport_key)
        print(f"  > {sport_title} ({sport_key})")

        odds_events = _request_json(
            SPORT_URL_TEMPLATE.format(sport_key=sport_key, resource="odds"),
            params={
                "apiKey": API_KEY,
                "regions": "us",
                "markets": "h2h",
                "oddsFormat": "american",
                "dateFormat": "iso",
            },
        )
        scores_events = _request_json(
            SPORT_URL_TEMPLATE.format(sport_key=sport_key, resource="scores"),
            params={
                "apiKey": API_KEY,
                "daysFrom": "3",
                "dateFormat": "iso",
            },
        )

        odds_by_id = {
            str(event.get("id") or ""): _extract_consensus_odds(event)
            for event in odds_events
        }

        merged_by_id = {}
        for event in scores_events:
            merged_by_id[str(event.get("id") or "")] = dict(event)
        for event in odds_events:
            event_id = str(event.get("id") or "")
            base = merged_by_id.get(event_id, {})
            base.update(event)
            merged_by_id[event_id] = base

        for event in merged_by_id.values():
            all_rows.append(_row_from_event(event, odds_by_id))

    return all_rows


def main() -> None:
    rows = _fetch_tennis_events()
    backfill_history = _fetch_sackmann_backfill()
    if not rows:
        if RAW_HISTORY_FILE.exists():
            try:
                existing = pd.read_csv(RAW_HISTORY_FILE, dtype={"match_id": str})
            except Exception:
                existing = pd.DataFrame(columns=HISTORY_COLUMNS)
            if not backfill_history.empty:
                merged = pd.concat([existing, backfill_history], ignore_index=True)
                merged = merged.drop_duplicates(subset=["match_id"], keep="last")
                merged = merged.sort_values(["date", "time", "match_id"], kind="stable")
                merged.to_csv(RAW_HISTORY_FILE, index=False)
        else:
            seed_history = backfill_history if not backfill_history.empty else pd.DataFrame(columns=HISTORY_COLUMNS)
            seed_history.to_csv(RAW_HISTORY_FILE, index=False)
        if not RAW_UPCOMING_FILE.exists():
            pd.DataFrame(columns=UPCOMING_COLUMNS).to_csv(RAW_UPCOMING_FILE, index=False)
        print("Tennis ingest: no se encontraron eventos ATP/WTA activos. Se conservan los archivos existentes.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["date", "time", "match_id"], kind="stable").drop_duplicates(subset=["match_id"], keep="last")

    history_df = df[df["status_completed"] == 1][HISTORY_COLUMNS].copy()
    if not backfill_history.empty:
        history_df = pd.concat([backfill_history, history_df], ignore_index=True)
    upcoming_df = df[df["status_completed"] != 1][UPCOMING_COLUMNS].copy()

    if RAW_HISTORY_FILE.exists():
        try:
            existing = pd.read_csv(RAW_HISTORY_FILE, dtype={"match_id": str})
            history_df = pd.concat([existing, history_df], ignore_index=True)
            history_df = history_df.drop_duplicates(subset=["match_id"], keep="last")
            history_df = history_df.sort_values(["date", "time", "match_id"], kind="stable")
        except Exception:
            pass

    history_df.to_csv(RAW_HISTORY_FILE, index=False)
    upcoming_df.to_csv(RAW_UPCOMING_FILE, index=False)

    print("Tennis ingest completado.")
    print(f"  Historico actualizado : {len(history_df)} filas -> {RAW_HISTORY_FILE}")
    print(f"  Upcoming actualizado  : {len(upcoming_df)} filas -> {RAW_UPCOMING_FILE}")


if __name__ == "__main__":
    main()
