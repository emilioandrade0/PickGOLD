import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

import sys
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
RAW_DATA_DIR = BASE_DIR / "data" / "euroleague" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

FILE_PATH_ADVANCED = RAW_DATA_DIR / "euroleague_advanced_history.csv"
FILE_PATH_UPCOMING = RAW_DATA_DIR / "euroleague_upcoming_schedule.csv"

SEASON_CODES = {
    "2025-26": "E2025",
}

MAX_GAMECODE = 390
CONSECUTIVE_EMPTY_STOP = 35
REQUEST_SLEEP_SECONDS = 0.0
CHECKPOINT_EVERY = 25
RECENT_BACKFILL_GAMES = 20
FORWARD_SCAN_GAMES = 45


def _safe_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default


def _extract_gamecode(game_id: str, season_code: str):
    txt = str(game_id or "")
    prefix = f"{season_code}-"
    if not txt.startswith(prefix):
        return None
    part = txt[len(prefix):]
    if not part.isdigit():
        return None
    return int(part)


def _team_code_from_header(header: dict, side: str) -> str:
    tv_key = "TVCodeA" if side == "A" else "TVCodeB"
    code_key = "CodeTeamA" if side == "A" else "CodeTeamB"
    team_key = "TeamA" if side == "A" else "TeamB"

    tv_code = str(header.get(tv_key, "") or "").strip().upper()
    if tv_code:
        return tv_code

    code = str(header.get(code_key, "") or "").strip().upper()
    if code:
        return code

    team = str(header.get(team_key, "") or "").strip().upper()
    if not team:
        return "UNK"
    return "".join(ch for ch in team if ch.isalnum())[:3] or "UNK"


def _parse_header_date(value: str) -> str:
    txt = str(value or "").strip()
    if not txt:
        return ""
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(txt, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return ""


def _fetch_json(session: requests.Session, url: str):
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    text = resp.text.strip()
    if not text.startswith("{"):
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_quarters(box: dict, team_a_name: str, team_b_name: str):
    by_quarter = box.get("ByQuarter") if isinstance(box, dict) else None
    if not isinstance(by_quarter, list) or len(by_quarter) < 2:
        return (0, 0, 0, 0), (0, 0, 0, 0)

    rows = [r for r in by_quarter if isinstance(r, dict)]
    if len(rows) < 2:
        return (0, 0, 0, 0), (0, 0, 0, 0)

    team_a = None
    team_b = None

    name_a = str(team_a_name or "").strip().upper()
    name_b = str(team_b_name or "").strip().upper()

    for row in rows:
        row_name = str(row.get("Team", "") or "").strip().upper()
        if row_name == name_a:
            team_a = row
        elif row_name == name_b:
            team_b = row

    if team_a is None:
        team_a = rows[0]
    if team_b is None:
        team_b = rows[1] if len(rows) > 1 else rows[0]

    qa = (
        _safe_int(team_a.get("Quarter1")),
        _safe_int(team_a.get("Quarter2")),
        _safe_int(team_a.get("Quarter3")),
        _safe_int(team_a.get("Quarter4")),
    )
    qb = (
        _safe_int(team_b.get("Quarter1")),
        _safe_int(team_b.get("Quarter2")),
        _safe_int(team_b.get("Quarter3")),
        _safe_int(team_b.get("Quarter4")),
    )
    return qa, qb


def _extract_top_points(box: dict):
    stats = box.get("Stats") if isinstance(box, dict) else None
    if not isinstance(stats, list) or len(stats) < 2:
        return 0, 0

    def _top_points(item):
        players = item.get("PlayersStats") if isinstance(item, dict) else None
        if not isinstance(players, list):
            return 0
        best = 0
        for p in players:
            if not isinstance(p, dict):
                continue
            best = max(best, _safe_int(p.get("Points"), 0))
        return best

    return _top_points(stats[0]), _top_points(stats[1])


def _collect_season_rows(
    session: requests.Session,
    season_label: str,
    season_code: str,
    start_gamecode: int,
    end_gamecode: int,
):
    rows_completed = []
    rows_upcoming = []

    seen_any = False
    empty_streak = 0
    today_str = datetime.now().strftime("%Y-%m-%d")

    for gamecode in range(start_gamecode, end_gamecode + 1):
        header_url = f"https://live.euroleague.net/api/Header?gamecode={gamecode}&seasoncode={season_code}"
        header = _fetch_json(session, header_url)

        if not isinstance(header, dict):
            continue

        game_date = _parse_header_date(header.get("Date"))
        team_a_name = str(header.get("TeamA", "") or "").strip().upper()
        team_b_name = str(header.get("TeamB", "") or "").strip().upper()

        if not game_date or not team_a_name or not team_b_name:
            if seen_any:
                empty_streak += 1
                if empty_streak >= CONSECUTIVE_EMPTY_STOP:
                    break
            continue

        seen_any = True
        empty_streak = 0

        home_team = _team_code_from_header(header, "A")
        away_team = _team_code_from_header(header, "B")
        game_id = f"{season_code}-{gamecode}"

        home_score = _safe_int(header.get("ScoreA"), 0)
        away_score = _safe_int(header.get("ScoreB"), 0)
        has_final_score = (home_score + away_score) > 0

        box = None
        if game_date <= today_str:
            box = _fetch_json(session, f"https://live.euroleague.net/api/Boxscore?gamecode={gamecode}&seasoncode={season_code}")
        q_home, q_away = _extract_quarters(box or {}, team_a_name, team_b_name)
        top_home, top_away = _extract_top_points(box or {})

        base_row = {
            "game_id": game_id,
            "date": game_date,
            "season": season_label,
            "home_team": home_team,
            "away_team": away_team,
            "home_pts_total": home_score,
            "away_pts_total": away_score,
            "home_q1": q_home[0],
            "home_q2": q_home[1],
            "home_q3": q_home[2],
            "home_q4": q_home[3],
            "away_q1": q_away[0],
            "away_q2": q_away[1],
            "away_q3": q_away[2],
            "away_q4": q_away[3],
            "attendance": _safe_int(header.get("Capacity") or header.get("Attendance"), 0),
            "odds_spread": "No Line",
            "home_spread": 0.0,
            "spread_abs": 0.0,
            "home_is_favorite": 0,
            "odds_over_under": 0.0,
            "home_top_scorer_pts": top_home,
            "away_top_scorer_pts": top_away,
        }

        if game_date <= today_str and has_final_score:
            rows_completed.append(base_row)
        elif game_date >= today_str:
            rows_upcoming.append(
                {
                    "game_id": game_id,
                    "date": game_date,
                    "time": str(header.get("Hour", "") or "").strip(),
                    "season": season_label,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": home_score,
                    "away_score": away_score,
                    "home_q1_score": q_home[0],
                    "away_q1_score": q_away[0],
                    "status_completed": int(has_final_score),
                    "status_state": "post" if has_final_score else "pre",
                    "status_description": "Final" if has_final_score else "Scheduled",
                    "status_detail": "Final" if has_final_score else "Scheduled",
                    "spread": "No Line",
                    "home_spread": 0.0,
                    "spread_abs": 0.0,
                    "home_is_favorite": 0,
                    "odds_over_under": 0.0,
                    "market_missing": 1,
                }
            )

        local_index = gamecode - start_gamecode + 1
        local_total = max(end_gamecode - start_gamecode + 1, 1)
        if local_index % CHECKPOINT_EVERY == 0 or gamecode == end_gamecode:
            print(
                f"     · progress {season_code}: gamecode={gamecode}/{end_gamecode} "
                f"scan={local_index}/{local_total} "
                f"completed={len(rows_completed)} upcoming={len(rows_upcoming)}"
            )

        time.sleep(REQUEST_SLEEP_SECONDS)

    return rows_completed, rows_upcoming


def _load_existing_rows():
    completed = []
    upcoming = []

    if FILE_PATH_ADVANCED.exists():
        try:
            completed = pd.read_csv(FILE_PATH_ADVANCED, dtype={"game_id": str}).to_dict("records")
        except Exception:
            completed = []

    if FILE_PATH_UPCOMING.exists():
        try:
            upcoming = pd.read_csv(FILE_PATH_UPCOMING, dtype={"game_id": str}).to_dict("records")
        except Exception:
            upcoming = []

    return completed, upcoming


def _scan_bounds_for_season(season_code: str, all_completed: list, all_upcoming: list, full_refresh: bool):
    if full_refresh:
        return 1, MAX_GAMECODE

    known_codes = []
    for row in all_completed:
        code = _extract_gamecode(row.get("game_id"), season_code)
        if code is not None:
            known_codes.append(code)
    for row in all_upcoming:
        code = _extract_gamecode(row.get("game_id"), season_code)
        if code is not None:
            known_codes.append(code)

    if not known_codes:
        return 1, MAX_GAMECODE

    max_known = max(known_codes)
    start = max(1, max_known - RECENT_BACKFILL_GAMES)
    end = min(MAX_GAMECODE, max_known + FORWARD_SCAN_GAMES)
    return start, end


def main():
    parser = argparse.ArgumentParser(description="Ingest Euroliga desde fuente oficial")
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Escanea toda la temporada en lugar de modo incremental",
    )
    args = parser.parse_args()

    print("🏀 Ingest Euroliga (fuente oficial live.euroleague.net)...")
    session = requests.Session()

    all_completed, all_upcoming = _load_existing_rows()
    if not args.full_refresh and (all_completed or all_upcoming):
        print(
            f"   > Modo incremental: base existente completed={len(all_completed)} "
            f"upcoming={len(all_upcoming)}"
        )
    elif args.full_refresh:
        print("   > Modo full-refresh: escaneo completo")

    # Ensure raw rows keep the current season values if files have mixed schema versions.
    all_completed = [dict(r) for r in all_completed]
    all_upcoming = [dict(r) for r in all_upcoming]

    def _save_checkpoint(tag: str):
        if all_completed:
            df = pd.DataFrame(all_completed).drop_duplicates(subset=["game_id"], keep="last")
            df = df.sort_values(["date", "game_id"]).reset_index(drop=True)
            FILE_PATH_ADVANCED.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(FILE_PATH_ADVANCED, index=False)
            print(f"💾 [{tag}] histórico parcial guardado: {FILE_PATH_ADVANCED} ({len(df)} juegos)")

        if all_upcoming:
            up = pd.DataFrame(all_upcoming).drop_duplicates(subset=["game_id"], keep="last")
            up = up.sort_values(["date", "game_id"]).reset_index(drop=True)
            FILE_PATH_UPCOMING.parent.mkdir(parents=True, exist_ok=True)
            up.to_csv(FILE_PATH_UPCOMING, index=False)
            print(f"💾 [{tag}] schedule parcial guardado: {FILE_PATH_UPCOMING} ({len(up)} juegos)")

    try:
        for season_label, season_code in SEASON_CODES.items():
            print(f"   > Season {season_label} ({season_code})")
            start_gc, end_gc = _scan_bounds_for_season(
                season_code,
                all_completed,
                all_upcoming,
                args.full_refresh,
            )
            print(f"     - scan window: {start_gc}..{end_gc}")
            comp, upc = _collect_season_rows(
                session,
                season_label,
                season_code,
                start_gc,
                end_gc,
            )
            print(f"     - completed: {len(comp)} | upcoming: {len(upc)}")
            all_completed.extend(comp)
            all_upcoming.extend(upc)
            _save_checkpoint(f"season-{season_code}")
    except KeyboardInterrupt:
        print("\n⚠️ Ingest interrumpida por usuario. Guardando checkpoint...")
        _save_checkpoint("interrupt")
        return

    if all_completed:
        df = pd.DataFrame(all_completed).drop_duplicates(subset=["game_id"], keep="last")
        df = df.sort_values(["date", "game_id"]).reset_index(drop=True)
        FILE_PATH_ADVANCED.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(FILE_PATH_ADVANCED, index=False)
        print(f"✅ Histórico Euroliga guardado: {FILE_PATH_ADVANCED} ({len(df)} juegos)")
    else:
        print("⚠️ No se obtuvieron juegos completados para Euroliga.")

    if all_upcoming:
        up = pd.DataFrame(all_upcoming).drop_duplicates(subset=["game_id"], keep="last")
        up = up.sort_values(["date", "game_id"]).reset_index(drop=True)
        FILE_PATH_UPCOMING.parent.mkdir(parents=True, exist_ok=True)
        up.to_csv(FILE_PATH_UPCOMING, index=False)
        print(f"✅ Schedule Euroliga guardado: {FILE_PATH_UPCOMING} ({len(up)} juegos)")


if __name__ == "__main__":
    main()
