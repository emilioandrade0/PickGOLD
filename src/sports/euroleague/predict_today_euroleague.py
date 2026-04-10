from datetime import datetime, timedelta
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
import requests

from sports.nba import predict_today as nba_predict

BASE_DIR = SRC_ROOT
RAW_DATA = BASE_DIR / "data" / "euroleague" / "raw" / "euroleague_advanced_history.csv"
UPCOMING_SCHEDULE = BASE_DIR / "data" / "euroleague" / "raw" / "euroleague_upcoming_schedule.csv"
MODELS_DIR = BASE_DIR / "data" / "euroleague" / "models"
PREDICTIONS_DIR = BASE_DIR / "data" / "euroleague" / "predictions"
CALIBRATION_FILE = MODELS_DIR / "calibration_params.json"

SEASON_CODES = ["E2025", "E2026"]
MAX_GAMECODE = 460


def _safe_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default


def _parse_date(txt: str) -> str:
    raw = str(txt or "").strip()
    if not raw:
        return ""
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return ""


def _team_code(header: dict, side: str) -> str:
    tv = str(header.get("TVCodeA" if side == "A" else "TVCodeB", "") or "").strip().upper()
    if tv:
        return tv
    code = str(header.get("CodeTeamA" if side == "A" else "CodeTeamB", "") or "").strip().upper()
    if code:
        return code
    team = str(header.get("TeamA" if side == "A" else "TeamB", "") or "").strip().upper()
    return ("".join(ch for ch in team if ch.isalnum())[:3] or "UNK") if team else "UNK"


def _from_schedule_csv(days_ahead: int):
    if not UPCOMING_SCHEDULE.exists():
        return pd.DataFrame()

    df = pd.read_csv(UPCOMING_SCHEDULE, dtype={"game_id": str})
    if df.empty:
        return pd.DataFrame()

    today = datetime.now().date()
    limit = today + timedelta(days=days_ahead)

    df["date"] = df["date"].astype(str)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df[(df["date_dt"] >= today) & (df["date_dt"] <= limit)].copy()
    df = df.drop(columns=["date_dt"], errors="ignore")
    return df.reset_index(drop=True)


def _fetch_upcoming_live(days_ahead: int):
    today = datetime.now().date()
    limit = today + timedelta(days=days_ahead)
    rows = []

    session = requests.Session()

    for season_code in SEASON_CODES:
        empty = 0
        seen = False
        for gamecode in range(1, MAX_GAMECODE + 1):
            try:
                url = f"https://live.euroleague.net/api/Header?gamecode={gamecode}&seasoncode={season_code}"
                resp = session.get(url, timeout=10)
                resp.raise_for_status()
                text = resp.text.strip()
                if not text.startswith("{"):
                    continue
                header = resp.json()
            except Exception:
                continue

            game_date = _parse_date(header.get("Date"))
            if not game_date:
                if seen:
                    empty += 1
                    if empty >= 30:
                        break
                continue

            seen = True
            empty = 0

            dt = datetime.strptime(game_date, "%Y-%m-%d").date()
            if dt < today or dt > limit:
                continue

            home_score = _safe_int(header.get("ScoreA"), 0)
            away_score = _safe_int(header.get("ScoreB"), 0)
            completed = int((home_score + away_score) > 0 and dt <= today)

            home_team = _team_code(header, "A")
            away_team = _team_code(header, "B")

            rows.append(
                {
                    "game_id": f"{season_code}-{gamecode}",
                    "date": game_date,
                    "time": str(header.get("Hour", "") or "").strip(),
                    "game_name": f"{away_team} @ {home_team}",
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": home_score,
                    "away_score": away_score,
                    "home_q1_score": 0,
                    "away_q1_score": 0,
                    "status_completed": completed,
                    "status_state": "post" if completed else "pre",
                    "status_description": "Final" if completed else "Scheduled",
                    "status_detail": "Final" if completed else "Scheduled",
                    "spread": "No Line",
                    "home_spread": 0.0,
                    "spread_abs": 0.0,
                    "home_is_favorite": 0,
                    "odds_over_under": 0.0,
                    "market_missing": 1,
                }
            )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).drop_duplicates(subset=["game_id"], keep="last")
    out = out.sort_values(["date", "time", "game_id"]).reset_index(drop=True)
    return out


def fetch_upcoming_games(days_ahead: int = 14):
    from_csv = _from_schedule_csv(days_ahead)
    if not from_csv.empty:
        return from_csv
    return _fetch_upcoming_live(days_ahead)


def main():
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    nba_predict.RAW_DATA = RAW_DATA
    nba_predict.MODELS_DIR = MODELS_DIR
    nba_predict.PREDICTIONS_DIR = PREDICTIONS_DIR
    nba_predict.CALIBRATION_FILE = CALIBRATION_FILE
    nba_predict.SPORT_KEY = "euroleague"
    nba_predict.LEAGUE_LABEL = "EuroLeague"
    nba_predict.fetch_upcoming_games = fetch_upcoming_games

    nba_predict.predict_today()


if __name__ == "__main__":
    main()
