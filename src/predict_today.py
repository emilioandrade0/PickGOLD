import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from calibration import calibrate_probability, load_calibration_config
from pattern_engine import aggregate_pattern_edge
from pattern_engine_nba import generate_nba_patterns
from pick_selector import fuse_with_pattern_score, recommendation_score

# --- RUTAS ---
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA = BASE_DIR / "data" / "raw" / "nba_advanced_history.csv"
MODELS_DIR = BASE_DIR / "models"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_FILE = MODELS_DIR / "calibration_params.json"
SPORT_KEY = "nba"
LEAGUE_LABEL = "NBA"

ESPN_TO_NBA = {
    "GS": "GSW",
    "NY": "NYK",
    "SA": "SAS",
    "NO": "NOP",
    "WSH": "WAS",
    "UTAH": "UTA",
    "CHA": "CHA",
    "BKN": "BKN",
}

TEAM_CONF_DIV = {
    "ATL": ("EAST", "SOUTHEAST"), "BOS": ("EAST", "ATLANTIC"), "BKN": ("EAST", "ATLANTIC"),
    "CHA": ("EAST", "SOUTHEAST"), "CHI": ("EAST", "CENTRAL"), "CLE": ("EAST", "CENTRAL"),
    "DAL": ("WEST", "SOUTHWEST"), "DEN": ("WEST", "NORTHWEST"), "DET": ("EAST", "CENTRAL"),
    "GSW": ("WEST", "PACIFIC"), "HOU": ("WEST", "SOUTHWEST"), "IND": ("EAST", "CENTRAL"),
    "LAC": ("WEST", "PACIFIC"), "LAL": ("WEST", "PACIFIC"), "MEM": ("WEST", "SOUTHWEST"),
    "MIA": ("EAST", "SOUTHEAST"), "MIL": ("EAST", "CENTRAL"), "MIN": ("WEST", "NORTHWEST"),
    "NOP": ("WEST", "SOUTHWEST"), "NYK": ("EAST", "ATLANTIC"), "OKC": ("WEST", "NORTHWEST"),
    "ORL": ("EAST", "SOUTHEAST"), "PHI": ("EAST", "ATLANTIC"), "PHX": ("WEST", "PACIFIC"),
    "POR": ("WEST", "NORTHWEST"), "SAC": ("WEST", "PACIFIC"), "SAS": ("WEST", "SOUTHWEST"),
    "TOR": ("EAST", "ATLANTIC"), "UTA": ("WEST", "NORTHWEST"), "WAS": ("EAST", "SOUTHEAST"),
}

TEAM_TIMEZONE = {
    "ATL": "ET", "BOS": "ET", "BKN": "ET", "CHA": "ET", "CHI": "CT", "CLE": "ET",
    "DAL": "CT", "DEN": "MT", "DET": "ET", "GSW": "PT", "HOU": "CT", "IND": "ET",
    "LAC": "PT", "LAL": "PT", "MEM": "CT", "MIA": "ET", "MIL": "CT", "MIN": "CT",
    "NOP": "CT", "NYK": "ET", "OKC": "CT", "ORL": "ET", "PHI": "ET", "PHX": "MT",
    "POR": "PT", "SAC": "PT", "SAS": "CT", "TOR": "ET", "UTA": "MT", "WAS": "ET",
}

TZ_OFFSET = {"ET": 0, "CT": -1, "MT": -2, "PT": -3}


def days_to_playoffs(game_date: pd.Timestamp) -> int:
    playoff_year = game_date.year + 1 if game_date.month >= 7 else game_date.year
    playoff_start = pd.Timestamp(playoff_year, 4, 15)
    return int(max((playoff_start - game_date).days, 0))


def add_context_features(df_predict: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    home_conf = df_predict["home_team"].map(lambda t: TEAM_CONF_DIV.get(str(t), ("UNK", "UNK"))[0])
    away_conf = df_predict["away_team"].map(lambda t: TEAM_CONF_DIV.get(str(t), ("UNK", "UNK"))[0])
    home_div = df_predict["home_team"].map(lambda t: TEAM_CONF_DIV.get(str(t), ("UNK", "UNK"))[1])
    away_div = df_predict["away_team"].map(lambda t: TEAM_CONF_DIV.get(str(t), ("UNK", "UNK"))[1])

    home_tz = df_predict["home_team"].map(lambda t: TEAM_TIMEZONE.get(str(t), "ET"))
    away_tz = df_predict["away_team"].map(lambda t: TEAM_TIMEZONE.get(str(t), "ET"))

    df_predict["same_conference"] = (home_conf == away_conf).astype(int)
    df_predict["same_division"] = (home_div == away_div).astype(int)
    df_predict["away_tz_diff"] = (home_tz.map(TZ_OFFSET) - away_tz.map(TZ_OFFSET)).abs().astype(int)
    df_predict["interconference_travel"] = (
        (df_predict["same_conference"] == 0) & (df_predict["away_tz_diff"] >= 2)
    ).astype(int)

    d2p = days_to_playoffs(pd.Timestamp(target_date))
    df_predict["days_to_playoffs"] = d2p
    df_predict["playoff_pressure"] = float(min(max(1 - (d2p / 120.0), 0), 1))

    return df_predict


def parse_home_spread(spread_text: str, home_abbr: str, away_abbr: str) -> float:
    if not spread_text:
        return 0.0

    txt = str(spread_text).strip().upper()

    if txt in {"N/A", "NO LINE", "PK", "PICK", "PICKEM", "PICK'EM"}:
        return 0.0

    m = re.match(r"^([A-Z]+)\s*([+-]?\d+(?:\.\d+)?)$", txt)
    if not m:
        return 0.0

    fav_team = m.group(1)
    fav_line = -abs(float(m.group(2)))

    if fav_team == home_abbr:
        return fav_line
    elif fav_team == away_abbr:
        return abs(fav_line)

    return 0.0


def parse_over_under(value) -> float:
    try:
        if value in [None, "", "N/A"]:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def get_pick_tier(conf: float) -> str:
    if conf >= 70:
        return "ELITE"
    elif conf >= 65:
        return "PREMIUM"
    elif conf >= 60:
        return "STRONG"
    elif conf >= 57:
        return "NORMAL"
    return "PASS"


def get_pick_tier_label(conf: float) -> str:
    if conf >= 70:
        return "🔥 ELITE PICK"
    elif conf >= 65:
        return "🔴 PREMIUM PICK"
    elif conf >= 60:
        return "🟢 STRONG PICK"
    elif conf >= 57:
        return "🔵 NORMAL PICK"
    return "⚪ PASS"


def get_q1_action(conf: float) -> str:
    return "JUGAR Q1" if conf >= 62 else "PASAR Q1"


def get_q1_action_label(conf: float) -> str:
    return "🟢 JUGAR Q1" if conf >= 62 else "⚪ PASAR Q1"


def load_pick_params(models_dir: Path) -> dict:
    default_params = {
        "game": {"xgb_weight": 0.5, "lgb_weight": 0.5, "threshold": 0.5},
        "q1": {"xgb_weight": 0.5, "lgb_weight": 0.5, "threshold": 0.5},
    }

    params_file = models_dir / "pick_params.pkl"
    if not params_file.exists():
        return default_params

    try:
        loaded = joblib.load(params_file)
        if not isinstance(loaded, dict):
            return default_params

        for key in ["game", "q1"]:
            if key not in loaded:
                loaded[key] = default_params[key]
            for p in ["xgb_weight", "lgb_weight", "threshold"]:
                if p not in loaded[key]:
                    loaded[key][p] = default_params[key][p]

        return loaded
    except Exception:
        return default_params


def calculate_elo_map(df: pd.DataFrame, k: float = 20, home_advantage: float = 100):
    elo_dict = {}
    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date_dt", "game_id"]).reset_index(drop=True)

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        if home not in elo_dict:
            elo_dict[home] = 1500.0
        if away not in elo_dict:
            elo_dict[away] = 1500.0

        home_elo_pre = elo_dict[home]
        away_elo_pre = elo_dict[away]

        elo_diff = away_elo_pre - (home_elo_pre + home_advantage)
        expected_home = 1 / (1 + 10 ** (elo_diff / 400))
        expected_away = 1 - expected_home

        actual_home = 1 if row["home_pts_total"] > row["away_pts_total"] else 0
        actual_away = 1 - actual_home

        elo_dict[home] = home_elo_pre + k * (actual_home - expected_home)
        elo_dict[away] = away_elo_pre + k * (actual_away - expected_away)

    return elo_dict


def get_current_team_stats(df_history: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    def _weighted_tail(series: pd.Series, window: int = 10, decay: float = 0.85) -> float:
        vals = pd.to_numeric(series.tail(window), errors="coerce").dropna().to_numpy(dtype=float)
        n = len(vals)
        if n <= 0:
            return 0.0
        weights = np.power(decay, np.arange(n - 1, -1, -1, dtype=float))
        return float(np.dot(vals, weights) / weights.sum())

    home_df = df_history[
        ["date", "game_id", "home_team", "home_pts_total", "away_pts_total", "home_q1", "away_q1"]
    ].copy()
    home_df.columns = [
        "date", "game_id", "team", "pts_scored", "pts_conceded", "q1_scored", "q1_conceded"
    ]
    home_df["is_home"] = 1

    away_df = df_history[
        ["date", "game_id", "away_team", "away_pts_total", "home_pts_total", "away_q1", "home_q1"]
    ].copy()
    away_df.columns = [
        "date", "game_id", "team", "pts_scored", "pts_conceded", "q1_scored", "q1_conceded"
    ]
    away_df["is_home"] = 0

    all_stats = pd.concat([home_df, away_df], ignore_index=True)
    all_stats["date_dt"] = pd.to_datetime(all_stats["date"])
    all_stats = all_stats.sort_values(["team", "date_dt", "game_id"]).reset_index(drop=True)

    all_stats["won_game"] = (all_stats["pts_scored"] > all_stats["pts_conceded"]).astype(int)
    all_stats["pts_diff"] = all_stats["pts_scored"] - all_stats["pts_conceded"]
    all_stats["won_q1"] = (all_stats["q1_scored"] > all_stats["q1_conceded"]).astype(int)
    all_stats["q1_diff"] = all_stats["q1_scored"] - all_stats["q1_conceded"]

    latest_stats = []

    for team, group in all_stats.groupby("team"):
        group = group.sort_values(["date_dt", "game_id"]).reset_index(drop=True)
        l5 = group.tail(5)
        l10 = group.tail(10)

        if l10.empty:
            continue

        last_game_dt = group["date_dt"].iloc[-1]
        rest_days = min(max((target_date - last_game_dt).days, 0), 7)
        is_b2b = int(rest_days == 1)

        day_diffs = (target_date - group["date_dt"]).dt.days
        games_last_3_days = int(day_diffs.between(1, 3).sum())
        games_last_5_days = int(day_diffs.between(1, 5).sum())
        games_last_7_days = int(day_diffs.between(1, 7).sum())

        home_only = group[group["is_home"] == 1].tail(5)
        away_only = group[group["is_home"] == 0].tail(5)

        win_pct_D10 = _weighted_tail(group["won_game"], window=10, decay=0.85)
        pts_diff_D10 = _weighted_tail(group["pts_diff"], window=10, decay=0.85)
        q1_diff_D10 = _weighted_tail(group["q1_diff"], window=10, decay=0.85)

        momentum_win = float(l5["won_game"].mean()) - win_pct_D10
        momentum_pts = float(l5["pts_diff"].mean()) - pts_diff_D10
        momentum_q1 = float(l5["q1_diff"].mean()) - q1_diff_D10
        regression_alert = abs(momentum_win) + (abs(momentum_pts) / 20.0)

        latest_stats.append(
            {
                "team": team,
                "rest_days": rest_days,
                "is_b2b": is_b2b,
                "games_last_3_days": games_last_3_days,
                "games_last_5_days": games_last_5_days,
                "games_last_7_days": games_last_7_days,
                "win_pct_L5": l5["won_game"].mean(),
                "win_pct_L10": l10["won_game"].mean(),
                "pts_diff_L5": l5["pts_diff"].mean(),
                "pts_diff_L10": l10["pts_diff"].mean(),
                "q1_win_pct_L5": l5["won_q1"].mean(),
                "q1_diff_L5": l5["q1_diff"].mean(),
                "pts_scored_L5": l5["pts_scored"].mean(),
                "pts_conceded_L5": l5["pts_conceded"].mean(),
                "win_pct_D10": win_pct_D10,
                "pts_diff_D10": pts_diff_D10,
                "q1_diff_D10": q1_diff_D10,
                "momentum_win": momentum_win,
                "momentum_pts": momentum_pts,
                "momentum_q1": momentum_q1,
                "regression_alert": regression_alert,
                "home_only_win_pct_L5": home_only["won_game"].mean() if not home_only.empty else 0.0,
                "home_only_pts_diff_L5": home_only["pts_diff"].mean() if not home_only.empty else 0.0,
                "home_only_q1_win_pct_L5": home_only["won_q1"].mean() if not home_only.empty else 0.0,
                "away_only_win_pct_L5": away_only["won_game"].mean() if not away_only.empty else 0.0,
                "away_only_pts_diff_L5": away_only["pts_diff"].mean() if not away_only.empty else 0.0,
                "away_only_q1_win_pct_L5": away_only["won_q1"].mean() if not away_only.empty else 0.0,
            }
        )

    return pd.DataFrame(latest_stats)


def get_matchup_stats(df_history: pd.DataFrame, home_team: str, away_team: str) -> dict:
    played = df_history[
        (
            ((df_history["home_team"] == home_team) & (df_history["away_team"] == away_team))
            | ((df_history["home_team"] == away_team) & (df_history["away_team"] == home_team))
        )
    ].copy()

    if played.empty:
        return {
            "matchup_home_win_pct_L5": 0.5,
            "matchup_home_win_pct_L10": 0.5,
            "matchup_home_q1_win_pct_L5": 0.5,
            "matchup_home_pts_diff_L5": 0.0,
        }

    played["date_dt"] = pd.to_datetime(played["date"])
    played = played.sort_values(["date_dt", "game_id"]).reset_index(drop=True)

    def _winner(r):
        if int(r["home_pts_total"]) > int(r["away_pts_total"]):
            return r["home_team"]
        if int(r["away_pts_total"]) > int(r["home_pts_total"]):
            return r["away_team"]
        return "TIE"

    def _q1_winner(r):
        if int(r["home_q1"]) > int(r["away_q1"]):
            return r["home_team"]
        if int(r["away_q1"]) > int(r["home_q1"]):
            return r["away_team"]
        return "TIE"

    played["full_winner"] = played.apply(_winner, axis=1)
    played["q1_winner"] = played.apply(_q1_winner, axis=1)
    played["home_team_pts_diff_from_current_home"] = np.where(
        played["home_team"] == home_team,
        played["home_pts_total"] - played["away_pts_total"],
        played["away_pts_total"] - played["home_pts_total"],
    )

    l5 = played.tail(5)
    l10 = played.tail(10)

    return {
        "matchup_home_win_pct_L5": float((l5["full_winner"] == home_team).mean()) if len(l5) else 0.5,
        "matchup_home_win_pct_L10": float((l10["full_winner"] == home_team).mean()) if len(l10) else 0.5,
        "matchup_home_q1_win_pct_L5": float((l5["q1_winner"] == home_team).mean()) if len(l5) else 0.5,
        "matchup_home_pts_diff_L5": float(l5["home_team_pts_diff_from_current_home"].mean()) if len(l5) else 0.0,
    }


def fetch_games_for_date(target_dt):
    target_date_str = target_dt.strftime("%Y%m%d")
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={target_date_str}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    events = resp.json().get("events", [])

    upcoming_games = []

    for event in events:
        competitions = event.get("competitions", [])
        if not competitions:
            continue

        comp = competitions[0]
        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            continue

        home_data = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away_data = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home_data or not away_data:
            continue

        status = event.get("status", {}).get("type", {})
        status_state = str(status.get("state", "") or "")
        status_description = str(status.get("description", "") or "")
        status_detail = str(status.get("detail", "") or "")
        status_completed = int(bool(status.get("completed", False)))

        home_score = int(home_data.get("score", 0) or 0)
        away_score = int(away_data.get("score", 0) or 0)

        def _first_period_score(competitor):
            linescores = competitor.get("linescores", [])
            if not linescores:
                return 0
            first = linescores[0] or {}
            return int(first.get("value", 0) or 0)

        home_q1_score = _first_period_score(home_data)
        away_q1_score = _first_period_score(away_data)

        h_abbr = ESPN_TO_NBA.get(home_data["team"]["abbreviation"], home_data["team"]["abbreviation"])
        a_abbr = ESPN_TO_NBA.get(away_data["team"]["abbreviation"], away_data["team"]["abbreviation"])

        odds = comp.get("odds", [{}])
        odds = odds[0] if odds else {}
        spread_text = odds.get("details", "No Line")
        over_under = parse_over_under(odds.get("overUnder", 0))

        home_spread = parse_home_spread(spread_text, h_abbr, a_abbr)

        # Hora local simple UTC-5, consistente con tu ingest
        raw_dt = event.get("date")
        game_time = ""
        if raw_dt:
            try:
                dt_utc = datetime.strptime(raw_dt, "%Y-%m-%dT%H:%MZ")
                dt_local = dt_utc - timedelta(hours=5)
                game_time = dt_local.strftime("%H:%M")
            except Exception:
                game_time = ""

        upcoming_games.append(
            {
                "game_id": str(event.get("id", f"{target_dt}_{a_abbr}_{h_abbr}")),
                "date": str(target_dt),
                "time": game_time,
                "game_name": f"{a_abbr} @ {h_abbr}",
                "home_team": h_abbr,
                "away_team": a_abbr,
                "home_score": home_score,
                "away_score": away_score,
                "home_q1_score": home_q1_score,
                "away_q1_score": away_q1_score,
                "status_completed": status_completed,
                "status_state": status_state,
                "status_description": status_description,
                "status_detail": status_detail,
                "spread": spread_text,
                "home_spread": home_spread,
                "spread_abs": abs(home_spread),
                "home_is_favorite": int(home_spread < 0),
                "odds_over_under": over_under,
                "market_missing": 0 if spread_text != "No Line" else 1,
            }
        )

    return pd.DataFrame(upcoming_games)


def fetch_upcoming_games(days_ahead: int = 14):
    base_dt = datetime.now().date()
    all_games = []

    for day_offset in range(days_ahead + 1):
        day_dt = base_dt + timedelta(days=day_offset)
        try:
            df_day = fetch_games_for_date(day_dt)
            if not df_day.empty:
                all_games.append(df_day)
            print(f"📅 NBA {day_dt}: {len(df_day)} juegos")
        except Exception as e:
            print(f"⚠️ Error NBA {day_dt}: {e}")

    if not all_games:
        return pd.DataFrame()

    merged = pd.concat(all_games, ignore_index=True)
    merged["game_id"] = merged["game_id"].astype(str)
    merged["date"] = merged["date"].astype(str)
    return (
        merged.sort_values(["date", "time", "game_id"])
        .drop_duplicates(subset=["game_id"], keep="last")
        .reset_index(drop=True)
    )


def build_prediction_features(df_history: pd.DataFrame, todays_games: pd.DataFrame, target_date: pd.Timestamp):
    elo_map = calculate_elo_map(df_history)
    current_stats = get_current_team_stats(df_history, target_date)

    if todays_games.empty:
        return pd.DataFrame()

    df_predict = todays_games.copy()

    df_predict["home_elo_pre"] = df_predict["home_team"].map(lambda t: elo_map.get(t, 1500.0))
    df_predict["away_elo_pre"] = df_predict["away_team"].map(lambda t: elo_map.get(t, 1500.0))

    # HOME JOIN
    df_predict = pd.merge(
        df_predict,
        current_stats,
        left_on="home_team",
        right_on="team",
        how="left",
    )
    df_predict = df_predict.rename(
        columns={c: f"home_{c}" for c in current_stats.columns if c != "team"}
    ).drop(columns=["team"])

    # AWAY JOIN
    df_predict = pd.merge(
        df_predict,
        current_stats,
        left_on="away_team",
        right_on="team",
        how="left",
    )
    df_predict = df_predict.rename(
        columns={c: f"away_{c}" for c in current_stats.columns if c != "team"}
    ).drop(columns=["team"])

    fill_cols = [
        "home_rest_days", "away_rest_days",
        "home_is_b2b", "away_is_b2b",
        "home_games_last_3_days", "away_games_last_3_days",
        "home_games_last_5_days", "away_games_last_5_days",
        "home_games_last_7_days", "away_games_last_7_days",
        "home_win_pct_L5", "away_win_pct_L5",
        "home_win_pct_L10", "away_win_pct_L10",
        "home_pts_diff_L5", "away_pts_diff_L5",
        "home_pts_diff_L10", "away_pts_diff_L10",
        "home_q1_win_pct_L5", "away_q1_win_pct_L5",
        "home_q1_diff_L5", "away_q1_diff_L5",
        "home_pts_scored_L5", "away_pts_scored_L5",
        "home_pts_conceded_L5", "away_pts_conceded_L5",
        "home_win_pct_D10", "away_win_pct_D10",
        "home_pts_diff_D10", "away_pts_diff_D10",
        "home_q1_diff_D10", "away_q1_diff_D10",
        "home_momentum_win", "away_momentum_win",
        "home_momentum_pts", "away_momentum_pts",
        "home_momentum_q1", "away_momentum_q1",
        "home_regression_alert", "away_regression_alert",
        "home_home_only_win_pct_L5", "away_away_only_win_pct_L5",
        "home_home_only_pts_diff_L5", "away_away_only_pts_diff_L5",
        "home_home_only_q1_win_pct_L5", "away_away_only_q1_win_pct_L5",
    ]

    for col in fill_cols:
        if col not in df_predict.columns:
            df_predict[col] = 0.0
        else:
            df_predict[col] = df_predict[col].fillna(0.0)

    df_predict["diff_elo"] = df_predict["home_elo_pre"] - df_predict["away_elo_pre"]
    df_predict["diff_rest_days"] = df_predict["home_rest_days"] - df_predict["away_rest_days"]
    df_predict["diff_is_b2b"] = df_predict["home_is_b2b"] - df_predict["away_is_b2b"]
    df_predict["diff_games_last_3_days"] = df_predict["home_games_last_3_days"] - df_predict["away_games_last_3_days"]
    df_predict["diff_games_last_5_days"] = df_predict["home_games_last_5_days"] - df_predict["away_games_last_5_days"]
    df_predict["diff_games_last_7_days"] = df_predict["home_games_last_7_days"] - df_predict["away_games_last_7_days"]

    df_predict["diff_win_pct_L5"] = df_predict["home_win_pct_L5"] - df_predict["away_win_pct_L5"]
    df_predict["diff_win_pct_L10"] = df_predict["home_win_pct_L10"] - df_predict["away_win_pct_L10"]
    df_predict["diff_pts_diff_L5"] = df_predict["home_pts_diff_L5"] - df_predict["away_pts_diff_L5"]
    df_predict["diff_pts_diff_L10"] = df_predict["home_pts_diff_L10"] - df_predict["away_pts_diff_L10"]
    df_predict["diff_q1_win_pct_L5"] = df_predict["home_q1_win_pct_L5"] - df_predict["away_q1_win_pct_L5"]
    df_predict["diff_q1_diff_L5"] = df_predict["home_q1_diff_L5"] - df_predict["away_q1_diff_L5"]
    df_predict["diff_pts_scored_L5"] = df_predict["home_pts_scored_L5"] - df_predict["away_pts_scored_L5"]
    df_predict["diff_pts_conceded_L5"] = df_predict["home_pts_conceded_L5"] - df_predict["away_pts_conceded_L5"]
    df_predict["diff_win_pct_D10"] = df_predict["home_win_pct_D10"] - df_predict["away_win_pct_D10"]
    df_predict["diff_pts_diff_D10"] = df_predict["home_pts_diff_D10"] - df_predict["away_pts_diff_D10"]
    df_predict["diff_q1_diff_D10"] = df_predict["home_q1_diff_D10"] - df_predict["away_q1_diff_D10"]
    df_predict["diff_momentum_win"] = df_predict["home_momentum_win"] - df_predict["away_momentum_win"]
    df_predict["diff_momentum_pts"] = df_predict["home_momentum_pts"] - df_predict["away_momentum_pts"]
    df_predict["diff_momentum_q1"] = df_predict["home_momentum_q1"] - df_predict["away_momentum_q1"]
    df_predict["diff_regression_alert"] = (
        df_predict["home_regression_alert"] - df_predict["away_regression_alert"]
    )

    df_predict["diff_surface_win_pct_L5"] = (
        df_predict["home_home_only_win_pct_L5"] - df_predict["away_away_only_win_pct_L5"]
    )
    df_predict["diff_surface_pts_diff_L5"] = (
        df_predict["home_home_only_pts_diff_L5"] - df_predict["away_away_only_pts_diff_L5"]
    )
    df_predict["diff_surface_q1_win_pct_L5"] = (
        df_predict["home_home_only_q1_win_pct_L5"] - df_predict["away_away_only_q1_win_pct_L5"]
    )

    matchup_rows = []
    for _, row in df_predict.iterrows():
        matchup_rows.append(get_matchup_stats(df_history, row["home_team"], row["away_team"]))

    df_matchup = pd.DataFrame(matchup_rows)
    for col in df_matchup.columns:
        df_predict[col] = df_matchup[col].values

    df_predict["diff_matchup_home_edge_L5"] = (df_predict["matchup_home_win_pct_L5"] - 0.5) * 2
    df_predict["diff_matchup_home_edge_L10"] = (df_predict["matchup_home_win_pct_L10"] - 0.5) * 2

    league_win_pct_L10 = float(current_stats["win_pct_L10"].mean()) if not current_stats.empty else 0.5
    league_pts_diff_L10 = float(current_stats["pts_diff_L10"].mean()) if not current_stats.empty else 0.0
    league_q1_win_pct_L5 = float(current_stats["q1_win_pct_L5"].mean()) if not current_stats.empty else 0.5

    df_predict["home_win_pct_L10_vs_league"] = df_predict["home_win_pct_L10"] - league_win_pct_L10
    df_predict["away_win_pct_L10_vs_league"] = df_predict["away_win_pct_L10"] - league_win_pct_L10
    df_predict["diff_win_pct_L10_vs_league"] = (
        df_predict["home_win_pct_L10_vs_league"] - df_predict["away_win_pct_L10_vs_league"]
    )

    df_predict["home_pts_diff_L10_vs_league"] = df_predict["home_pts_diff_L10"] - league_pts_diff_L10
    df_predict["away_pts_diff_L10_vs_league"] = df_predict["away_pts_diff_L10"] - league_pts_diff_L10
    df_predict["diff_pts_diff_L10_vs_league"] = (
        df_predict["home_pts_diff_L10_vs_league"] - df_predict["away_pts_diff_L10_vs_league"]
    )

    df_predict["home_q1_win_pct_L5_vs_league"] = df_predict["home_q1_win_pct_L5"] - league_q1_win_pct_L5
    df_predict["away_q1_win_pct_L5_vs_league"] = df_predict["away_q1_win_pct_L5"] - league_q1_win_pct_L5
    df_predict["diff_q1_win_pct_L5_vs_league"] = (
        df_predict["home_q1_win_pct_L5_vs_league"] - df_predict["away_q1_win_pct_L5_vs_league"]
    )

    df_predict = add_context_features(df_predict, target_date)

    return df_predict


def predict_today():
    print("🔮 INICIANDO MOTOR IA NBA...")

    try:
        xgb_game = joblib.load(MODELS_DIR / "xgb_game.pkl")
        lgb_game = joblib.load(MODELS_DIR / "lgb_game.pkl")
        xgb_q1 = joblib.load(MODELS_DIR / "xgb_q1.pkl")
        lgb_q1 = joblib.load(MODELS_DIR / "lgb_q1.pkl")
        feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
        pick_params = load_pick_params(MODELS_DIR)
    except Exception as e:
        print(f"❌ Faltan modelos o feature_names.pkl. Error: {e}")
        return

    if not RAW_DATA.exists():
        print(f"❌ No existe el histórico raw: {RAW_DATA}")
        return

    df_history = pd.read_csv(RAW_DATA)
    upcoming_games = fetch_upcoming_games(days_ahead=14)

    if upcoming_games.empty:
        print("📭 No hay partidos en ventana rolling NBA.")
        return

    game_wx = float(pick_params["game"]["xgb_weight"])
    game_wl = float(pick_params["game"]["lgb_weight"])
    game_th = float(pick_params["game"]["threshold"])

    q1_wx = float(pick_params["q1"]["xgb_weight"])
    q1_wl = float(pick_params["q1"]["lgb_weight"])
    q1_th = float(pick_params["q1"]["threshold"])

    calibration_cfg = load_calibration_config(CALIBRATION_FILE)

    total_games = 0
    for date_str, games_df in upcoming_games.groupby("date"):
        target_date = pd.Timestamp(date_str)
        df_predict = build_prediction_features(df_history, games_df, target_date)
        X_day = df_predict.reindex(columns=feature_names, fill_value=0)

        prob_game = game_wx * xgb_game.predict_proba(X_day)[:, 1] + game_wl * lgb_game.predict_proba(X_day)[:, 1]
        prob_q1 = q1_wx * xgb_q1.predict_proba(X_day)[:, 1] + q1_wl * lgb_q1.predict_proba(X_day)[:, 1]

        print("======================================================")
        print(f"🏀 CARTELERA {LEAGUE_LABEL} {date_str} ({len(df_predict)} juegos)")
        print("======================================================")

        predictions_output = []

        for i, row in df_predict.iterrows():
            model_prob_game_home = float(prob_game[i])
            calibrated_prob_game_home = calibrate_probability(
                model_prob_game_home,
                sport=SPORT_KEY,
                market="full_game",
                calibration_config=calibration_cfg,
            )
            prob_game_home = calibrated_prob_game_home * 100
            prob_game_away = 100 - prob_game_home
            ganador_juego = row["home_team"] if (calibrated_prob_game_home >= game_th) else row["away_team"]
            conf_juego = max(prob_game_home, prob_game_away)
            tier_game = get_pick_tier(conf_juego)
            tier_game_label = get_pick_tier_label(conf_juego)
            base_game_score = recommendation_score(calibrated_prob_game_home)
            nba_patterns = generate_nba_patterns(row.to_dict())
            pattern_edge = aggregate_pattern_edge(nba_patterns)
            game_rec_score = fuse_with_pattern_score(base_game_score, pattern_edge)
            game_recommended = bool(game_rec_score >= 56.0)

            model_prob_q1_home = float(prob_q1[i])
            calibrated_prob_q1_home = calibrate_probability(
                model_prob_q1_home,
                sport=SPORT_KEY,
                market="q1",
                calibration_config=calibration_cfg,
            )
            prob_q1_home = calibrated_prob_q1_home * 100
            prob_q1_away = 100 - prob_q1_home
            ganador_q1 = row["home_team"] if (calibrated_prob_q1_home >= q1_th) else row["away_team"]
            conf_q1 = max(prob_q1_home, prob_q1_away)
            q1_action = get_q1_action(conf_q1)
            q1_action_label = get_q1_action_label(conf_q1)
            q1_rec_score = recommendation_score(calibrated_prob_q1_home)

            total_line = float(row.get("odds_over_under", 0) or 0)
            total_pick = "Pendiente"
            if total_line > 0:
                total_pick = f"Lean total {total_line}"

            spread_pick = "Pendiente"
            if str(row.get("spread", "No Line")) != "No Line":
                spread_pick = ganador_juego

            assists_pick = "Pendiente"

            print(f"👉 {row['game_name']} | Spread Vegas: {row['spread']}")
            print(f"   ⏱️ 1er Cuarto: Gana {ganador_q1} (score modelo: {conf_q1:.1f}%) | {q1_action_label}")
            print(f"   🏆 Partido:    Gana {ganador_juego} (score modelo: {conf_juego:.1f}%) | {tier_game_label}")
            print("-" * 54)

            predictions_output.append(
                {
                    "game_id": str(row.get("game_id", f"{date_str}_{row['away_team']}_{row['home_team']}")),
                    "date": date_str,
                    "time": row.get("time", ""),
                    "game_name": row["game_name"],
                    "away_team": row["away_team"],
                    "home_team": row["home_team"],
                    "home_score": int(row.get("home_score", 0) or 0),
                    "away_score": int(row.get("away_score", 0) or 0),
                    "home_q1_score": int(row.get("home_q1_score", 0) or 0),
                    "away_q1_score": int(row.get("away_q1_score", 0) or 0),
                    "status_completed": int(row.get("status_completed", 0) or 0),
                    "status_state": str(row.get("status_state", "") or ""),
                    "status_description": str(row.get("status_description", "") or ""),
                    "status_detail": str(row.get("status_detail", "") or ""),
                    "spread_market": row.get("spread", "No Line"),
                    "home_spread": float(row.get("home_spread", 0) or 0),
                    "spread_abs": float(row.get("spread_abs", 0) or 0),
                    "odds_over_under": total_line,
                    "market_missing": int(row.get("market_missing", 0) or 0),
                    "full_game_pick": ganador_juego,
                    "full_game_confidence": round(conf_juego, 1),
                    "full_game_tier": tier_game,
                    "full_game_model_prob_home": round(model_prob_game_home, 4),
                    "full_game_calibrated_prob_home": round(calibrated_prob_game_home, 4),
                    "full_game_pattern_edge": round(pattern_edge, 4),
                    "full_game_detected_patterns": nba_patterns,
                    "full_game_recommended_score": round(game_rec_score, 1),
                    "full_game_recommended": game_recommended,
                    "full_game_prob_home": round(prob_game_home, 2),
                    "full_game_prob_away": round(prob_game_away, 2),
                    "q1_pick": ganador_q1,
                    "q1_confidence": round(conf_q1, 1),
                    "q1_action": q1_action,
                    "q1_model_prob_home": round(model_prob_q1_home, 4),
                    "q1_calibrated_prob_home": round(calibrated_prob_q1_home, 4),
                    "q1_recommended_score": round(q1_rec_score, 1),
                    "q1_prob_home": round(prob_q1_home, 2),
                    "q1_prob_away": round(prob_q1_away, 2),
                    "total_pick": total_pick,
                    "spread_pick": spread_pick,
                    "assists_pick": assists_pick,
                }
            )

        output_file = PREDICTIONS_DIR / f"{date_str}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(predictions_output, f, ensure_ascii=False, indent=2)

        total_games += len(predictions_output)
        print(f"\n💾 Predicciones guardadas en: {output_file}")

    print(f"\n✅ Total juegos NBA predichos (rolling): {total_games}")


if __name__ == "__main__":
    predict_today()