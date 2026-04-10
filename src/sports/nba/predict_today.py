import json
import re
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project `src` root is on sys.path so local modules import correctly
import sys
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import joblib
import numpy as np
import pandas as pd
import requests
from calibration import calibrate_probability, load_calibration_config
from pattern_engine import aggregate_pattern_edge
from sports.nba.pattern_engine_nba import generate_nba_patterns
from pick_selector import fuse_with_pattern_score, recommendation_score
try:
    from sports.nba.error_risk_utils import apply_error_risk_override, load_error_risk_bundle
except Exception:
    def load_error_risk_bundle(path):
        return None

    def apply_error_risk_override(bundle, row, calibrated_prob_home, threshold_used, current_pick, current_rule):
        return current_pick, current_rule, None

# --- RUTAS ---
import sys
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
RAW_DATA = BASE_DIR / "data" / "raw" / "nba_advanced_history.csv"
MODELS_DIR = BASE_DIR / "models"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_FILE = MODELS_DIR / "calibration_params.json"
ERROR_RISK_MODEL_FILE = MODELS_DIR / "error_risk_fullgame.pkl"
PROCESSED_DATA = BASE_DIR / "data" / "processed" / "model_ready_features.csv"
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


def add_full_game_signal_features(df_predict: pd.DataFrame) -> pd.DataFrame:
    def _num(col: str, default: float = 0.0) -> pd.Series:
        if col in df_predict.columns:
            return pd.to_numeric(df_predict[col], errors="coerce").fillna(default)
        return pd.Series(default, index=df_predict.index, dtype=float)

    home_elo = _num("home_elo_pre")
    away_elo = _num("away_elo_pre")
    home_spread = _num("home_spread")

    home_rest = _num("home_rest_days")
    away_rest = _num("away_rest_days")
    home_b2b = _num("home_is_b2b")
    away_b2b = _num("away_is_b2b")
    home_g5 = _num("home_games_last_5_days")
    away_g5 = _num("away_games_last_5_days")

    diff_win_l10 = _num("diff_win_pct_L10")
    diff_pts_l10 = _num("diff_pts_diff_L10")
    diff_surface_pts = _num("diff_surface_pts_diff_L5")
    diff_matchup_l5 = _num("diff_matchup_home_edge_L5")
    diff_pts_vs_league = _num("diff_pts_diff_L10_vs_league")
    diff_rest = _num("diff_rest_days")
    diff_is_b2b = _num("diff_is_b2b")
    diff_g5 = _num("diff_games_last_5_days")

    elo_expected_spread = (away_elo - (home_elo + 100.0)) / 28.0
    df_predict["elo_spread_gap"] = home_spread - elo_expected_spread
    df_predict["elo_spread_gap_abs"] = df_predict["elo_spread_gap"].abs()

    df_predict["fullgame_form_strength_edge"] = (
        (diff_win_l10 * 0.45)
        + (diff_pts_l10 * 0.20)
        + (diff_surface_pts * 0.20)
        + (diff_matchup_l5 * 0.15)
    )

    home_stress = (home_g5 * 0.70) + (home_b2b * 1.40) - (home_rest * 0.50)
    away_stress = (away_g5 * 0.70) + (away_b2b * 1.40) - (away_rest * 0.50)
    df_predict["schedule_stress_edge"] = away_stress - home_stress

    df_predict["fullgame_context_interaction"] = diff_win_l10 * diff_matchup_l5
    df_predict["rest_form_interaction"] = diff_rest * diff_pts_vs_league
    df_predict["fatigue_penalty_edge"] = (diff_is_b2b * 0.8) + (diff_g5 * 0.25)
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


def get_pick_tier(conf: float, row: dict = None) -> str:
    """
    Tiering para picks. ELITE es más estricto que antes (70 -> 74).
    Si se pasa `row`, aplicamos exclusiones simples (market_missing, volatilidad, fatiga, 4in6).
    """
    ELITE_THRESHOLD = 74.0

    if row is None:
        if conf >= ELITE_THRESHOLD:
            return "ELITE"
        elif conf >= 65:
            return "PREMIUM"
        elif conf >= 60:
            return "STRONG"
        elif conf >= 57:
            return "NORMAL"
        return "PASS"

    try:
        market_missing = infer_market_missing(pd.Series(row) if not isinstance(row, pd.Series) else row)
        home_vol = float(row.get("home_volatility", 0.0) or 0.0)
        away_vol = float(row.get("away_volatility", 0.0) or 0.0)
        home_fat = float(row.get("home_fatigue_load", 0.0) or 0.0)
        away_fat = float(row.get("away_fatigue_load", 0.0) or 0.0)
        home_4in6 = int(row.get("home_4in6_flag", 0) or 0)
        away_4in6 = int(row.get("away_4in6_flag", 0) or 0)
    except Exception:
        market_missing = 0
        home_vol = away_vol = 0.0
        home_fat = away_fat = 0.0
        home_4in6 = away_4in6 = 0

    high_volatility = max(home_vol, away_vol) > 0.6
    high_fatigue = max(home_fat, away_fat) > 2.5
    any_4in6 = (home_4in6 == 1) or (away_4in6 == 1)

    if conf >= ELITE_THRESHOLD and not market_missing and not high_volatility and not high_fatigue and not any_4in6:
        return "ELITE"

    if conf >= 65:
        return "PREMIUM"
    elif conf >= 60:
        return "STRONG"
    elif conf >= 57:
        return "NORMAL"
    return "PASS"


def get_pick_tier_label(conf: float, row: dict = None) -> str:
    tier = get_pick_tier(conf, row=row)
    if tier == "ELITE":
        return "🔥 ELITE PICK"
    if tier == "PREMIUM":
        return "🔴 PREMIUM PICK"
    if tier == "STRONG":
        return "🟢 STRONG PICK"
    if tier == "NORMAL":
        return "🔵 NORMAL PICK"
    return "⚪ PASS"


def get_q1_action(conf: float) -> str:
    return "JUGAR Q1" if conf >= 62 else "PASAR Q1"


def get_q1_action_label(conf: float) -> str:
    return "🟢 JUGAR Q1" if conf >= 62 else "⚪ PASAR Q1"


def load_pick_params(models_dir: Path) -> dict:
    default_mode = {
        "xgb_weight": 1.0 / 3.0,
        "lgb_weight": 1.0 / 3.0,
        "cat_weight": 1.0 / 3.0,
        "threshold": 0.5,
        "market_threshold": 0.515,
        "use_meta": True,
        "market_tiebreak_band": 0.10,
        "min_edge": 0.07,
        "alt_min_edge": 0.085,
        "preferred_line_min": 3.5,
        "preferred_line_max": 6.0,
        "long_line_min": 10.5,
    }
    default_params = {
        "game": default_mode.copy(),
        "q1": default_mode.copy(),
        "h1": default_mode.copy(),
        "spread": default_mode.copy(),
        "total": default_mode.copy(),
        "splits": {},
    }

    def _normalize_mode_block(block: dict) -> dict:
        if not isinstance(block, dict):
            return default_mode.copy()

        if "weights" in block and isinstance(block["weights"], dict):
            wx = float(block["weights"].get("xgb", default_mode["xgb_weight"]))
            wl = float(block["weights"].get("lgb", default_mode["lgb_weight"]))
            wc = float(block["weights"].get("cat", default_mode["cat_weight"]))
        else:
            wx = float(block.get("xgb_weight", default_mode["xgb_weight"]))
            wl = float(block.get("lgb_weight", default_mode["lgb_weight"]))
            wc = float(block.get("cat_weight", default_mode["cat_weight"]))

        total_w = wx + wl + wc
        if total_w <= 0:
            wx = wl = wc = 1.0 / 3.0
        else:
            wx, wl, wc = wx / total_w, wl / total_w, wc / total_w

        return {
            "xgb_weight": float(wx),
            "lgb_weight": float(wl),
            "cat_weight": float(wc),
            "threshold": float(block.get("threshold", default_mode["threshold"])),
            "market_threshold": float(block.get("market_threshold", block.get("threshold", default_mode["market_threshold"]))),
            "use_meta": bool(block.get("use_meta", True)),
            "market_tiebreak_band": float(block.get("market_tiebreak_band", default_mode["market_tiebreak_band"])),
            "min_edge": float(block.get("min_edge", default_mode["min_edge"])),
            "alt_min_edge": float(block.get("alt_min_edge", default_mode["alt_min_edge"])),
            "preferred_line_min": float(block.get("preferred_line_min", default_mode["preferred_line_min"])),
            "preferred_line_max": float(block.get("preferred_line_max", default_mode["preferred_line_max"])),
            "long_line_min": float(block.get("long_line_min", default_mode["long_line_min"])),
        }

    params_file = models_dir / "pick_params.pkl"
    if not params_file.exists():
        return default_params

    try:
        loaded = joblib.load(params_file)
        if not isinstance(loaded, dict):
            return default_params

        norm = {
            "game": _normalize_mode_block(loaded.get("game", {})),
            "q1": _normalize_mode_block(loaded.get("q1", {})),
            "h1": _normalize_mode_block(loaded.get("h1", {})),
            "spread": _normalize_mode_block(loaded.get("spread", {})),
            "total": _normalize_mode_block(loaded.get("total", {})),
            "splits": {},
        }

        loaded_splits = loaded.get("splits", {})
        if isinstance(loaded_splits, dict):
            for split_key, split_block in loaded_splits.items():
                if not isinstance(split_block, dict):
                    continue
                norm["splits"][str(split_key)] = {
                    "game": _normalize_mode_block(split_block.get("game", {})),
                    "q1": _normalize_mode_block(split_block.get("q1", {})),
                    "h1": _normalize_mode_block(split_block.get("h1", {})),
                }

        return norm
    except Exception:
        return default_params


def infer_market_missing(row: pd.Series) -> int:
    def _f(name: str):
        try:
            v = pd.to_numeric(row.get(name), errors="coerce")
            if pd.isna(v):
                return None
            return float(v)
        except Exception:
            return None

    h_spread = _f("home_spread")
    h_ml = _f("home_moneyline_odds")
    a_ml = _f("away_moneyline_odds")
    total = _f("odds_over_under")
    spread_odds = _f("closing_spread_odds")
    total_odds = _f("closing_total_odds")

    has_spread = (h_spread is not None) and (abs(h_spread) > 0)
    has_moneyline = (h_ml is not None) and (a_ml is not None)
    has_total = (total is not None) and (total > 0)
    has_prices = (spread_odds is not None) or (total_odds is not None)

    return 0 if (has_spread or has_moneyline or has_total or has_prices) else 1


def choose_runtime_split(row: pd.Series, pick_params: dict, models: dict) -> str:
    splits = pick_params.get("splits", {}) if isinstance(pick_params, dict) else {}

    has_with_market_cfg = isinstance(splits.get("with_market"), dict)
    has_no_market_cfg = isinstance(splits.get("no_market"), dict)
    has_with_market_models = bool(models.get("has_with_market_models", False))
    has_no_market_models = bool(models.get("has_no_market_models", False))

    market_missing = infer_market_missing(row)

    if market_missing == 0 and has_with_market_cfg and has_with_market_models:
        return "with_market"
    if market_missing == 1 and has_no_market_cfg and has_no_market_models:
        return "no_market"
    return "global"


def allow_spread_pick(row: pd.Series, spread_prob_home_cover: float | None, spread_params: dict) -> bool:
    if spread_prob_home_cover is None:
        return False
    try:
        h_spread = float(row.get("home_spread", 0) or 0)
    except Exception:
        return False
    if h_spread == 0:
        return False

    spread_abs = abs(h_spread)
    edge = abs(float(spread_prob_home_cover) - 0.5)
    preferred_min = float(spread_params.get("preferred_line_min", 3.5))
    preferred_max = float(spread_params.get("preferred_line_max", 6.0))
    long_line_min = float(spread_params.get("long_line_min", 10.5))
    min_edge = float(spread_params.get("min_edge", 0.07))
    alt_min_edge = float(spread_params.get("alt_min_edge", 0.085))

    if preferred_min <= spread_abs <= preferred_max:
        return edge >= min_edge
    if spread_abs >= long_line_min:
        return edge >= alt_min_edge
    return False


def get_mode_params(pick_params: dict, mode: str, split_key: str = "global") -> dict:
    default_mode = {
        "xgb_weight": 1.0 / 3.0,
        "lgb_weight": 1.0 / 3.0,
        "cat_weight": 1.0 / 3.0,
        "threshold": 0.5,
        "market_threshold": 0.515,
        "use_meta": True,
        "market_tiebreak_band": 0.10,
    }
    if split_key != "global":
        split_block = pick_params.get("splits", {}).get(split_key, {})
        if isinstance(split_block, dict) and isinstance(split_block.get(mode), dict):
            return {**default_mode, **split_block[mode]}

    top_block = pick_params.get(mode, {})
    if isinstance(top_block, dict):
        return {**default_mode, **top_block}
    return default_mode


def choose_full_game_pick(
    row: pd.Series,
    calibrated_prob_home: float,
    threshold: float,
    uncertainty_band: float = 0.035,
):
    model_pick = row["home_team"] if calibrated_prob_home >= threshold else row["away_team"]

    market_missing = infer_market_missing(row)

    if market_missing == 1:
        return model_pick, "model_no_market"

    try:
        h_spread = float(row.get("home_spread", 0) or 0)
    except Exception:
        h_spread = 0.0

    if h_spread < 0:
        market_pick = row["home_team"]
    elif h_spread > 0:
        market_pick = row["away_team"]
    else:
        market_pick = None

    if market_pick is None:
        return model_pick, "model_pickem"

    uncertainty = abs(float(calibrated_prob_home) - float(threshold))
    if uncertainty <= float(uncertainty_band):
        return market_pick, "market_tiebreak"
    return model_pick, "model_confident"


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
        if str(h_abbr).strip().upper() == "TBD" or str(a_abbr).strip().upper() == "TBD":
            continue

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
    merged = merged[
        merged["home_team"].fillna("").str.upper().ne("TBD")
        & merged["away_team"].fillna("").str.upper().ne("TBD")
    ].copy()
    return (
        merged.sort_values(["date", "time", "game_id"])
        .drop_duplicates(subset=["game_id"], keep="last")
        .reset_index(drop=True)
    )


def enrich_upcoming_games_with_market_data(df_history: pd.DataFrame, upcoming_games: pd.DataFrame):
    if upcoming_games is None or upcoming_games.empty or df_history is None or df_history.empty:
        return upcoming_games

    market_cols = [
        "game_id",
        "date",
        "home_team",
        "away_team",
        "home_spread",
        "spread_abs",
        "home_is_favorite",
        "odds_over_under",
        "home_moneyline_odds",
        "away_moneyline_odds",
        "spread_parse_success",
        "spread_is_pickem",
        "spread_missing",
        "odds_data_quality",
    ]
    available_cols = [col for col in market_cols if col in df_history.columns]
    if len(available_cols) < 5:
        return upcoming_games

    market_df = df_history[available_cols].copy()
    market_df["game_id"] = market_df["game_id"].astype(str)
    market_df["date"] = market_df["date"].astype(str)

    merged = upcoming_games.copy()
    merged["game_id"] = merged["game_id"].astype(str)
    merged["date"] = merged["date"].astype(str)

    by_game = market_df.drop_duplicates(subset=["game_id"], keep="last")
    merged = merged.merge(by_game, on="game_id", how="left", suffixes=("", "_hist"))

    by_matchup = market_df.drop_duplicates(subset=["date", "away_team", "home_team"], keep="last")
    merged = merged.merge(by_matchup, on=["date", "away_team", "home_team"], how="left", suffixes=("", "_fallback"))

    for col in available_cols:
        if col in {"game_id", "date", "home_team", "away_team"}:
            continue
        hist_col = f"{col}_hist"
        fallback_col = f"{col}_fallback"
        if hist_col in merged.columns:
            merged[col] = merged[hist_col].combine_first(merged.get(col))
        if fallback_col in merged.columns:
            merged[col] = merged[col].combine_first(merged[fallback_col])

    drop_cols = [col for col in merged.columns if col.endswith("_hist") or col.endswith("_fallback")]
    return merged.drop(columns=drop_cols, errors="ignore")


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
    # EXPERIMENTAL OFF: df_predict = add_full_game_signal_features(df_predict)

    return df_predict


def load_full_stack():
    models = {
        "xgb_game": joblib.load(MODELS_DIR / "xgb_game.pkl"),
        "lgb_game": joblib.load(MODELS_DIR / "lgb_game.pkl"),
        "cat_game": joblib.load(MODELS_DIR / "cat_game.pkl"),
        "meta_game": joblib.load(MODELS_DIR / "meta_game.pkl"),
        "xgb_q1": joblib.load(MODELS_DIR / "xgb_q1.pkl"),
        "lgb_q1": joblib.load(MODELS_DIR / "lgb_q1.pkl"),
        "cat_q1": joblib.load(MODELS_DIR / "cat_q1.pkl"),
        "meta_q1": joblib.load(MODELS_DIR / "meta_q1.pkl"),
        "xgb_h1": joblib.load(MODELS_DIR / "xgb_h1.pkl"),
        "lgb_h1": joblib.load(MODELS_DIR / "lgb_h1.pkl"),
        "cat_h1": joblib.load(MODELS_DIR / "cat_h1.pkl"),
        "meta_h1": joblib.load(MODELS_DIR / "meta_h1.pkl"),
        "feature_names": joblib.load(MODELS_DIR / "feature_names.pkl"),
    }
    models["has_with_market_models"] = False
    models["has_no_market_models"] = False

    for split_key, suffix in [("with_market", "_with_market"), ("no_market", "_no_market")]:
        required = [
            MODELS_DIR / f"xgb_game{suffix}.pkl",
            MODELS_DIR / f"lgb_game{suffix}.pkl",
            MODELS_DIR / f"cat_game{suffix}.pkl",
            MODELS_DIR / f"meta_game{suffix}.pkl",
        ]
        if not all(p.exists() for p in required):
            continue
        try:
            models[f"xgb_game{suffix}"] = joblib.load(MODELS_DIR / f"xgb_game{suffix}.pkl")
            models[f"lgb_game{suffix}"] = joblib.load(MODELS_DIR / f"lgb_game{suffix}.pkl")
            models[f"cat_game{suffix}"] = joblib.load(MODELS_DIR / f"cat_game{suffix}.pkl")
            models[f"meta_game{suffix}"] = joblib.load(MODELS_DIR / f"meta_game{suffix}.pkl")
            models[f"has_{split_key}_models"] = True
        except Exception:
            continue

    # Mercados adicionales (opcionales)
    for mk in ["spread", "total"]:
        try:
            models[f"xgb_{mk}"] = joblib.load(MODELS_DIR / f"xgb_{mk}.pkl")
            models[f"lgb_{mk}"] = joblib.load(MODELS_DIR / f"lgb_{mk}.pkl")
            models[f"cat_{mk}"] = joblib.load(MODELS_DIR / f"cat_{mk}.pkl")
            models[f"meta_{mk}"] = joblib.load(MODELS_DIR / f"meta_{mk}.pkl")
            models[f"has_{mk}_models"] = True
        except Exception:
            models[f"has_{mk}_models"] = False

    return models


def get_final_prediction(models, X_input, mode="game", suffix: str = "", mode_params: dict = None):
    key_suffix = suffix or ""
    p_xgb = models[f"xgb_{mode}{key_suffix}"].predict_proba(X_input)[:, 1]
    p_lgb = models[f"lgb_{mode}{key_suffix}"].predict_proba(X_input)[:, 1]
    p_cat = models[f"cat_{mode}{key_suffix}"].predict_proba(X_input)[:, 1]

    use_meta = True
    wx = wl = wc = 1.0 / 3.0
    if isinstance(mode_params, dict):
        use_meta = bool(mode_params.get("use_meta", True))
        wx = float(mode_params.get("xgb_weight", wx))
        wl = float(mode_params.get("lgb_weight", wl))
        wc = float(mode_params.get("cat_weight", wc))
        wsum = wx + wl + wc
        if wsum > 0:
            wx, wl, wc = wx / wsum, wl / wsum, wc / wsum
        else:
            wx = wl = wc = 1.0 / 3.0

    if use_meta:
        X_meta = np.column_stack([p_xgb, p_lgb, p_cat])
        final_prob = models[f"meta_{mode}{key_suffix}"].predict_proba(X_meta)[:, 1]
    else:
        final_prob = (wx * p_xgb) + (wl * p_lgb) + (wc * p_cat)
    return final_prob


def predict_today():
    print("🔮 INICIANDO MOTOR IA NBA...")

    try:
        models = load_full_stack()
        feature_names = models.get("feature_names", joblib.load(MODELS_DIR / "feature_names.pkl"))
        pick_params = load_pick_params(MODELS_DIR)
    except Exception as e:
        print(f"❌ Faltan modelos o feature_names.pkl. Error: {e}")
        return

    if not RAW_DATA.exists():
        print(f"❌ No existe el histórico raw: {RAW_DATA}")
        return

    df_history = pd.read_csv(RAW_DATA)
    upcoming_games = fetch_upcoming_games(days_ahead=14)
    upcoming_games = enrich_upcoming_games_with_market_data(df_history, upcoming_games)

    if upcoming_games.empty:
        print("📭 No hay partidos en ventana rolling NBA.")
        return

    q1_params = get_mode_params(pick_params, mode="q1", split_key="global")
    q1_th = float(q1_params["threshold"])
    h1_params = get_mode_params(pick_params, mode="h1", split_key="global")
    h1_th = float(h1_params["threshold"])
    spread_params = get_mode_params(pick_params, mode="spread", split_key="global")
    total_params = get_mode_params(pick_params, mode="total", split_key="global")

    calibration_cfg = load_calibration_config(CALIBRATION_FILE)
    error_risk_bundle = load_error_risk_bundle(ERROR_RISK_MODEL_FILE)

    total_games = 0
    for date_str, games_df in upcoming_games.groupby("date"):
        target_date = pd.Timestamp(date_str)
        df_predict = build_prediction_features(df_history, games_df, target_date)
        # Enriquecer con columnas ya calculadas por feature_engineering si están en el CSV procesado
        try:
            proc = pd.read_csv(PROCESSED_DATA)
            proc_day = proc[proc["date"] == str(target_date.date())]
            if not proc_day.empty:
                proc_day = proc_day[proc_day["game_id"].astype(str).isin(df_predict["game_id"].astype(str))]
                if not proc_day.empty:
                    proc_day = proc_day.set_index(proc_day["game_id"].astype(str))
                    df_predict = df_predict.set_index(df_predict["game_id"].astype(str))
                    df_predict = proc_day.combine_first(df_predict)
                    df_predict = df_predict.reset_index(drop=True)
        except Exception:
            pass

        X_day = df_predict.reindex(columns=feature_names, fill_value=0)

        # Mantener Q1 en ruta global para no afectar comportamiento actual
        prob_q1 = get_final_prediction(models, X_day, mode="q1", suffix="", mode_params=q1_params)
        prob_h1 = get_final_prediction(models, X_day, mode="h1", suffix="", mode_params=h1_params)

        print("======================================================")
        print(f"🏀 CARTELERA {LEAGUE_LABEL} {date_str} ({len(df_predict)} juegos)")
        print("======================================================")

        predictions_output = []

        for i, row in df_predict.iterrows():
            split_key = choose_runtime_split(row, pick_params, models)
            split_suffix = "" if split_key == "global" else f"_{split_key}"
            game_params = get_mode_params(pick_params, mode="game", split_key=split_key)
            game_th = float(game_params["threshold"])

            X_row = X_day.iloc[[i]]
            model_prob_game_home = float(
                get_final_prediction(
                    models=models,
                    X_input=X_row,
                    mode="game",
                    suffix=split_suffix,
                    mode_params=game_params,
                )[0]
            )
            calibrated_prob_game_home = calibrate_probability(
                model_prob_game_home,
                sport=SPORT_KEY,
                market="full_game",
                calibration_config=calibration_cfg,
            )
            prob_game_home = calibrated_prob_game_home * 100
            prob_game_away = 100 - prob_game_home
            market_threshold = float(game_params.get("market_threshold", game_th))
            try:
                row_home_spread = float(row.get("home_spread", 0) or 0)
            except Exception:
                row_home_spread = 0.0
            row_has_market_side = abs(row_home_spread) > 0
            threshold_for_pick = market_threshold if row_has_market_side else game_th
            ganador_juego, full_pick_rule = choose_full_game_pick(
                row=row,
                calibrated_prob_home=calibrated_prob_game_home,
                threshold=threshold_for_pick,
                uncertainty_band=float(game_params.get("market_tiebreak_band", 0.10)),
            )
            ganador_juego, full_pick_rule, error_risk_prob = apply_error_risk_override(
                bundle=error_risk_bundle,
                row=row,
                calibrated_prob_home=calibrated_prob_game_home,
                threshold_used=threshold_for_pick,
                current_pick=ganador_juego,
                current_rule=full_pick_rule,
            )
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

            model_prob_h1_home = float(prob_h1[i])
            calibrated_prob_h1_home = calibrate_probability(
                model_prob_h1_home,
                sport=SPORT_KEY,
                market="h1",
                calibration_config=calibration_cfg,
            )
            prob_h1_home = calibrated_prob_h1_home * 100
            prob_h1_away = 100 - prob_h1_home
            ganador_h1 = row["home_team"] if (calibrated_prob_h1_home >= h1_th) else row["away_team"]
            conf_h1 = max(prob_h1_home, prob_h1_away)
            h1_action = "JUGAR 1H" if conf_h1 >= 58 else "PASAR 1H"

            total_line = float(row.get("odds_over_under", 0) or 0)
            h_spread = float(row.get("home_spread", 0) or 0)

            spread_pick = "Pendiente"
            spread_prob_home_cover = None
            if models.get("has_spread_models", False) and abs(h_spread) > 0:
                model_prob_home_cover = float(
                    get_final_prediction(
                        models=models,
                        X_input=X_row,
                        mode="spread",
                        suffix="",
                        mode_params=spread_params,
                    )[0]
                )
                calibrated_prob_home_cover = calibrate_probability(
                    model_prob_home_cover, sport=SPORT_KEY, market="spread", calibration_config=calibration_cfg
                )
                spread_prob_home_cover = float(calibrated_prob_home_cover)
                if allow_spread_pick(row, spread_prob_home_cover, spread_params):
                    spread_th = float(spread_params.get("threshold", 0.5))
                    spread_pick = f"{row['home_team']} {h_spread:+g}" if calibrated_prob_home_cover >= spread_th else f"{row['away_team']} {-h_spread:+g}"
                else:
                    spread_pick = "N/A"
            elif abs(h_spread) > 0:
                spread_pick = "N/A"

            total_pick = "Pendiente"
            total_prob_over = None
            if models.get("has_total_models", False) and total_line > 0:
                model_prob_over = float(
                    get_final_prediction(
                        models=models,
                        X_input=X_row,
                        mode="total",
                        suffix="",
                        mode_params=total_params,
                    )[0]
                )
                calibrated_prob_over = calibrate_probability(
                    model_prob_over, sport=SPORT_KEY, market="total", calibration_config=calibration_cfg
                )
                total_prob_over = float(calibrated_prob_over)
                total_th = float(total_params.get("threshold", 0.5))
                total_pick = f"OVER {total_line}" if calibrated_prob_over >= total_th else f"UNDER {total_line}"
            elif total_line > 0:
                total_pick = f"Lean total {total_line}"

            assists_pick = "Pendiente"
            home_ml_odds = row.get("home_moneyline_odds")
            away_ml_odds = row.get("away_moneyline_odds")
            selected_ml_odds = None
            if ganador_juego == row["home_team"]:
                selected_ml_odds = home_ml_odds
            elif ganador_juego == row["away_team"]:
                selected_ml_odds = away_ml_odds

            print(f"👉 {row['game_name']} | Spread Vegas: {row['spread']}")
            print(f"   ⏱️ 1er Cuarto: Gana {ganador_q1} (score modelo: {conf_q1:.1f}%) | {q1_action_label}")
            print(f"   🕐 1ra Mitad:  Gana {ganador_h1} (score modelo: {conf_h1:.1f}%) | {h1_action}")
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
                    "home_spread": h_spread,
                    "spread_abs": float(row.get("spread_abs", 0) or 0),
                    "odds_over_under": total_line,
                    "home_moneyline_odds": float(home_ml_odds) if pd.notna(home_ml_odds) else None,
                    "away_moneyline_odds": float(away_ml_odds) if pd.notna(away_ml_odds) else None,
                    "moneyline_odds": round(float(selected_ml_odds), 4) if selected_ml_odds is not None and pd.notna(selected_ml_odds) else None,
                    "pick_ml_odds": round(float(selected_ml_odds), 4) if selected_ml_odds is not None and pd.notna(selected_ml_odds) else None,
                    "market_missing": int(infer_market_missing(row)),
                    "full_game_pick": ganador_juego,
                    "full_game_pick_rule": full_pick_rule,
                    "full_game_error_risk_prob": round(float(error_risk_prob), 4) if error_risk_prob is not None else None,
                    "full_game_confidence": round(conf_juego, 1),
                    "full_game_tier": tier_game,
                    "full_game_model_prob_home": round(model_prob_game_home, 4),
                    "full_game_calibrated_prob_home": round(calibrated_prob_game_home, 4),
                    "full_game_model_route": split_key,
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
                    "h1_pick": ganador_h1,
                    "h1_confidence": round(conf_h1, 1),
                    "h1_action": h1_action,
                    "h1_model_prob_home": round(model_prob_h1_home, 4),
                    "h1_calibrated_prob_home": round(calibrated_prob_h1_home, 4),
                    "h1_prob_home": round(prob_h1_home, 2),
                    "h1_prob_away": round(prob_h1_away, 2),
                    "total_pick": total_pick,
                    "total_prob_over": round(float(total_prob_over), 4) if total_prob_over is not None else None,
                    "spread_pick": spread_pick,
                    "spread_prob_home_cover": round(float(spread_prob_home_cover), 4) if spread_prob_home_cover is not None else None,
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
