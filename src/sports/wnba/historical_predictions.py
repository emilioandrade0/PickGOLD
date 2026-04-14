import json
import math
import os
from pathlib import Path

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from tqdm import tqdm
try:
    from calibration import calibrate_probability, load_calibration_config
except Exception:
    # Fallbacks cuando no exista el módulo de calibración en el entorno.
    def calibrate_probability(p, sport=None, market=None, calibration_config=None):
        try:
            return float(p)
        except Exception:
            return p

    def load_calibration_config(path):
        return None

# --- RUTAS ---
import sys
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
RAW_DATA = BASE_DIR / "data" / "wnba" / "raw" / "wnba_advanced_history.csv"
PROCESSED_DATA = BASE_DIR / "data" / "wnba" / "processed" / "model_ready_features.csv"
MODELS_DIR = BASE_DIR / "data" / "wnba" / "models"
HIST_PRED_DIR = BASE_DIR / "data" / "wnba" / "historical_predictions"
HIST_PRED_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_FILE = MODELS_DIR / "calibration_params.json"
ERROR_RISK_MODEL_FILE = MODELS_DIR / "error_risk_fullgame.pkl"

try:
    from sports.wnba.error_risk_utils import apply_error_risk_override, load_error_risk_bundle
except Exception:
    def load_error_risk_bundle(path):
        return None

    def apply_error_risk_override(bundle, row, calibrated_prob_home, threshold_used, current_pick, current_rule):
        return current_pick, current_rule, None

TEAM_CONF_DIV = {
    "ATL": ("EAST", "EAST"),
    "CHI": ("EAST", "EAST"),
    "CON": ("EAST", "EAST"),
    "DAL": ("WEST", "WEST"),
    "GSV": ("WEST", "WEST"),
    "IND": ("EAST", "EAST"),
    "LA": ("WEST", "WEST"),
    "LAS": ("WEST", "WEST"),
    "LV": ("WEST", "WEST"),
    "LVA": ("WEST", "WEST"),
    "MIN": ("WEST", "WEST"),
    "NY": ("EAST", "EAST"),
    "NYL": ("EAST", "EAST"),
    "PHO": ("WEST", "WEST"),
    "PHX": ("WEST", "WEST"),
    "SEA": ("WEST", "WEST"),
    "WAS": ("EAST", "EAST"),
    "WSH": ("EAST", "EAST"),
}

TEAM_TIMEZONE = {
    "ATL": "ET",
    "CHI": "CT",
    "CON": "ET",
    "DAL": "CT",
    "GSV": "PT",
    "IND": "ET",
    "LA": "PT",
    "LAS": "PT",
    "LV": "PT",
    "LVA": "PT",
    "MIN": "CT",
    "NY": "ET",
    "NYL": "ET",
    "PHO": "MT",
    "PHX": "MT",
    "SEA": "PT",
    "WAS": "ET",
    "WSH": "ET",
}

TZ_OFFSET = {"ET": 0, "CT": -1, "MT": -2, "PT": -3}

class ConstantBinaryModel:
    def __init__(self, prob_class_1: float):
        self.prob_class_1 = float(min(max(prob_class_1, 0.0), 1.0))

    def predict_proba(self, X):
        n = len(X)
        p1 = np.full(n, self.prob_class_1, dtype=float)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

def days_to_playoffs(game_date: pd.Timestamp) -> int:
    playoff_year = game_date.year + 1 if game_date.month >= 7 else game_date.year
    playoff_start = pd.Timestamp(playoff_year, 4, 15)
    return int(max((playoff_start - game_date).days, 0))

def american_to_decimal(american_odds):
    """Convierte momios americanos a decimales para la UI."""
    try:
        val = float(american_odds)
        if val == 0 or pd.isna(val): return "N/A"
        if 1.0 < val < 25.0:
            return f"{val:.2f}"
        if val > 0:
            return f"{(val / 100 + 1):.2f}"
        else:
            return f"{(100 / abs(val) + 1):.2f}"
    except:
        return "N/A"

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
    df_predict["interconference_travel"] = ((df_predict["same_conference"] == 0) & (df_predict["away_tz_diff"] >= 2)).astype(int)
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


def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
    return obj


def safe_int(x, default=0):
    try:
        if x is None:
            return int(default)
        if pd.isna(x):
            return int(default)
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return int(default)

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([float("inf"), float("-inf")], 0).infer_objects(copy=False).fillna(0)


def get_mode_params(pick_params: dict, mode: str) -> dict:
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
    if not isinstance(pick_params, dict):
        return default_mode

    block = pick_params.get(mode, {})
    if not isinstance(block, dict):
        return default_mode

    if "weights" in block and isinstance(block["weights"], dict):
        wx = float(block["weights"].get("xgb", default_mode["xgb_weight"]))
        wl = float(block["weights"].get("lgb", default_mode["lgb_weight"]))
        wc = float(block["weights"].get("cat", default_mode["cat_weight"]))
    else:
        wx = float(block.get("xgb_weight", default_mode["xgb_weight"]))
        wl = float(block.get("lgb_weight", default_mode["lgb_weight"]))
        wc = float(block.get("cat_weight", default_mode["cat_weight"]))

    wsum = wx + wl + wc
    if wsum <= 0:
        wx = wl = wc = 1.0 / 3.0
    else:
        wx, wl, wc = wx / wsum, wl / wsum, wc / wsum

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

def get_pick_tier(conf: float, row: dict = None) -> str:
    """
    Determina el `tier` de un pick a partir de `conf` y (opcional) contexto del `row`.
    Cambios: ELITE más estricto — threshold subido de 70 -> 74.
    Además, si se pasa `row`, se evita marcar ELITE en casos de alta volatilidad,
    falta de mercado o fatiga evidente.
    """
    # Nuevo umbral ELITE (más conservador)
    ELITE_THRESHOLD = 74.0

    # Si no hay contexto, aplicar solo el threshold
    if row is None:
        if conf >= ELITE_THRESHOLD:
            return "ELITE"
        if conf >= 65:
            return "PREMIUM"
        if conf >= 60:
            return "STRONG"
        if conf >= 57:
            return "NORMAL"
        return "PASS"

    # Si tenemos contexto, evaluamos condiciones adicionales
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

    # Fallback: mantener los thresholds anteriores para otras categorías
    if conf >= 65:
        return "PREMIUM"
    if conf >= 60:
        return "STRONG"
    if conf >= 57:
        return "NORMAL"
    return "PASS"

def calculate_elo_map(df: pd.DataFrame, k: float = 20, home_advantage: float = 100):
    elo_dict = {}
    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date_dt", "game_id"]).reset_index(drop=True)
    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        if home not in elo_dict: elo_dict[home] = 1500.0
        if away not in elo_dict: elo_dict[away] = 1500.0
        elo_diff = elo_dict[away] - (elo_dict[home] + home_advantage)
        expected_home = 1 / (1 + 10 ** (elo_diff / 400))
        actual_home = 1 if row["home_pts_total"] > row["away_pts_total"] else 0
        elo_dict[home] += k * (actual_home - expected_home)
        elo_dict[away] += k * ((1 - actual_home) - (1 - expected_home))
    return elo_dict

def get_current_team_stats(df_history: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    home_df = df_history[["date", "game_id", "home_team", "home_pts_total", "away_pts_total", "home_q1", "away_q1"]].copy()
    home_df.columns = ["date", "game_id", "team", "pts_scored", "pts_conceded", "q1_scored", "q1_conceded"]
    home_df["is_home"] = 1

    away_df = df_history[["date", "game_id", "away_team", "away_pts_total", "home_pts_total", "away_q1", "home_q1"]].copy()
    away_df.columns = ["date", "game_id", "team", "pts_scored", "pts_conceded", "q1_scored", "q1_conceded"]
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
        l5, l10 = group.tail(5), group.tail(10)
        if l10.empty: continue
        rest_days = min(max((target_date - group["date_dt"].iloc[-1]).days, 0), 7)
        day_diffs = (target_date - group["date_dt"]).dt.days
        home_only = group[group["is_home"] == 1].tail(5)
        away_only = group[group["is_home"] == 0].tail(5)

        latest_stats.append({
            "team": team,
            "rest_days": rest_days,
            "is_b2b": int(rest_days == 1),
            "games_last_3_days": int(day_diffs.between(1, 3).sum()),
            "games_last_5_days": int(day_diffs.between(1, 5).sum()),
            "games_last_7_days": int(day_diffs.between(1, 7).sum()),
            "win_pct_L5": l5["won_game"].mean(),
            "win_pct_L10": l10["won_game"].mean(),
            "pts_diff_L5": l5["pts_diff"].mean(),
            "pts_diff_L10": l10["pts_diff"].mean(),
            "q1_win_pct_L5": l5["won_q1"].mean(),
            "q1_diff_L5": l5["q1_diff"].mean(),
            "pts_scored_L5": l5["pts_scored"].mean(),
            "pts_conceded_L5": l5["pts_conceded"].mean(),
            "home_only_win_pct_L5": home_only["won_game"].mean() if not home_only.empty else 0.0,
            "home_only_pts_diff_L5": home_only["pts_diff"].mean() if not home_only.empty else 0.0,
            "home_only_q1_win_pct_L5": home_only["won_q1"].mean() if not home_only.empty else 0.0,
            "away_only_win_pct_L5": away_only["won_game"].mean() if not away_only.empty else 0.0,
            "away_only_pts_diff_L5": away_only["pts_diff"].mean() if not away_only.empty else 0.0,
            "away_only_q1_win_pct_L5": away_only["won_q1"].mean() if not away_only.empty else 0.0,
        })
    return clean_dataframe(pd.DataFrame(latest_stats))

def get_matchup_stats(df_history: pd.DataFrame, home_team: str, away_team: str) -> dict:
    played = df_history[(((df_history["home_team"] == home_team) & (df_history["away_team"] == away_team)) | ((df_history["home_team"] == away_team) & (df_history["away_team"] == home_team)))].copy()
    if played.empty:
        return {"matchup_home_win_pct_L5": 0.5, "matchup_home_win_pct_L10": 0.5, "matchup_home_q1_win_pct_L5": 0.5, "matchup_home_pts_diff_L5": 0.0}
    
    played["date_dt"] = pd.to_datetime(played["date"])
    played = played.sort_values(["date_dt", "game_id"]).reset_index(drop=True)
    played["full_winner"] = np.where(played["home_pts_total"] > played["away_pts_total"], played["home_team"], played["away_team"])
    played["q1_winner"] = np.where(played["home_q1"] > played["away_q1"], played["home_team"], played["away_team"])
    played["home_team_pts_diff_from_current_home"] = np.where(played["home_team"] == home_team, played["home_pts_total"] - played["away_pts_total"], played["away_pts_total"] - played["home_pts_total"])

    l5, l10 = played.tail(5), played.tail(10)
    return {
        "matchup_home_win_pct_L5": float((l5["full_winner"] == home_team).mean()) if len(l5) else 0.5,
        "matchup_home_win_pct_L10": float((l10["full_winner"] == home_team).mean()) if len(l10) else 0.5,
        "matchup_home_q1_win_pct_L5": float((l5["q1_winner"] == home_team).mean()) if len(l5) else 0.5,
        "matchup_home_pts_diff_L5": float(l5["home_team_pts_diff_from_current_home"].mean()) if len(l5) else 0.0,
    }

def build_prediction_features_for_games(train_history: pd.DataFrame, target_games: pd.DataFrame, target_date: pd.Timestamp):
    elo_map = calculate_elo_map(train_history)
    current_stats = get_current_team_stats(train_history, target_date)

    df_predict = target_games.copy()
    df_predict["home_elo_pre"] = df_predict["home_team"].map(lambda t: elo_map.get(t, 1500.0))
    df_predict["away_elo_pre"] = df_predict["away_team"].map(lambda t: elo_map.get(t, 1500.0))

    df_predict = pd.merge(df_predict, current_stats, left_on="home_team", right_on="team", how="left").rename(columns={c: f"home_{c}" for c in current_stats.columns if c != "team"}).drop(columns=["team"])
    df_predict = pd.merge(df_predict, current_stats, left_on="away_team", right_on="team", how="left").rename(columns={c: f"away_{c}" for c in current_stats.columns if c != "team"}).drop(columns=["team"])

    fill_cols = ["home_rest_days", "away_rest_days", "home_is_b2b", "away_is_b2b", "home_games_last_3_days", "away_games_last_3_days", "home_games_last_5_days", "away_games_last_5_days", "home_games_last_7_days", "away_games_last_7_days", "home_win_pct_L5", "away_win_pct_L5", "home_win_pct_L10", "away_win_pct_L10", "home_pts_diff_L5", "away_pts_diff_L5", "home_pts_diff_L10", "away_pts_diff_L10", "home_q1_win_pct_L5", "away_q1_win_pct_L5", "home_q1_diff_L5", "away_q1_diff_L5", "home_pts_scored_L5", "away_pts_scored_L5", "home_pts_conceded_L5", "away_pts_conceded_L5", "home_home_only_win_pct_L5", "away_away_only_win_pct_L5", "home_home_only_pts_diff_L5", "away_away_only_pts_diff_L5", "home_home_only_q1_win_pct_L5", "away_away_only_q1_win_pct_L5"]
    for col in fill_cols: df_predict[col] = df_predict[col].fillna(0.0) if col in df_predict.columns else 0.0

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
    df_predict["diff_surface_win_pct_L5"] = df_predict["home_home_only_win_pct_L5"] - df_predict["away_away_only_win_pct_L5"]
    df_predict["diff_surface_pts_diff_L5"] = df_predict["home_home_only_pts_diff_L5"] - df_predict["away_away_only_pts_diff_L5"]
    df_predict["diff_surface_q1_win_pct_L5"] = df_predict["home_home_only_q1_win_pct_L5"] - df_predict["away_away_only_q1_win_pct_L5"]

    # Precompute matchup stats once per unique matchup to avoid repeated filtering
    matchup_keys = df_predict.apply(lambda r: (r["home_team"], r["away_team"]), axis=1).tolist()
    unique_pairs = list({(h, a) for h, a in matchup_keys})
    matchup_map = {}
    for h, a in unique_pairs:
        matchup_map[(h, a)] = get_matchup_stats(train_history, h, a)

    df_matchup = pd.DataFrame([matchup_map[(r["home_team"], r["away_team"])] for _, r in df_predict.iterrows()])
    for col in df_matchup.columns:
        df_predict[col] = df_matchup[col].values

    df_predict["diff_matchup_home_edge_L5"] = (df_predict["matchup_home_win_pct_L5"] - 0.5) * 2
    df_predict["diff_matchup_home_edge_L10"] = (df_predict["matchup_home_win_pct_L10"] - 0.5) * 2

    league_win_pct_L10 = float(current_stats["win_pct_L10"].mean()) if not current_stats.empty else 0.5
    league_pts_diff_L10 = float(current_stats["pts_diff_L10"].mean()) if not current_stats.empty else 0.0
    league_q1_win_pct_L5 = float(current_stats["q1_win_pct_L5"].mean()) if not current_stats.empty else 0.5

    df_predict["home_win_pct_L10_vs_league"] = df_predict["home_win_pct_L10"] - league_win_pct_L10
    df_predict["away_win_pct_L10_vs_league"] = df_predict["away_win_pct_L10"] - league_win_pct_L10
    df_predict["diff_win_pct_L10_vs_league"] = df_predict["home_win_pct_L10_vs_league"] - df_predict["away_win_pct_L10_vs_league"]
    df_predict["home_pts_diff_L10_vs_league"] = df_predict["home_pts_diff_L10"] - league_pts_diff_L10
    df_predict["away_pts_diff_L10_vs_league"] = df_predict["away_pts_diff_L10"] - league_pts_diff_L10
    df_predict["diff_pts_diff_L10_vs_league"] = df_predict["home_pts_diff_L10_vs_league"] - df_predict["away_pts_diff_L10_vs_league"]
    df_predict["home_q1_win_pct_L5_vs_league"] = df_predict["home_q1_win_pct_L5"] - league_q1_win_pct_L5
    df_predict["away_q1_win_pct_L5_vs_league"] = df_predict["away_q1_win_pct_L5"] - league_q1_win_pct_L5
    df_predict["diff_q1_win_pct_L5_vs_league"] = df_predict["home_q1_win_pct_L5_vs_league"] - df_predict["away_q1_win_pct_L5_vs_league"]

    df_predict = add_context_features(df_predict, target_date)
    # --- NUEVO: Lógica de Lesiones para Predicción ---
    # Como target_games ya trae las columnas del CSV, solo calculamos los diferenciales
    df_predict["diff_injuries"] = df_predict["home_injuries_count"] - df_predict["away_injuries_count"]
    df_predict["injury_advantage_home"] = (df_predict["diff_injuries"] <= -3).astype(int)
    df_predict["injury_advantage_away"] = (df_predict["diff_injuries"] >= 3).astype(int)
    
    # Aseguramos que home_is_favorite sea numérico para la interacción
    df_predict["home_is_favorite"] = pd.to_numeric(df_predict["home_is_favorite"], errors='coerce').fillna(0)
    df_predict["injury_surprise_upset"] = ((df_predict["injury_advantage_home"] == 1) & (df_predict["home_is_favorite"] == 0)).astype(int)

    # EXPERIMENTAL OFF: df_predict = add_full_game_signal_features(df_predict)
    return clean_dataframe(df_predict)

def train_models_from_past(train_df: pd.DataFrame):
    # LISTA MAESTRA PARA BLOQUEAR LA "TRAMPA"
    cols_to_drop = [
        "game_id", "date", "season", "home_team", "away_team",
        "TARGET_home_win", "TARGET_home_win_q1", "TARGET_home_win_h1", "TARGET_home_cover_spread", "TARGET_over_total",
        "home_pts_total", "away_pts_total", "home_q1", "away_q1"
    ]
    
    train_df = clean_dataframe(train_df).sort_values("date").reset_index(drop=True)

    # Solo dropeamos las que existan para evitar errores
    existing_drop = [c for c in cols_to_drop if c in train_df.columns]
    X = train_df.drop(columns=existing_drop)
    
    # --- SELECCIÓN DINÁMICA (solo columnas numéricas) ---
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        num_selector = X[num_cols].var() > 1e-6
        keep_numeric = num_selector[num_selector].index.tolist()
    else:
        keep_numeric = []
    X = X.loc[:, keep_numeric]

    y_game = train_df["TARGET_home_win"].astype(int)
    y_q1 = train_df["TARGET_home_win_q1"].astype(int)
    y_h1 = pd.to_numeric(train_df.get("TARGET_home_win_h1"), errors="coerce")
    y_spread = pd.to_numeric(train_df.get("TARGET_home_cover_spread"), errors="coerce")
    y_total = pd.to_numeric(train_df.get("TARGET_over_total"), errors="coerce")

    # El resto de la función (X_train, fit, etc.) se queda igual...
    split_idx = int(len(train_df) * 0.85)
    X_train, X_calib = X.iloc[:split_idx], X.iloc[split_idx:]
    yg_train, yg_calib = y_game.iloc[:split_idx], y_game.iloc[split_idx:]
    yq_train, yq_calib = y_q1.iloc[:split_idx], y_q1.iloc[split_idx:]
    yh_train, yh_calib = y_h1.iloc[:split_idx], y_h1.iloc[split_idx:]

    # Modelos base (CPU Multi-hilo)
    xgb_g = xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=4, subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", random_state=42, n_jobs=-1, base_score=0.5)
    lgb_g = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=4, num_leaves=31, subsample=0.85, colsample_bytree=0.85, random_state=42, verbosity=-1, n_jobs=-1)
    cat_g = CatBoostClassifier(iterations=300, learning_rate=0.03, depth=4, subsample=0.85, random_state=42, verbose=0, thread_count=-1)

    xgb_q = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", random_state=42, n_jobs=-1, base_score=0.5)
    lgb_q = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, num_leaves=15, subsample=0.85, colsample_bytree=0.85, random_state=42, verbosity=-1, n_jobs=-1)
    cat_q = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=3, subsample=0.85, random_state=42, verbose=0, thread_count=-1)
    xgb_h = xgb.XGBClassifier(n_estimators=220, learning_rate=0.04, max_depth=3, subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", random_state=42, n_jobs=-1, base_score=0.5)
    lgb_h = lgb.LGBMClassifier(n_estimators=220, learning_rate=0.04, max_depth=3, num_leaves=15, subsample=0.85, colsample_bytree=0.85, random_state=42, verbosity=-1, n_jobs=-1)
    cat_h = CatBoostClassifier(iterations=220, learning_rate=0.04, depth=3, subsample=0.85, random_state=42, verbose=0, thread_count=-1)

    models = {"feature_names": X.columns.tolist()}

    # --- LÓGICA PARTIDO COMPLETO ---
    if yg_train.nunique() < 2:
        val = 1.0 if int(yg_train.iloc[0]) == 1 else 0.0
        models["xgb_game"] = ConstantBinaryModel(val); models["lgb_game"] = ConstantBinaryModel(val)
        models["cat_game"] = ConstantBinaryModel(val); models["meta_game"] = ConstantBinaryModel(val)
    else:
        xgb_g.fit(X_train, yg_train); lgb_g.fit(X_train, yg_train); cat_g.fit(X_train, yg_train)
        models["xgb_game"] = xgb_g; models["lgb_game"] = lgb_g; models["cat_game"] = cat_g
        
        p_x = xgb_g.predict_proba(X_calib)[:, 1]
        p_l = lgb_g.predict_proba(X_calib)[:, 1]
        p_c = cat_g.predict_proba(X_calib)[:, 1]
        
        if yg_calib.nunique() < 2:
            models["meta_game"] = ConstantBinaryModel(1.0 if int(yg_calib.iloc[0]) == 1 else 0.0)
        else:
            # AQUÍ VA EL BLOQUE SIGMOIDE PARA PARTIDO COMPLETO
            meta_base = LogisticRegression()
            meta_sig = CalibratedClassifierCV(estimator=meta_base, method='sigmoid', cv=min(5, len(yg_calib)//2))
            meta_sig.fit(np.column_stack([p_x, p_l, p_c]), yg_calib)
            models["meta_game"] = meta_sig

    # --- LÓGICA PRIMER CUARTO ---
    if yq_train.nunique() < 2:
        val = 1.0 if int(yq_train.iloc[0]) == 1 else 0.0
        models["xgb_q1"] = ConstantBinaryModel(val); models["lgb_q1"] = ConstantBinaryModel(val)
        models["cat_q1"] = ConstantBinaryModel(val); models["meta_q1"] = ConstantBinaryModel(val)
    else:
        xgb_q.fit(X_train, yq_train); lgb_q.fit(X_train, yq_train); cat_q.fit(X_train, yq_train)
        models["xgb_q1"] = xgb_q; models["lgb_q1"] = lgb_q; models["cat_q1"] = cat_q
        
        p_x_q = xgb_q.predict_proba(X_calib)[:, 1]
        p_l_q = lgb_q.predict_proba(X_calib)[:, 1]
        p_c_q = cat_q.predict_proba(X_calib)[:, 1]
        
        if yq_calib.nunique() < 2:
            models["meta_q1"] = ConstantBinaryModel(1.0 if int(yq_calib.iloc[0]) == 1 else 0.0)
        else:
            # AQUÍ VA EL BLOQUE SIGMOIDE PARA Q1
            meta_base_q = LogisticRegression()
            meta_sig_q = CalibratedClassifierCV(estimator=meta_base_q, method='sigmoid', cv=min(5, len(yq_calib)//2))
            meta_sig_q.fit(np.column_stack([p_x_q, p_l_q, p_c_q]), yq_calib)
            models["meta_q1"] = meta_sig_q

    # --- LÓGICA PRIMERA MITAD ---
    mask_h1 = y_h1.notna()
    X_h1 = X.loc[mask_h1]
    y_h1_clean = y_h1.loc[mask_h1].astype(int)
    split_h1 = int(len(X_h1) * 0.85) if len(X_h1) else 0
    X_h1_train, X_h1_calib = X_h1.iloc[:split_h1], X_h1.iloc[split_h1:]
    yh_train_clean, yh_calib_clean = y_h1_clean.iloc[:split_h1], y_h1_clean.iloc[split_h1:]
    if len(X_h1_train) == 0 or yh_train_clean.nunique() < 2:
        val = float(y_h1_clean.mean()) if len(y_h1_clean) else 0.5
        models["xgb_h1"] = ConstantBinaryModel(val); models["lgb_h1"] = ConstantBinaryModel(val)
        models["cat_h1"] = ConstantBinaryModel(val); models["meta_h1"] = ConstantBinaryModel(val)
    else:
        xgb_h.fit(X_h1_train, yh_train_clean); lgb_h.fit(X_h1_train, yh_train_clean); cat_h.fit(X_h1_train, yh_train_clean)
        models["xgb_h1"] = xgb_h; models["lgb_h1"] = lgb_h; models["cat_h1"] = cat_h

        p_x_h = xgb_h.predict_proba(X_h1_calib)[:, 1]
        p_l_h = lgb_h.predict_proba(X_h1_calib)[:, 1]
        p_c_h = cat_h.predict_proba(X_h1_calib)[:, 1]

        if len(yh_calib_clean) == 0 or yh_calib_clean.nunique() < 2:
            models["meta_h1"] = ConstantBinaryModel(float(yh_train_clean.mean()))
        else:
            meta_base_h = LogisticRegression()
            meta_sig_h = CalibratedClassifierCV(estimator=meta_base_h, method='sigmoid', cv=min(5, max(2, len(yh_calib_clean)//2)))
            meta_sig_h.fit(np.column_stack([p_x_h, p_l_h, p_c_h]), yh_calib_clean)
            models["meta_h1"] = meta_sig_h

    def _fit_optional_market(y_ser: pd.Series, market_key: str):
        mask = y_ser.notna()
        X_m = X.loc[mask]
        y_m = y_ser.loc[mask].astype(int)
        if len(X_m) < 200 or y_m.nunique() < 2:
            models[f"has_{market_key}_models"] = False
            return

        split_m = int(len(X_m) * 0.85)
        X_m_train, X_m_calib = X_m.iloc[:split_m], X_m.iloc[split_m:]
        y_m_train, y_m_calib = y_m.iloc[:split_m], y_m.iloc[split_m:]

        if len(X_m_train) == 0 or y_m_train.nunique() < 2:
            val = float(y_m.mean()) if len(y_m) else 0.5
            models[f"xgb_{market_key}"] = ConstantBinaryModel(val)
            models[f"lgb_{market_key}"] = ConstantBinaryModel(val)
            models[f"cat_{market_key}"] = ConstantBinaryModel(val)
            models[f"meta_{market_key}"] = ConstantBinaryModel(val)
            models[f"has_{market_key}_models"] = True
            return

        xgb_m = xgb.XGBClassifier(
            n_estimators=220, learning_rate=0.04, max_depth=4, subsample=0.85,
            colsample_bytree=0.85, eval_metric="logloss", random_state=42, n_jobs=-1, base_score=0.5
        )
        lgb_m = lgb.LGBMClassifier(
            n_estimators=220, learning_rate=0.04, max_depth=4, num_leaves=31,
            subsample=0.85, colsample_bytree=0.85, random_state=42, verbosity=-1, n_jobs=-1
        )
        cat_m = CatBoostClassifier(
            iterations=220, learning_rate=0.04, depth=4, subsample=0.85,
            random_state=42, verbose=0, thread_count=-1
        )
        xgb_m.fit(X_m_train, y_m_train)
        lgb_m.fit(X_m_train, y_m_train)
        cat_m.fit(X_m_train, y_m_train)

        p_x_m = xgb_m.predict_proba(X_m_calib)[:, 1]
        p_l_m = lgb_m.predict_proba(X_m_calib)[:, 1]
        p_c_m = cat_m.predict_proba(X_m_calib)[:, 1]

        if y_m_calib.nunique() < 2:
            meta_m = ConstantBinaryModel(1.0 if int(y_m_calib.iloc[0]) == 1 else 0.0)
        else:
            meta_base_m = LogisticRegression()
            meta_m = CalibratedClassifierCV(estimator=meta_base_m, method='sigmoid', cv=min(5, max(2, len(y_m_calib)//2)))
            meta_m.fit(np.column_stack([p_x_m, p_l_m, p_c_m]), y_m_calib)

        models[f"xgb_{market_key}"] = xgb_m
        models[f"lgb_{market_key}"] = lgb_m
        models[f"cat_{market_key}"] = cat_m
        models[f"meta_{market_key}"] = meta_m
        models[f"has_{market_key}_models"] = True

    _fit_optional_market(y_spread, "spread")
    _fit_optional_market(y_total, "total")

    return models

def predict_for_date(
    target_date_str: str,
    df_raw: pd.DataFrame = None,
    df_feat: pd.DataFrame = None,
    pick_params: dict = None,
    calibration_cfg: dict = None,
    models: dict = None,
    error_risk_bundle: dict | None = None,
):
    """Predict for a single date. If `df_raw` and `df_feat` are provided, reuse them to avoid repeated I/O.
    Otherwise the function will read CSVs from disk (backwards-compatible).
    """
    if df_raw is None:
        if not RAW_DATA.exists():
            return False
        df_raw = pd.read_csv(RAW_DATA)

    if df_feat is None:
        if not PROCESSED_DATA.exists():
            return False
        df_feat = pd.read_csv(PROCESSED_DATA)

    target_date = pd.Timestamp(target_date_str)
    # reuse precomputed datetimes if present, but ensure they are real datetimes
    raw_dates = pd.to_datetime(df_raw["date_dt"] if "date_dt" in df_raw.columns else df_raw["date"]) 
    feat_dates = pd.to_datetime(df_feat["date_dt"] if "date_dt" in df_feat.columns else df_feat["date"]) 

    train_history = df_raw[raw_dates < target_date].copy()
    target_games = df_raw[raw_dates == target_date].copy()
    train_features = df_feat[feat_dates < target_date].copy()

    if train_history.empty or target_games.empty or train_features.empty:
        return False

    # If models are provided (fast mode / cached), reuse them to avoid retraining.
    if models is None:
        models = train_models_from_past(train_features)
    df_predict = build_prediction_features_for_games(train_history, target_games, target_date)

    # Enriquecer df_predict con columnas ya calculadas por feature_engineering (si existen)
    try:
        proc = df_feat
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

    X_pred = df_predict.reindex(columns=models["feature_names"], fill_value=0)
    X_pred = clean_dataframe(X_pred)

    # Use cached pick_params/calibration_cfg if provided, otherwise load once
    if pick_params is None:
        try:
            pick_params = joblib.load(MODELS_DIR / "pick_params.pkl")
        except Exception:
            pick_params = {"game": {"threshold": 0.5}, "q1": {"threshold": 0.5}, "h1": {"threshold": 0.5}}

    game_params = get_mode_params(pick_params, mode="game")
    q1_params = get_mode_params(pick_params, mode="q1")
    h1_params = get_mode_params(pick_params, mode="h1")
    spread_params = get_mode_params(pick_params, mode="spread")
    total_params = get_mode_params(pick_params, mode="total")

    # Predicciones Partido Completo
    p_x_g = models["xgb_game"].predict_proba(X_pred)[:, 1]
    p_l_g = models["lgb_game"].predict_proba(X_pred)[:, 1]
    p_c_g = models["cat_game"].predict_proba(X_pred)[:, 1]

    if bool(game_params.get("use_meta", True)):
        if isinstance(models["meta_game"], ConstantBinaryModel):
            prob_game = models["meta_game"].predict_proba(X_pred)[:, 1]
        else:
            prob_game = models["meta_game"].predict_proba(np.column_stack([p_x_g, p_l_g, p_c_g]))[:, 1]
    else:
        wx = float(game_params.get("xgb_weight", 1.0 / 3.0))
        wl = float(game_params.get("lgb_weight", 1.0 / 3.0))
        wc = float(game_params.get("cat_weight", 1.0 / 3.0))
        wsum = wx + wl + wc
        if wsum <= 0:
            wx = wl = wc = 1.0 / 3.0
        else:
            wx, wl, wc = wx / wsum, wl / wsum, wc / wsum
        prob_game = (wx * p_x_g) + (wl * p_l_g) + (wc * p_c_g)

    # Guardar también las probabilidades base (promedio de modelos) y calibradas
    if calibration_cfg is None:
        try:
            calibration_cfg = load_calibration_config(CALIBRATION_FILE)
        except Exception:
            calibration_cfg = None

    model_avg_game = (p_x_g + p_l_g + p_c_g) / 3.0
    # Calibrated meta probabilities (use same function que predict_today)
    if calibration_cfg is not None:
        calibrated_meta_game = np.array(list(map(lambda v: calibrate_probability(float(v), sport="wnba", market="full_game", calibration_config=calibration_cfg), prob_game)))
    else:
        calibrated_meta_game = prob_game.copy()

    # Predicciones Q1
    p_x_q = models["xgb_q1"].predict_proba(X_pred)[:, 1]
    p_l_q = models["lgb_q1"].predict_proba(X_pred)[:, 1]
    p_c_q = models["cat_q1"].predict_proba(X_pred)[:, 1]
    
    if isinstance(models["meta_q1"], ConstantBinaryModel):
        prob_q1 = models["meta_q1"].predict_proba(X_pred)[:, 1]
    else:
        prob_q1 = models["meta_q1"].predict_proba(np.column_stack([p_x_q, p_l_q, p_c_q]))[:, 1]

    model_avg_q1 = (p_x_q + p_l_q + p_c_q) / 3.0
    if calibration_cfg is not None:
        calibrated_meta_q1 = np.array(list(map(lambda v: calibrate_probability(float(v), sport="wnba", market="q1", calibration_config=calibration_cfg), prob_q1)))
    else:
        calibrated_meta_q1 = prob_q1.copy()

    p_x_h = models["xgb_h1"].predict_proba(X_pred)[:, 1]
    p_l_h = models["lgb_h1"].predict_proba(X_pred)[:, 1]
    p_c_h = models["cat_h1"].predict_proba(X_pred)[:, 1]
    if bool(h1_params.get("use_meta", True)):
        if isinstance(models["meta_h1"], ConstantBinaryModel):
            prob_h1 = models["meta_h1"].predict_proba(X_pred)[:, 1]
        else:
            prob_h1 = models["meta_h1"].predict_proba(np.column_stack([p_x_h, p_l_h, p_c_h]))[:, 1]
    else:
        wx = float(h1_params.get("xgb_weight", 1.0 / 3.0))
        wl = float(h1_params.get("lgb_weight", 1.0 / 3.0))
        wc = float(h1_params.get("cat_weight", 1.0 / 3.0))
        wsum = wx + wl + wc
        if wsum <= 0:
            wx = wl = wc = 1.0 / 3.0
        else:
            wx, wl, wc = wx / wsum, wl / wsum, wc / wsum
        prob_h1 = (wx * p_x_h) + (wl * p_l_h) + (wc * p_c_h)

    model_avg_h1 = (p_x_h + p_l_h + p_c_h) / 3.0
    if calibration_cfg is not None:
        calibrated_meta_h1 = np.array(list(map(lambda v: calibrate_probability(float(v), sport="wnba", market="h1", calibration_config=calibration_cfg), prob_h1)))
    else:
        calibrated_meta_h1 = prob_h1.copy()

    if bool(models.get("has_spread_models", False)):
        p_x_s = models["xgb_spread"].predict_proba(X_pred)[:, 1]
        p_l_s = models["lgb_spread"].predict_proba(X_pred)[:, 1]
        p_c_s = models["cat_spread"].predict_proba(X_pred)[:, 1]
        if isinstance(models["meta_spread"], ConstantBinaryModel):
            prob_spread = models["meta_spread"].predict_proba(X_pred)[:, 1]
        else:
            prob_spread = models["meta_spread"].predict_proba(np.column_stack([p_x_s, p_l_s, p_c_s]))[:, 1]
        if calibration_cfg is not None:
            calibrated_meta_spread = np.array(list(map(lambda v: calibrate_probability(float(v), sport="wnba", market="spread", calibration_config=calibration_cfg), prob_spread)))
        else:
            calibrated_meta_spread = prob_spread.copy()
    else:
        prob_spread = np.full(len(X_pred), np.nan)
        calibrated_meta_spread = np.full(len(X_pred), np.nan)

    if bool(models.get("has_total_models", False)):
        p_x_t = models["xgb_total"].predict_proba(X_pred)[:, 1]
        p_l_t = models["lgb_total"].predict_proba(X_pred)[:, 1]
        p_c_t = models["cat_total"].predict_proba(X_pred)[:, 1]
        if isinstance(models["meta_total"], ConstantBinaryModel):
            prob_total = models["meta_total"].predict_proba(X_pred)[:, 1]
        else:
            prob_total = models["meta_total"].predict_proba(np.column_stack([p_x_t, p_l_t, p_c_t]))[:, 1]
        if calibration_cfg is not None:
            calibrated_meta_total = np.array(list(map(lambda v: calibrate_probability(float(v), sport="wnba", market="total", calibration_config=calibration_cfg), prob_total)))
        else:
            calibrated_meta_total = prob_total.copy()
    else:
        prob_total = np.full(len(X_pred), np.nan)
        calibrated_meta_total = np.full(len(X_pred), np.nan)

    # --- PARCHE 1: PEGADO DE PROBABILIDADES ---
    # Esto evita que los datos de un juego se mezclen con otro
    df_predict["prob_game_safe"] = prob_game
    df_predict["prob_q1_safe"] = prob_q1
    # Añadimos columnas adicionales para auditing/historico
    df_predict["full_game_model_prob_home"] = model_avg_game
    df_predict["full_game_meta_prob_home"] = prob_game
    df_predict["full_game_calibrated_prob_home"] = calibrated_meta_game
    df_predict["full_q1_model_prob_home"] = model_avg_q1
    df_predict["full_q1_meta_prob_home"] = prob_q1
    df_predict["full_q1_calibrated_prob_home"] = calibrated_meta_q1
    df_predict["full_h1_model_prob_home"] = model_avg_h1
    df_predict["full_h1_meta_prob_home"] = prob_h1
    df_predict["full_h1_calibrated_prob_home"] = calibrated_meta_h1
    df_predict["spread_model_prob_home_cover"] = prob_spread
    df_predict["spread_calibrated_prob_home_cover"] = calibrated_meta_spread
    df_predict["total_model_prob_over"] = prob_total
    df_predict["total_calibrated_prob_over"] = calibrated_meta_total

    output = []
    # Usamos iterrows directamente sobre el DataFrame sincronizado
    for _, row in df_predict.iterrows():
        # Recuperar probs del DataFrame
        p_game_h = row["prob_game_safe"]
        p_q1_h = row["prob_q1_safe"]

        # Probabilidades históricas detalladas
        model_prob_game_h = float(row.get("full_game_model_prob_home", 0.0))
        meta_prob_game_h = float(row.get("full_game_meta_prob_home", p_game_h))
        calibrated_prob_game_h = float(row.get("full_game_calibrated_prob_home", p_game_h))

        model_prob_q1_h = float(row.get("full_q1_model_prob_home", 0.0))
        meta_prob_q1_h = float(row.get("full_q1_meta_prob_home", p_q1_h))
        calibrated_prob_q1_h = float(row.get("full_q1_calibrated_prob_home", p_q1_h))

        # 1. Probabilidades y Picks
        # Usar probabilidad calibrada final para decisión de pick
        game_threshold = float(game_params.get("threshold", 0.5))
        market_threshold = float(game_params.get("market_threshold", game_threshold))
        try:
            row_home_spread = float(row.get("home_spread", 0) or 0)
        except Exception:
            row_home_spread = 0.0
        row_has_market_side = abs(row_home_spread) > 0
        threshold_for_pick = market_threshold if row_has_market_side else game_threshold
        prob_game_home = float(calibrated_prob_game_h * 100)
        prob_game_away = float(100 - prob_game_home)
        full_pick, full_pick_rule = choose_full_game_pick(
            row=row,
            calibrated_prob_home=calibrated_prob_game_h,
            threshold=threshold_for_pick,
            uncertainty_band=float(game_params.get("market_tiebreak_band", 0.10)),
        )
        full_pick, full_pick_rule, error_risk_prob = apply_error_risk_override(
            bundle=error_risk_bundle,
            row=row,
            calibrated_prob_home=calibrated_prob_game_h,
            threshold_used=threshold_for_pick,
            current_pick=full_pick,
            current_rule=full_pick_rule,
        )
        full_conf = float(max(prob_game_home, prob_game_away))

        q1_threshold = float(q1_params.get("threshold", 0.5))
        prob_q1_home = float(calibrated_prob_q1_h * 100)
        prob_q1_away = float(100 - prob_q1_home)
        q1_pick = row["home_team"] if calibrated_prob_q1_h >= q1_threshold else row["away_team"]
        q1_conf = float(max(prob_q1_home, prob_q1_away))

        calibrated_prob_h1_h = float(row.get("full_h1_calibrated_prob_home", 0.5))
        model_prob_h1_h = float(row.get("full_h1_model_prob_home", calibrated_prob_h1_h))
        h1_threshold = float(h1_params.get("threshold", 0.5))
        prob_h1_home = float(calibrated_prob_h1_h * 100)
        prob_h1_away = float(100 - prob_h1_home)
        h1_pick = row["home_team"] if calibrated_prob_h1_h >= h1_threshold else row["away_team"]
        h1_conf = float(max(prob_h1_home, prob_h1_away))

        # 2. Datos de Mercado
        h_spread = float(row.get("home_spread", 0) or 0)
        o_u_line = float(row.get("odds_over_under", 0) or 0)

        # 3. Mercados extra (spread / total)
        spread_prob_home_cover = float(row.get("spread_calibrated_prob_home_cover")) if pd.notna(row.get("spread_calibrated_prob_home_cover")) else None
        spread_pick = "N/A"
        if allow_spread_pick(row, spread_prob_home_cover, spread_params):
            spread_thr = float(spread_params.get("threshold", 0.5))
            spread_pick = f"{row['home_team']} {h_spread:+g}" if spread_prob_home_cover >= spread_thr else f"{row['away_team']} {-h_spread:+g}"

        total_prob_over = float(row.get("total_calibrated_prob_over")) if pd.notna(row.get("total_calibrated_prob_over")) else None
        total_pick = "N/A"
        if total_prob_over is not None and o_u_line > 0:
            total_thr = float(total_params.get("threshold", 0.5))
            total_pick = f"OVER {o_u_line}" if total_prob_over >= total_thr else f"UNDER {o_u_line}"
        elif o_u_line > 0:
            total_pick = f"O/U {o_u_line}"

        # --- PARCHE 2: LÓGICA DE AUDITORÍA ROBUSTA ---
        h_pts = safe_int(row.get("home_pts_total", 0))
        a_pts = safe_int(row.get("away_pts_total", 0))
        h_q1 = safe_int(row.get("home_q1", 0))
        a_q1 = safe_int(row.get("away_q1", 0))
        h_q2 = safe_int(row.get("home_q2", 0))
        a_q2 = safe_int(row.get("away_q2", 0))
        result_available = (h_pts > 0 or a_pts > 0)
        
        full_game_hit = None
        q1_hit = None
        h1_hit = None
        correct_spread = None
        total_hit = None

        if result_available:
            # Marcador Final Text
            final_score_text = f"{row['home_team']} {h_pts} - {row['away_team']} {a_pts}"
            
            # Ganador Real Partido (ML) con detección de empate
            real_winner = row["home_team"] if h_pts > a_pts else (row["away_team"] if a_pts > h_pts else "TIE")
            full_game_hit = (full_pick == real_winner)
            
            # Ganador Real Q1 (Aquí arreglamos el 31-31 de Toronto)
            real_q1_winner = row["home_team"] if h_q1 > a_q1 else (row["away_team"] if a_q1 > h_q1 else "TIE")
            q1_hit = (q1_pick == real_q1_winner)

            home_h1 = h_q1 + h_q2
            away_h1 = a_q1 + a_q2
            real_h1_winner = row["home_team"] if home_h1 > away_h1 else (row["away_team"] if away_h1 > home_h1 else "TIE")
            h1_hit = (h1_pick == real_h1_winner)

            # Auditoría Spread
            if h_spread != 0:
                if str(spread_pick).startswith(str(row["home_team"])):
                    correct_spread = (h_pts + h_spread) > a_pts
                else:
                    correct_spread = (a_pts - h_spread) > h_pts
            if o_u_line > 0:
                total_pts = h_pts + a_pts
                if str(total_pick).startswith("OVER"):
                    total_hit = total_pts > o_u_line
                elif str(total_pick).startswith("UNDER"):
                    total_hit = total_pts < o_u_line
        else:
            final_score_text = ""

        # --- 4. CONSTRUCCIÓN DEL OBJETO FINAL (TODOS TUS CAMPOS) ---
        output.append({
            "game_id": str(row["game_id"]),
            "date": target_date_str,
            "time": "",
            "game_name": f"{row['away_team']} @ {row['home_team']}",
            "away_team": str(row["away_team"]),
            "home_team": str(row["home_team"]),
            "spread_market": str(row.get("odds_spread", "N/A")),
            "home_spread": h_spread,
            "spread_abs": float(row.get("spread_abs", 0) or 0),
            "odds_over_under": o_u_line,
            "odds_data_quality": str(row.get("odds_data_quality", "fallback")),
            "market_missing": int(infer_market_missing(row)),
            "full_game_pick": str(full_pick),
            "full_game_pick_rule": str(full_pick_rule),
            "full_game_error_risk_prob": round(float(error_risk_prob), 4) if error_risk_prob is not None else None,
            "full_game_confidence": round(full_conf, 1),
            "full_game_tier": get_pick_tier(full_conf),
            "q1_pick": str(q1_pick),
            "q1_confidence": round(q1_conf, 1),
            "q1_action": "JUGAR Q1" if q1_conf >= 62 else "PASAR Q1",
            "h1_pick": str(h1_pick),
            "h1_confidence": round(h1_conf, 1),
            "h1_action": "JUGAR 1H" if h1_conf >= 58 else "PASAR 1H",
            "total_pick": total_pick,
            "total_prob_over": round(float(total_prob_over), 4) if total_prob_over is not None else None,
            "spread_pick": spread_pick,
            "spread_prob_home_cover": round(float(spread_prob_home_cover), 4) if spread_prob_home_cover is not None else None,
            "assists_pick": "N/A",
            "prediction_mode": "historical_rebuild",
            "trained_until": str((target_date - pd.Timedelta(days=1)).date()),
            "result_available": result_available,
            "full_game_hit": full_game_hit,
            "q1_hit": q1_hit,
            "h1_hit": h1_hit,
            "correct_spread": correct_spread,
            "total_hit": total_hit,
            "final_score_text": final_score_text,
            "home_moneyline_odds": american_to_decimal(row.get("home_moneyline_odds")),
            "away_moneyline_odds": american_to_decimal(row.get("away_moneyline_odds")),
            "closing_spread_odds": american_to_decimal(row.get("closing_spread_odds")),
            "closing_total_odds": american_to_decimal(row.get("closing_total_odds")),
            # --- Probabilidades para auditoría ---
            "full_game_model_prob_home": round(model_prob_game_h, 4),
            "full_game_calibrated_prob_home": round(calibrated_prob_game_h, 4),
            "full_game_prob_home": round(float(calibrated_prob_game_h * 100), 2),
            "full_game_prob_away": round(float(100 - (calibrated_prob_game_h * 100)), 2),
            "q1_model_prob_home": round(model_prob_q1_h, 4),
            "q1_calibrated_prob_home": round(calibrated_prob_q1_h, 4),
            "q1_prob_home": round(float(calibrated_prob_q1_h * 100), 2),
            "q1_prob_away": round(float(100 - (calibrated_prob_q1_h * 100)), 2),
            "h1_model_prob_home": round(model_prob_h1_h, 4),
            "h1_calibrated_prob_home": round(calibrated_prob_h1_h, 4),
            "h1_prob_home": round(prob_h1_home, 2),
            "h1_prob_away": round(prob_h1_away, 2),
        })

    # Guardar archivo
    output_file = HIST_PRED_DIR / f"{target_date_str}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(clean_for_json(output), f, ensure_ascii=False, indent=2)

    return True

def generate_range(
    start_date: str,
    end_date: str,
    fast: bool = False,
    retrain_every_n_days: int = 1,
    only_game_days: bool = False,
):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if end < start:
        print("❌ end_date no puede ser menor que start_date")
        return

    retrain_every_n_days = max(1, int(retrain_every_n_days))
    fechas = pd.date_range(start=start, end=end)
    ok = 0
    skipped = 0

    print(f"Iniciando generación desde {start_date} hasta {end_date}...")

    # Preload CSVs and precompute datetime columns and pick/calibration configs once
    df_raw_all = pd.read_csv(RAW_DATA)
    if "date_dt" not in df_raw_all.columns:
        df_raw_all["date_dt"] = pd.to_datetime(df_raw_all["date"]) 
    df_feat_all = pd.read_csv(PROCESSED_DATA)
    if "date_dt" not in df_feat_all.columns:
        df_feat_all["date_dt"] = pd.to_datetime(df_feat_all["date"]) 

    if only_game_days:
        available_game_days = set(pd.to_datetime(df_raw_all["date"]).dt.normalize().unique())
        fechas = [d for d in fechas if d.normalize() in available_game_days]
        print(f"📅 Modo only_game_days activo: {len(fechas)} días con juegos en rango.")

    try:
        cached_pick_params = joblib.load(MODELS_DIR / "pick_params.pkl")
    except Exception:
        cached_pick_params = None

    try:
        cached_calib_cfg = load_calibration_config(CALIBRATION_FILE)
    except Exception:
        cached_calib_cfg = None
    cached_error_risk_bundle = load_error_risk_bundle(ERROR_RISK_MODEL_FILE)

    models_cache = None
    days_since_retrain = retrain_every_n_days
    if fast:
        # Train a single model on the full features history and reuse for all dates.
        try:
            train_features_all = df_feat_all[df_feat_all["date_dt"] < fechas[-1]]
            models_cache = train_models_from_past(train_features_all)
            print("⚡ Fast mode: modelos entrenados una vez y reutilizados para todo el rango")
        except Exception as e:
            print(f"⚠️ Fast mode training failed, falling back to per-day training: {e}")
    elif retrain_every_n_days > 1:
        print(f"⏱️ Modo rápido sin leakage: reentrenar cada {retrain_every_n_days} días.")

    for current in tqdm(fechas, desc="Procesando días", unit="día"):
        try:
            if not fast:
                need_retrain = (models_cache is None) or (days_since_retrain >= retrain_every_n_days)
                if need_retrain:
                    train_features = df_feat_all[df_feat_all["date_dt"] < current]
                    if not train_features.empty:
                        models_cache = train_models_from_past(train_features)
                        days_since_retrain = 0

            success = predict_for_date(
                str(current.date()),
                df_raw=df_raw_all,
                df_feat=df_feat_all,
                pick_params=cached_pick_params,
                calibration_cfg=cached_calib_cfg,
                models=models_cache,
                error_risk_bundle=cached_error_risk_bundle,
            )
            if success: ok += 1
            else: skipped += 1
            if not fast:
                days_since_retrain += 1
        except Exception as e:
            tqdm.write(f"❌ Error procesando {current.date()}: {e}")

    print(f"\n🏁 Rango terminado\n   Generadas: {ok}\n   Saltadas : {skipped}")

from datetime import datetime, timedelta
if __name__ == "__main__":
    start_date = "2025-05-01"
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Generando historial WNBA desde {start_date} hasta {end_date}...")
    mode = str(os.getenv("WNBA_HIST_MODE", "hybrid")).strip().lower()
    if mode == "precise":
        cfg = {"fast": False, "retrain_every_n_days": 1, "only_game_days": True}
    elif mode == "fast":
        cfg = {"fast": True, "retrain_every_n_days": 9999, "only_game_days": True}
    else:
        # hybrid (default): much faster, minimal accuracy tradeoff vs precise.
        cfg = {"fast": False, "retrain_every_n_days": 2, "only_game_days": True}
    print(f"Modo: {mode} | cfg={cfg}")
    generate_range(
        start_date,
        end_date,
        fast=cfg["fast"],
        retrain_every_n_days=cfg["retrain_every_n_days"],
        only_game_days=cfg["only_game_days"],
    )
