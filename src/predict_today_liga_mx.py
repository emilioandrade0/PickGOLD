import json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from pattern_engine import aggregate_pattern_edge
from pattern_engine_liga_mx import generate_liga_mx_patterns
from pick_selector import fuse_with_pattern_score, recommendation_score
from feature_engineering_liga_mx_v3 import add_v3_features
from liga_mx_market_source import load_market_feature_sources, resolve_market_model_dir
from event_adjustments_liga_mx import (
    USE_EVENT_ADJUSTMENTS,
    apply_probability_adjustment,
    detect_pre_match_events,
    get_h2h_features,
    get_recent_team_form_features,
    probability_to_confidence,
)

BASE_DIR = Path(__file__).resolve().parent

FEATURES_FILE = BASE_DIR / "data" / "liga_mx" / "processed" / "model_ready_features_liga_mx.csv"
UPCOMING_FILE = BASE_DIR / "data" / "liga_mx" / "raw" / "liga_mx_upcoming_schedule.csv"
RAW_HISTORY_FILE = BASE_DIR / "data" / "liga_mx" / "raw" / "liga_mx_advanced_history.csv"
MODELS_DIR = BASE_DIR / "data" / "liga_mx" / "models"
SELECTIVE_MODELS_DIR = BASE_DIR / "data" / "liga_mx" / "models_selective"
SELECTIVE_PLAN_FILE = BASE_DIR / "data" / "liga_mx" / "reports" / "liga_mx_selective_upgrade_plan.csv"
PREDICTIONS_DIR = BASE_DIR / "data" / "liga_mx" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)



def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_market_assets(market_key: str, market_sources: dict):
    """Carga modelos, features y metadata de un mercado específico."""
    market_dir = resolve_market_model_dir(
        market_key=market_key,
        market_sources=market_sources,
        baseline_models_dir=MODELS_DIR,
        selective_models_dir=SELECTIVE_MODELS_DIR,
    )
    xgb = joblib.load(market_dir / "xgb_model.pkl")
    lgbm = joblib.load(market_dir / "lgbm_model.pkl")
    lgbm_secondary = joblib.load(market_dir / "lgbm_secondary_model.pkl")
    catboost_path = market_dir / "catboost_model.pkl"
    if catboost_path.exists():
        try:
            catboost = joblib.load(catboost_path)
        except Exception:
            catboost = None
    else:
        catboost = None
    feature_columns = load_json(market_dir / "feature_columns.json")
    metadata = load_json(market_dir / "metadata.json")
    threshold = metadata.get("ensemble_threshold", 0.5)
    return xgb, lgbm, lgbm_secondary, catboost, feature_columns, metadata, threshold


def predict_market(df: pd.DataFrame, market_key: str, market_sources: dict):
    """Predice probabilidades usando ensemble: XGB + LGBM Primary + LGBM Secondary + CatBoost."""
    xgb, lgbm, lgbm_secondary, catboost, feature_columns, metadata, threshold = load_market_assets(
        market_key,
        market_sources,
    )

    # feature_columns puede ser dict o list
    if isinstance(feature_columns, dict):
        feat_list = feature_columns.get("features", [])
    else:
        feat_list = feature_columns

    X = df[feat_list].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    if market_key == "full_game":
        # Para multiclass: promedio de 3/4 modelos según disponibilidad de CatBoost.
        xgb_probs = xgb.predict_proba(X)
        lgbm_probs = lgbm.predict_proba(X)
        lgbm_sec_probs = lgbm_secondary.predict_proba(X)
        probs = xgb_probs + lgbm_probs + lgbm_sec_probs
        model_count = 3.0
        if catboost is not None:
            probs = probs + catboost.predict_proba(X)
            model_count += 1.0
        probs = probs / model_count
        preds = np.argmax(probs, axis=1)
        return probs, preds, threshold, metadata
    else:
        # Para binary (over_25, btts): promedio de 3/4 modelos según disponibilidad de CatBoost.
        xgb_probs = xgb.predict_proba(X)[:, 1]
        lgbm_probs = lgbm.predict_proba(X)[:, 1]
        lgbm_sec_probs = lgbm_secondary.predict_proba(X)[:, 1]
        probs = xgb_probs + lgbm_probs + lgbm_sec_probs
        model_count = 3.0
        if catboost is not None:
            probs = probs + catboost.predict_proba(X)[:, 1]
            model_count += 1.0
        probs = probs / model_count
        preds = (probs >= threshold).astype(int)
        return probs, preds, threshold, metadata


def confidence_from_prob(prob: float) -> int:
    prob = float(np.clip(prob, 0.0, 1.0))
    return int(round(max(prob, 1 - prob) * 100))


def normalize_multiclass_probs(home_p: float, away_p: float, draw_p: float):
    arr = np.array([away_p, home_p, draw_p], dtype=float)
    arr = np.clip(arr, 1e-9, None)
    arr = arr / arr.sum()
    return float(arr[1]), float(arr[0]), float(arr[2])


def tier_from_conf(conf: int) -> str:
    if conf >= 72:
        return "ELITE"
    if conf >= 66:
        return "PREMIUM"
    if conf >= 60:
        return "STRONG"
    if conf >= 54:
        return "NORMAL"
    return "PASS"


def get_team_snapshot(history_df: pd.DataFrame, team: str, cutoff_date: str):
    """Obtiene el snapshot pregame de un equipo antes de una fecha."""
    prior = history_df[history_df["date"] < cutoff_date].copy()
    if prior.empty:
        return None

    home_rows = prior[prior["home_team"] == team].copy()
    away_rows = prior[prior["away_team"] == team].copy()

    candidates = []

    if not home_rows.empty:
        row = home_rows.sort_values(["date", "game_id"]).iloc[-1]
        snap = {
            "elo_pre": row["home_elo_pre"],
            "rest_days": row["home_rest_days"],
            "is_b2b": row["home_is_b2b"],
            "games_last_3_days": row["home_games_last_3_days"],
            "games_last_5_days": row["home_games_last_5_days"],
            "games_last_7_days": row["home_games_last_7_days"],
            "games_last_10_days": row["home_games_last_10_days"],
            "games_last_14_days": row["home_games_last_14_days"],
            "win_pct_L5": row["home_win_pct_L5"],
            "win_pct_L10": row["home_win_pct_L10"],
            "draw_pct_L5": row["home_draw_pct_L5"],
            "draw_pct_L10": row["home_draw_pct_L10"],
            "goal_diff_L3": row["home_goal_diff_L3"],
            "goal_diff_L5": row["home_goal_diff_L5"],
            "goal_diff_L10": row["home_goal_diff_L10"],
            "goals_scored_L3": row["home_goals_scored_L3"],
            "goals_scored_L5": row["home_goals_scored_L5"],
            "goals_scored_L10": row["home_goals_scored_L10"],
            "goals_allowed_L3": row["home_goals_allowed_L3"],
            "goals_allowed_L5": row["home_goals_allowed_L5"],
            "goals_allowed_L10": row["home_goals_allowed_L10"],
            "goal_diff_std_L10": row["home_goal_diff_std_L10"],
            "goals_scored_std_L10": row["home_goals_scored_std_L10"],
            "goals_allowed_std_L10": row["home_goals_allowed_std_L10"],
            "btts_rate_L10": row["home_btts_rate_L10"],
            "over_25_rate_L10": row["home_over_25_rate_L10"],
            "surface_win_pct_L5": row.get("home_home_only_win_pct_L5", 0),
            "surface_goal_diff_L5": row.get("home_home_only_goal_diff_L5", 0),
            "surface_draw_pct_L5": row.get("home_home_only_draw_pct_L5", 0),
            "surface_btts_rate_L10": row.get("home_home_only_btts_rate_L10", 0),
            "surface_over_25_rate_L10": row.get("home_home_only_over_25_rate_L10", 0),
            "_date": row["date"],
            "_game_id": row["game_id"],
        }
        candidates.append(snap)

    if not away_rows.empty:
        row = away_rows.sort_values(["date", "game_id"]).iloc[-1]
        snap = {
            "elo_pre": row["away_elo_pre"],
            "rest_days": row["away_rest_days"],
            "is_b2b": row["away_is_b2b"],
            "games_last_3_days": row["away_games_last_3_days"],
            "games_last_5_days": row["away_games_last_5_days"],
            "games_last_7_days": row["away_games_last_7_days"],
            "games_last_10_days": row["away_games_last_10_days"],
            "games_last_14_days": row["away_games_last_14_days"],
            "win_pct_L5": row["away_win_pct_L5"],
            "win_pct_L10": row["away_win_pct_L10"],
            "draw_pct_L5": row["away_draw_pct_L5"],
            "draw_pct_L10": row["away_draw_pct_L10"],
            "goal_diff_L3": row["away_goal_diff_L3"],
            "goal_diff_L5": row["away_goal_diff_L5"],
            "goal_diff_L10": row["away_goal_diff_L10"],
            "goals_scored_L3": row["away_goals_scored_L3"],
            "goals_scored_L5": row["away_goals_scored_L5"],
            "goals_scored_L10": row["away_goals_scored_L10"],
            "goals_allowed_L3": row["away_goals_allowed_L3"],
            "goals_allowed_L5": row["away_goals_allowed_L5"],
            "goals_allowed_L10": row["away_goals_allowed_L10"],
            "goal_diff_std_L10": row["away_goal_diff_std_L10"],
            "goals_scored_std_L10": row["away_goals_scored_std_L10"],
            "goals_allowed_std_L10": row["away_goals_allowed_std_L10"],
            "btts_rate_L10": row["away_btts_rate_L10"],
            "over_25_rate_L10": row["away_over_25_rate_L10"],
            "surface_win_pct_L5": row.get("away_away_only_win_pct_L5", 0),
            "surface_goal_diff_L5": row.get("away_away_only_goal_diff_L5", 0),
            "surface_draw_pct_L5": row.get("away_away_only_draw_pct_L5", 0),
            "surface_btts_rate_L10": row.get("away_away_only_btts_rate_L10", 0),
            "surface_over_25_rate_L10": row.get("away_away_only_over_25_rate_L10", 0),
            "_date": row["date"],
            "_game_id": row["game_id"],
        }
        candidates.append(snap)

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda x: (x["_date"], str(x["_game_id"])))
    best = candidates[-1].copy()
    best.pop("_date", None)
    best.pop("_game_id", None)
    return best


def build_pregame_feature_row(history_df: pd.DataFrame, schedule_row: pd.Series):
    """Construye una fila de features pregame para un partido."""
    date_str = str(schedule_row["date"])
    home_team = str(schedule_row["home_team"])
    away_team = str(schedule_row["away_team"])

    home_snap = get_team_snapshot(history_df, home_team, date_str)
    away_snap = get_team_snapshot(history_df, away_team, date_str)

    if home_snap is None or away_snap is None:
        return None

    row = {
        "game_id": str(schedule_row["game_id"]),
        "date": date_str,
        "season": str(date_str[:4]),
        "home_team": home_team,
        "away_team": away_team,

        "home_elo_pre": home_snap["elo_pre"],
        "away_elo_pre": away_snap["elo_pre"],
        "diff_elo": home_snap["elo_pre"] - away_snap["elo_pre"],

        "home_rest_days": home_snap["rest_days"],
        "away_rest_days": away_snap["rest_days"],
        "diff_rest_days": home_snap["rest_days"] - away_snap["rest_days"],

        "home_is_b2b": home_snap["is_b2b"],
        "away_is_b2b": away_snap["is_b2b"],
        "diff_is_b2b": home_snap["is_b2b"] - away_snap["is_b2b"],

        "home_games_last_3_days": home_snap["games_last_3_days"],
        "away_games_last_3_days": away_snap["games_last_3_days"],
        "diff_games_last_3_days": home_snap["games_last_3_days"] - away_snap["games_last_3_days"],

        "home_games_last_5_days": home_snap["games_last_5_days"],
        "away_games_last_5_days": away_snap["games_last_5_days"],
        "diff_games_last_5_days": home_snap["games_last_5_days"] - away_snap["games_last_5_days"],

        "home_games_last_7_days": home_snap["games_last_7_days"],
        "away_games_last_7_days": away_snap["games_last_7_days"],
        "diff_games_last_7_days": home_snap["games_last_7_days"] - away_snap["games_last_7_days"],

        "home_games_last_10_days": home_snap["games_last_10_days"],
        "away_games_last_10_days": away_snap["games_last_10_days"],
        "diff_games_last_10_days": home_snap["games_last_10_days"] - away_snap["games_last_10_days"],

        "home_games_last_14_days": home_snap["games_last_14_days"],
        "away_games_last_14_days": away_snap["games_last_14_days"],
        "diff_games_last_14_days": home_snap["games_last_14_days"] - away_snap["games_last_14_days"],

        "home_win_pct_L5": home_snap["win_pct_L5"],
        "away_win_pct_L5": away_snap["win_pct_L5"],
        "diff_win_pct_L5": home_snap["win_pct_L5"] - away_snap["win_pct_L5"],

        "home_win_pct_L10": home_snap["win_pct_L10"],
        "away_win_pct_L10": away_snap["win_pct_L10"],
        "diff_win_pct_L10": home_snap["win_pct_L10"] - away_snap["win_pct_L10"],

        "home_draw_pct_L5": home_snap["draw_pct_L5"],
        "away_draw_pct_L5": away_snap["draw_pct_L5"],
        "diff_draw_pct_L5": home_snap["draw_pct_L5"] - away_snap["draw_pct_L5"],

        "home_draw_pct_L10": home_snap["draw_pct_L10"],
        "away_draw_pct_L10": away_snap["draw_pct_L10"],
        "diff_draw_pct_L10": home_snap["draw_pct_L10"] - away_snap["draw_pct_L10"],

        "home_goal_diff_L3": home_snap["goal_diff_L3"],
        "away_goal_diff_L3": away_snap["goal_diff_L3"],
        "diff_goal_diff_L3": home_snap["goal_diff_L3"] - away_snap["goal_diff_L3"],

        "home_goal_diff_L5": home_snap["goal_diff_L5"],
        "away_goal_diff_L5": away_snap["goal_diff_L5"],
        "diff_goal_diff_L5": home_snap["goal_diff_L5"] - away_snap["goal_diff_L5"],

        "home_goal_diff_L10": home_snap["goal_diff_L10"],
        "away_goal_diff_L10": away_snap["goal_diff_L10"],
        "diff_goal_diff_L10": home_snap["goal_diff_L10"] - away_snap["goal_diff_L10"],

        "home_goals_scored_L3": home_snap["goals_scored_L3"],
        "away_goals_scored_L3": away_snap["goals_scored_L3"],
        "diff_goals_scored_L3": home_snap["goals_scored_L3"] - away_snap["goals_scored_L3"],

        "home_goals_scored_L5": home_snap["goals_scored_L5"],
        "away_goals_scored_L5": away_snap["goals_scored_L5"],
        "diff_goals_scored_L5": home_snap["goals_scored_L5"] - away_snap["goals_scored_L5"],

        "home_goals_scored_L10": home_snap["goals_scored_L10"],
        "away_goals_scored_L10": away_snap["goals_scored_L10"],
        "diff_goals_scored_L10": home_snap["goals_scored_L10"] - away_snap["goals_scored_L10"],

        "home_goals_allowed_L3": home_snap["goals_allowed_L3"],
        "away_goals_allowed_L3": away_snap["goals_allowed_L3"],
        "diff_goals_allowed_L3": home_snap["goals_allowed_L3"] - away_snap["goals_allowed_L3"],

        "home_goals_allowed_L5": home_snap["goals_allowed_L5"],
        "away_goals_allowed_L5": away_snap["goals_allowed_L5"],
        "diff_goals_allowed_L5": home_snap["goals_allowed_L5"] - away_snap["goals_allowed_L5"],

        "home_goals_allowed_L10": home_snap["goals_allowed_L10"],
        "away_goals_allowed_L10": away_snap["goals_allowed_L10"],
        "diff_goals_allowed_L10": home_snap["goals_allowed_L10"] - away_snap["goals_allowed_L10"],

        "home_goal_diff_std_L10": home_snap["goal_diff_std_L10"],
        "away_goal_diff_std_L10": away_snap["goal_diff_std_L10"],
        "diff_goal_diff_std_L10": home_snap["goal_diff_std_L10"] - away_snap["goal_diff_std_L10"],

        "home_goals_scored_std_L10": home_snap["goals_scored_std_L10"],
        "away_goals_scored_std_L10": away_snap["goals_scored_std_L10"],
        "diff_goals_scored_std_L10": home_snap["goals_scored_std_L10"] - away_snap["goals_scored_std_L10"],

        "home_goals_allowed_std_L10": home_snap["goals_allowed_std_L10"],
        "away_goals_allowed_std_L10": away_snap["goals_allowed_std_L10"],
        "diff_goals_allowed_std_L10": home_snap["goals_allowed_std_L10"] - away_snap["goals_allowed_std_L10"],

        "home_btts_rate_L10": home_snap["btts_rate_L10"],
        "away_btts_rate_L10": away_snap["btts_rate_L10"],
        "diff_btts_rate_L10": home_snap["btts_rate_L10"] - away_snap["btts_rate_L10"],

        "home_over_25_rate_L10": home_snap["over_25_rate_L10"],
        "away_over_25_rate_L10": away_snap["over_25_rate_L10"],
        "diff_over_25_rate_L10": home_snap["over_25_rate_L10"] - away_snap["over_25_rate_L10"],

        "home_home_only_win_pct_L5": home_snap["surface_win_pct_L5"],
        "away_away_only_win_pct_L5": away_snap["surface_win_pct_L5"],
        "diff_surface_win_pct_L5": home_snap["surface_win_pct_L5"] - away_snap["surface_win_pct_L5"],

        "home_home_only_goal_diff_L5": home_snap["surface_goal_diff_L5"],
        "away_away_only_goal_diff_L5": away_snap["surface_goal_diff_L5"],
        "diff_surface_goal_diff_L5": home_snap["surface_goal_diff_L5"] - away_snap["surface_goal_diff_L5"],

        "home_home_only_draw_pct_L5": home_snap["surface_draw_pct_L5"],
        "away_away_only_draw_pct_L5": away_snap["surface_draw_pct_L5"],
        "diff_surface_draw_pct_L5": home_snap["surface_draw_pct_L5"] - away_snap["surface_draw_pct_L5"],

        "home_home_only_btts_rate_L10": home_snap["surface_btts_rate_L10"],
        "away_away_only_btts_rate_L10": away_snap["surface_btts_rate_L10"],
        "diff_surface_btts_rate_L10": home_snap["surface_btts_rate_L10"] - away_snap["surface_btts_rate_L10"],

        "home_home_only_over_25_rate_L10": home_snap["surface_over_25_rate_L10"],
        "away_away_only_over_25_rate_L10": away_snap["surface_over_25_rate_L10"],
        "diff_surface_over_25_rate_L10": home_snap["surface_over_25_rate_L10"] - away_snap["surface_over_25_rate_L10"],

        "home_momentum_win": home_snap["win_pct_L5"] - home_snap["win_pct_L10"],
        "away_momentum_win": away_snap["win_pct_L5"] - away_snap["win_pct_L10"],
        "diff_momentum_win": (
            (home_snap["win_pct_L5"] - home_snap["win_pct_L10"])
            - (away_snap["win_pct_L5"] - away_snap["win_pct_L10"])
        ),

        "home_momentum_goal_diff": home_snap["goal_diff_L5"] - home_snap["goal_diff_L10"],
        "away_momentum_goal_diff": away_snap["goal_diff_L5"] - away_snap["goal_diff_L10"],
        "diff_momentum_goal_diff": (
            (home_snap["goal_diff_L5"] - home_snap["goal_diff_L10"])
            - (away_snap["goal_diff_L5"] - away_snap["goal_diff_L10"])
        ),

        "home_surface_edge": home_snap["surface_win_pct_L5"] - home_snap["win_pct_L10"],
        "away_surface_edge": away_snap["surface_win_pct_L5"] - away_snap["win_pct_L10"],
        "diff_surface_edge": (
            (home_snap["surface_win_pct_L5"] - home_snap["win_pct_L10"])
            - (away_snap["surface_win_pct_L5"] - away_snap["win_pct_L10"])
        ),

        "home_fatigue_index": home_snap["games_last_7_days"] - home_snap["rest_days"],
        "away_fatigue_index": away_snap["games_last_7_days"] - away_snap["rest_days"],
        "diff_fatigue_index": (
            (home_snap["games_last_7_days"] - home_snap["rest_days"])
            - (away_snap["games_last_7_days"] - away_snap["rest_days"])
        ),

        "home_form_power": home_snap["win_pct_L10"] * home_snap["goal_diff_L10"],
        "away_form_power": away_snap["win_pct_L10"] * away_snap["goal_diff_L10"],
        "diff_form_power": (
            (home_snap["win_pct_L10"] * home_snap["goal_diff_L10"])
            - (away_snap["win_pct_L10"] * away_snap["goal_diff_L10"])
        ),

        "match_parity_elo": abs(home_snap["elo_pre"] - away_snap["elo_pre"]),
        "match_parity_goal_diff": abs(home_snap["goal_diff_L10"] - away_snap["goal_diff_L10"]),
        "draw_pressure_avg": (home_snap["draw_pct_L10"] + away_snap["draw_pct_L10"]) / 2.0,
        "draw_pressure_surface_avg": (
            home_snap["surface_draw_pct_L5"] + away_snap["surface_draw_pct_L5"]
        ) / 2.0,

        "home_schedule_load_exp": (
            1.00 * home_snap["games_last_3_days"]
            + 0.70 * home_snap["games_last_5_days"]
            + 0.40 * home_snap["games_last_7_days"]
            + 0.25 * home_snap["games_last_10_days"]
            + 0.15 * home_snap["games_last_14_days"]
        ),
        "away_schedule_load_exp": (
            1.00 * away_snap["games_last_3_days"]
            + 0.70 * away_snap["games_last_5_days"]
            + 0.40 * away_snap["games_last_7_days"]
            + 0.25 * away_snap["games_last_10_days"]
            + 0.15 * away_snap["games_last_14_days"]
        ),
        "diff_schedule_load_exp": (
            (1.00 * home_snap["games_last_3_days"] + 0.70 * home_snap["games_last_5_days"] + 0.40 * home_snap["games_last_7_days"] + 0.25 * home_snap["games_last_10_days"] + 0.15 * home_snap["games_last_14_days"])
            - (1.00 * away_snap["games_last_3_days"] + 0.70 * away_snap["games_last_5_days"] + 0.40 * away_snap["games_last_7_days"] + 0.25 * away_snap["games_last_10_days"] + 0.15 * away_snap["games_last_14_days"])
        ),

        "home_attack_closing_trend": home_snap["goals_scored_L3"] - home_snap["goals_scored_L10"],
        "away_attack_closing_trend": away_snap["goals_scored_L3"] - away_snap["goals_scored_L10"],
        "diff_attack_closing_trend": (
            (home_snap["goals_scored_L3"] - home_snap["goals_scored_L10"])
            - (away_snap["goals_scored_L3"] - away_snap["goals_scored_L10"])
        ),

        "home_defense_closing_trend": home_snap["goals_allowed_L10"] - home_snap["goals_allowed_L3"],
        "away_defense_closing_trend": away_snap["goals_allowed_L10"] - away_snap["goals_allowed_L3"],
        "diff_defense_closing_trend": (
            (home_snap["goals_allowed_L10"] - home_snap["goals_allowed_L3"])
            - (away_snap["goals_allowed_L10"] - away_snap["goals_allowed_L3"])
        ),

        "home_match_stability_L10": 1.0 / (1.0 + home_snap["goal_diff_std_L10"]),
        "away_match_stability_L10": 1.0 / (1.0 + away_snap["goal_diff_std_L10"]),
        "diff_match_stability_L10": (
            (1.0 / (1.0 + home_snap["goal_diff_std_L10"]))
            - (1.0 / (1.0 + away_snap["goal_diff_std_L10"]))
        ),

        "draw_equilibrium_index": (
            (
                home_snap["draw_pct_L10"]
                + away_snap["draw_pct_L10"]
                + home_snap["surface_draw_pct_L5"]
                + away_snap["surface_draw_pct_L5"]
            )
            / 4.0
        )
        * (
            (
                np.exp(-abs(home_snap["elo_pre"] - away_snap["elo_pre"]) / 120.0)
                + np.exp(-abs(home_snap["goal_diff_L10"] - away_snap["goal_diff_L10"]) / 1.5)
            )
            / 2.0
        ),

        "odds_over_under": 2.5,
        "market_missing": 0,
    }

    return row


def build_output_rows(
    df_day: pd.DataFrame,
    schedule_df: pd.DataFrame,
    raw_history_df: pd.DataFrame,
    market_sources: dict,
):
    """Construye las predicciones finales usando los modelos ML."""
    fg_probs, fg_preds, _, _ = predict_market(df_day, "full_game", market_sources)
    over_probs, over_preds, over_threshold, _ = predict_market(df_day, "over_25", market_sources)
    btts_probs, btts_preds, btts_threshold, _ = predict_market(df_day, "btts", market_sources)
    try:
        corners_probs, corners_preds, corners_threshold, _ = predict_market(
            df_day,
            "corners_over_95",
            market_sources,
        )
        corners_model_available = True
    except Exception:
        corners_probs = np.full(len(df_day), 0.5, dtype=float)
        corners_preds = np.zeros(len(df_day), dtype=int)
        corners_threshold = 0.5
        corners_model_available = False

    output = []

    for i, row in df_day.reset_index(drop=True).iterrows():
        sched = schedule_df[schedule_df["game_id"].astype(str) == str(row["game_id"])]
        sched_row = sched.iloc[0] if not sched.empty else None

        home_team = row["home_team"]
        away_team = row["away_team"]
        date_str = row["date"]

        # Full Game (Multiclass: 0=away_win, 1=home_win, 2=draw)
        fg_pred = fg_preds[i]
        fg_probs_arr = fg_probs[i]
        fg_prob_away_base = float(fg_probs_arr[0])
        fg_prob_home_base = float(fg_probs_arr[1])
        fg_prob_draw_base = float(fg_probs_arr[2])
        
        if fg_pred == 1:
            full_game_pick = home_team
            fg_prob_pick = fg_prob_home_base
        elif fg_pred == 2:
            full_game_pick = "DRAW"
            fg_prob_pick = fg_prob_draw_base
        else:
            full_game_pick = away_team
            fg_prob_pick = fg_prob_away_base
        
        full_game_conf = confidence_from_prob(fg_prob_pick)

        # Over 2.5
        over_prob = float(over_probs[i])
        total_pick = "OVER 2.5" if over_preds[i] == 1 else "UNDER 2.5"
        over_conf = confidence_from_prob(over_prob)

        # BTTS
        btts_prob = float(btts_probs[i])
        btts_pick = "YES" if btts_preds[i] == 1 else "NO"
        btts_conf = confidence_from_prob(btts_prob)

        detected_events = []
        full_adj_breakdown = []
        total_adj_breakdown = []
        btts_adj_breakdown = []

        fg_prob_home_adj = fg_prob_home_base
        fg_prob_away_adj = fg_prob_away_base
        fg_prob_draw_adj = fg_prob_draw_base
        over_prob_adj = over_prob
        btts_prob_adj = btts_prob

        if USE_EVENT_ADJUSTMENTS and raw_history_df is not None and not raw_history_df.empty:
            recent_features = get_recent_team_form_features(
                raw_history_df,
                home_team=home_team,
                away_team=away_team,
                match_date=date_str,
                lookback=5,
            )
            h2h_features = get_h2h_features(
                raw_history_df,
                home_team=home_team,
                away_team=away_team,
                match_date=date_str,
            )

            events_full = detect_pre_match_events(recent_features, h2h_features, market_type="full_game")
            events_over = detect_pre_match_events(recent_features, h2h_features, market_type="over_25")
            events_btts = detect_pre_match_events(recent_features, h2h_features, market_type="btts")
            detected_events = events_full

            adj_home = apply_probability_adjustment(
                fg_prob_home_base, events_full, h2h_features, market_type="full_game_home"
            )
            adj_away = apply_probability_adjustment(
                fg_prob_away_base, events_full, h2h_features, market_type="full_game_away"
            )
            adj_draw = apply_probability_adjustment(
                fg_prob_draw_base, events_full, h2h_features, market_type="full_game_draw"
            )

            fg_prob_home_adj, fg_prob_away_adj, fg_prob_draw_adj = normalize_multiclass_probs(
                adj_home["adjusted_prob"], adj_away["adjusted_prob"], adj_draw["adjusted_prob"]
            )

            over_adj_info = apply_probability_adjustment(
                over_prob, events_over, h2h_features, market_type="over_25"
            )
            btts_adj_info = apply_probability_adjustment(
                btts_prob, events_btts, h2h_features, market_type="btts"
            )
            over_prob_adj = over_adj_info["adjusted_prob"]
            btts_prob_adj = btts_adj_info["adjusted_prob"]

            if full_game_pick == home_team:
                full_adj_breakdown = adj_home["adjustment_breakdown"]
                full_adj_amount = fg_prob_home_adj - fg_prob_home_base
            elif full_game_pick == away_team:
                full_adj_breakdown = adj_away["adjustment_breakdown"]
                full_adj_amount = fg_prob_away_adj - fg_prob_away_base
            else:
                full_adj_breakdown = adj_draw["adjustment_breakdown"]
                full_adj_amount = fg_prob_draw_adj - fg_prob_draw_base

            total_adj_breakdown = over_adj_info["adjustment_breakdown"]
            btts_adj_breakdown = btts_adj_info["adjustment_breakdown"]
        else:
            if full_game_pick == home_team:
                full_adj_amount = 0.0
            elif full_game_pick == away_team:
                full_adj_amount = 0.0
            else:
                full_adj_amount = 0.0

        fg_adj_idx = int(np.argmax([fg_prob_away_adj, fg_prob_home_adj, fg_prob_draw_adj]))
        if fg_adj_idx == 1:
            full_game_recommended_pick = home_team
            full_game_adjusted_prob = fg_prob_home_adj
            full_game_base_prob = fg_prob_home_base
        elif fg_adj_idx == 2:
            full_game_recommended_pick = "DRAW"
            full_game_adjusted_prob = fg_prob_draw_adj
            full_game_base_prob = fg_prob_draw_base
        else:
            full_game_recommended_pick = away_team
            full_game_adjusted_prob = fg_prob_away_adj
            full_game_base_prob = fg_prob_away_base

        full_game_adjusted_conf = probability_to_confidence(full_game_adjusted_prob)
        liga_patterns = generate_liga_mx_patterns(row.to_dict())
        pattern_edge = aggregate_pattern_edge(liga_patterns)
        full_game_score = fuse_with_pattern_score(recommendation_score(full_game_adjusted_prob), pattern_edge)

        total_recommended_pick = "OVER 2.5" if over_prob_adj >= over_threshold else "UNDER 2.5"
        btts_recommended_pick = "BTTS YES" if btts_prob_adj >= btts_threshold else "BTTS NO"
        total_score = fuse_with_pattern_score(recommendation_score(over_prob_adj), pattern_edge)
        btts_score = fuse_with_pattern_score(recommendation_score(btts_prob_adj), pattern_edge)
        over_adj_amount = over_prob_adj - over_prob
        btts_adj_amount = btts_prob_adj - btts_prob

        corners_line = 9.5
        corners_prob_over = float(corners_probs[i])
        if corners_model_available:
            corners_pick = f"OVER {corners_line}" if corners_preds[i] == 1 else f"UNDER {corners_line}"
        else:
            corners_pick = f"OVER {corners_line}" if corners_prob_over >= corners_threshold else f"UNDER {corners_line}"
        corners_confidence = confidence_from_prob(corners_prob_over)
        corners_score = fuse_with_pattern_score(
            recommendation_score(corners_prob_over),
            pattern_edge,
        )

        # Spread pick
        if full_game_pick == "DRAW":
            spread_pick = f"1X {home_team}"
        else:
            spread_pick = f"DNB {full_game_pick}"

        schedule_total_line = 2.5
        if sched_row is not None:
            try:
                schedule_total_line = float(sched_row.get("odds_over_under", 2.5) or 2.5)
            except Exception:
                schedule_total_line = 2.5

        schedule_odds_quality = "fallback"
        if sched_row is not None:
            schedule_odds_quality = str(sched_row.get("odds_data_quality", "fallback") or "fallback")

        game_name = f"{away_team} @ {home_team}"

        output.append(
            {
                "game_id": str(row["game_id"]),
                "date": str(date_str),
                "time": "" if sched_row is None else str(sched_row.get("time", "") or ""),
                "game_name": game_name,
                "home_team": home_team,
                "away_team": away_team,

                "full_game_pick": full_game_pick,
                "full_game_confidence": full_game_conf,
                "full_game_tier": tier_from_conf(full_game_conf),

                "base_probability": round(float(full_game_base_prob), 6),
                "adjusted_probability": round(float(full_game_adjusted_prob), 6),
                "adjustment_amount": round(float(full_game_adjusted_prob - full_game_base_prob), 6),
                "adjustment_breakdown": full_adj_breakdown,
                "detected_events": detected_events,
                "detected_patterns": liga_patterns,
                "pattern_edge": round(float(pattern_edge), 4),
                "recommended_pick": full_game_recommended_pick,
                "recommended_confidence": full_game_adjusted_conf,
                "recommended_score": round(float(full_game_score), 1),
                "recommended": bool(full_game_score >= 56.0),

                "spread_pick": spread_pick,
                "spread_market": "DNB / 1X",

                "total_pick": total_pick,
                "total_confidence": over_conf,
                "total_action": "JUGAR" if over_conf >= 56 else "PASS",
                "total_base_probability": round(float(over_prob), 6),
                "total_adjusted_probability": round(float(over_prob_adj), 6),
                "total_adjustment_amount": round(float(over_adj_amount), 6),
                "total_adjustment_breakdown": total_adj_breakdown,
                "total_recommended_pick": total_recommended_pick,
                "total_recommended_score": round(float(total_score), 1),
                "odds_over_under": schedule_total_line,
                "closing_moneyline_odds": None if sched_row is None else sched_row.get("closing_moneyline_odds"),
                "home_moneyline_odds": None if sched_row is None else sched_row.get("home_moneyline_odds"),
                "away_moneyline_odds": None if sched_row is None else sched_row.get("away_moneyline_odds"),
                "closing_spread_odds": None if sched_row is None else sched_row.get("closing_spread_odds"),
                "closing_total_odds": None if sched_row is None else sched_row.get("closing_total_odds"),
                "odds_data_quality": schedule_odds_quality,

                "btts_pick": f"BTTS {btts_pick}",
                "btts_confidence": btts_conf,
                "btts_tier": tier_from_conf(btts_conf),
                "btts_base_probability": round(float(btts_prob), 6),
                "btts_adjusted_probability": round(float(btts_prob_adj), 6),
                "btts_adjustment_amount": round(float(btts_adj_amount), 6),
                "btts_adjustment_breakdown": btts_adj_breakdown,
                "btts_recommended_pick": btts_recommended_pick,
                "btts_recommended_score": round(float(btts_score), 1),

                "corners_market": "TOTAL CORNERS O/U",
                "corners_line": corners_line,
                "corners_pick": corners_pick,
                "corners_confidence": corners_confidence,
                "corners_model_prob_over": round(float(corners_prob_over), 6),
                "corners_recommended_pick": corners_pick,
                "corners_recommended_score": round(float(corners_score), 1),
                "corners_action": "JUGAR" if corners_confidence >= 56 else "PASS",

                "status_state": "" if sched_row is None else str(sched_row.get("status_state", "") or ""),
                "status_description": "" if sched_row is None else str(sched_row.get("status_description", "") or ""),
            }
        )

    return output



def write_empty_predictions(output_path: Path, date_str: str):
    """Escribe un JSON vacío cuando no hay partidos."""
    payload = {
        "date": date_str,
        "sport": "liga_mx",
        "generated_at": datetime.now().isoformat(),
        "games": [],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[OK] Archivo vacio generado en: {output_path}")


def main():
    today_date = datetime.now().strftime("%Y-%m-%d")

    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"No existe el archivo de features: {FEATURES_FILE}")

    if not UPCOMING_FILE.exists():
        print(f"⚠️ No existe el archivo de agenda: {UPCOMING_FILE}")
        output_path = PREDICTIONS_DIR / f"{today_date}.json"
        write_empty_predictions(output_path, today_date)
        return

    history_df = pd.read_csv(FEATURES_FILE, dtype={"game_id": str})
    schedule_df = pd.read_csv(UPCOMING_FILE, dtype={"game_id": str})

    raw_history_df = pd.DataFrame()
    if RAW_HISTORY_FILE.exists():
        try:
            raw_history_df = pd.read_csv(RAW_HISTORY_FILE, dtype={"game_id": str})
            raw_history_df["date"] = raw_history_df["date"].astype(str)
        except Exception:
            raw_history_df = pd.DataFrame()

    if history_df.empty or schedule_df.empty:
        print("⚠️ Dataset vacío")
        output_path = PREDICTIONS_DIR / f"{today_date}.json"
        write_empty_predictions(output_path, today_date)
        return

    history_df["date"] = history_df["date"].astype(str)
    schedule_df["date"] = schedule_df["date"].astype(str)
    market_sources = load_market_feature_sources(SELECTIVE_PLAN_FILE)
    print(f"[INFO] Market sources Liga MX: {market_sources}")

    if "status_completed" not in schedule_df.columns:
        schedule_df["status_completed"] = 0

    schedule_df["status_completed"] = (
        pd.to_numeric(schedule_df["status_completed"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    total_predictions = 0

    for date_str, sched in schedule_df.groupby("date"):
        sched = sched.sort_values(["time", "game_id"]).copy()
        output_path = PREDICTIONS_DIR / f"{date_str}.json"

        feature_rows = []
        for _, srow in sched.iterrows():
            feat_row = build_pregame_feature_row(history_df, srow)
            if feat_row is not None:
                feature_rows.append(feat_row)

        if not feature_rows:
            write_empty_predictions(output_path, date_str)
            print(f"ℹ️ {date_str}: sin features suficientes")
            continue

        df_features = pd.DataFrame(feature_rows)
        if any(source == "v3" for source in market_sources.values()):
            df_features = add_v3_features(df_features)

        predictions = build_output_rows(df_features, sched, raw_history_df, market_sources)

        payload = {
            "date": date_str,
            "sport": "liga_mx",
            "generated_at": datetime.now().isoformat(),
            "games": predictions,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        total_predictions += len(predictions)
        print(f"[OK] {date_str}: {len(predictions)} predicciones -> {output_path.name}")

    print(f"\n[OK] Total predicciones Liga MX: {total_predictions}")


if __name__ == "__main__":
    main()