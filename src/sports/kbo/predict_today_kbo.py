import json
from pathlib import Path
from datetime import datetime
import sys

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import joblib
import numpy as np
import pandas as pd
from calibration import calibrate_probability, load_calibration_config
from pattern_engine import aggregate_pattern_edge
from pattern_engine_kbo import generate_kbo_patterns
from pick_selector import recommendation_score
from pick_selector import fuse_with_pattern_score

BASE_DIR = SRC_ROOT

FEATURES_FILE = BASE_DIR / "data" / "kbo" / "processed" / "model_ready_features_kbo.csv"
UPCOMING_FILE = BASE_DIR / "data" / "kbo" / "raw" / "kbo_upcoming_schedule.csv"
MODELS_DIR = BASE_DIR / "data" / "kbo" / "models"
PREDICTIONS_DIR = BASE_DIR / "data" / "kbo" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_FILE = MODELS_DIR / "calibration_params.json"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_market_assets(market_key: str):
    market_dir = MODELS_DIR / market_key
    xgb = joblib.load(market_dir / "xgb_model.pkl")
    lgbm = joblib.load(market_dir / "lgbm_model.pkl")
    feature_columns = load_json(market_dir / "feature_columns.json")
    metadata = load_json(market_dir / "metadata.json")
    threshold = metadata.get("ensemble_threshold", 0.5)
    return xgb, lgbm, feature_columns, metadata, threshold


def predict_market(df: pd.DataFrame, market_key: str):
    xgb, lgbm, feature_columns, metadata, threshold = load_market_assets(market_key)

    X = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    xgb_probs = xgb.predict_proba(X)[:, 1]
    lgbm_probs = lgbm.predict_proba(X)[:, 1]
    weights = metadata.get("ensemble_weights", {"xgboost": 0.5, "lightgbm": 0.5})
    wx = float(weights.get("xgboost", 0.5))
    wl = float(weights.get("lightgbm", 0.5))
    probs = wx * xgb_probs + wl * lgbm_probs
    preds = (probs >= threshold).astype(int)
    return probs, preds, threshold, metadata


def confidence_from_prob(prob: float) -> int:
    return int(round(max(prob, 1 - prob) * 100))


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
    """
    Devuelve el Ãºltimo snapshot pregame disponible del equipo antes de cutoff_date.
    Usa cualquier partido previo donde ese equipo haya sido local o visitante.
    """
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
            "win_pct_L5": row["home_win_pct_L5"],
            "win_pct_L10": row["home_win_pct_L10"],
            "run_diff_L5": row["home_run_diff_L5"],
            "run_diff_L10": row["home_run_diff_L10"],
            "runs_scored_L5": row["home_runs_scored_L5"],
            "runs_allowed_L5": row["home_runs_allowed_L5"],
            "yrfi_rate_L10": row["home_yrfi_rate_L10"],
            "r1_scored_rate_L10": row["home_r1_scored_rate_L10"],
            "r1_allowed_rate_L10": row["home_r1_allowed_rate_L10"],
            "f5_win_pct_L5": row["home_f5_win_pct_L5"],
            "f5_diff_L5": row["home_f5_diff_L5"],
            "hits_L5": row["home_hits_L5"],
            "hits_allowed_L5": row["home_hits_allowed_L5"],
            "surface_win_pct_L5": row.get("home_home_only_win_pct_L5", 0),
            "surface_run_diff_L5": row.get("home_home_only_run_diff_L5", 0),
            "surface_yrfi_rate_L10": row.get("home_home_only_yrfi_rate_L10", 0),
            "surface_f5_win_pct_L5": row.get("home_home_only_f5_win_pct_L5", 0),
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
            "win_pct_L5": row["away_win_pct_L5"],
            "win_pct_L10": row["away_win_pct_L10"],
            "run_diff_L5": row["away_run_diff_L5"],
            "run_diff_L10": row["away_run_diff_L10"],
            "runs_scored_L5": row["away_runs_scored_L5"],
            "runs_allowed_L5": row["away_runs_allowed_L5"],
            "yrfi_rate_L10": row["away_yrfi_rate_L10"],
            "r1_scored_rate_L10": row["away_r1_scored_rate_L10"],
            "r1_allowed_rate_L10": row["away_r1_allowed_rate_L10"],
            "f5_win_pct_L5": row["away_f5_win_pct_L5"],
            "f5_diff_L5": row["away_f5_diff_L5"],
            "hits_L5": row["away_hits_L5"],
            "hits_allowed_L5": row["away_hits_allowed_L5"],
            "surface_win_pct_L5": row.get("away_away_only_win_pct_L5", 0),
            "surface_run_diff_L5": row.get("away_away_only_run_diff_L5", 0),
            "surface_yrfi_rate_L10": row.get("away_away_only_yrfi_rate_L10", 0),
            "surface_f5_win_pct_L5": row.get("away_away_only_f5_win_pct_L5", 0),
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


def get_league_means_before_date(history_df: pd.DataFrame, cutoff_date: str):
    prior = history_df[history_df["date"] < cutoff_date].copy()
    if prior.empty:
        return {
            "league_win_pct_L10": 0.5,
            "league_run_diff_L10": 0.0,
            "league_yrfi_rate_L10": 0.5,
            "league_f5_win_pct_L5": 0.5,
        }

    win_vals = pd.concat([prior["home_win_pct_L10"], prior["away_win_pct_L10"]], ignore_index=True)
    run_vals = pd.concat([prior["home_run_diff_L10"], prior["away_run_diff_L10"]], ignore_index=True)
    yrfi_vals = pd.concat([prior["home_yrfi_rate_L10"], prior["away_yrfi_rate_L10"]], ignore_index=True)
    f5_vals = pd.concat([prior["home_f5_win_pct_L5"], prior["away_f5_win_pct_L5"]], ignore_index=True)

    return {
        "league_win_pct_L10": float(win_vals.mean()) if not win_vals.empty else 0.5,
        "league_run_diff_L10": float(run_vals.mean()) if not run_vals.empty else 0.0,
        "league_yrfi_rate_L10": float(yrfi_vals.mean()) if not yrfi_vals.empty else 0.5,
        "league_f5_win_pct_L5": float(f5_vals.mean()) if not f5_vals.empty else 0.5,
    }


def build_pregame_feature_row(history_df: pd.DataFrame, schedule_row: pd.Series):
    date_str = str(schedule_row["date"])
    home_team = str(schedule_row["home_team"])
    away_team = str(schedule_row["away_team"])

    home_snap = get_team_snapshot(history_df, home_team, date_str)
    away_snap = get_team_snapshot(history_df, away_team, date_str)
    league_means = get_league_means_before_date(history_df, date_str)

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

        "home_win_pct_L5": home_snap["win_pct_L5"],
        "away_win_pct_L5": away_snap["win_pct_L5"],
        "diff_win_pct_L5": home_snap["win_pct_L5"] - away_snap["win_pct_L5"],

        "home_win_pct_L10": home_snap["win_pct_L10"],
        "away_win_pct_L10": away_snap["win_pct_L10"],
        "diff_win_pct_L10": home_snap["win_pct_L10"] - away_snap["win_pct_L10"],

        "home_run_diff_L5": home_snap["run_diff_L5"],
        "away_run_diff_L5": away_snap["run_diff_L5"],
        "diff_run_diff_L5": home_snap["run_diff_L5"] - away_snap["run_diff_L5"],

        "home_run_diff_L10": home_snap["run_diff_L10"],
        "away_run_diff_L10": away_snap["run_diff_L10"],
        "diff_run_diff_L10": home_snap["run_diff_L10"] - away_snap["run_diff_L10"],

        "home_runs_scored_L5": home_snap["runs_scored_L5"],
        "away_runs_scored_L5": away_snap["runs_scored_L5"],
        "diff_runs_scored_L5": home_snap["runs_scored_L5"] - away_snap["runs_scored_L5"],

        "home_runs_allowed_L5": home_snap["runs_allowed_L5"],
        "away_runs_allowed_L5": away_snap["runs_allowed_L5"],
        "diff_runs_allowed_L5": home_snap["runs_allowed_L5"] - away_snap["runs_allowed_L5"],

        "home_yrfi_rate_L10": home_snap["yrfi_rate_L10"],
        "away_yrfi_rate_L10": away_snap["yrfi_rate_L10"],
        "diff_yrfi_rate_L10": home_snap["yrfi_rate_L10"] - away_snap["yrfi_rate_L10"],

        "home_r1_scored_rate_L10": home_snap["r1_scored_rate_L10"],
        "away_r1_scored_rate_L10": away_snap["r1_scored_rate_L10"],
        "diff_r1_scored_rate_L10": home_snap["r1_scored_rate_L10"] - away_snap["r1_scored_rate_L10"],

        "home_r1_allowed_rate_L10": home_snap["r1_allowed_rate_L10"],
        "away_r1_allowed_rate_L10": away_snap["r1_allowed_rate_L10"],
        "diff_r1_allowed_rate_L10": home_snap["r1_allowed_rate_L10"] - away_snap["r1_allowed_rate_L10"],

        "home_f5_win_pct_L5": home_snap["f5_win_pct_L5"],
        "away_f5_win_pct_L5": away_snap["f5_win_pct_L5"],
        "diff_f5_win_pct_L5": home_snap["f5_win_pct_L5"] - away_snap["f5_win_pct_L5"],

        "home_f5_diff_L5": home_snap["f5_diff_L5"],
        "away_f5_diff_L5": away_snap["f5_diff_L5"],
        "diff_f5_diff_L5": home_snap["f5_diff_L5"] - away_snap["f5_diff_L5"],

        "home_hits_L5": home_snap["hits_L5"],
        "away_hits_L5": away_snap["hits_L5"],
        "diff_hits_L5": home_snap["hits_L5"] - away_snap["hits_L5"],

        "home_hits_allowed_L5": home_snap["hits_allowed_L5"],
        "away_hits_allowed_L5": away_snap["hits_allowed_L5"],
        "diff_hits_allowed_L5": home_snap["hits_allowed_L5"] - away_snap["hits_allowed_L5"],

        "home_home_only_win_pct_L5": home_snap["surface_win_pct_L5"],
        "away_away_only_win_pct_L5": away_snap["surface_win_pct_L5"],
        "diff_surface_win_pct_L5": home_snap["surface_win_pct_L5"] - away_snap["surface_win_pct_L5"],

        "home_home_only_run_diff_L5": home_snap["surface_run_diff_L5"],
        "away_away_only_run_diff_L5": away_snap["surface_run_diff_L5"],
        "diff_surface_run_diff_L5": home_snap["surface_run_diff_L5"] - away_snap["surface_run_diff_L5"],

        "home_home_only_yrfi_rate_L10": home_snap["surface_yrfi_rate_L10"],
        "away_away_only_yrfi_rate_L10": away_snap["surface_yrfi_rate_L10"],
        "diff_surface_yrfi_rate_L10": home_snap["surface_yrfi_rate_L10"] - away_snap["surface_yrfi_rate_L10"],

        "home_home_only_f5_win_pct_L5": home_snap["surface_f5_win_pct_L5"],
        "away_away_only_f5_win_pct_L5": away_snap["surface_f5_win_pct_L5"],
        "diff_surface_f5_win_pct_L5": home_snap["surface_f5_win_pct_L5"] - away_snap["surface_f5_win_pct_L5"],

        "home_win_pct_L10_vs_league": home_snap["win_pct_L10"] - league_means["league_win_pct_L10"],
        "away_win_pct_L10_vs_league": away_snap["win_pct_L10"] - league_means["league_win_pct_L10"],
        "diff_win_pct_L10_vs_league": (
            (home_snap["win_pct_L10"] - league_means["league_win_pct_L10"])
            - (away_snap["win_pct_L10"] - league_means["league_win_pct_L10"])
        ),

        "home_run_diff_L10_vs_league": home_snap["run_diff_L10"] - league_means["league_run_diff_L10"],
        "away_run_diff_L10_vs_league": away_snap["run_diff_L10"] - league_means["league_run_diff_L10"],
        "diff_run_diff_L10_vs_league": (
            (home_snap["run_diff_L10"] - league_means["league_run_diff_L10"])
            - (away_snap["run_diff_L10"] - league_means["league_run_diff_L10"])
        ),

        "home_yrfi_rate_L10_vs_league": home_snap["yrfi_rate_L10"] - league_means["league_yrfi_rate_L10"],
        "away_yrfi_rate_L10_vs_league": away_snap["yrfi_rate_L10"] - league_means["league_yrfi_rate_L10"],
        "diff_yrfi_rate_L10_vs_league": (
            (home_snap["yrfi_rate_L10"] - league_means["league_yrfi_rate_L10"])
            - (away_snap["yrfi_rate_L10"] - league_means["league_yrfi_rate_L10"])
        ),

        "home_f5_win_pct_L5_vs_league": home_snap["f5_win_pct_L5"] - league_means["league_f5_win_pct_L5"],
        "away_f5_win_pct_L5_vs_league": away_snap["f5_win_pct_L5"] - league_means["league_f5_win_pct_L5"],
        "diff_f5_win_pct_L5_vs_league": (
            (home_snap["f5_win_pct_L5"] - league_means["league_f5_win_pct_L5"])
            - (away_snap["f5_win_pct_L5"] - league_means["league_f5_win_pct_L5"])
        ),

        "home_momentum_win": home_snap["win_pct_L5"] - home_snap["win_pct_L10"],
        "away_momentum_win": away_snap["win_pct_L5"] - away_snap["win_pct_L10"],
        "diff_momentum_win": (
            (home_snap["win_pct_L5"] - home_snap["win_pct_L10"])
            - (away_snap["win_pct_L5"] - away_snap["win_pct_L10"])
        ),

        "home_momentum_run_diff": home_snap["run_diff_L5"] - home_snap["run_diff_L10"],
        "away_momentum_run_diff": away_snap["run_diff_L5"] - away_snap["run_diff_L10"],
        "diff_momentum_run_diff": (
            (home_snap["run_diff_L5"] - home_snap["run_diff_L10"])
            - (away_snap["run_diff_L5"] - away_snap["run_diff_L10"])
        ),

        "home_surface_edge": home_snap["surface_win_pct_L5"] - home_snap["win_pct_L10"],
        "away_surface_edge": away_snap["surface_win_pct_L5"] - away_snap["win_pct_L10"],
        "diff_surface_edge": (
            (home_snap["surface_win_pct_L5"] - home_snap["win_pct_L10"])
            - (away_snap["surface_win_pct_L5"] - away_snap["win_pct_L10"])
        ),

        "home_fatigue_index": home_snap["games_last_5_days"] - home_snap["rest_days"],
        "away_fatigue_index": away_snap["games_last_5_days"] - away_snap["rest_days"],
        "diff_fatigue_index": (
            (home_snap["games_last_5_days"] - home_snap["rest_days"])
            - (away_snap["games_last_5_days"] - away_snap["rest_days"])
        ),

        "home_form_power": home_snap["win_pct_L10"] * home_snap["run_diff_L10"],
        "away_form_power": away_snap["win_pct_L10"] * away_snap["run_diff_L10"],
        "diff_form_power": (
            (home_snap["win_pct_L10"] * home_snap["run_diff_L10"])
            - (away_snap["win_pct_L10"] * away_snap["run_diff_L10"])
        ),

        # Mercado pregame: si no hay lÃ­nea, no inventamos
        "home_is_favorite": -1,
        "odds_over_under": 0.0,
        "market_missing": 1,
    }

    return row


def build_output_rows(df_day: pd.DataFrame, schedule_df: pd.DataFrame):
    calibration_cfg = load_calibration_config(CALIBRATION_FILE)

    fg_probs, fg_preds, _, _ = predict_market(df_day, "full_game")
    yrfi_probs, yrfi_preds, _, _ = predict_market(df_day, "yrfi")
    f5_probs, f5_preds, _, _ = predict_market(df_day, "f5")

    output = []

    for i, row in df_day.reset_index(drop=True).iterrows():
        sched = schedule_df[schedule_df["game_id"].astype(str) == str(row["game_id"])]
        sched_row = sched.iloc[0] if not sched.empty else None

        home_team = row["home_team"]
        away_team = row["away_team"]
        date_str = row["date"]

        fg_model_prob_home = float(fg_probs[i])
        f5_model_prob_home = float(f5_probs[i])
        yrfi_model_prob_yes = float(yrfi_probs[i])

        fg_prob_home = calibrate_probability(fg_model_prob_home, "kbo", "full_game", calibration_cfg)
        f5_prob_home = calibrate_probability(f5_model_prob_home, "kbo", "f5", calibration_cfg)
        yrfi_prob_yes = calibrate_probability(yrfi_model_prob_yes, "kbo", "yrfi", calibration_cfg)

        kbo_patterns = generate_kbo_patterns(row.to_dict())
        pattern_edge = aggregate_pattern_edge(kbo_patterns)

        full_game_pick = home_team if fg_preds[i] == 1 else away_team
        full_game_conf = confidence_from_prob(fg_prob_home)
        full_game_score = fuse_with_pattern_score(recommendation_score(fg_prob_home), pattern_edge)

        f5_pick = f"{home_team} F5" if f5_preds[i] == 1 else f"{away_team} F5"
        f5_conf = confidence_from_prob(f5_prob_home)
        f5_score = fuse_with_pattern_score(recommendation_score(f5_prob_home), pattern_edge)

        yrfi_pick = "YRFI" if yrfi_preds[i] == 1 else "NRFI"
        yrfi_conf = confidence_from_prob(yrfi_prob_yes)
        yrfi_score = fuse_with_pattern_score(recommendation_score(yrfi_prob_yes), pattern_edge)

        total_line = 0.0
        if sched_row is not None:
            try:
                total_line = float(sched_row.get("odds_over_under", 0) or 0)
            except Exception:
                total_line = 0.0

        total_pick = "OVER" if yrfi_preds[i] == 1 else "UNDER"
        spread_pick = f"{full_game_pick} ML"

        game_name = f"{away_team} @ {home_team}"

        home_ml_odds = None if sched_row is None else sched_row.get("home_moneyline_odds")
        away_ml_odds = None if sched_row is None else sched_row.get("away_moneyline_odds")
        closing_ml_odds = None if sched_row is None else sched_row.get("closing_moneyline_odds")
        closing_spread_odds = None if sched_row is None else sched_row.get("closing_spread_odds")
        closing_total_odds = None if sched_row is None else sched_row.get("closing_total_odds")
        odds_details = "No Line" if sched_row is None else str(sched_row.get("odds_details", "No Line") or "No Line")
        odds_data_quality = "none" if sched_row is None else str(sched_row.get("odds_data_quality", "none") or "none")
        selected_ml_odds = home_ml_odds if full_game_pick == home_team else away_ml_odds

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
                "full_game_model_prob_home": round(fg_model_prob_home, 4),
                "full_game_calibrated_prob_home": round(fg_prob_home, 4),
                "full_game_pattern_edge": round(pattern_edge, 4),
                "full_game_detected_patterns": kbo_patterns,
                "full_game_recommended_score": round(full_game_score, 1),
                "full_game_recommended": bool(full_game_score >= 56.0),

                "q1_pick": yrfi_pick,
                "q1_confidence": yrfi_conf,
                "q1_action": "JUGAR" if yrfi_conf >= 56 else "PASS",
                "q1_model_prob_yes": round(yrfi_model_prob_yes, 4),
                "q1_calibrated_prob_yes": round(yrfi_prob_yes, 4),
                "q1_recommended_score": round(yrfi_score, 1),

                "spread_pick": spread_pick,
                "spread_market": "ML",

                "total_pick": total_pick,
                "odds_over_under": total_line,
                "closing_total_line": total_line if total_line > 0 else None,
                "odds_details": odds_details,
                "odds_data_quality": odds_data_quality,
                "home_moneyline_odds": None if pd.isna(home_ml_odds) else float(home_ml_odds),
                "away_moneyline_odds": None if pd.isna(away_ml_odds) else float(away_ml_odds),
                "closing_moneyline_odds": None if pd.isna(closing_ml_odds) else float(closing_ml_odds),
                "closing_spread_odds": None if pd.isna(closing_spread_odds) else float(closing_spread_odds),
                "closing_total_odds": None if pd.isna(closing_total_odds) else float(closing_total_odds),
                "moneyline_odds": None if pd.isna(selected_ml_odds) else float(selected_ml_odds),
                "pick_ml_odds": None if pd.isna(selected_ml_odds) else float(selected_ml_odds),

                "assists_pick": f5_pick,
                "extra_f5_confidence": f5_conf,
                "extra_f5_tier": tier_from_conf(f5_conf),
                "extra_f5_model_prob_home": round(f5_model_prob_home, 4),
                "extra_f5_calibrated_prob_home": round(f5_prob_home, 4),
                "extra_f5_recommended_score": round(f5_score, 1),

                "status_state": "" if sched_row is None else str(sched_row.get("status_state", "") or ""),
                "status_description": "" if sched_row is None else str(sched_row.get("status_description", "") or ""),
                "status_completed": 0 if sched_row is None else int(sched_row.get("status_completed", 0) or 0),
                "home_score": None if sched_row is None else int(sched_row.get("home_runs_total", 0) or 0),
                "away_score": None if sched_row is None else int(sched_row.get("away_runs_total", 0) or 0),
            }
        )

    return output


def main():
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"No existe el archivo: {FEATURES_FILE}")
    if not UPCOMING_FILE.exists():
        raise FileNotFoundError(
            f"No existe el archivo de agenda del dÃ­a: {UPCOMING_FILE}\n"
            f"Primero corre data_ingest_kbo.py"
        )

    history_df = pd.read_csv(FEATURES_FILE)
    history_df["date"] = history_df["date"].astype(str)

    schedule_df = pd.read_csv(UPCOMING_FILE, dtype={"game_id": str})
    if schedule_df.empty:
        raise ValueError("La agenda del dÃ­a estÃ¡ vacÃ­a.")

    schedule_df["date"] = schedule_df["date"].astype(str)

    schedule_df["status_completed"] = pd.to_numeric(
        schedule_df.get("status_completed", 0), errors="coerce"
    ).fillna(0).astype(int)

    total_rows = 0
    total_skipped = 0

    for date_str, sched in schedule_df.groupby("date"):
        sched = sched.sort_values(["time", "game_id"]).copy()

        pregame_rows = []
        skipped = []

        for _, srow in sched.iterrows():
            row = build_pregame_feature_row(history_df, srow)
            if row is None:
                skipped.append(f"{srow['away_team']} @ {srow['home_team']}")
                continue
            pregame_rows.append(row)

        output_path = PREDICTIONS_DIR / f"{date_str}.json"

        if not pregame_rows:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2, ensure_ascii=False)
            print(f"âš ï¸ {date_str}: sin features suficientes, se guardÃ³ archivo vacÃ­o")
            total_skipped += len(skipped)
            continue

        df_day = pd.DataFrame(pregame_rows)
        rows = build_output_rows(df_day, sched)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

        print(f"âœ… {date_str}: predicciones kbo {len(rows)} juegos -> {output_path.name}")
        total_rows += len(rows)
        total_skipped += len(skipped)

    print(f"ðŸ“¦ Total predicciones kbo generadas: {total_rows}")
    if total_skipped:
        print(f"âš ï¸ Juegos omitidos por falta de histÃ³rico: {total_skipped}")


if __name__ == "__main__":
    main()
