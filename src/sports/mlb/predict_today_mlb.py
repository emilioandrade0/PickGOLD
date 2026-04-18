import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Ensure project `src` root is on sys.path so imports of shared modules work
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import joblib
import numpy as np
import pandas as pd
from calibration import calibrate_probability, load_calibration_config
from pattern_engine import aggregate_pattern_edge
from pattern_engine_mlb import generate_mlb_patterns
from pick_selector import recommendation_score
from pick_selector import fuse_with_pattern_score

# Base directory should point to the project `src` root so data lives under src/data
BASE_DIR = SRC_ROOT

FEATURES_FILE = BASE_DIR / "data" / "mlb" / "processed" / "model_ready_features_mlb.csv"
UPCOMING_FILE = BASE_DIR / "data" / "mlb" / "raw" / "mlb_upcoming_schedule.csv"
LINE_MOVEMENT_FILE = BASE_DIR / "data" / "mlb" / "cache" / "line_movement.csv"
MODELS_DIR = BASE_DIR / "data" / "mlb" / "models"
PREDICTIONS_DIR = BASE_DIR / "data" / "mlb" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_FILE = MODELS_DIR / "calibration_params.json"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


FULL_GAME_VOL_NORM_ENABLED = int(np.clip(_env_float("NBA_MLB_FULL_GAME_VOL_NORM_ENABLED", 0.0), 0.0, 1.0))
FULL_GAME_VOL_NORM_ALPHA = float(np.clip(_env_float("NBA_MLB_FULL_GAME_VOL_NORM_ALPHA", 0.18), 0.0, 0.45))
FULL_GAME_VOL_NORM_THRESHOLD_BONUS = float(np.clip(_env_float("NBA_MLB_FULL_GAME_VOL_NORM_THRESHOLD_BONUS", 0.02), 0.0, 0.08))

_full_game_vol_decision_default = 1.0 if FULL_GAME_VOL_NORM_ENABLED > 0 else 0.0
FULL_GAME_VOL_DECISION_ENABLED = int(
    np.clip(
        _env_float("NBA_MLB_FULL_GAME_VOL_DECISION_ENABLED", _full_game_vol_decision_default),
        0.0,
        1.0,
    )
)
FULL_GAME_VOL_DECISION_BETA = float(np.clip(_env_float("NBA_MLB_FULL_GAME_VOL_DECISION_BETA", 0.0), -0.30, 0.30))
FULL_GAME_VOL_DECISION_CENTER = float(np.clip(_env_float("NBA_MLB_FULL_GAME_VOL_DECISION_CENTER", 0.50), 0.0, 1.0))
FULL_GAME_VOL_DECISION_MAX_SHIFT = float(np.clip(_env_float("NBA_MLB_FULL_GAME_VOL_DECISION_MAX_SHIFT", 0.06), 0.0, 0.20))

FULL_GAME_RELIABILITY_ENABLED = int(
    np.clip(_env_float("NBA_MLB_FULL_GAME_RELIABILITY_ENABLED", 0.0), 0.0, 1.0)
)
FULL_GAME_RELIABILITY_SHRINK_ALPHA = float(
    np.clip(_env_float("NBA_MLB_FULL_GAME_RELIABILITY_SHRINK_ALPHA", 0.14), 0.0, 0.60)
)
FULL_GAME_RELIABILITY_SIDE_SHIFT = float(
    np.clip(_env_float("NBA_MLB_FULL_GAME_RELIABILITY_SIDE_SHIFT", 0.035), 0.0, 0.25)
)
FULL_GAME_RELIABILITY_CONFLICT_SHIFT = float(
    np.clip(_env_float("NBA_MLB_FULL_GAME_RELIABILITY_CONFLICT_SHIFT", 0.018), 0.0, 0.25)
)


def get_latest_prediction_dependency_mtime() -> float:
    dependency_paths = [
        FEATURES_FILE,
        UPCOMING_FILE,
        LINE_MOVEMENT_FILE,
        CALIBRATION_FILE,
        Path(__file__),
    ]

    for market_key in ["full_game", "yrfi", "f5", "totals", "total_hits_event", "run_line"]:
        market_dir = MODELS_DIR / market_key
        dependency_paths.extend(
            [
                market_dir / "xgb_model.pkl",
                market_dir / "lgbm_model.pkl",
                market_dir / "feature_columns.json",
                market_dir / "metadata.json",
            ]
        )

    latest_mtime = 0.0
    for path in dependency_paths:
        try:
            if path.exists():
                latest_mtime = max(latest_mtime, float(path.stat().st_mtime))
        except Exception:
            continue

    return latest_mtime


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


def load_regression_assets(market_key: str):
    market_dir = MODELS_DIR / market_key
    xgb = joblib.load(market_dir / "xgb_model.pkl")
    lgbm = joblib.load(market_dir / "lgbm_model.pkl")
    feature_columns = load_json(market_dir / "feature_columns.json")
    metadata = load_json(market_dir / "metadata.json")
    weights = metadata.get("ensemble_weights", {"xgboost": 0.5, "lightgbm": 0.5})
    return xgb, lgbm, feature_columns, metadata, weights


def align_features_for_market(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in feature_columns:
        if col not in out.columns:
            out[col] = 0.0
    return out

def predict_market(df: pd.DataFrame, market_key: str):
    xgb, lgbm, feature_columns, metadata, threshold = load_market_assets(market_key)

    df = align_features_for_market(df, feature_columns)
    X = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    xgb_probs = xgb.predict_proba(X)[:, 1]
    lgbm_probs = lgbm.predict_proba(X)[:, 1]
    weights = metadata.get("ensemble_weights", {"xgboost": 0.5, "lightgbm": 0.5})
    wx = float(weights.get("xgboost", 0.5))
    wl = float(weights.get("lightgbm", 0.5))
    probs = wx * xgb_probs + wl * lgbm_probs
    preds = (probs >= threshold).astype(int)
    return probs, preds, threshold, metadata


def predict_regression_market(df: pd.DataFrame, market_key: str):
    xgb, lgbm, feature_columns, metadata, weights = load_regression_assets(market_key)
    df = align_features_for_market(df, feature_columns)
    X = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    xgb_preds = np.asarray(xgb.predict(X), dtype=float)
    lgbm_preds = np.asarray(lgbm.predict(X), dtype=float)
    wx = float(weights.get("xgboost", 0.5))
    wl = float(weights.get("lightgbm", 0.5))
    preds = (wx * xgb_preds) + (wl * lgbm_preds)
    return preds, metadata


def confidence_from_prob(prob: float) -> int:
    return int(round(max(prob, 1 - prob) * 100))


def confidence_from_edge(edge: float, base: int = 52, scale: float = 11.0, cap: int = 82) -> int:
    edge_value = max(0.0, float(edge))
    return int(max(50, min(cap, round(base + (edge_value * scale)))))


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


def _safe_float(value, default: float = 0.0) -> float:
    try:
        num = float(pd.to_numeric(value, errors="coerce"))
    except Exception:
        return float(default)
    if not np.isfinite(num):
        return float(default)
    return float(num)


def _signed_squash_value(value: float) -> float:
    val = _safe_float(value, 0.0)
    return val / (1.0 + abs(val))


def compute_full_game_volatility_score(feature_row: pd.Series) -> float:
    components = []

    def _scaled_abs(col: str, scale: float, weight: float) -> None:
        value = abs(_safe_float(feature_row.get(col, 0.0), 0.0))
        norm = np.clip(value / max(float(scale), 1e-6), 0.0, 1.0)
        components.append((float(norm), float(weight)))

    _scaled_abs("diff_runs_scored_std_L10", scale=2.5, weight=1.00)
    _scaled_abs("diff_runs_allowed_std_L10", scale=2.5, weight=1.00)
    _scaled_abs("diff_r1_scored_std_L10", scale=0.30, weight=0.70)
    _scaled_abs("diff_r1_allowed_std_L10", scale=0.30, weight=0.70)
    _scaled_abs("diff_pitcher_blowup_rate_L10", scale=0.35, weight=1.20)
    _scaled_abs("diff_pitcher_era_trend", scale=0.40, weight=0.90)
    _scaled_abs("diff_pitcher_whip_trend", scale=0.08, weight=0.90)
    _scaled_abs("umpire_volatility_risk", scale=0.60, weight=0.70)

    both_pitchers_available = _safe_float(feature_row.get("both_pitchers_available", 0.0), 0.0)
    if both_pitchers_available < 1.0:
        components.append((1.0, 1.25))

    if not components:
        return 0.0

    numer = float(sum(v * w for v, w in components))
    denom = float(sum(w for _, w in components))
    if denom <= 0:
        return 0.0
    return float(np.clip(numer / denom, 0.0, 1.0))


def compute_full_game_decision_threshold(vol_score: float) -> float:
    if FULL_GAME_VOL_DECISION_ENABLED <= 0:
        return 0.5

    shift = float(FULL_GAME_VOL_DECISION_BETA) * (float(vol_score) - float(FULL_GAME_VOL_DECISION_CENTER))
    shift = float(np.clip(shift, -float(FULL_GAME_VOL_DECISION_MAX_SHIFT), float(FULL_GAME_VOL_DECISION_MAX_SHIFT)))
    return float(np.clip(0.5 + shift, 0.35, 0.65))


def _signed_scaled(value: float, scale: float, invert: bool = False) -> float:
    signed = _safe_float(value, 0.0)
    if invert:
        signed = -signed
    return float(np.clip(signed / max(float(scale), 1e-6), -1.0, 1.0))


def compute_full_game_reliability_adjustment(
    feature_row: pd.Series,
    prob_home: float,
    vol_score: float,
):
    prob = float(np.clip(prob_home, 1e-6, 1.0 - 1e-6))
    if FULL_GAME_RELIABILITY_ENABLED <= 0:
        return prob, 1.0, 0.0, 0.0, 0.0, 1.0

    quality_signal = _signed_scaled(feature_row.get("diff_pitcher_recent_quality_score", 0.0), scale=4.0)
    quality_start_signal = _signed_scaled(feature_row.get("diff_pitcher_quality_start_rate_L10", 0.0), scale=0.30)
    blowup_signal = _signed_scaled(feature_row.get("diff_pitcher_blowup_rate_L10", 0.0), scale=0.35, invert=True)
    era_trend_signal = _signed_scaled(feature_row.get("diff_pitcher_era_trend", 0.0), scale=0.40, invert=True)
    whip_trend_signal = _signed_scaled(feature_row.get("diff_pitcher_whip_trend", 0.0), scale=0.08, invert=True)
    form_signal = _signed_scaled(feature_row.get("diff_form_power", 0.0), scale=3.0)
    elo_signal = _signed_scaled(feature_row.get("diff_elo", 0.0), scale=120.0)
    favorite_signal = 1.0 if _safe_float(feature_row.get("home_is_favorite", 0.0), 0.0) >= 0.5 else -1.0

    weights = np.array([1.20, 1.00, 1.00, 0.80, 0.80, 0.70, 0.55, 0.35], dtype=float)
    signals = np.array(
        [
            quality_signal,
            quality_start_signal,
            blowup_signal,
            era_trend_signal,
            whip_trend_signal,
            form_signal,
            elo_signal,
            favorite_signal,
        ],
        dtype=float,
    )
    side_hint = float(np.clip(np.sum(weights * signals) / max(float(weights.sum()), 1e-6), -1.0, 1.0))
    context_strength = abs(side_hint)

    model_side = 1.0 if prob >= 0.5 else -1.0
    conflict_flag = 1.0 if ((np.sign(side_hint) * model_side) < 0 and context_strength >= 0.20) else 0.0

    edge = float(np.clip(abs(prob - 0.5) * 2.0, 0.0, 1.0))
    missing_pitchers = 1.0 if _safe_float(feature_row.get("both_pitchers_available", 0.0), 0.0) < 1.0 else 0.0

    base_reliability = float(np.clip((0.55 * edge) + (0.45 * context_strength), 0.0, 1.0))
    risk = 1.0 - base_reliability
    risk += 0.22 * conflict_flag * context_strength
    risk += 0.18 * float(np.clip(vol_score, 0.0, 1.0))
    risk += 0.24 * missing_pitchers
    risk = float(np.clip(risk, 0.0, 1.0))
    reliability = float(np.clip(1.0 - risk, 0.0, 1.0))
    unreliability = 1.0 - reliability

    shrink_factor = float(np.clip(1.0 - (FULL_GAME_RELIABILITY_SHRINK_ALPHA * unreliability), 0.35, 1.0))
    prob_shrunk = float(np.clip(0.5 + ((prob - 0.5) * shrink_factor), 1e-6, 1.0 - 1e-6))

    side_shift = float(FULL_GAME_RELIABILITY_SIDE_SHIFT) * unreliability * side_hint
    side_shift += float(FULL_GAME_RELIABILITY_CONFLICT_SHIFT) * conflict_flag * float(np.sign(side_hint))
    prob_adjusted = float(np.clip(prob_shrunk + side_shift, 1e-6, 1.0 - 1e-6))

    return prob_adjusted, reliability, side_hint, conflict_flag, side_shift, shrink_factor


def compute_full_game_publish_policy(feature_row: pd.Series, prob_home: float):
    vol_score = compute_full_game_volatility_score(feature_row) if FULL_GAME_VOL_NORM_ENABLED > 0 else 0.0

    prob_adjusted = float(np.clip(float(prob_home), 1e-6, 1.0 - 1e-6))
    if FULL_GAME_VOL_NORM_ENABLED > 0 and vol_score > 0:
        shrink = np.clip(1.0 - (FULL_GAME_VOL_NORM_ALPHA * vol_score), 0.35, 1.0)
        prob_adjusted = float(np.clip(0.5 + ((prob_adjusted - 0.5) * shrink), 1e-6, 1.0 - 1e-6))

    prob_after_vol_norm = float(prob_adjusted)
    (
        prob_adjusted,
        reliability_score,
        reliability_side_hint,
        reliability_conflict_flag,
        reliability_side_shift,
        reliability_shrink,
    ) = compute_full_game_reliability_adjustment(
        feature_row=feature_row,
        prob_home=prob_adjusted,
        vol_score=vol_score,
    )

    confidence = max(prob_adjusted, 1.0 - prob_adjusted)
    threshold = 0.56

    both_pitchers_available = _safe_float(feature_row.get("both_pitchers_available", 0.0), 0.0)
    if both_pitchers_available < 1.0:
        threshold += 0.05

    decision_threshold = compute_full_game_decision_threshold(vol_score)
    pred_side = 1.0 if prob_adjusted >= decision_threshold else -1.0

    momentum_penalty = 0.0
    diff_quality = _safe_float(feature_row.get("diff_pitcher_recent_quality_score", 0.0), 0.0)
    quality_side = np.sign(diff_quality)
    quality_conflict = (quality_side * pred_side) < 0
    if quality_conflict:
        momentum_penalty += 0.004
    if quality_conflict and abs(diff_quality) >= 2.0:
        momentum_penalty += 0.005

    diff_era_trend = _safe_float(feature_row.get("diff_pitcher_era_trend", 0.0), 0.0)
    if (prob_adjusted >= 0.5 and diff_era_trend > 0.12) or (prob_adjusted < 0.5 and diff_era_trend < -0.12):
        momentum_penalty += 0.002

    diff_whip_trend = _safe_float(feature_row.get("diff_pitcher_whip_trend", 0.0), 0.0)
    if (prob_adjusted >= 0.5 and diff_whip_trend > 0.025) or (prob_adjusted < 0.5 and diff_whip_trend < -0.025):
        momentum_penalty += 0.002

    threshold += min(0.03, momentum_penalty)
    threshold += float(FULL_GAME_VOL_NORM_THRESHOLD_BONUS) * float(vol_score)
    threshold = float(np.clip(threshold, 0.50, 0.75))

    publish = confidence >= threshold
    if 0.60 <= confidence < 0.62:
        publish = False

    return (
        bool(publish),
        float(threshold),
        float(min(0.03, momentum_penalty)),
        float(vol_score),
        float(prob_after_vol_norm),
        float(prob_adjusted),
        float(decision_threshold),
        float(reliability_score),
        float(reliability_side_hint),
        float(reliability_conflict_flag),
        float(reliability_side_shift),
        float(reliability_shrink),
    )


def get_team_snapshot(history_df: pd.DataFrame, team: str, cutoff_date: str):
    """
    Devuelve el último snapshot pregame disponible del equipo antes de cutoff_date.
    `cutoff_date` puede ser 'YYYY-MM-DD' o 'YYYY-MM-DD HH:MM'.
    Se intenta usar la columna `time` de `history_df` si está disponible
    para resolver juegos múltiples el mismo día (doubleheaders).
    """
    if history_df.empty:
        return None

    # Build datetime index for history rows (use 'time' when available)
    hist = history_df.copy()
    if "time" in hist.columns:
        hist["_date_dt"] = pd.to_datetime(hist["date"].astype(str) + " " + hist["time"].astype(str), errors="coerce")
    else:
        hist["_date_dt"] = pd.to_datetime(hist["date"].astype(str), errors="coerce")

    # Parse cutoff into datetime; if cutoff lacks time, it will be midnight
    try:
        cutoff_dt = pd.to_datetime(str(cutoff_date), errors="coerce")
    except Exception:
        cutoff_dt = None

    if cutoff_dt is not None and pd.notna(cutoff_dt):
        prior = hist[hist["_date_dt"] < cutoff_dt].copy()
    else:
        # fallback to date-only comparison
        prior = hist[hist["date"] < str(cutoff_date)].copy()

    if prior.empty:
        return None

    home_rows = prior[prior["home_team"] == team].copy()
    away_rows = prior[prior["away_team"] == team].copy()

    candidates = []

    if not home_rows.empty:
        # prefer ordering by parsed datetime if available
        if "_date_dt" in home_rows.columns:
            row = home_rows.sort_values(["_date_dt", "game_id"]).iloc[-1]
        else:
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
            "runs_scored_std_L10": row.get("home_runs_scored_std_L10", 0.0),
            "runs_allowed_std_L10": row.get("home_runs_allowed_std_L10", 0.0),
            "yrfi_rate_L10": row["home_yrfi_rate_L10"],
            "r1_scored_rate_L10": row["home_r1_scored_rate_L10"],
            "r1_allowed_rate_L10": row["home_r1_allowed_rate_L10"],
            "f5_win_pct_L5": row["home_f5_win_pct_L5"],
            "f5_diff_L5": row["home_f5_diff_L5"],
            "win_pct_L10_blend": row.get("home_win_pct_L10_blend", row.get("home_win_pct_L10", 0.0)),
            "run_diff_L10_blend": row.get("home_run_diff_L10_blend", row.get("home_run_diff_L10", 0.0)),
            "runs_scored_L5_blend": row.get("home_runs_scored_L5_blend", row.get("home_runs_scored_L5", 0.0)),
            "runs_allowed_L5_blend": row.get("home_runs_allowed_L5_blend", row.get("home_runs_allowed_L5", 0.0)),
            "hits_L5": row["home_hits_L5"],
            "hits_L10": row.get("home_hits_L10", row.get("home_hits_L5", 0)),
            "hits_allowed_L5": row["home_hits_allowed_L5"],
            "hits_allowed_L10": row.get("home_hits_allowed_L10", row.get("home_hits_allowed_L5", 0)),
            "baserunners_L10": row.get("home_baserunners_L10", 0.0),
            "baserunners_allowed_L10": row.get("home_baserunners_allowed_L10", 0.0),
            "runs_per_baserunner_L10": row.get("home_runs_per_baserunner_L10", 0.0),
            "player_hits_L10": row.get("home_player_hits_L10", 0.0),
            "player_hits_allowed_L10": row.get("home_player_hits_allowed_L10", 0.0),
            "player_walks_L10": row.get("home_player_walks_L10", 0.0),
            "player_walks_allowed_L10": row.get("home_player_walks_allowed_L10", 0.0),
            "player_total_bases_L10": row.get("home_player_total_bases_L10", 0.0),
            "player_total_bases_allowed_L10": row.get("home_player_total_bases_allowed_L10", 0.0),
            "player_obp_proxy_L10": row.get("home_player_obp_proxy_L10", 0.0),
            "player_slg_proxy_L10": row.get("home_player_slg_proxy_L10", 0.0),
            "player_k_rate_L10": row.get("home_player_k_rate_L10", 0.0),
            "top4_hits_share_L10": row.get("home_top4_hits_share_L10", 0.0),
            "surface_win_pct_L5": row.get("home_home_only_win_pct_L5", 0),
            "surface_run_diff_L5": row.get("home_home_only_run_diff_L5", 0),
            "surface_yrfi_rate_L10": row.get("home_home_only_yrfi_rate_L10", 0),
            "surface_f5_win_pct_L5": row.get("home_home_only_f5_win_pct_L5", 0),
            "_date": row["date"],
            "_game_id": row["game_id"],

            "pitcher_data_available": row.get("home_pitcher_data_available", 0),
            "pitcher_rest_days": row.get("home_pitcher_rest_days", 5),
            "pitcher_games_started_L5": row.get("home_pitcher_games_started_L5", 0),
            "pitcher_games_started_L10": row.get("home_pitcher_games_started_L10", 0),
            "pitcher_era_L5": row.get("home_pitcher_era_L5", 0),
            "pitcher_whip_L5": row.get("home_pitcher_whip_L5", 0),
            "pitcher_k_bb_L5": row.get("home_pitcher_k_bb_L5", 0),
            "pitcher_bb_allowed_L5": row.get("home_pitcher_bb_allowed_L5", 0),
            "pitcher_baserunners_allowed_L5": row.get("home_pitcher_baserunners_allowed_L5", 0),
            "pitcher_total_bases_allowed_L5": row.get("home_pitcher_total_bases_allowed_L5", 0),
            "pitcher_hr9_L5": row.get("home_pitcher_hr9_L5", 0),
            "pitcher_ip_L5": row.get("home_pitcher_ip_L5", 0),
            "pitcher_runs_allowed_L5": row.get("home_pitcher_runs_allowed_L5", 0),
            "pitcher_runs_allowed_L10": row.get("home_pitcher_runs_allowed_L10", 0),
            "pitcher_r1_allowed_rate_L10": row.get("home_pitcher_r1_allowed_rate_L10", 0),
            "pitcher_r1_allowed_rate_L5": row.get("home_pitcher_r1_allowed_rate_L5", 0),
            "pitcher_f5_runs_allowed_L5": row.get("home_pitcher_f5_runs_allowed_L5", 0),
            "pitcher_quality_start_rate_L10": row.get("home_pitcher_quality_start_rate_L10", 0),
            "pitcher_blowup_rate_L10": row.get("home_pitcher_blowup_rate_L10", 0),
            "pitcher_era_trend": row.get("home_pitcher_era_trend", 0),
            "pitcher_whip_trend": row.get("home_pitcher_whip_trend", 0),
            "pitcher_recent_quality_score": row.get("home_pitcher_recent_quality_score", 0),
            "pitcher_team_run_support_L5": row.get("home_pitcher_team_run_support_L5", 0),
            "pitcher_start_win_rate_L10": row.get("home_pitcher_start_win_rate_L10", 0),

            "bullpen_runs_allowed_L5": row.get("home_bullpen_runs_allowed_L5", 0),
            "bullpen_runs_allowed_L10": row.get("home_bullpen_runs_allowed_L10", 0),
            "bullpen_load_L3": row.get("home_bullpen_load_L3", 0),
            "bullpen_load_L5": row.get("home_bullpen_load_L5", 0),

            "offense_vs_pitcher": row.get("home_offense_vs_away_pitcher", 0),
            "r1_vs_pitcher": row.get("home_r1_vs_away_pitcher", 0),
            "r1_vs_pitcher_L5_proxy": row.get("home_r1_vs_away_pitcher_L5_proxy", 0),
            "f5_vs_pitcher": row.get("home_f5_vs_away_pitcher", 0),

            "r1_scored_rate_L5": row.get("home_r1_scored_rate_L5", row.get("home_r1_scored_rate_L10", 0)),
            "r1_allowed_rate_L5": row.get("home_r1_allowed_rate_L5", row.get("home_r1_allowed_rate_L10", 0)),
            "r1_scored_std_L10": row.get("home_r1_scored_std_L10", 0),
            "r1_allowed_std_L10": row.get("home_r1_allowed_std_L10", 0),

            "yrfi_pressure": row.get("yrfi_pressure_home", 0),
            "yrfi_pressure_L5": row.get("yrfi_pressure_home_L5", 0),
            "yrfi_consistency_L10": row.get("home_yrfi_consistency_L10", 0),
        }
        candidates.append(snap)

    if not away_rows.empty:
        if "_date_dt" in away_rows.columns:
            row = away_rows.sort_values(["_date_dt", "game_id"]).iloc[-1]
        else:
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
            "runs_scored_std_L10": row.get("away_runs_scored_std_L10", 0.0),
            "runs_allowed_std_L10": row.get("away_runs_allowed_std_L10", 0.0),
            "yrfi_rate_L10": row["away_yrfi_rate_L10"],
            "r1_scored_rate_L10": row["away_r1_scored_rate_L10"],
            "r1_allowed_rate_L10": row["away_r1_allowed_rate_L10"],
            "f5_win_pct_L5": row["away_f5_win_pct_L5"],
            "f5_diff_L5": row["away_f5_diff_L5"],
            "win_pct_L10_blend": row.get("away_win_pct_L10_blend", row.get("away_win_pct_L10", 0.0)),
            "run_diff_L10_blend": row.get("away_run_diff_L10_blend", row.get("away_run_diff_L10", 0.0)),
            "runs_scored_L5_blend": row.get("away_runs_scored_L5_blend", row.get("away_runs_scored_L5", 0.0)),
            "runs_allowed_L5_blend": row.get("away_runs_allowed_L5_blend", row.get("away_runs_allowed_L5", 0.0)),
            "hits_L5": row["away_hits_L5"],
            "hits_L10": row.get("away_hits_L10", row.get("away_hits_L5", 0)),
            "hits_allowed_L5": row["away_hits_allowed_L5"],
            "hits_allowed_L10": row.get("away_hits_allowed_L10", row.get("away_hits_allowed_L5", 0)),
            "baserunners_L10": row.get("away_baserunners_L10", 0.0),
            "baserunners_allowed_L10": row.get("away_baserunners_allowed_L10", 0.0),
            "runs_per_baserunner_L10": row.get("away_runs_per_baserunner_L10", 0.0),
            "player_hits_L10": row.get("away_player_hits_L10", 0.0),
            "player_hits_allowed_L10": row.get("away_player_hits_allowed_L10", 0.0),
            "player_walks_L10": row.get("away_player_walks_L10", 0.0),
            "player_walks_allowed_L10": row.get("away_player_walks_allowed_L10", 0.0),
            "player_total_bases_L10": row.get("away_player_total_bases_L10", 0.0),
            "player_total_bases_allowed_L10": row.get("away_player_total_bases_allowed_L10", 0.0),
            "player_obp_proxy_L10": row.get("away_player_obp_proxy_L10", 0.0),
            "player_slg_proxy_L10": row.get("away_player_slg_proxy_L10", 0.0),
            "player_k_rate_L10": row.get("away_player_k_rate_L10", 0.0),
            "top4_hits_share_L10": row.get("away_top4_hits_share_L10", 0.0),
            "surface_win_pct_L5": row.get("away_away_only_win_pct_L5", 0),
            "surface_run_diff_L5": row.get("away_away_only_run_diff_L5", 0),
            "surface_yrfi_rate_L10": row.get("away_away_only_yrfi_rate_L10", 0),
            "surface_f5_win_pct_L5": row.get("away_away_only_f5_win_pct_L5", 0),
            "_date": row["date"],
            "_game_id": row["game_id"],
          
            "pitcher_data_available": row.get("away_pitcher_data_available", 0),
            "pitcher_rest_days": row.get("away_pitcher_rest_days", 5),
            "pitcher_games_started_L5": row.get("away_pitcher_games_started_L5", 0),
            "pitcher_games_started_L10": row.get("away_pitcher_games_started_L10", 0),
            "pitcher_era_L5": row.get("away_pitcher_era_L5", 0),
            "pitcher_whip_L5": row.get("away_pitcher_whip_L5", 0),
            "pitcher_k_bb_L5": row.get("away_pitcher_k_bb_L5", 0),
            "pitcher_bb_allowed_L5": row.get("away_pitcher_bb_allowed_L5", 0),
            "pitcher_baserunners_allowed_L5": row.get("away_pitcher_baserunners_allowed_L5", 0),
            "pitcher_total_bases_allowed_L5": row.get("away_pitcher_total_bases_allowed_L5", 0),
            "pitcher_hr9_L5": row.get("away_pitcher_hr9_L5", 0),
            "pitcher_ip_L5": row.get("away_pitcher_ip_L5", 0),
            "pitcher_runs_allowed_L5": row.get("away_pitcher_runs_allowed_L5", 0),
            "pitcher_runs_allowed_L10": row.get("away_pitcher_runs_allowed_L10", 0),
            "pitcher_r1_allowed_rate_L10": row.get("away_pitcher_r1_allowed_rate_L10", 0),
            "pitcher_r1_allowed_rate_L5": row.get("away_pitcher_r1_allowed_rate_L5", 0),
            "pitcher_f5_runs_allowed_L5": row.get("away_pitcher_f5_runs_allowed_L5", 0),
            "pitcher_quality_start_rate_L10": row.get("away_pitcher_quality_start_rate_L10", 0),
            "pitcher_blowup_rate_L10": row.get("away_pitcher_blowup_rate_L10", 0),
            "pitcher_era_trend": row.get("away_pitcher_era_trend", 0),
            "pitcher_whip_trend": row.get("away_pitcher_whip_trend", 0),
            "pitcher_recent_quality_score": row.get("away_pitcher_recent_quality_score", 0),
            "pitcher_team_run_support_L5": row.get("away_pitcher_team_run_support_L5", 0),
            "pitcher_start_win_rate_L10": row.get("away_pitcher_start_win_rate_L10", 0),

            "bullpen_runs_allowed_L5": row.get("away_bullpen_runs_allowed_L5", 0),
            "bullpen_runs_allowed_L10": row.get("away_bullpen_runs_allowed_L10", 0),
            "bullpen_load_L3": row.get("away_bullpen_load_L3", 0),
            "bullpen_load_L5": row.get("away_bullpen_load_L5", 0),

            "offense_vs_pitcher": row.get("away_offense_vs_home_pitcher", 0),
            "r1_vs_pitcher": row.get("away_r1_vs_home_pitcher", 0),
            "r1_vs_pitcher_L5_proxy": row.get("away_r1_vs_home_pitcher_L5_proxy", 0),
            "f5_vs_pitcher": row.get("away_f5_vs_home_pitcher", 0),

            "r1_scored_rate_L5": row.get("away_r1_scored_rate_L5", row.get("away_r1_scored_rate_L10", 0)),
            "r1_allowed_rate_L5": row.get("away_r1_allowed_rate_L5", row.get("away_r1_allowed_rate_L10", 0)),
            "r1_scored_std_L10": row.get("away_r1_scored_std_L10", 0),
            "r1_allowed_std_L10": row.get("away_r1_allowed_std_L10", 0),

            "yrfi_pressure": row.get("yrfi_pressure_away", 0),
            "yrfi_pressure_L5": row.get("yrfi_pressure_away_L5", 0),
            "yrfi_consistency_L10": row.get("away_yrfi_consistency_L10", 0),
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

    # build cutoff including time when available to handle same-day ordering (doubleheaders)
    time_str = str(schedule_row.get("time", "")).strip()
    if time_str:
        cutoff = f"{date_str} {time_str}"
    else:
        cutoff = date_str

    home_snap = get_team_snapshot(history_df, home_team, cutoff)
    away_snap = get_team_snapshot(history_df, away_team, cutoff)
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

        "home_runs_scored_std_L10": float(home_snap.get("runs_scored_std_L10", 0.0)),
        "away_runs_scored_std_L10": float(away_snap.get("runs_scored_std_L10", 0.0)),
        "diff_runs_scored_std_L10": float(home_snap.get("runs_scored_std_L10", 0.0)) - float(away_snap.get("runs_scored_std_L10", 0.0)),

        "home_runs_allowed_L5": home_snap["runs_allowed_L5"],
        "away_runs_allowed_L5": away_snap["runs_allowed_L5"],
        "diff_runs_allowed_L5": home_snap["runs_allowed_L5"] - away_snap["runs_allowed_L5"],

        "home_runs_allowed_std_L10": float(home_snap.get("runs_allowed_std_L10", 0.0)),
        "away_runs_allowed_std_L10": float(away_snap.get("runs_allowed_std_L10", 0.0)),
        "diff_runs_allowed_std_L10": float(home_snap.get("runs_allowed_std_L10", 0.0)) - float(away_snap.get("runs_allowed_std_L10", 0.0)),

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

        "home_win_pct_L10_blend": float(home_snap.get("win_pct_L10_blend", home_snap.get("win_pct_L10", 0.0))),
        "away_win_pct_L10_blend": float(away_snap.get("win_pct_L10_blend", away_snap.get("win_pct_L10", 0.0))),
        "diff_win_pct_L10_blend": float(home_snap.get("win_pct_L10_blend", home_snap.get("win_pct_L10", 0.0))) - float(away_snap.get("win_pct_L10_blend", away_snap.get("win_pct_L10", 0.0))),

        "home_run_diff_L10_blend": float(home_snap.get("run_diff_L10_blend", home_snap.get("run_diff_L10", 0.0))),
        "away_run_diff_L10_blend": float(away_snap.get("run_diff_L10_blend", away_snap.get("run_diff_L10", 0.0))),
        "diff_run_diff_L10_blend": float(home_snap.get("run_diff_L10_blend", home_snap.get("run_diff_L10", 0.0))) - float(away_snap.get("run_diff_L10_blend", away_snap.get("run_diff_L10", 0.0))),

        "home_runs_scored_L5_blend": float(home_snap.get("runs_scored_L5_blend", home_snap.get("runs_scored_L5", 0.0))),
        "away_runs_scored_L5_blend": float(away_snap.get("runs_scored_L5_blend", away_snap.get("runs_scored_L5", 0.0))),
        "diff_runs_scored_L5_blend": float(home_snap.get("runs_scored_L5_blend", home_snap.get("runs_scored_L5", 0.0))) - float(away_snap.get("runs_scored_L5_blend", away_snap.get("runs_scored_L5", 0.0))),

        "home_runs_allowed_L5_blend": float(home_snap.get("runs_allowed_L5_blend", home_snap.get("runs_allowed_L5", 0.0))),
        "away_runs_allowed_L5_blend": float(away_snap.get("runs_allowed_L5_blend", away_snap.get("runs_allowed_L5", 0.0))),
        "diff_runs_allowed_L5_blend": float(home_snap.get("runs_allowed_L5_blend", home_snap.get("runs_allowed_L5", 0.0))) - float(away_snap.get("runs_allowed_L5_blend", away_snap.get("runs_allowed_L5", 0.0))),

        "home_hits_L5": home_snap["hits_L5"],
        "away_hits_L5": away_snap["hits_L5"],
        "diff_hits_L5": home_snap["hits_L5"] - away_snap["hits_L5"],

        "home_hits_L10": home_snap.get("hits_L10", home_snap["hits_L5"]),
        "away_hits_L10": away_snap.get("hits_L10", away_snap["hits_L5"]),
        "diff_hits_L10": home_snap.get("hits_L10", home_snap["hits_L5"]) - away_snap.get("hits_L10", away_snap["hits_L5"]),

        "home_hits_allowed_L5": home_snap["hits_allowed_L5"],
        "away_hits_allowed_L5": away_snap["hits_allowed_L5"],
        "diff_hits_allowed_L5": home_snap["hits_allowed_L5"] - away_snap["hits_allowed_L5"],

        "home_hits_allowed_L10": home_snap.get("hits_allowed_L10", home_snap["hits_allowed_L5"]),
        "away_hits_allowed_L10": away_snap.get("hits_allowed_L10", away_snap["hits_allowed_L5"]),
        "diff_hits_allowed_L10": home_snap.get("hits_allowed_L10", home_snap["hits_allowed_L5"]) - away_snap.get("hits_allowed_L10", away_snap["hits_allowed_L5"]),

        "home_baserunners_L10": float(home_snap.get("baserunners_L10", 0.0)),
        "away_baserunners_L10": float(away_snap.get("baserunners_L10", 0.0)),
        "diff_baserunners_L10": float(home_snap.get("baserunners_L10", 0.0)) - float(away_snap.get("baserunners_L10", 0.0)),

        "home_baserunners_allowed_L10": float(home_snap.get("baserunners_allowed_L10", 0.0)),
        "away_baserunners_allowed_L10": float(away_snap.get("baserunners_allowed_L10", 0.0)),
        "diff_baserunners_allowed_L10": float(home_snap.get("baserunners_allowed_L10", 0.0)) - float(away_snap.get("baserunners_allowed_L10", 0.0)),

        "home_runs_per_baserunner_L10": float(home_snap.get("runs_per_baserunner_L10", 0.0)),
        "away_runs_per_baserunner_L10": float(away_snap.get("runs_per_baserunner_L10", 0.0)),
        "diff_runs_per_baserunner_L10": float(home_snap.get("runs_per_baserunner_L10", 0.0)) - float(away_snap.get("runs_per_baserunner_L10", 0.0)),

        "home_player_hits_L10": float(home_snap.get("player_hits_L10", 0.0)),
        "away_player_hits_L10": float(away_snap.get("player_hits_L10", 0.0)),
        "diff_player_hits_L10": float(home_snap.get("player_hits_L10", 0.0)) - float(away_snap.get("player_hits_L10", 0.0)),

        "home_player_hits_allowed_L10": float(home_snap.get("player_hits_allowed_L10", 0.0)),
        "away_player_hits_allowed_L10": float(away_snap.get("player_hits_allowed_L10", 0.0)),
        "diff_player_hits_allowed_L10": float(home_snap.get("player_hits_allowed_L10", 0.0)) - float(away_snap.get("player_hits_allowed_L10", 0.0)),

        "home_player_walks_L10": float(home_snap.get("player_walks_L10", 0.0)),
        "away_player_walks_L10": float(away_snap.get("player_walks_L10", 0.0)),
        "diff_player_walks_L10": float(home_snap.get("player_walks_L10", 0.0)) - float(away_snap.get("player_walks_L10", 0.0)),

        "home_player_walks_allowed_L10": float(home_snap.get("player_walks_allowed_L10", 0.0)),
        "away_player_walks_allowed_L10": float(away_snap.get("player_walks_allowed_L10", 0.0)),
        "diff_player_walks_allowed_L10": float(home_snap.get("player_walks_allowed_L10", 0.0)) - float(away_snap.get("player_walks_allowed_L10", 0.0)),

        "home_player_total_bases_L10": float(home_snap.get("player_total_bases_L10", 0.0)),
        "away_player_total_bases_L10": float(away_snap.get("player_total_bases_L10", 0.0)),
        "diff_player_total_bases_L10": float(home_snap.get("player_total_bases_L10", 0.0)) - float(away_snap.get("player_total_bases_L10", 0.0)),

        "home_player_total_bases_allowed_L10": float(home_snap.get("player_total_bases_allowed_L10", 0.0)),
        "away_player_total_bases_allowed_L10": float(away_snap.get("player_total_bases_allowed_L10", 0.0)),
        "diff_player_total_bases_allowed_L10": float(home_snap.get("player_total_bases_allowed_L10", 0.0)) - float(away_snap.get("player_total_bases_allowed_L10", 0.0)),

        "home_player_obp_proxy_L10": float(home_snap.get("player_obp_proxy_L10", 0.0)),
        "away_player_obp_proxy_L10": float(away_snap.get("player_obp_proxy_L10", 0.0)),
        "diff_player_obp_proxy_L10": float(home_snap.get("player_obp_proxy_L10", 0.0)) - float(away_snap.get("player_obp_proxy_L10", 0.0)),

        "home_player_slg_proxy_L10": float(home_snap.get("player_slg_proxy_L10", 0.0)),
        "away_player_slg_proxy_L10": float(away_snap.get("player_slg_proxy_L10", 0.0)),
        "diff_player_slg_proxy_L10": float(home_snap.get("player_slg_proxy_L10", 0.0)) - float(away_snap.get("player_slg_proxy_L10", 0.0)),

        "home_player_k_rate_L10": float(home_snap.get("player_k_rate_L10", 0.0)),
        "away_player_k_rate_L10": float(away_snap.get("player_k_rate_L10", 0.0)),
        "diff_player_k_rate_L10": float(home_snap.get("player_k_rate_L10", 0.0)) - float(away_snap.get("player_k_rate_L10", 0.0)),

        "home_top4_hits_share_L10": float(home_snap.get("top4_hits_share_L10", 0.0)),
        "away_top4_hits_share_L10": float(away_snap.get("top4_hits_share_L10", 0.0)),
        "diff_top4_hits_share_L10": float(home_snap.get("top4_hits_share_L10", 0.0)) - float(away_snap.get("top4_hits_share_L10", 0.0)),

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

        # Nuevas columnas de disponibilidad y diferenciales de pitchers, bullpen y matchup
        "both_pitchers_available": int(
            float(home_snap.get("pitcher_data_available", 0)) > 0
            and float(away_snap.get("pitcher_data_available", 0)) > 0
        ),
        "diff_pitcher_data_available": float(home_snap.get("pitcher_data_available", 0)) - float(away_snap.get("pitcher_data_available", 0)),

        # Diferenciales de pitcher
        "home_pitcher_era_L5": float(home_snap.get("pitcher_era_L5", 0)),
        "away_pitcher_era_L5": float(away_snap.get("pitcher_era_L5", 0)),
        "diff_pitcher_rest_days": float(home_snap.get("pitcher_rest_days", 0)) - float(away_snap.get("pitcher_rest_days", 0)),
        "diff_pitcher_games_started_L5": float(home_snap.get("pitcher_games_started_L5", 0)) - float(away_snap.get("pitcher_games_started_L5", 0)),
        "diff_pitcher_games_started_L10": float(home_snap.get("pitcher_games_started_L10", 0)) - float(away_snap.get("pitcher_games_started_L10", 0)),
        "diff_pitcher_era_L5": float(home_snap.get("pitcher_era_L5", 0)) - float(away_snap.get("pitcher_era_L5", 0)),
        "home_pitcher_whip_L5": float(home_snap.get("pitcher_whip_L5", 0)),
        "away_pitcher_whip_L5": float(away_snap.get("pitcher_whip_L5", 0)),
        "diff_pitcher_whip_L5": float(home_snap.get("pitcher_whip_L5", 0)) - float(away_snap.get("pitcher_whip_L5", 0)),
        "diff_pitcher_k_bb_L5": float(home_snap.get("pitcher_k_bb_L5", 0)) - float(away_snap.get("pitcher_k_bb_L5", 0)),
        "diff_pitcher_bb_allowed_L5": float(home_snap.get("pitcher_bb_allowed_L5", 0)) - float(away_snap.get("pitcher_bb_allowed_L5", 0)),
        "diff_pitcher_baserunners_allowed_L5": float(home_snap.get("pitcher_baserunners_allowed_L5", 0)) - float(away_snap.get("pitcher_baserunners_allowed_L5", 0)),
        "diff_pitcher_total_bases_allowed_L5": float(home_snap.get("pitcher_total_bases_allowed_L5", 0)) - float(away_snap.get("pitcher_total_bases_allowed_L5", 0)),
        "home_pitcher_hr9_L5": float(home_snap.get("pitcher_hr9_L5", 0)),
        "away_pitcher_hr9_L5": float(away_snap.get("pitcher_hr9_L5", 0)),
        "diff_pitcher_hr9_L5": float(home_snap.get("pitcher_hr9_L5", 0)) - float(away_snap.get("pitcher_hr9_L5", 0)),
        "home_pitcher_ip_L5": float(home_snap.get("pitcher_ip_L5", 0)),
        "away_pitcher_ip_L5": float(away_snap.get("pitcher_ip_L5", 0)),
        "diff_pitcher_ip_L5": float(home_snap.get("pitcher_ip_L5", 0)) - float(away_snap.get("pitcher_ip_L5", 0)),
        "diff_pitcher_runs_allowed_L5": float(home_snap.get("pitcher_runs_allowed_L5", 0)) - float(away_snap.get("pitcher_runs_allowed_L5", 0)),
        "diff_pitcher_runs_allowed_L10": float(home_snap.get("pitcher_runs_allowed_L10", 0)) - float(away_snap.get("pitcher_runs_allowed_L10", 0)),
        "home_pitcher_r1_allowed_rate_L10": float(home_snap.get("pitcher_r1_allowed_rate_L10", 0)),
        "away_pitcher_r1_allowed_rate_L10": float(away_snap.get("pitcher_r1_allowed_rate_L10", 0)),
        "diff_pitcher_r1_allowed_rate_L10": float(home_snap.get("pitcher_r1_allowed_rate_L10", 0)) - float(away_snap.get("pitcher_r1_allowed_rate_L10", 0)),
        "home_pitcher_r1_allowed_rate_L5": float(home_snap.get("pitcher_r1_allowed_rate_L5", 0)),
        "away_pitcher_r1_allowed_rate_L5": float(away_snap.get("pitcher_r1_allowed_rate_L5", 0)),
        "diff_pitcher_r1_allowed_rate_L5": float(home_snap.get("pitcher_r1_allowed_rate_L5", 0)) - float(away_snap.get("pitcher_r1_allowed_rate_L5", 0)),
        "home_pitcher_f5_runs_allowed_L5": float(home_snap.get("pitcher_f5_runs_allowed_L5", 0)),
        "away_pitcher_f5_runs_allowed_L5": float(away_snap.get("pitcher_f5_runs_allowed_L5", 0)),
        "diff_pitcher_f5_runs_allowed_L5": float(home_snap.get("pitcher_f5_runs_allowed_L5", 0)) - float(away_snap.get("pitcher_f5_runs_allowed_L5", 0)),
        "diff_pitcher_quality_start_rate_L10": float(home_snap.get("pitcher_quality_start_rate_L10", 0)) - float(away_snap.get("pitcher_quality_start_rate_L10", 0)),
        "diff_pitcher_blowup_rate_L10": float(home_snap.get("pitcher_blowup_rate_L10", 0)) - float(away_snap.get("pitcher_blowup_rate_L10", 0)),
        "diff_pitcher_era_trend": float(home_snap.get("pitcher_era_trend", 0)) - float(away_snap.get("pitcher_era_trend", 0)),
        "diff_pitcher_whip_trend": float(home_snap.get("pitcher_whip_trend", 0)) - float(away_snap.get("pitcher_whip_trend", 0)),
        "diff_pitcher_recent_quality_score": float(home_snap.get("pitcher_recent_quality_score", 0)) - float(away_snap.get("pitcher_recent_quality_score", 0)),
        "diff_pitcher_team_run_support_L5": float(home_snap.get("pitcher_team_run_support_L5", 0)) - float(away_snap.get("pitcher_team_run_support_L5", 0)),
        "diff_pitcher_start_win_rate_L10": float(home_snap.get("pitcher_start_win_rate_L10", 0)) - float(away_snap.get("pitcher_start_win_rate_L10", 0)),

        # Diferenciales de bullpen
        "home_bullpen_runs_allowed_L5": float(home_snap.get("bullpen_runs_allowed_L5", 0)),
        "away_bullpen_runs_allowed_L5": float(away_snap.get("bullpen_runs_allowed_L5", 0)),
        "diff_bullpen_runs_allowed_L5": float(home_snap.get("bullpen_runs_allowed_L5", 0)) - float(away_snap.get("bullpen_runs_allowed_L5", 0)),
        "diff_bullpen_runs_allowed_L10": float(home_snap.get("bullpen_runs_allowed_L10", 0)) - float(away_snap.get("bullpen_runs_allowed_L10", 0)),
        "home_bullpen_load_L3": float(home_snap.get("bullpen_load_L3", 0)),
        "away_bullpen_load_L3": float(away_snap.get("bullpen_load_L3", 0)),
        "diff_bullpen_load_L3": float(home_snap.get("bullpen_load_L3", 0)) - float(away_snap.get("bullpen_load_L3", 0)),
        "diff_bullpen_load_L5": float(home_snap.get("bullpen_load_L5", 0)) - float(away_snap.get("bullpen_load_L5", 0)),

        # Diferenciales matchup vs pitcher
        "home_offense_vs_away_pitcher": float(home_snap.get("offense_vs_pitcher", 0)),
        "away_offense_vs_home_pitcher": float(away_snap.get("offense_vs_pitcher", 0)),
        "diff_offense_vs_pitcher": float(home_snap.get("offense_vs_pitcher", 0)) - float(away_snap.get("offense_vs_pitcher", 0)),
        "home_r1_vs_away_pitcher": float(home_snap.get("r1_vs_pitcher", 0)),
        "away_r1_vs_home_pitcher": float(away_snap.get("r1_vs_pitcher", 0)),
        "diff_r1_vs_pitcher": float(home_snap.get("r1_vs_pitcher", 0)) - float(away_snap.get("r1_vs_pitcher", 0)),
        "diff_r1_vs_pitcher_L5": float(home_snap.get("r1_vs_pitcher_L5_proxy", 0)) - float(away_snap.get("r1_vs_pitcher_L5_proxy", 0)),
        "diff_f5_vs_pitcher": float(home_snap.get("f5_vs_pitcher", 0)) - float(away_snap.get("f5_vs_pitcher", 0)),

        # Diferenciales R1 nuevos
        "home_r1_scored_rate_L5": float(home_snap.get("r1_scored_rate_L5", 0)),
        "away_r1_scored_rate_L5": float(away_snap.get("r1_scored_rate_L5", 0)),
        "diff_r1_scored_rate_L5": float(home_snap.get("r1_scored_rate_L5", 0)) - float(away_snap.get("r1_scored_rate_L5", 0)),
        "home_r1_allowed_rate_L5": float(home_snap.get("r1_allowed_rate_L5", 0)),
        "away_r1_allowed_rate_L5": float(away_snap.get("r1_allowed_rate_L5", 0)),
        "diff_r1_allowed_rate_L5": float(home_snap.get("r1_allowed_rate_L5", 0)) - float(away_snap.get("r1_allowed_rate_L5", 0)),
        "diff_r1_scored_std_L10": float(home_snap.get("r1_scored_std_L10", 0)) - float(away_snap.get("r1_scored_std_L10", 0)),
        "diff_r1_allowed_std_L10": float(home_snap.get("r1_allowed_std_L10", 0)) - float(away_snap.get("r1_allowed_std_L10", 0)),

        # YRFI pressure y consistencia
        "yrfi_pressure_home": float(home_snap.get("yrfi_pressure", 0)),
        "yrfi_pressure_away": float(away_snap.get("yrfi_pressure", 0)),
        "diff_yrfi_pressure": float(home_snap.get("yrfi_pressure", 0)) - float(away_snap.get("yrfi_pressure", 0)),
        "total_yrfi_pressure": float(home_snap.get("yrfi_pressure", 0)) + float(away_snap.get("yrfi_pressure", 0)),

        "yrfi_pressure_home_L5": float(home_snap.get("yrfi_pressure_L5", 0)),
        "yrfi_pressure_away_L5": float(away_snap.get("yrfi_pressure_L5", 0)),
        "diff_yrfi_pressure_L5": float(home_snap.get("yrfi_pressure_L5", 0)) - float(away_snap.get("yrfi_pressure_L5", 0)),
        "total_yrfi_pressure_L5": float(home_snap.get("yrfi_pressure_L5", 0)) + float(away_snap.get("yrfi_pressure_L5", 0)),

        "home_yrfi_consistency_L10": float(home_snap.get("yrfi_consistency_L10", 0)),
        "away_yrfi_consistency_L10": float(away_snap.get("yrfi_consistency_L10", 0)),
        "diff_yrfi_consistency_L10": float(home_snap.get("yrfi_consistency_L10", 0)) - float(away_snap.get("yrfi_consistency_L10", 0)),

        # Mercado pregame desde ingest.
        "home_is_favorite": float(np.clip(_safe_float(schedule_row.get("home_is_favorite", 0), 0.0), 0.0, 1.0)),
        "odds_over_under": float(pd.to_numeric(schedule_row.get("odds_over_under", 0), errors="coerce") or 0),
        "weather_temp": _safe_float(schedule_row.get("weather_temp", np.nan), 0.0),
        "weather_wind": _safe_float(schedule_row.get("weather_wind", np.nan), 0.0),
        "umpire_zone_delta": _safe_float(schedule_row.get("umpire_zone_delta", np.nan), 0.0),
        "umpire_sample_log": _safe_float(schedule_row.get("umpire_sample_log", np.nan), 0.0),
        "market_missing": int(not np.isfinite(pd.to_numeric(schedule_row.get("odds_over_under", np.nan), errors="coerce"))),
    }

    home_favorite_flag = float(np.clip(_safe_float(row.get("home_is_favorite", 0.0), 0.0), 0.0, 1.0))
    away_favorite_flag = 1.0 - home_favorite_flag
    row["favorite_elo_gap_signed"] = row["diff_elo"] if home_favorite_flag >= 0.5 else -row["diff_elo"]

    home_quality_edge = (
        0.65 * _safe_float(row.get("home_win_pct_L10_vs_league", 0.0), 0.0)
        + 0.35 * _signed_squash_value(_safe_float(row.get("home_run_diff_L10_vs_league", 0.0), 0.0))
    )
    away_quality_edge = (
        0.65 * _safe_float(row.get("away_win_pct_L10_vs_league", 0.0), 0.0)
        + 0.35 * _signed_squash_value(_safe_float(row.get("away_run_diff_L10_vs_league", 0.0), 0.0))
    )

    home_variance_pressure = 1.0 + _safe_float(row.get("home_runs_scored_std_L10", 0.0), 0.0) + _safe_float(row.get("home_runs_allowed_std_L10", 0.0), 0.0)
    away_variance_pressure = 1.0 + _safe_float(row.get("away_runs_scored_std_L10", 0.0), 0.0) + _safe_float(row.get("away_runs_allowed_std_L10", 0.0), 0.0)

    home_momentum_win = _safe_float(row.get("home_momentum_win", 0.0), 0.0)
    away_momentum_win = _safe_float(row.get("away_momentum_win", 0.0), 0.0)
    home_momentum_run_diff = _safe_float(row.get("home_momentum_run_diff", 0.0), 0.0)
    away_momentum_run_diff = _safe_float(row.get("away_momentum_run_diff", 0.0), 0.0)
    home_rest_days = _safe_float(row.get("home_rest_days", 0.0), 0.0)
    away_rest_days = _safe_float(row.get("away_rest_days", 0.0), 0.0)

    home_regression_risk = (
        max(0.0, home_momentum_win)
        * home_variance_pressure
        * (1.0 + max(0.0, home_quality_edge))
    ) + (0.35 * max(0.0, home_momentum_run_diff))
    away_regression_risk = (
        max(0.0, away_momentum_win)
        * away_variance_pressure
        * (1.0 + max(0.0, away_quality_edge))
    ) + (0.35 * max(0.0, away_momentum_run_diff))

    home_bounce_back_signal = (
        max(0.0, -home_momentum_win)
        * (1.0 + max(0.0, home_quality_edge))
        * (1.0 + max(0.0, home_rest_days))
    )
    away_bounce_back_signal = (
        max(0.0, -away_momentum_win)
        * (1.0 + max(0.0, away_quality_edge))
        * (1.0 + max(0.0, away_rest_days))
    )

    row["home_regression_risk"] = home_regression_risk
    row["away_regression_risk"] = away_regression_risk
    row["diff_regression_risk"] = home_regression_risk - away_regression_risk
    row["home_bounce_back_signal"] = home_bounce_back_signal
    row["away_bounce_back_signal"] = away_bounce_back_signal
    row["diff_bounce_back_signal"] = home_bounce_back_signal - away_bounce_back_signal
    row["favorite_trap_signal"] = (
        home_favorite_flag * (home_regression_risk - away_bounce_back_signal)
        + away_favorite_flag * (away_regression_risk - home_bounce_back_signal)
    )
    row["underdog_upset_signal"] = (
        home_favorite_flag * (away_bounce_back_signal - home_regression_risk)
        + away_favorite_flag * (home_bounce_back_signal - away_regression_risk)
    )

    return row


def load_line_movement_frame() -> pd.DataFrame:
    if not LINE_MOVEMENT_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(LINE_MOVEMENT_FILE, dtype={"game_id": str})
        if "game_id" in df.columns:
            df["game_id"] = df["game_id"].astype(str)
        return df
    except Exception:
        return pd.DataFrame()


def resolve_total_market_line(sched_row, line_row) -> float:
    candidates = []
    if sched_row is not None:
        candidates.extend([
            sched_row.get("odds_over_under", np.nan),
            sched_row.get("closing_total_line", np.nan),
        ])
    if line_row is not None:
        candidates.extend([
            line_row.get("current_total_line", np.nan),
            line_row.get("open_total", np.nan),
        ])

    for value in candidates:
        try:
            numeric = float(value)
        except Exception:
            continue
        if np.isfinite(numeric) and numeric > 0:
            return numeric
    return 0.0


def resolve_total_hits_market_line(sched_row, line_row) -> float:
    candidates = []
    if sched_row is not None:
        candidates.extend([
            sched_row.get("odds_total_hits_event", np.nan),
            sched_row.get("odds_total_hits", np.nan),
            sched_row.get("closing_total_hits_line", np.nan),
        ])
    if line_row is not None:
        candidates.extend([
            line_row.get("current_total_hits_line", np.nan),
            line_row.get("open_total_hits", np.nan),
            line_row.get("closing_total_hits_line", np.nan),
        ])

    for value in candidates:
        try:
            numeric = float(value)
        except Exception:
            continue
        if np.isfinite(numeric) and numeric > 0:
            return numeric
    return 0.0


def resolve_home_spread_line(sched_row, line_row, home_is_favorite: float) -> float:
    candidates = []
    if sched_row is not None:
        candidates.extend([
            sched_row.get("closing_spread_line", np.nan),
            sched_row.get("home_spread", np.nan),
        ])
    if line_row is not None:
        candidates.extend([
            line_row.get("current_home_spread", np.nan),
            line_row.get("open_line", np.nan),
        ])

    for value in candidates:
        try:
            numeric = float(value)
        except Exception:
            continue
        if np.isfinite(numeric) and numeric != 0:
            return numeric

    if float(home_is_favorite) > 0:
        return -1.5
    if float(home_is_favorite) < 0:
        return 1.5
    return 0.0


def build_run_line_pick(home_team: str, away_team: str, home_spread_line: float, predicted_margin: float):
    line_abs = abs(float(home_spread_line)) if float(home_spread_line) != 0 else 1.5
    cover_edge = float(predicted_margin) + float(home_spread_line)
    if cover_edge >= 0:
        return f"{home_team} {-line_abs:.1f}", home_team, line_abs
    return f"{away_team} +{line_abs:.1f}", away_team, line_abs


def build_output_rows(df_day: pd.DataFrame, schedule_df: pd.DataFrame, line_movement_df: pd.DataFrame):
    calibration_cfg = load_calibration_config(CALIBRATION_FILE)

    fg_probs, fg_preds, _, _ = predict_market(df_day, "full_game")
    yrfi_probs, yrfi_preds, _, _ = predict_market(df_day, "yrfi")
    f5_probs, f5_preds, _, _ = predict_market(df_day, "f5")
    total_preds, _ = predict_regression_market(df_day, "totals")
    total_hits_preds = np.full(len(df_day), np.nan, dtype=float)
    try:
        total_hits_preds, _ = predict_regression_market(df_day, "total_hits_event")
    except Exception as exc:
        print(f"WARNING: mercado total_hits_event no disponible ({exc}). Se deja en PASS.")
    margin_preds, _ = predict_regression_market(df_day, "run_line")

    output = []

    for i, row in df_day.reset_index(drop=True).iterrows():
        sched = schedule_df[schedule_df["game_id"].astype(str) == str(row["game_id"])]
        sched_row = sched.iloc[0] if not sched.empty else None
        line = line_movement_df[line_movement_df["game_id"].astype(str) == str(row["game_id"])] if not line_movement_df.empty else pd.DataFrame()
        line_row = line.iloc[0] if not line.empty else None

        home_team = row["home_team"]
        away_team = row["away_team"]
        date_str = row["date"]

        fg_model_prob_home = float(fg_probs[i])
        f5_model_prob_home = float(f5_probs[i])
        yrfi_model_prob_yes = float(yrfi_probs[i])
        predicted_total_runs = float(total_preds[i])
        predicted_total_hits_event = float(total_hits_preds[i]) if np.isfinite(total_hits_preds[i]) else None
        predicted_home_margin = float(margin_preds[i])

        fg_prob_home = calibrate_probability(fg_model_prob_home, "mlb", "full_game", calibration_cfg)
        f5_prob_home = calibrate_probability(f5_model_prob_home, "mlb", "f5", calibration_cfg)
        yrfi_prob_yes = calibrate_probability(yrfi_model_prob_yes, "mlb", "yrfi", calibration_cfg)

        mlb_patterns = generate_mlb_patterns(row.to_dict())
        pattern_edge = aggregate_pattern_edge(mlb_patterns)

        (
            full_game_publish_ok,
            full_game_threshold_effective,
            full_game_momentum_penalty,
            full_game_volatility_score,
            fg_prob_home_vol_norm,
            fg_prob_home_reliability,
            full_game_decision_threshold,
            full_game_reliability_score,
            full_game_reliability_side_hint,
            full_game_reliability_conflict_flag,
            full_game_reliability_side_shift,
            full_game_reliability_shrink,
        ) = compute_full_game_publish_policy(row, fg_prob_home)
        full_game_pick = home_team if fg_prob_home_reliability >= full_game_decision_threshold else away_team
        full_game_pick_prob = fg_prob_home_reliability if full_game_pick == home_team else (1.0 - fg_prob_home_reliability)
        full_game_conf = int(round(np.clip(full_game_pick_prob, 0.50, 1.0) * 100))
        full_game_score = fuse_with_pattern_score(recommendation_score(fg_prob_home_reliability), pattern_edge)

        f5_pick = home_team if f5_preds[i] == 1 else away_team
        f5_conf = confidence_from_prob(f5_prob_home)
        f5_score = fuse_with_pattern_score(recommendation_score(f5_prob_home), pattern_edge)

        yrfi_pick = "YRFI" if yrfi_preds[i] == 1 else "NRFI"
        yrfi_conf = confidence_from_prob(yrfi_prob_yes)
        yrfi_score = fuse_with_pattern_score(recommendation_score(yrfi_prob_yes), pattern_edge)

        total_line = resolve_total_market_line(sched_row, line_row)
        total_hits_line = resolve_total_hits_market_line(sched_row, line_row)
        home_spread_line = resolve_home_spread_line(sched_row, line_row, row.get("home_is_favorite", 0))

        home_ml_odds = None if sched_row is None else sched_row.get("home_moneyline_odds")
        away_ml_odds = None if sched_row is None else sched_row.get("away_moneyline_odds")
        closing_ml_odds = None if sched_row is None else sched_row.get("closing_moneyline_odds")
        closing_spread_odds = None if sched_row is None else sched_row.get("closing_spread_odds")
        closing_total_odds = None if sched_row is None else sched_row.get("closing_total_odds")
        odds_details = "No Line" if sched_row is None else str(sched_row.get("odds_details", "No Line") or "No Line")
        odds_data_quality = "none" if sched_row is None else str(sched_row.get("odds_data_quality", "none") or "none")

        if total_line > 0:
            total_pick = "OVER" if predicted_total_runs >= total_line else "UNDER"
            total_confidence = confidence_from_edge(abs(predicted_total_runs - total_line))
        else:
            total_pick = "PASS"
            total_confidence = 50

        if total_hits_line > 0 and predicted_total_hits_event is not None:
            total_hits_pick = "OVER" if predicted_total_hits_event >= total_hits_line else "UNDER"
            total_hits_confidence = confidence_from_edge(abs(predicted_total_hits_event - total_hits_line))
        else:
            total_hits_pick = "PASS"
            total_hits_confidence = 50

        spread_pick, spread_side_team, spread_line_abs = build_run_line_pick(
            home_team=home_team,
            away_team=away_team,
            home_spread_line=home_spread_line,
            predicted_margin=predicted_home_margin,
        )
        spread_confidence = confidence_from_edge(abs(predicted_home_margin + home_spread_line))
        selected_spread_line = -spread_line_abs if spread_side_team == home_team else spread_line_abs

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
                "full_game_model_prob_home": round(fg_model_prob_home, 4),
                "full_game_calibrated_prob_home": round(fg_prob_home, 4),
                "full_game_vol_norm_prob_home": round(fg_prob_home_vol_norm, 4),
                "full_game_reliability_prob_home": round(fg_prob_home_reliability, 4),
                "full_game_volatility_score": round(full_game_volatility_score, 4),
                "full_game_vol_norm_enabled": bool(FULL_GAME_VOL_NORM_ENABLED > 0),
                "full_game_decision_threshold": round(full_game_decision_threshold, 4),
                "full_game_vol_decision_enabled": bool(FULL_GAME_VOL_DECISION_ENABLED > 0),
                "full_game_vol_decision_beta": round(FULL_GAME_VOL_DECISION_BETA, 4),
                "full_game_reliability_enabled": bool(FULL_GAME_RELIABILITY_ENABLED > 0),
                "full_game_reliability_score": round(full_game_reliability_score, 4),
                "full_game_reliability_side_hint": round(full_game_reliability_side_hint, 4),
                "full_game_reliability_conflict_flag": bool(full_game_reliability_conflict_flag >= 0.5),
                "full_game_reliability_side_shift": round(full_game_reliability_side_shift, 5),
                "full_game_reliability_shrink": round(full_game_reliability_shrink, 4),
                "full_game_pattern_edge": round(pattern_edge, 4),
                "full_game_detected_patterns": mlb_patterns,
                "full_game_recommended_score": round(full_game_score, 1),
                "full_game_publish_threshold_effective": round(full_game_threshold_effective, 4),
                "full_game_momentum_penalty": round(full_game_momentum_penalty, 4),
                "full_game_action": "JUGAR" if full_game_publish_ok else "PASS",
                "full_game_recommended": bool(full_game_publish_ok),

                "q1_pick": yrfi_pick,
                "q1_confidence": yrfi_conf,
                "q1_action": "JUGAR" if yrfi_conf >= 56 else "PASS",
                "q1_model_prob_yes": round(yrfi_model_prob_yes, 4),
                "q1_calibrated_prob_yes": round(yrfi_prob_yes, 4),
                "q1_recommended_score": round(yrfi_score, 1),

                "f5_pick": f5_pick,
                "f5_confidence": f5_conf,
                "f5_tier": tier_from_conf(f5_conf),
                "f5_model_prob_home": round(f5_model_prob_home, 4),
                "f5_calibrated_prob_home": round(f5_prob_home, 4),
                "f5_recommended_score": round(f5_score, 1),

                "total_pick": total_pick,
                "total_recommended_pick": f"{total_pick.title()} {total_line:.1f}" if total_line > 0 else total_pick.title(),
                "total_confidence": total_confidence,
                "predicted_total_runs": round(predicted_total_runs, 2),
                "odds_over_under": total_line,
                "closing_total_line": total_line,

                "total_hits_pick": total_hits_pick,
                "total_hits_recommended_pick": (
                    f"{total_hits_pick.title()} {total_hits_line:.1f}" if total_hits_line > 0 else total_hits_pick.title()
                ),
                "total_hits_confidence": total_hits_confidence,
                "predicted_total_hits_event": (
                    round(predicted_total_hits_event, 2) if predicted_total_hits_event is not None else None
                ),
                "odds_total_hits_event": total_hits_line,
                "closing_total_hits_line": total_hits_line,
                "total_hits_model_available": bool(predicted_total_hits_event is not None),

                "odds_details": odds_details,
                "odds_data_quality": odds_data_quality,
                "home_moneyline_odds": None if pd.isna(home_ml_odds) else float(home_ml_odds),
                "away_moneyline_odds": None if pd.isna(away_ml_odds) else float(away_ml_odds),
                "closing_moneyline_odds": None if pd.isna(closing_ml_odds) else float(closing_ml_odds),
                "closing_spread_odds": None if pd.isna(closing_spread_odds) else float(closing_spread_odds),
                "closing_total_odds": None if pd.isna(closing_total_odds) else float(closing_total_odds),
                "moneyline_odds": (
                    None if pd.isna(home_ml_odds if full_game_pick == home_team else away_ml_odds)
                    else float(home_ml_odds if full_game_pick == home_team else away_ml_odds)
                ),
                "pick_ml_odds": (
                    None if pd.isna(home_ml_odds if full_game_pick == home_team else away_ml_odds)
                    else float(home_ml_odds if full_game_pick == home_team else away_ml_odds)
                ),

                "spread_pick": spread_pick,
                "spread_confidence": spread_confidence,
                "spread_market": "RUN_LINE",
                "predicted_home_margin": round(predicted_home_margin, 2),
                "closing_spread_line": round(selected_spread_line, 2),
                "home_spread": round(home_spread_line, 2),
                "spread_abs": round(spread_line_abs, 1),
                "spread_side_team": spread_side_team,

                "status_state": "" if sched_row is None else str(sched_row.get("status_state", "") or ""),
                "status_description": "" if sched_row is None else str(sched_row.get("status_description", "") or ""),
            }
        )

    return output


def main():
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"No existe el archivo: {FEATURES_FILE}")
    if not UPCOMING_FILE.exists():
        raise FileNotFoundError(
            f"No existe el archivo de agenda del día: {UPCOMING_FILE}\n"
            f"Primero corre data_ingest_mlb.py"
        )

    history_df = pd.read_csv(FEATURES_FILE)
    history_df["date"] = history_df["date"].astype(str)

    schedule_df = pd.read_csv(UPCOMING_FILE, dtype={"game_id": str})
    if schedule_df.empty:
        raise ValueError("La agenda del día está vacía.")

    # Defensive cleanup: drop blank/invalid date rows that can create nan.json outputs.
    before_rows = len(schedule_df)
    schedule_df = schedule_df.dropna(how="all").copy()
    schedule_df["date"] = pd.to_datetime(schedule_df.get("date"), errors="coerce").dt.strftime("%Y-%m-%d")
    invalid_date_mask = schedule_df["date"].isna()
    invalid_date_count = int(invalid_date_mask.sum())
    if invalid_date_count:
        schedule_df = schedule_df.loc[~invalid_date_mask].copy()
        print(f"WARNING: Se omitieron {invalid_date_count} filas de agenda con fecha inválida.")

    if schedule_df.empty:
        raise ValueError("La agenda del día no contiene filas válidas con fecha.")

    cleaned_rows = before_rows - len(schedule_df)
    if cleaned_rows and not invalid_date_count:
        print(f"INFO: Se limpiaron {cleaned_rows} filas vacías de agenda MLB.")
    line_movement_df = load_line_movement_frame()

    schedule_df["status_completed"] = pd.to_numeric(
        schedule_df.get("status_completed", 0), errors="coerce"
    ).fillna(0).astype(int)

    latest_dependency_mtime = get_latest_prediction_dependency_mtime()

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

        # Incremental: skip generation if output is newer than all relevant inputs
        if output_path.exists():
            try:
                out_mtime = output_path.stat().st_mtime
            except Exception:
                out_mtime = 0
            if out_mtime >= latest_dependency_mtime:
                print(f"ℹ️ {date_str}: predicción ya existente y actualizada, se omite")
                total_skipped += len(skipped)
                continue

        if not pregame_rows:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2, ensure_ascii=False)
            print(f"⚠️ {date_str}: sin features suficientes, se guardó archivo vacío")
            total_skipped += len(skipped)
            continue

        df_day = pd.DataFrame(pregame_rows)
        rows = build_output_rows(df_day, sched, line_movement_df)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

        print(f"✅ {date_str}: predicciones MLB {len(rows)} juegos -> {output_path.name}")
        total_rows += len(rows)
        total_skipped += len(skipped)

    print(f"📦 Total predicciones MLB generadas: {total_rows}")
    if total_skipped:
        print(f"⚠️ Juegos omitidos por falta de histórico: {total_skipped}")


if __name__ == "__main__":
    main()
