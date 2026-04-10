from __future__ import annotations

import json
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from .train_models_mlb import (
        ConstantBinaryModel,
        REGRESSION_TARGET_CONFIG,
        TARGET_CONFIG,
        build_lgbm,
        build_lgbm_regressor,
        build_xgb,
        build_xgb_regressor,
        get_feature_columns,
        get_market_feature_columns,
        get_scale_pos_weight,
        load_dataset,
    )
except ImportError:
    from sports.mlb.train_models_mlb import (
        ConstantBinaryModel,
        REGRESSION_TARGET_CONFIG,
        TARGET_CONFIG,
        build_lgbm,
        build_lgbm_regressor,
        build_xgb,
        build_xgb_regressor,
        get_feature_columns,
        get_market_feature_columns,
        get_scale_pos_weight,
        load_dataset,
    )
from walk_forward_utils import (
    choose_threshold_from_calibration,
    generate_date_walk_forward_splits,
    safe_binary_metrics,
    sanitize_feature_frame,
    summarize_predictions,
)

BASE_DIR = SRC_ROOT
OUTPUT_DIR = BASE_DIR / "data" / "mlb" / "walkforward"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuración inicial OPTIMIZADA
MIN_TRAIN_DATES = 120
CALIBRATION_DATES = 21
TEST_DATES = 1
STEP_DATES = 7

MIN_CALIBRATION_ROWS = 50

MIN_PUBLISH_THRESHOLD = {
    "full_game": 0.56,
    "yrfi": 0.56,
    "f5": 0.56,
}

MARKET_THRESHOLD_CONFIG = {
    "full_game": {
        "min_threshold": 0.56,
        "max_threshold": 0.68,
        "min_coverage": 0.08,
        "prob_shrink": 0.00,
        "fallback_penalty": 0.01,
        "missing_pitcher_penalty": 0.05,
        "dead_zones": [(0.60, 0.62)],
    },
    "yrfi": {
        "min_threshold": 0.55,
        "max_threshold": 0.63,
        "min_coverage": 0.02,
        "prob_shrink": 0.06,
        "fallback_penalty": 0.02,
        "missing_pitcher_penalty": 0.08,
    },
    "f5": {
        "min_threshold": 0.56,
        "max_threshold": 0.66,
        "min_coverage": 0.02,
        "prob_shrink": 0.00,
        "fallback_penalty": 0.015,
        "missing_pitcher_penalty": 0.08,
        "dead_zones": [(0.60, 0.65)],
    },
    "totals": {
        "min_edge": 0.75,
        "publish_edge": 1.50,
    },
    "run_line": {
        "min_edge": 0.60,
        "publish_edge": 1.60,
    },
}

class WeightedEnsembleModel:
    def __init__(self, xgb_model, lgbm_model, xgb_weight: float = 0.5, lgbm_weight: float = 0.5):
        self.xgb_model = xgb_model
        self.lgbm_model = lgbm_model
        self.xgb_weight = float(xgb_weight)
        self.lgbm_weight = float(lgbm_weight)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        xgb_probs = self.xgb_model.predict_proba(X)[:, 1]
        lgbm_probs = self.lgbm_model.predict_proba(X)[:, 1]
        probs = (self.xgb_weight * xgb_probs) + (self.lgbm_weight * lgbm_probs)
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - probs, probs])


def get_market_runtime_config(market_key: str) -> Dict[str, float]:
    base = MARKET_THRESHOLD_CONFIG.get(market_key, {}).copy()
    if not base:
        base = {
            "min_threshold": MIN_PUBLISH_THRESHOLD.get(market_key, 0.56),
            "max_threshold": 0.68,
            "min_coverage": 0.15,
            "prob_shrink": 0.00,
            "fallback_penalty": 0.01,
            "missing_pitcher_penalty": 0.00,
        }
    return base


def shrink_probs_toward_half(probs: np.ndarray, shrink: float) -> np.ndarray:
    shrink = float(np.clip(shrink, 0.0, 0.49))
    return np.clip(0.5 + ((probs - 0.5) * (1.0 - shrink)), 1e-6, 1.0 - 1e-6)


def get_missing_pitcher_flag(df_like: pd.DataFrame) -> np.ndarray:
    if "both_pitchers_available" in df_like.columns:
        vals = df_like["both_pitchers_available"].fillna(0).astype(float).to_numpy()
        return (vals < 1.0).astype(int)

    if "diff_pitcher_data_available" in df_like.columns:
        vals = df_like["diff_pitcher_data_available"].fillna(0).astype(float).to_numpy()
        return (np.abs(vals) < 0.5).astype(int)

    return np.zeros(len(df_like), dtype=int)


def _safe_numeric_array(df_like: pd.DataFrame, col: str) -> np.ndarray | None:
    if not isinstance(df_like, pd.DataFrame) or col not in df_like.columns:
        return None
    return pd.to_numeric(df_like[col], errors="coerce").to_numpy(dtype=float)


def _collect_mean_arrays(arrays: List[np.ndarray]) -> np.ndarray | None:
    valid = [a for a in arrays if a is not None and len(a) > 0]
    if not valid:
        return None
    stacked = np.vstack(valid)
    return np.nanmean(stacked, axis=0)


def compute_pitcher_momentum_publish_penalty(
    probs_calibrated: np.ndarray,
    df_context: pd.DataFrame,
    market_key: str,
) -> Tuple[np.ndarray, Dict[str, float]]:
    n = len(probs_calibrated)
    penalty = np.zeros(n, dtype=float)
    meta = {
        "momentum_penalty_mean": 0.0,
        "momentum_penalty_max": 0.0,
        "momentum_penalty_active_rate": 0.0,
    }

    if n == 0 or not isinstance(df_context, pd.DataFrame) or df_context.empty:
        return penalty, meta

    pred_home_or_yrfi = probs_calibrated >= 0.5

    if market_key in {"full_game", "f5"}:
        pred_side = np.where(pred_home_or_yrfi, 1.0, -1.0)

        diff_quality = _safe_numeric_array(df_context, "diff_pitcher_recent_quality_score")
        if diff_quality is not None:
            diff_quality = np.nan_to_num(diff_quality, nan=0.0)
            quality_side = np.sign(diff_quality)
            quality_conflict = (quality_side * pred_side) < 0
            abs_quality = np.abs(diff_quality)
            strong_thr = float(np.nanquantile(abs_quality, 0.65)) if np.any(np.isfinite(abs_quality)) else 0.0
            strong_conflict = quality_conflict & (abs_quality >= strong_thr) & (abs_quality > 0)

            penalty += quality_conflict.astype(float) * 0.004
            penalty += strong_conflict.astype(float) * 0.005

        diff_era_trend = _safe_numeric_array(df_context, "diff_pitcher_era_trend")
        if diff_era_trend is not None:
            diff_era_trend = np.nan_to_num(diff_era_trend, nan=0.0)
            era_conflict = (
                (pred_home_or_yrfi & (diff_era_trend > 0.12))
                | ((~pred_home_or_yrfi) & (diff_era_trend < -0.12))
            )
            penalty += era_conflict.astype(float) * 0.002

        diff_whip_trend = _safe_numeric_array(df_context, "diff_pitcher_whip_trend")
        if diff_whip_trend is not None:
            diff_whip_trend = np.nan_to_num(diff_whip_trend, nan=0.0)
            whip_conflict = (
                (pred_home_or_yrfi & (diff_whip_trend > 0.025))
                | ((~pred_home_or_yrfi) & (diff_whip_trend < -0.025))
            )
            penalty += whip_conflict.astype(float) * 0.002

    elif market_key == "yrfi":
        risk_components: List[np.ndarray] = []

        for col in [
            "home_pitcher_r1_allowed_rate_L10",
            "away_pitcher_r1_allowed_rate_L10",
            "home_pitcher_r1_allowed_rate_L5",
            "away_pitcher_r1_allowed_rate_L5",
            "home_pitcher_blowup_rate_L10",
            "away_pitcher_blowup_rate_L10",
        ]:
            arr = _safe_numeric_array(df_context, col)
            if arr is not None:
                risk_components.append(np.clip(np.nan_to_num(arr, nan=0.0), 0.0, 1.0))

        avg_pitcher_quality = _collect_mean_arrays(
            [
                _safe_numeric_array(df_context, "home_pitcher_recent_quality_score"),
                _safe_numeric_array(df_context, "away_pitcher_recent_quality_score"),
            ]
        )
        if avg_pitcher_quality is not None:
            q = np.nan_to_num(avg_pitcher_quality, nan=0.0)
            q10 = float(np.nanquantile(q, 0.10)) if np.any(np.isfinite(q)) else 0.0
            q90 = float(np.nanquantile(q, 0.90)) if np.any(np.isfinite(q)) else 1.0
            denom = max(q90 - q10, 1e-6)
            quality_risk = 1.0 - np.clip((q - q10) / denom, 0.0, 1.0)
            risk_components.append(quality_risk)

        if risk_components:
            risk = np.nanmean(np.vstack(risk_components), axis=0)
            hi_thr = float(np.nanquantile(risk, 0.60))
            lo_thr = float(np.nanquantile(risk, 0.40))

            pred_yrfi = pred_home_or_yrfi
            nrfi_conflict = (~pred_yrfi) & (risk >= hi_thr)
            yrfi_conflict = pred_yrfi & (risk <= lo_thr)

            penalty += nrfi_conflict.astype(float) * 0.008
            penalty += yrfi_conflict.astype(float) * 0.004

        diff_quality = _safe_numeric_array(df_context, "diff_pitcher_quality_start_rate_L10")
        if diff_quality is not None:
            diff_quality = np.nan_to_num(diff_quality, nan=0.0)
            nrfi_quality_conflict = (~pred_home_or_yrfi) & (diff_quality > 0.10)
            nrfi_quality_strong_conflict = (~pred_home_or_yrfi) & (diff_quality > 0.25)
            penalty += nrfi_quality_conflict.astype(float) * 0.010
            penalty += nrfi_quality_strong_conflict.astype(float) * 0.006

    penalty = np.clip(penalty, 0.0, 0.03)
    if len(penalty):
        meta["momentum_penalty_mean"] = float(np.mean(penalty))
        meta["momentum_penalty_max"] = float(np.max(penalty))
        meta["momentum_penalty_active_rate"] = float(np.mean(penalty > 1e-8))
    return penalty, meta


def apply_publish_policy(
    probs_calibrated: np.ndarray,
    df_context: pd.DataFrame,
    base_threshold: float,
    market_key: str,
    used_fallback: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    cfg = get_market_runtime_config(market_key)

    confidence = np.maximum(probs_calibrated, 1.0 - probs_calibrated)
    missing_pitcher_flag = get_missing_pitcher_flag(df_context)
    effective_threshold = np.full(len(df_context), float(base_threshold), dtype=float)

    if int(used_fallback) == 1:
        effective_threshold += float(cfg.get("fallback_penalty", 0.0))

    effective_threshold += missing_pitcher_flag * float(cfg.get("missing_pitcher_penalty", 0.0))

    momentum_penalty, momentum_meta = compute_pitcher_momentum_publish_penalty(
        probs_calibrated=probs_calibrated,
        df_context=df_context,
        market_key=market_key,
    )
    if int(used_fallback) == 1:
        momentum_penalty = np.zeros_like(momentum_penalty)
        momentum_meta = {
            "momentum_penalty_mean": 0.0,
            "momentum_penalty_max": 0.0,
            "momentum_penalty_active_rate": 0.0,
        }
    effective_threshold += momentum_penalty

    effective_threshold = np.clip(
        effective_threshold,
        float(cfg.get("min_threshold", 0.56)),
        float(cfg.get("max_threshold", 0.75)),
    )

    publish_pick = (confidence >= effective_threshold).astype(int)
    for dead_zone in cfg.get("dead_zones", []) or []:
        if not isinstance(dead_zone, (list, tuple)) or len(dead_zone) != 2:
            continue
        low, high = float(dead_zone[0]), float(dead_zone[1])
        publish_pick[(confidence >= low) & (confidence < high)] = 0

    meta = {
        "fallback_penalty": float(cfg.get("fallback_penalty", 0.0)) if int(used_fallback) == 1 else 0.0,
        "missing_pitcher_penalty": float(cfg.get("missing_pitcher_penalty", 0.0)),
        "prob_shrink": float(cfg.get("prob_shrink", 0.0)),
        "missing_pitchers_rate": float(missing_pitcher_flag.mean()) if len(missing_pitcher_flag) else 0.0,
        "momentum_penalty_mean": float(momentum_meta.get("momentum_penalty_mean", 0.0)),
        "momentum_penalty_max": float(momentum_meta.get("momentum_penalty_max", 0.0)),
        "momentum_penalty_active_rate": float(momentum_meta.get("momentum_penalty_active_rate", 0.0)),
    }
    return publish_pick, effective_threshold, meta


def add_publish_buckets(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty or "confidence" not in detail_df.columns:
        return detail_df

    out = detail_df.copy()
    bins = [0.0, 0.53, 0.56, 0.60, 0.65, 1.01]
    labels = ["<=0.53", "0.53-0.56", "0.56-0.60", "0.60-0.65", "0.65+"]
    out["confidence_bucket"] = pd.cut(
        out["confidence"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    ).astype(str)
    return out


def compute_published_accuracy(detail_df: pd.DataFrame) -> float:
    if detail_df.empty or "publish_pick" not in detail_df.columns:
        return 0.0
    pub = detail_df[detail_df["publish_pick"] == 1].copy()
    if pub.empty:
        return 0.0
    return float((pub["pred_label"].astype(int) == pub["y_true"].astype(int)).mean())


def compute_published_coverage(detail_df: pd.DataFrame) -> float:
    if detail_df.empty or "publish_pick" not in detail_df.columns or len(detail_df) == 0:
        return 0.0
    return float(detail_df["publish_pick"].fillna(0).astype(int).mean())


def summarize_confidence_buckets(detail_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if detail_df.empty or "confidence_bucket" not in detail_df.columns:
        return {}

    out: Dict[str, Dict[str, float]] = {}
    for bucket, grp in detail_df.groupby("confidence_bucket", dropna=False):
        if grp.empty:
            continue
        bucket_pub = grp[grp["publish_pick"] == 1].copy()
        out[str(bucket)] = {
            "rows": int(len(grp)),
            "coverage": float(grp["publish_pick"].fillna(0).astype(int).mean()) if len(grp) else 0.0,
            "accuracy": float((grp["pred_label"].astype(int) == grp["y_true"].astype(int)).mean()) if len(grp) else 0.0,
            "published_rows": int(len(bucket_pub)),
            "published_accuracy": float((bucket_pub["pred_label"].astype(int) == bucket_pub["y_true"].astype(int)).mean()) if len(bucket_pub) else 0.0,
        }
    return out


def _summarize_quantile_buckets(
    detail_df: pd.DataFrame,
    value_col: str,
    q: int = 3,
) -> Dict[str, Dict[str, float]]:
    if detail_df.empty or value_col not in detail_df.columns:
        return {}

    tmp = detail_df.copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[value_col, "pred_label", "y_true"]).copy()
    if len(tmp) < 30:
        return {}

    try:
        tmp["bucket"] = pd.qcut(tmp[value_col], q=q, duplicates="drop")
    except Exception:
        return {}

    out: Dict[str, Dict[str, float]] = {}
    for bucket_name, grp in tmp.groupby("bucket", dropna=False):
        if grp.empty:
            continue
        pub = grp[grp.get("publish_pick", 0).fillna(0).astype(int) == 1].copy() if "publish_pick" in grp.columns else grp.iloc[0:0].copy()
        out[str(bucket_name)] = {
            "rows": int(len(grp)),
            "accuracy": float((grp["pred_label"].astype(int) == grp["y_true"].astype(int)).mean()) if len(grp) else 0.0,
            "published_rows": int(len(pub)),
            "published_accuracy": float((pub["pred_label"].astype(int) == pub["y_true"].astype(int)).mean()) if len(pub) else 0.0,
        }
    return out


def summarize_pitcher_momentum_analysis(detail_df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
    metrics_map = {
        "diff_pitcher_recent_quality_score": "recent_quality_diff",
        "diff_pitcher_quality_start_rate_L10": "quality_start_rate_diff",
        "diff_pitcher_blowup_rate_L10": "blowup_rate_diff",
        "diff_pitcher_era_trend": "era_trend_diff",
        "diff_pitcher_whip_trend": "whip_trend_diff",
    }

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for col, key_name in metrics_map.items():
        summary = _summarize_quantile_buckets(detail_df, col, q=3)
        if summary:
            out[key_name] = summary
    return out


def build_market_feature_map(df: pd.DataFrame) -> Dict[str, List[str]]:
    all_feature_cols = get_feature_columns(df)

    # Conservador: ajusta un poco por mercado sin romper compatibilidad
    yrfi_keep = [
        "home_yrfi_rate_L10", "away_yrfi_rate_L10", "diff_yrfi_rate_L10",
        "home_r1_scored_rate_L10", "away_r1_scored_rate_L10", "diff_r1_scored_rate_L10",
        "home_r1_scored_rate_L5", "away_r1_scored_rate_L5", "diff_r1_scored_rate_L5",
        "home_r1_allowed_rate_L10", "away_r1_allowed_rate_L10", "diff_r1_allowed_rate_L10",
        "home_r1_allowed_rate_L5", "away_r1_allowed_rate_L5", "diff_r1_allowed_rate_L5",
        "home_pitcher_r1_allowed_rate_L10", "away_pitcher_r1_allowed_rate_L10", "diff_pitcher_r1_allowed_rate_L10",
        "home_pitcher_r1_allowed_rate_L5", "away_pitcher_r1_allowed_rate_L5", "diff_pitcher_r1_allowed_rate_L5",
        "home_r1_vs_away_pitcher", "away_r1_vs_home_pitcher", "diff_r1_vs_pitcher",
        "home_r1_vs_away_pitcher_L5_proxy", "away_r1_vs_home_pitcher_L5_proxy", "diff_r1_vs_pitcher_L5",
        "yrfi_pressure_home", "yrfi_pressure_away", "diff_yrfi_pressure", "total_yrfi_pressure",
        "yrfi_pressure_home_L5", "yrfi_pressure_away_L5", "diff_yrfi_pressure_L5", "total_yrfi_pressure_L5",
        "home_r1_scored_std_L10", "away_r1_scored_std_L10", "diff_r1_scored_std_L10",
        "home_r1_allowed_std_L10", "away_r1_allowed_std_L10", "diff_r1_allowed_std_L10",
        "home_yrfi_consistency_L10", "away_yrfi_consistency_L10", "diff_yrfi_consistency_L10",
        "both_pitchers_available", "diff_pitcher_data_available",
    ]

    full_game_keep = [
        "diff_elo", "diff_rest_days", "diff_games_last_5_days", "diff_win_pct_L10",
        "diff_run_diff_L10", "diff_runs_scored_L5", "diff_runs_allowed_L5",
        "diff_runs_scored_std_L10", "diff_runs_allowed_std_L10", "diff_surface_win_pct_L5",
        "diff_surface_run_diff_L5", "diff_surface_edge", "diff_win_pct_L10_vs_league",
        "diff_run_diff_L10_vs_league", "diff_fatigue_index", "diff_form_power",
        "diff_pitcher_data_available", "diff_pitcher_rest_days", "diff_pitcher_runs_allowed_L5",
        "diff_pitcher_runs_allowed_L10", "diff_pitcher_start_win_rate_L10", "diff_bullpen_runs_allowed_L5",
        "diff_bullpen_runs_allowed_L10", "diff_bullpen_load_L3", "diff_offense_vs_pitcher",
        "home_is_favorite", "odds_over_under", "market_missing", "both_pitchers_available",
    ]

    f5_keep = [
        "diff_elo", "diff_rest_days", "diff_win_pct_L5", "diff_run_diff_L5", "diff_f5_win_pct_L5",
        "diff_f5_diff_L5", "diff_surface_f5_win_pct_L5", "diff_f5_win_pct_L5_vs_league",
        "diff_pitcher_rest_days", "diff_pitcher_f5_runs_allowed_L5", "diff_pitcher_runs_allowed_L5",
        "diff_pitcher_start_win_rate_L10", "diff_f5_vs_pitcher", "diff_offense_vs_pitcher",
        "home_is_favorite", "odds_over_under", "market_missing", "both_pitchers_available",
        "diff_pitcher_data_available",
    ]

    market_feature_map = {
        "full_game": [c for c in full_game_keep if c in all_feature_cols],
        "yrfi": [c for c in yrfi_keep if c in all_feature_cols],
        "f5": [c for c in f5_keep if c in all_feature_cols],
    }

    return market_feature_map


def fit_market_models(X_train: pd.DataFrame, y_train: pd.Series, market_key: str):
    if y_train.nunique() < 2:
        fixed_prob = 1.0 if int(y_train.iloc[0]) == 1 else 0.0
        return ConstantBinaryModel(fixed_prob), ConstantBinaryModel(fixed_prob)

    scale_pos_weight = get_scale_pos_weight(y_train)
    xgb_model = build_xgb(scale_pos_weight, market_key)
    lgbm_model = build_lgbm(scale_pos_weight, market_key)

    xgb_model.fit(X_train, y_train)
    lgbm_model.fit(X_train, y_train)
    return xgb_model, lgbm_model


def fit_regression_market_models(X_train: pd.DataFrame, y_train: pd.Series, market_key: str):
    xgb_model = build_xgb_regressor(market_key)
    lgbm_model = build_lgbm_regressor(market_key)
    xgb_model.fit(X_train, y_train)
    lgbm_model.fit(X_train, y_train)
    return xgb_model, lgbm_model


def evaluate_regression(y_true: pd.Series, preds: np.ndarray) -> Dict[str, float]:
    y_true_numeric = pd.to_numeric(y_true, errors="coerce").fillna(0.0)
    preds_arr = np.asarray(preds, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true_numeric, preds_arr)))
    return {
        "mae": float(mean_absolute_error(y_true_numeric, preds_arr)),
        "rmse": rmse,
        "r2": float(r2_score(y_true_numeric, preds_arr)),
    }


def get_standard_run_line_for_home(row: pd.Series) -> float | None:
    home_is_favorite = pd.to_numeric(row.get("home_is_favorite", np.nan), errors="coerce")
    if not np.isfinite(home_is_favorite) or float(home_is_favorite) == 0:
        return None
    return -1.5 if float(home_is_favorite) > 0 else 1.5


def build_regression_prediction_rows(
    df_test: pd.DataFrame,
    market_key: str,
    preds: np.ndarray,
    split_id: int,
) -> pd.DataFrame:
    out = df_test[[c for c in ["game_id", "date", "home_team", "away_team", "odds_over_under", "home_is_favorite"] if c in df_test.columns]].copy()
    out["market_key"] = market_key
    out["split_id"] = int(split_id)
    out["pred_value"] = np.asarray(preds, dtype=float)

    if market_key == "totals":
        line = pd.to_numeric(df_test.get("odds_over_under", np.nan), errors="coerce")
        actual_total = pd.to_numeric(df_test["TARGET_total_runs"], errors="coerce")
        valid_mask = np.isfinite(line) & (line > 0) & np.isfinite(actual_total)
        out["line_value"] = line
        out["actual_value"] = actual_total
        out["pred_label"] = np.where(valid_mask, (out["pred_value"] > line).astype(int), np.nan)
        out["y_true"] = np.where(valid_mask, (actual_total > line).astype(int), np.nan)
        out["edge_value"] = np.where(valid_mask, np.abs(out["pred_value"] - line), np.nan)
        publish_edge = float(MARKET_THRESHOLD_CONFIG.get("totals", {}).get("publish_edge", 1.5))
        out["publish_pick"] = np.where(valid_mask, (out["edge_value"] >= publish_edge).astype(int), 0)
    else:
        line_series = df_test.apply(get_standard_run_line_for_home, axis=1)
        actual_margin = pd.to_numeric(df_test["TARGET_home_run_margin"], errors="coerce")
        valid_mask = line_series.notna() & np.isfinite(actual_margin)
        out["line_value"] = pd.to_numeric(line_series, errors="coerce")
        out["actual_value"] = actual_margin
        out["pred_label"] = np.where(valid_mask, ((out["pred_value"] + out["line_value"]) > 0).astype(int), np.nan)
        out["y_true"] = np.where(valid_mask, ((actual_margin + out["line_value"]) > 0).astype(int), np.nan)
        out["edge_value"] = np.where(valid_mask, np.abs(out["pred_value"] + out["line_value"]), np.nan)
        publish_edge = float(MARKET_THRESHOLD_CONFIG.get("run_line", {}).get("publish_edge", 1.25))
        out["publish_pick"] = np.where(valid_mask, (out["edge_value"] >= publish_edge).astype(int), 0)

    out["confidence"] = np.where(np.isfinite(out["edge_value"]), np.clip(0.50 + (out["edge_value"] / 6.0), 0.50, 0.90), np.nan)
    return out


def summarize_regression_market(detail_df: pd.DataFrame, market_key: str) -> Dict[str, float]:
    if detail_df.empty:
        return {
            "rows": 0,
            "accuracy": 0.0,
            "published_accuracy": 0.0,
            "coverage": 0.0,
            "published_coverage": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "r2": 0.0,
            "published_confidence_buckets": {},
        }

    valid = detail_df.dropna(subset=["y_true", "pred_label", "actual_value", "pred_value"]).copy()
    if valid.empty:
        return {
            "rows": 0,
            "accuracy": 0.0,
            "published_accuracy": 0.0,
            "coverage": 0.0,
            "published_coverage": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "r2": 0.0,
            "published_confidence_buckets": {},
        }

    y_true = valid["y_true"].astype(int)
    pred_label = valid["pred_label"].astype(int)
    actual_value = pd.to_numeric(valid["actual_value"], errors="coerce").fillna(0.0)
    pred_value = pd.to_numeric(valid["pred_value"], errors="coerce").fillna(0.0)

    rmse = float(np.sqrt(mean_squared_error(actual_value, pred_value)))
    metrics = {
        "rows": int(len(valid)),
        "accuracy": float((pred_label == y_true).mean()),
        "published_accuracy": compute_published_accuracy(valid),
        "coverage": float(valid["publish_pick"].fillna(0).astype(int).mean()),
        "published_coverage": compute_published_coverage(valid),
        "mae": float(mean_absolute_error(actual_value, pred_value)),
        "rmse": rmse,
        "r2": float(r2_score(actual_value, pred_value)),
    }
    valid = add_publish_buckets(valid)
    metrics["published_confidence_buckets"] = summarize_confidence_buckets(valid)
    return metrics


def choose_ensemble_and_threshold(
    xgb_model,
    lgbm_model,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    market_key: str,
    calib_context: pd.DataFrame | None = None,
) -> Tuple[WeightedEnsembleModel, LogisticRegression, Dict[str, float], pd.DataFrame]:
    xgb_probs = xgb_model.predict_proba(X_calib)[:, 1]
    lgbm_probs = lgbm_model.predict_proba(X_calib)[:, 1]

    cfg = get_market_runtime_config(market_key)
    default_threshold = float(cfg["min_threshold"])

    best = {
        "xgb_weight": 0.5,
        "lgbm_weight": 0.5,
        "threshold": default_threshold,
        "score": -1e9,
        "coverage": 0.0,
        "accuracy": 0.0,
        "used_fallback": 1,
        "prob_shrink": float(cfg.get("prob_shrink", 0.0)),
    }
    best_calib_probs = None
    best_calibrator = None

    for xgb_weight in [0.20, 0.35, 0.50, 0.65, 0.80]:
        lgbm_weight = 1.0 - xgb_weight
        raw_probs = np.clip((xgb_weight * xgb_probs) + (lgbm_weight * lgbm_probs), 1e-6, 1.0 - 1e-6)

        lr_calibrator = LogisticRegression(C=1.0, solver='lbfgs')
        lr_calibrator.fit(raw_probs.reshape(-1, 1), y_calib.to_numpy())
        calib_probs = lr_calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
        
        if cfg.get("prob_shrink", 0.0) > 0:
            calib_probs = shrink_probs_toward_half(calib_probs, cfg["prob_shrink"])

        threshold_info = choose_threshold_from_calibration(
            y_true=y_calib.to_numpy(),
            probs=calib_probs,
            min_threshold=cfg["min_threshold"],
            max_threshold=cfg["max_threshold"],
            step=0.01,
            min_coverage=cfg["min_coverage"],
        )
        if market_key == "f5":
            print(
                f"[F5 CALIB] xgb_w={xgb_weight:.2f} | "
                f"thr={threshold_info.get('threshold', -1):.3f} | "
                f"acc={threshold_info.get('accuracy', 0.0):.4f} | "
                f"cov={threshold_info.get('coverage', 0.0):.4f} | "
                f"score={threshold_info.get('score', -999):.4f}"
            )

        invalid_threshold = (
            threshold_info.get("coverage", 0.0) <= 0.0
            or threshold_info.get("accuracy", 0.0) <= 0.0
            or threshold_info.get("score", -1e9) <= 0.0
        )
        if invalid_threshold:
            continue

        score = float(threshold_info["score"])
        if score > best["score"]:
            best = {
                "xgb_weight": float(round(xgb_weight, 2)),
                "lgbm_weight": float(round(lgbm_weight, 2)),
                **threshold_info,
                "used_fallback": 0,
                "prob_shrink": float(cfg.get("prob_shrink", 0.0)),
            }
            best_calib_probs = calib_probs
            best_calibrator = lr_calibrator

    if best_calib_probs is None or best_calibrator is None:
        if market_key == "f5":
            print(
                f"[F5 DEBUG] no valid threshold | "
                f"default_threshold={default_threshold:.3f} | "
                f"min_coverage={cfg['min_coverage']:.3f}"
            )

        fallback_raw = np.clip((0.5 * xgb_probs) + (0.5 * lgbm_probs), 1e-6, 1.0 - 1e-6)
        best_calibrator = LogisticRegression(C=1.0, solver='lbfgs')
        best_calibrator.fit(fallback_raw.reshape(-1, 1), y_calib.to_numpy())
        best_calib_probs = best_calibrator.predict_proba(fallback_raw.reshape(-1, 1))[:, 1]
        
        if cfg.get("prob_shrink", 0.0) > 0:
            best_calib_probs = shrink_probs_toward_half(best_calib_probs, cfg["prob_shrink"])

        best = {
            "xgb_weight": 0.5,
            "lgbm_weight": 0.5,
            "threshold": default_threshold,
            "score": -999.0,
            "coverage": 0.0,
            "accuracy": 0.0,
            "used_fallback": 1,
            "prob_shrink": float(cfg.get("prob_shrink", 0.0)),
        }

    ensemble_model = WeightedEnsembleModel(
        xgb_model=xgb_model,
        lgbm_model=lgbm_model,
        xgb_weight=best["xgb_weight"],
        lgbm_weight=best["lgbm_weight"],
    )

    calib_publish_pick, calib_effective_threshold, calib_policy_meta = apply_publish_policy(
        probs_calibrated=best_calib_probs,
        df_context=calib_context if isinstance(calib_context, pd.DataFrame) else (X_calib if isinstance(X_calib, pd.DataFrame) else pd.DataFrame(index=range(len(best_calib_probs)))),
        base_threshold=float(best["threshold"]),
        market_key=market_key,
        used_fallback=int(best.get("used_fallback", 0)),
    )

    calib_detail = pd.DataFrame(
        {
            "y_true": y_calib.to_numpy().astype(int),
            "ensemble_prob_calibrated": best_calib_probs,
            "pred_label": (best_calib_probs >= 0.5).astype(int),
            "publish_pick": calib_publish_pick.astype(int),
            "publish_threshold_effective": calib_effective_threshold.astype(float),
            "used_fallback": int(best.get("used_fallback", 0)),
        }
    )

    published_calib = calib_detail[calib_detail["publish_pick"] == 1].copy()
    best["published_accuracy"] = float((published_calib["pred_label"].astype(int) == published_calib["y_true"].astype(int)).mean()) if len(published_calib) else 0.0
    best["published_coverage"] = float(calib_detail["publish_pick"].mean()) if len(calib_detail) else 0.0
    best["fallback_penalty"] = float(calib_policy_meta.get("fallback_penalty", 0.0))
    best["missing_pitcher_penalty"] = float(calib_policy_meta.get("missing_pitcher_penalty", 0.0))
    best["missing_pitchers_rate_calib"] = float(calib_policy_meta.get("missing_pitchers_rate", 0.0))
    best["momentum_penalty_mean_calib"] = float(calib_policy_meta.get("momentum_penalty_mean", 0.0))
    best["momentum_penalty_active_rate_calib"] = float(calib_policy_meta.get("momentum_penalty_active_rate", 0.0))

    return ensemble_model, best_calibrator, best, calib_detail


def build_prediction_rows(
    df_test: pd.DataFrame,
    market_key: str,
    probs_raw: np.ndarray,
    probs_calibrated: np.ndarray,
    threshold: float,
    split_id: int,
    used_fallback: int,
    prob_shrink: float,
) -> pd.DataFrame:
    candidate_cols = [
        "game_id", "date", "home_team", "away_team",
        "both_pitchers_available", "diff_pitcher_data_available",
        "home_pitcher_quality_start_rate_L10", "away_pitcher_quality_start_rate_L10", "diff_pitcher_quality_start_rate_L10",
        "home_pitcher_blowup_rate_L10", "away_pitcher_blowup_rate_L10", "diff_pitcher_blowup_rate_L10",
        "home_pitcher_era_trend", "away_pitcher_era_trend", "diff_pitcher_era_trend",
        "home_pitcher_whip_trend", "away_pitcher_whip_trend", "diff_pitcher_whip_trend",
        "home_pitcher_recent_quality_score", "away_pitcher_recent_quality_score", "diff_pitcher_recent_quality_score",
    ]
    present_cols = [c for c in candidate_cols if c in df_test.columns]

    out = df_test[present_cols].copy()
    out["market_key"] = market_key
    out["split_id"] = int(split_id)
    out["ensemble_prob_raw"] = probs_raw
    out["ensemble_prob_calibrated"] = probs_calibrated
    out["pred_label"] = (probs_calibrated >= 0.5).astype(int)
    out["confidence"] = np.maximum(probs_calibrated, 1.0 - probs_calibrated)
    out["threshold"] = float(threshold)
    out["used_fallback"] = int(used_fallback)
    out["prob_shrink"] = float(prob_shrink)

    publish_pick, effective_threshold, policy_meta = apply_publish_policy(
        probs_calibrated=probs_calibrated,
        df_context=df_test,
        base_threshold=threshold,
        market_key=market_key,
        used_fallback=used_fallback,
    )

    out["publish_pick"] = publish_pick.astype(int)
    out["publish_threshold_effective"] = effective_threshold.astype(float)
    out["missing_pitchers_rate_test_split"] = float(policy_meta.get("missing_pitchers_rate", 0.0))
    out["y_true"] = df_test["target_col_runtime"].astype(int).to_numpy()
    return out


def run_market_walkforward(
    df: pd.DataFrame,
    market_key: str,
    target_col: str,
    feature_cols: List[str],
) -> Dict[str, pd.DataFrame]:
    market_df = df.dropna(subset=[target_col]).copy()
    market_df = market_df.sort_values(["date_dt", "game_id"]).reset_index(drop=True)
    market_df["target_col_runtime"] = market_df[target_col].astype(int)

    splits = generate_date_walk_forward_splits(
        market_df,
        date_col="date",
        min_train_dates=MIN_TRAIN_DATES,
        calibration_dates=CALIBRATION_DATES,
        test_dates=TEST_DATES,
        step_dates=STEP_DATES,
    )

    total_splits = len(splits)
    print(f"   total_splits={total_splits} | rows_market={len(market_df)}")

    split_rows: List[Dict] = []
    prediction_rows: List[pd.DataFrame] = []

    for i, split in enumerate(splits, start=1):
        print(
            f"   -> split {i}/{total_splits} | "
            f"train_dates={len(split.train_dates)} | "
            f"calib_dates={len(split.calib_dates)} | "
            f"test_dates={len(split.test_dates)} | "
            f"test_start={split.test_dates[0]} | test_end={split.test_dates[-1]}"
        )

        train_df = market_df[market_df["date"].astype(str).isin(split.train_dates)].copy()
        calib_df = market_df[market_df["date"].astype(str).isin(split.calib_dates)].copy()
        test_df = market_df[market_df["date"].astype(str).isin(split.test_dates)].copy()

        print(
            f"      rows -> train={len(train_df)} | calib={len(calib_df)} | test={len(test_df)}"
        )

        if len(train_df) < 100 or len(calib_df) < MIN_CALIBRATION_ROWS or test_df.empty:
            print("      split omitido por tamaño insuficiente")
            continue

        X_train = sanitize_feature_frame(train_df, feature_cols)
        y_train = train_df[target_col].astype(int)

        X_calib = sanitize_feature_frame(calib_df, feature_cols)
        y_calib = calib_df[target_col].astype(int)

        X_test = sanitize_feature_frame(test_df, feature_cols)
        y_test = test_df[target_col].astype(int)

        if y_train.nunique() < 2:
            print("      split omitido: y_train con una sola clase")
            continue
        if y_calib.nunique() < 2:
            print("      split omitido: y_calib con una sola clase")
            continue

        print("      entrenando XGB + LGBM...")
        xgb_model, lgbm_model = fit_market_models(X_train, y_train, market_key)
        print("      modelos entrenados, calibrando ensemble...")

        ensemble_model, calibrator, threshold_info, calib_detail = choose_ensemble_and_threshold(
            xgb_model=xgb_model,
            lgbm_model=lgbm_model,
            X_calib=X_calib,
            y_calib=y_calib,
            market_key=market_key,
            calib_context=calib_df,
        )

        print(
            f"      threshold={threshold_info['threshold']:.3f} | "
            f"calib_acc={threshold_info['accuracy']:.4f} | "
            f"calib_cov={threshold_info['coverage']:.4f} | "
            f"published_acc={threshold_info.get('published_accuracy', 0.0):.4f} | "
            f"published_cov={threshold_info.get('published_coverage', 0.0):.4f} | "
            f"xgb_w={threshold_info['xgb_weight']:.2f} | "
            f"lgbm_w={threshold_info['lgbm_weight']:.2f} | "
            f"fallback={threshold_info.get('used_fallback', 0)} | "
            f"shrink={threshold_info.get('prob_shrink', 0.0):.2f}"
        )

        raw_test_probs = ensemble_model.predict_proba(X_test)[:, 1]
        
        # Aquí también aplicamos reshape porque es de LogReg
        calibrated_test_probs = calibrator.predict_proba(raw_test_probs.reshape(-1, 1))[:, 1]
        
        if float(threshold_info.get("prob_shrink", 0.0)) > 0:
            calibrated_test_probs = shrink_probs_toward_half(
                calibrated_test_probs,
                float(threshold_info["prob_shrink"]),
            )
        test_metrics = safe_binary_metrics(y_test, calibrated_test_probs, threshold=0.5)

        print(
            f"      test_acc={test_metrics['accuracy']:.4f} | "
            f"brier={test_metrics['brier']:.4f} | "
            f"logloss={test_metrics['logloss']:.4f} | "
            f"auc={test_metrics['roc_auc']:.4f}"
        )

        pred_rows = build_prediction_rows(
            df_test=test_df,
            market_key=market_key,
            probs_raw=raw_test_probs,
            probs_calibrated=calibrated_test_probs,
            threshold=float(threshold_info["threshold"]),
            split_id=split.split_id,
            used_fallback=int(threshold_info.get("used_fallback", 0)),
            prob_shrink=float(threshold_info.get("prob_shrink", 0.0)),
        )
        prediction_rows.append(pred_rows)

        published_test = pred_rows[pred_rows["publish_pick"] == 1].copy()
        published_accuracy = float((published_test["pred_label"].astype(int) == published_test["y_true"].astype(int)).mean()) if len(published_test) else 0.0
        published_coverage = float(pred_rows["publish_pick"].mean()) if len(pred_rows) else 0.0

        if "both_pitchers_available" in test_df.columns:
            missing_pitchers_rate_test = float((test_df["both_pitchers_available"].fillna(0).astype(float) < 1.0).mean())
        elif "diff_pitcher_data_available" in test_df.columns:
            missing_pitchers_rate_test = float((test_df["diff_pitcher_data_available"].fillna(0).abs() < 0.5).mean())
        else:
            missing_pitchers_rate_test = 0.0

        split_rows.append(
            {
                "split_id": int(split.split_id),
                "market_key": market_key,
                "train_start": split.train_dates[0],
                "train_end": split.train_dates[-1],
                "calib_start": split.calib_dates[0],
                "calib_end": split.calib_dates[-1],
                "test_start": split.test_dates[0],
                "test_end": split.test_dates[-1],
                "train_rows": int(len(train_df)),
                "calib_rows": int(len(calib_df)),
                "test_rows": int(len(test_df)),
                "ensemble_xgb_weight": float(threshold_info["xgb_weight"]),
                "ensemble_lgbm_weight": float(threshold_info["lgbm_weight"]),
                "publish_threshold": float(threshold_info["threshold"]),
                "prob_shrink": float(threshold_info.get("prob_shrink", 0.0)),
                "calib_accuracy": float(threshold_info["accuracy"]),
                "calib_coverage": float(threshold_info["coverage"]),
                "published_calib_accuracy": float(threshold_info.get("published_accuracy", 0.0)),
                "published_calib_coverage": float(threshold_info.get("published_coverage", 0.0)),
                "used_fallback": int(threshold_info.get("used_fallback", 0)),
                "fallback_penalty": float(threshold_info.get("fallback_penalty", 0.0)),
                "missing_pitcher_penalty": float(threshold_info.get("missing_pitcher_penalty", 0.0)),
                "missing_pitchers_rate_calib": float(threshold_info.get("missing_pitchers_rate_calib", 0.0)),
                "momentum_penalty_mean_calib": float(threshold_info.get("momentum_penalty_mean_calib", 0.0)),
                "momentum_penalty_active_rate_calib": float(threshold_info.get("momentum_penalty_active_rate_calib", 0.0)),
                "missing_pitchers_rate_test": float(missing_pitchers_rate_test),
                "test_accuracy_at_050": float(test_metrics["accuracy"]),
                "test_brier": float(test_metrics["brier"]),
                "test_logloss": float(test_metrics["logloss"]),
                "test_auc": float(test_metrics["roc_auc"]),
                "published_test_accuracy": float(published_accuracy),
                "published_test_coverage": float(published_coverage),
            }
        )

    detail_df = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    detail_df = add_publish_buckets(detail_df)
    split_df = pd.DataFrame(split_rows)

    return {
        "detail": detail_df,
        "splits": split_df,
    }


def run_regression_market_walkforward(
    df: pd.DataFrame,
    market_key: str,
    target_col: str,
    feature_cols: List[str],
) -> Dict[str, pd.DataFrame]:
    market_df = df.dropna(subset=[target_col]).copy()
    market_df = market_df.sort_values(["date_dt", "game_id"]).reset_index(drop=True)

    splits = generate_date_walk_forward_splits(
        market_df,
        date_col="date",
        min_train_dates=MIN_TRAIN_DATES,
        calibration_dates=CALIBRATION_DATES,
        test_dates=TEST_DATES,
        step_dates=STEP_DATES,
    )

    total_splits = len(splits)
    print(f"   total_splits={total_splits} | rows_market={len(market_df)}")

    split_rows: List[Dict] = []
    prediction_rows: List[pd.DataFrame] = []

    for i, split in enumerate(splits, start=1):
        print(
            f"   -> split {i}/{total_splits} | "
            f"train_dates={len(split.train_dates)} | "
            f"calib_dates={len(split.calib_dates)} | "
            f"test_dates={len(split.test_dates)} | "
            f"test_start={split.test_dates[0]} | test_end={split.test_dates[-1]}"
        )

        train_df = market_df[market_df["date"].astype(str).isin(split.train_dates)].copy()
        calib_df = market_df[market_df["date"].astype(str).isin(split.calib_dates)].copy()
        test_df = market_df[market_df["date"].astype(str).isin(split.test_dates)].copy()

        print(f"      rows -> train={len(train_df)} | calib={len(calib_df)} | test={len(test_df)}")
        if len(train_df) < 100 or len(calib_df) < MIN_CALIBRATION_ROWS or test_df.empty:
            print("      split omitido por tamaño insuficiente")
            continue

        train_plus_calib = pd.concat([train_df, calib_df], ignore_index=True)
        X_train = sanitize_feature_frame(train_plus_calib, feature_cols)
        y_train = pd.to_numeric(train_plus_calib[target_col], errors="coerce").fillna(0.0)
        X_test = sanitize_feature_frame(test_df, feature_cols)
        y_test = pd.to_numeric(test_df[target_col], errors="coerce").fillna(0.0)

        print("      entrenando XGBReg + LGBMReg...")
        xgb_model, lgbm_model = fit_regression_market_models(X_train, y_train, market_key)

        xgb_preds = np.asarray(xgb_model.predict(X_test), dtype=float)
        lgbm_preds = np.asarray(lgbm_model.predict(X_test), dtype=float)
        ensemble_preds = (0.5 * xgb_preds) + (0.5 * lgbm_preds)
        regression_metrics = evaluate_regression(y_test, ensemble_preds)

        pred_rows = build_regression_prediction_rows(
            df_test=test_df,
            market_key=market_key,
            preds=ensemble_preds,
            split_id=split.split_id,
        )
        prediction_rows.append(pred_rows)

        valid_rows = pred_rows.dropna(subset=["y_true", "pred_label"]).copy()
        test_accuracy = float(
            (valid_rows["pred_label"].astype(int) == valid_rows["y_true"].astype(int)).mean()
        ) if len(valid_rows) else 0.0
        published_test = valid_rows[valid_rows["publish_pick"] == 1].copy()
        published_accuracy = float(
            (published_test["pred_label"].astype(int) == published_test["y_true"].astype(int)).mean()
        ) if len(published_test) else 0.0
        published_coverage = float(valid_rows["publish_pick"].mean()) if len(valid_rows) else 0.0

        print(
            f"      test_acc={test_accuracy:.4f} | "
            f"published_acc={published_accuracy:.4f} | "
            f"published_cov={published_coverage:.4f} | "
            f"mae={regression_metrics['mae']:.4f} | "
            f"rmse={regression_metrics['rmse']:.4f}"
        )

        split_rows.append(
            {
                "split_id": int(split.split_id),
                "market_key": market_key,
                "train_start": split.train_dates[0],
                "train_end": split.train_dates[-1],
                "calib_start": split.calib_dates[0],
                "calib_end": split.calib_dates[-1],
                "test_start": split.test_dates[0],
                "test_end": split.test_dates[-1],
                "train_rows": int(len(train_plus_calib)),
                "calib_rows": int(len(calib_df)),
                "test_rows": int(len(test_df)),
                "test_accuracy": float(test_accuracy),
                "published_test_accuracy": float(published_accuracy),
                "published_test_coverage": float(published_coverage),
                "test_mae": float(regression_metrics["mae"]),
                "test_rmse": float(regression_metrics["rmse"]),
                "test_r2": float(regression_metrics["r2"]),
            }
        )

    detail_df = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    detail_df = add_publish_buckets(detail_df)
    split_df = pd.DataFrame(split_rows)
    return {
        "detail": detail_df,
        "splits": split_df,
    }


def save_market_outputs(market_key: str, detail_df: pd.DataFrame, split_df: pd.DataFrame) -> Dict:
    market_dir = OUTPUT_DIR / market_key
    market_dir.mkdir(parents=True, exist_ok=True)

    detail_path = market_dir / "walkforward_predictions_detail.csv"
    splits_path = market_dir / "walkforward_splits_summary.csv"
    metrics_path = market_dir / "walkforward_metrics.json"

    detail_df.to_csv(detail_path, index=False)
    split_df.to_csv(splits_path, index=False)

    if market_key in REGRESSION_TARGET_CONFIG:
        metrics = summarize_regression_market(detail_df, market_key)
        metrics["splits"] = int(len(split_df))
        metrics["fallback_splits"] = 0
        metrics["avg_missing_pitchers_rate_test"] = 0.0
        metrics["avg_prob_shrink"] = 0.0
    else:
        metrics = summarize_predictions(detail_df)
        metrics["splits"] = int(len(split_df))
        metrics["fallback_splits"] = int(split_df["used_fallback"].fillna(0).sum()) if (not split_df.empty and "used_fallback" in split_df.columns) else 0
        metrics["published_accuracy"] = compute_published_accuracy(detail_df)
        metrics["published_coverage"] = compute_published_coverage(detail_df)
        metrics["avg_missing_pitchers_rate_test"] = float(split_df["missing_pitchers_rate_test"].fillna(0).mean()) if (not split_df.empty and "missing_pitchers_rate_test" in split_df.columns) else 0.0
        metrics["avg_prob_shrink"] = float(split_df["prob_shrink"].fillna(0).mean()) if (not split_df.empty and "prob_shrink" in split_df.columns) else 0.0
        metrics["avg_momentum_penalty_mean_calib"] = float(split_df["momentum_penalty_mean_calib"].fillna(0).mean()) if (not split_df.empty and "momentum_penalty_mean_calib" in split_df.columns) else 0.0
        metrics["avg_momentum_penalty_active_rate_calib"] = float(split_df["momentum_penalty_active_rate_calib"].fillna(0).mean()) if (not split_df.empty and "momentum_penalty_active_rate_calib" in split_df.columns) else 0.0
        metrics["published_confidence_buckets"] = summarize_confidence_buckets(detail_df)
        metrics["pitcher_momentum_analysis"] = summarize_pitcher_momentum_analysis(detail_df)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return {
        "detail_path": str(detail_path),
        "splits_path": str(splits_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }


def main() -> None:
    df = load_dataset()

    global_summary = {}

    for market_key, cfg in TARGET_CONFIG.items():
        target_col = cfg["target_col"]
        feature_cols = get_market_feature_columns(df, market_key)

        print(f"\nWalk-forward MLB | mercado={market_key} | features={len(feature_cols)}")
        outputs = run_market_walkforward(df, market_key, target_col, feature_cols)
        saved = save_market_outputs(market_key, outputs["detail"], outputs["splits"])
        global_summary[market_key] = saved["metrics"]
        print(f"OK {market_key}: {saved['metrics']}")

    for market_key, cfg in REGRESSION_TARGET_CONFIG.items():
        target_col = cfg["target_col"]
        feature_cols = get_market_feature_columns(df, market_key)

        print(f"\nWalk-forward MLB | mercado={market_key} | features={len(feature_cols)}")
        outputs = run_regression_market_walkforward(df, market_key, target_col, feature_cols)
        saved = save_market_outputs(market_key, outputs["detail"], outputs["splits"])
        global_summary[market_key] = saved["metrics"]
        print(f"OK {market_key}: {saved['metrics']}")

    summary_path = OUTPUT_DIR / "walkforward_summary_mlb.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)

    print(f"\nResumen global guardado en: {summary_path}")


if __name__ == "__main__":
    main()
