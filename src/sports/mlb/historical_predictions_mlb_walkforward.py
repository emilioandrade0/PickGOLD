from __future__ import annotations

import json
import os
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier

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
RAW_ADVANCED_HISTORY_PATH = BASE_DIR / "data" / "mlb" / "raw" / "mlb_advanced_history.csv"
_FULL_GAME_ODDS_LOOKUP_CACHE: pd.DataFrame | None = None

# Configuracion inicial OPTIMIZADA
MIN_TRAIN_DATES = 120
CALIBRATION_DATES = 21
TEST_DATES = 1
STEP_DATES = 7

MIN_CALIBRATION_ROWS = 50

MIN_PUBLISH_THRESHOLD = {
    "full_game": 0.56,
    "yrfi": 0.56,
    "f5": 0.55,
}

MARKET_THRESHOLD_CONFIG = {
    "full_game": {
        "min_threshold": 0.56,
        "max_threshold": 0.68,
        "min_coverage": 0.08,
        "target_coverage": 0.075,
        "min_published_rows": 18,
        "prob_shrink": 0.00,
        "fallback_penalty": 0.01,
        "missing_pitcher_penalty": 0.05,
        "dead_zones": [(0.60, 0.62)],
    },
    "yrfi": {
        "min_threshold": 0.55,
        "max_threshold": 0.63,
        "min_coverage": 0.02,
        "target_coverage": 0.04,
        "min_published_rows": 10,
        "prob_shrink": 0.06,
        "fallback_penalty": 0.02,
        "missing_pitcher_penalty": 0.08,
    },
    "f5": {
        "min_threshold": 0.55,
        "max_threshold": 0.66,
        "min_coverage": 0.02,
        "target_coverage": 0.09,
        "min_published_rows": 10,
        "prob_shrink": 0.00,
        "fallback_penalty": 0.015,
        "missing_pitcher_penalty": 0.08,
        "dead_zones": [(0.60, 0.65)],
    },
    "totals": {
        "min_edge": 0.75,
        "publish_edge": 1.50,
    },
    "total_hits_event": {
        "min_edge": 0.80,
        "publish_edge": 1.60,
    },
    "run_line": {
        "min_edge": 0.60,
        "publish_edge": 1.60,
    },
}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_float_list(name: str, default: List[float]) -> List[float]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return [float(v) for v in default]
    out: List[float] = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        try:
            out.append(float(t))
        except Exception:
            continue
    return [float(v) for v in out] if out else [float(v) for v in default]


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return str(default)
    value = str(raw).strip()
    return value if value else str(default)


def _env_market_filter(name: str = "NBA_MLB_MARKETS") -> Set[str] | None:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None

    values = {token.strip().lower() for token in raw.split(",") if token.strip()}
    return values if values else None


FULL_GAME_GLOBAL_BRIER_WEIGHT = float(
    np.clip(_env_float("NBA_MLB_FULL_GAME_BRIER_WEIGHT", 0.08), 0.0, 0.25)
)

FULL_GAME_PROB_SHIFT_ENABLED = int(np.clip(_env_float("NBA_MLB_FULL_GAME_PROB_SHIFT_ENABLED", 1.0), 0.0, 1.0))
FULL_GAME_PROB_SHIFT_MIN = float(np.clip(_env_float("NBA_MLB_FULL_GAME_PROB_SHIFT_MIN", -0.02), -0.20, 0.20))
FULL_GAME_PROB_SHIFT_MAX = float(np.clip(_env_float("NBA_MLB_FULL_GAME_PROB_SHIFT_MAX", 0.02), -0.20, 0.20))
FULL_GAME_PROB_SHIFT_STEP = float(np.clip(_env_float("NBA_MLB_FULL_GAME_PROB_SHIFT_STEP", 0.01), 0.001, 0.10))

FULL_GAME_META_GATE_ENABLED = int(np.clip(_env_float("NBA_MLB_META_GATE_ENABLED", 0.0), 0.0, 1.0))
FULL_GAME_META_GATE_MODEL_C = float(np.clip(_env_float("NBA_MLB_META_GATE_MODEL_C", 0.8), 0.1, 5.0))
FULL_GAME_META_GATE_MIN_CALIB_ROWS = int(max(50, _env_float("NBA_MLB_META_GATE_MIN_CALIB_ROWS", 180.0)))
FULL_GAME_META_GATE_MIN_BASE_ROWS = int(max(8, _env_float("NBA_MLB_META_GATE_MIN_BASE_ROWS", 25.0)))
FULL_GAME_META_GATE_THRESHOLD_MIN = float(np.clip(_env_float("NBA_MLB_META_GATE_THRESHOLD_MIN", 0.50), 0.30, 0.95))
FULL_GAME_META_GATE_THRESHOLD_MAX = float(np.clip(_env_float("NBA_MLB_META_GATE_THRESHOLD_MAX", 0.80), 0.35, 0.99))
FULL_GAME_META_GATE_THRESHOLD_STEP = float(np.clip(_env_float("NBA_MLB_META_GATE_THRESHOLD_STEP", 0.02), 0.005, 0.20))
FULL_GAME_META_GATE_MIN_KEEP_ROWS = int(max(5, _env_float("NBA_MLB_META_GATE_MIN_KEEP_ROWS", 12.0)))
FULL_GAME_META_GATE_COVERAGE_BONUS = float(np.clip(_env_float("NBA_MLB_META_GATE_COVERAGE_BONUS", 0.04), 0.0, 0.20))
FULL_GAME_META_GATE_RETENTION_TARGET = float(np.clip(_env_float("NBA_MLB_META_GATE_RETENTION_TARGET", 0.50), 0.0, 1.0))
FULL_GAME_META_GATE_RETENTION_PENALTY = float(np.clip(_env_float("NBA_MLB_META_GATE_RETENTION_PENALTY", 0.14), 0.0, 0.50))
FULL_GAME_META_GATE_MIN_ACC_GAIN = float(np.clip(_env_float("NBA_MLB_META_GATE_MIN_ACC_GAIN", 0.01), 0.0, 0.10))
FULL_GAME_META_GATE_MIN_COVERAGE_RETENTION = float(np.clip(_env_float("NBA_MLB_META_GATE_MIN_COVERAGE_RETENTION", 0.40), 0.0, 1.0))

# Optional threshold sweep controls for full_game.
FULL_GAME_THR_MIN = float(np.clip(_env_float("NBA_MLB_FULL_GAME_THR_MIN", 0.56), 0.30, 0.90))
FULL_GAME_THR_MAX = float(np.clip(_env_float("NBA_MLB_FULL_GAME_THR_MAX", 0.68), 0.35, 0.95))
FULL_GAME_THR_STEP = float(np.clip(_env_float("NBA_MLB_FULL_GAME_THR_STEP", 0.01), 0.001, 0.10))

FULL_GAME_XGB_WEIGHT_GRID = sorted(
    {
        float(np.clip(v, 0.0, 1.0))
        for v in _env_float_list("NBA_MLB_FULL_GAME_XGB_WEIGHT_GRID", [0.00, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00])
    }
)
if len(FULL_GAME_XGB_WEIGHT_GRID) == 0:
    FULL_GAME_XGB_WEIGHT_GRID = [0.00, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00]

_cal_mode_raw = _env_str("NBA_MLB_FULL_GAME_CALIBRATOR_MODE", "global_lr").lower()
if _cal_mode_raw not in {"auto", "global_lr", "regime_aware"}:
    _cal_mode_raw = "auto"
FULL_GAME_CALIBRATOR_MODE = _cal_mode_raw

_thr_obj_raw = _env_str("NBA_MLB_FULL_GAME_THRESHOLD_OBJECTIVE", "accuracy_cov").lower()
if _thr_obj_raw not in {"accuracy_cov", "roi"}:
    _thr_obj_raw = "accuracy_cov"
FULL_GAME_THRESHOLD_OBJECTIVE = _thr_obj_raw

FULL_GAME_ROI_MIN_EDGE = float(np.clip(_env_float("NBA_MLB_FULL_GAME_ROI_MIN_EDGE", 0.0), 0.0, 0.25))
FULL_GAME_ROI_MIN_ACCURACY = float(np.clip(_env_float("NBA_MLB_FULL_GAME_ROI_MIN_ACCURACY", 0.45), 0.0, 1.0))
FULL_GAME_ROI_MIN_PRICED_ROWS = int(max(5, _env_float("NBA_MLB_FULL_GAME_ROI_MIN_PRICED_ROWS", 8.0)))
FULL_GAME_ROI_SCORE_ROI_WEIGHT = float(np.clip(_env_float("NBA_MLB_FULL_GAME_ROI_SCORE_ROI_WEIGHT", 1.0), 0.0, 5.0))
FULL_GAME_ROI_SCORE_ACC_WEIGHT = float(np.clip(_env_float("NBA_MLB_FULL_GAME_ROI_SCORE_ACC_WEIGHT", 0.10), 0.0, 2.0))
FULL_GAME_ROI_SCORE_COV_WEIGHT = float(np.clip(_env_float("NBA_MLB_FULL_GAME_ROI_SCORE_COV_WEIGHT", 0.04), 0.0, 0.50))

FULL_GAME_VOL_NORM_ENABLED = int(np.clip(_env_float("NBA_MLB_FULL_GAME_VOL_NORM_ENABLED", 0.0), 0.0, 1.0))
FULL_GAME_VOL_NORM_ALPHA = float(np.clip(_env_float("NBA_MLB_FULL_GAME_VOL_NORM_ALPHA", 0.18), 0.0, 0.45))
FULL_GAME_VOL_NORM_THRESHOLD_BONUS = float(np.clip(_env_float("NBA_MLB_FULL_GAME_VOL_NORM_THRESHOLD_BONUS", 0.02), 0.0, 0.08))
FULL_GAME_VOL_NORM_HIGH_QUANTILE = float(np.clip(_env_float("NBA_MLB_FULL_GAME_VOL_NORM_HIGH_QUANTILE", 0.75), 0.50, 0.95))
FULL_GAME_VOL_NORM_MIN_ROWS = int(max(60, _env_float("NBA_MLB_FULL_GAME_VOL_NORM_MIN_ROWS", 160.0)))

_full_game_vol_decision_default = 1.0 if FULL_GAME_VOL_NORM_ENABLED > 0 else 0.0
FULL_GAME_VOL_DECISION_ENABLED = int(
    np.clip(
        _env_float("NBA_MLB_FULL_GAME_VOL_DECISION_ENABLED", _full_game_vol_decision_default),
        0.0,
        1.0,
    )
)
FULL_GAME_VOL_DECISION_CENTER = float(np.clip(_env_float("NBA_MLB_FULL_GAME_VOL_DECISION_CENTER", 0.50), 0.0, 1.0))
FULL_GAME_VOL_DECISION_MAX_SHIFT = float(np.clip(_env_float("NBA_MLB_FULL_GAME_VOL_DECISION_MAX_SHIFT", 0.06), 0.0, 0.20))
FULL_GAME_VOL_DECISION_BETA_PENALTY = float(np.clip(_env_float("NBA_MLB_FULL_GAME_VOL_DECISION_BETA_PENALTY", 0.002), 0.0, 0.05))
FULL_GAME_VOL_DECISION_BETA_GRID = sorted(
    {
        float(np.clip(v, -0.30, 0.30))
        for v in _env_float_list(
            "NBA_MLB_FULL_GAME_VOL_DECISION_BETA_GRID",
            [-0.12, -0.08, -0.05, -0.03, 0.00, 0.03, 0.05, 0.08, 0.12],
        )
    }
)
if len(FULL_GAME_VOL_DECISION_BETA_GRID) == 0:
    FULL_GAME_VOL_DECISION_BETA_GRID = [-0.12, -0.08, -0.05, -0.03, 0.00, 0.03, 0.05, 0.08, 0.12]
if not any(abs(v) <= 1e-12 for v in FULL_GAME_VOL_DECISION_BETA_GRID):
    FULL_GAME_VOL_DECISION_BETA_GRID.append(0.0)
FULL_GAME_VOL_DECISION_BETA_GRID = sorted(set(float(round(v, 6)) for v in FULL_GAME_VOL_DECISION_BETA_GRID))

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

FULL_GAME_VOLATILITY_COMPONENT_WEIGHTS = {
    "diff_runs_scored_std_L10": 1.00,
    "diff_runs_allowed_std_L10": 1.00,
    "diff_r1_scored_std_L10": 0.70,
    "diff_r1_allowed_std_L10": 0.70,
    "diff_pitcher_blowup_rate_L10": 1.20,
    "diff_pitcher_era_trend": 0.90,
    "diff_pitcher_whip_trend": 0.90,
    "umpire_volatility_risk": 0.70,
}

def build_full_game_stacking_feature_frame(
    xgb_probs: np.ndarray,
    lgbm_probs: np.ndarray,
    rf_probs: np.ndarray | None = None,
) -> pd.DataFrame:
    xgb = np.asarray(xgb_probs, dtype=float)
    lgbm = np.asarray(lgbm_probs, dtype=float)
    if rf_probs is None or len(rf_probs) != len(xgb):
        rf = np.full(len(xgb), 0.5, dtype=float)
    else:
        rf = np.asarray(rf_probs, dtype=float)

    stacked = np.vstack([xgb, lgbm, rf])
    out = pd.DataFrame(
        {
            "xgb_prob": xgb,
            "lgbm_prob": lgbm,
            "rf_prob": rf,
            "mean_prob": np.nanmean(stacked, axis=0),
            "std_prob": np.nanstd(stacked, axis=0),
            "max_prob": np.nanmax(stacked, axis=0),
            "min_prob": np.nanmin(stacked, axis=0),
            "xgb_lgbm_gap": np.abs(xgb - lgbm),
            "xgb_rf_gap": np.abs(xgb - rf),
            "lgbm_rf_gap": np.abs(lgbm - rf),
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.5)
    return out


class IdentityProbabilityCalibrator:
    def predict_proba(self, probs_2d: np.ndarray) -> np.ndarray:
        p = np.asarray(probs_2d, dtype=float).reshape(-1)
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - p, p])


class WeightedEnsembleModel:
    def __init__(
        self,
        xgb_model,
        lgbm_model,
        xgb_weight: float = 0.5,
        lgbm_weight: float = 0.5,
        rf_model=None,
        rf_weight: float = 0.0,
        stacking_model: LogisticRegression | None = None,
    ):
        self.xgb_model = xgb_model
        self.lgbm_model = lgbm_model
        self.rf_model = rf_model
        self.xgb_weight = float(xgb_weight)
        self.lgbm_weight = float(lgbm_weight)
        self.rf_weight = float(rf_weight)
        self.stacking_model = stacking_model

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        xgb_probs = self.xgb_model.predict_proba(X)[:, 1]
        lgbm_probs = self.lgbm_model.predict_proba(X)[:, 1]
        rf_probs = self.rf_model.predict_proba(X)[:, 1] if self.rf_model is not None else None

        if self.stacking_model is not None:
            X_meta = build_full_game_stacking_feature_frame(xgb_probs, lgbm_probs, rf_probs)
            probs = self.stacking_model.predict_proba(X_meta)[:, 1]
        else:
            total_weight = self.xgb_weight + self.lgbm_weight + (self.rf_weight if rf_probs is not None else 0.0)
            if total_weight <= 0:
                total_weight = 1.0
            weighted_sum = (self.xgb_weight * xgb_probs) + (self.lgbm_weight * lgbm_probs)
            if rf_probs is not None and self.rf_weight > 0:
                weighted_sum += self.rf_weight * rf_probs
            probs = weighted_sum / total_weight

        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - probs, probs])


class RegimeAwareLogisticCalibrator:
    def __init__(self, global_calibrator: LogisticRegression, regime_calibrators: Dict[str, LogisticRegression]):
        self.global_calibrator = global_calibrator
        self.regime_calibrators = regime_calibrators

    def predict_proba(self, probs_2d: np.ndarray, regimes: np.ndarray | None = None) -> np.ndarray:
        raw = np.asarray(probs_2d, dtype=float).reshape(-1, 1)
        out = self.global_calibrator.predict_proba(raw)[:, 1]

        if regimes is not None and len(regimes) == len(out):
            regime_arr = np.asarray(regimes, dtype=str)
            for regime_name, calibrator in self.regime_calibrators.items():
                mask = regime_arr == regime_name
                if np.any(mask):
                    out[mask] = calibrator.predict_proba(raw[mask])[:, 1]

        out = np.clip(out, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - out, out])


def build_calibration_regimes(df_context: pd.DataFrame, market_key: str) -> np.ndarray:
    if not isinstance(df_context, pd.DataFrame) or df_context.empty:
        return np.array([], dtype=str)

    if market_key != "full_game":
        return np.array(["global"] * len(df_context), dtype=str)

    has_pitchers = np.ones(len(df_context), dtype=bool)
    if "both_pitchers_available" in df_context.columns:
        has_pitchers = pd.to_numeric(df_context["both_pitchers_available"], errors="coerce").fillna(0).to_numpy(dtype=float) >= 1.0

    quality_gap = np.zeros(len(df_context), dtype=float)
    if "diff_pitcher_recent_quality_score" in df_context.columns:
        quality_gap = np.abs(pd.to_numeric(df_context["diff_pitcher_recent_quality_score"], errors="coerce").fillna(0).to_numpy(dtype=float))

    volatility_components: List[np.ndarray] = []
    for col in [
        "diff_runs_scored_std_L10",
        "diff_runs_allowed_std_L10",
        "diff_pitcher_blowup_rate_L10",
        "diff_pitcher_era_trend",
        "diff_pitcher_whip_trend",
        "umpire_volatility_risk",
    ]:
        arr = _safe_numeric_array(df_context, col)
        if arr is None:
            continue
        volatility_components.append(np.abs(np.nan_to_num(arr, nan=0.0)))

    volatility_score = np.zeros(len(df_context), dtype=float)
    if volatility_components:
        volatility_score = np.nanmean(np.vstack(volatility_components), axis=0)
        if np.any(np.isfinite(volatility_score)):
            vol_q10 = float(np.nanquantile(volatility_score, 0.10))
            vol_q90 = float(np.nanquantile(volatility_score, 0.90))
            vol_span = max(vol_q90 - vol_q10, 1e-6)
            volatility_score = np.clip((volatility_score - vol_q10) / vol_span, 0.0, 1.0)

    regime = np.full(len(df_context), "fg_normal", dtype=object)
    regime[~has_pitchers] = "fg_missing_pitchers"
    if np.any(np.isfinite(volatility_score)):
        vol_thr = float(np.nanquantile(volatility_score, 0.75))
        regime[has_pitchers & (volatility_score >= vol_thr)] = "fg_high_volatility"
    regime[has_pitchers & (quality_gap >= 2.5)] = "fg_high_quality_gap"
    return regime.astype(str)


def fit_regime_aware_calibrator(
    raw_probs: np.ndarray,
    y_calib: pd.Series,
    calib_context: pd.DataFrame,
    market_key: str,
) -> Tuple[object, np.ndarray, int]:
    y_arr = y_calib.to_numpy(dtype=int)
    global_cal = LogisticRegression(C=1.0, solver="lbfgs")
    global_cal.fit(raw_probs.reshape(-1, 1), y_arr)
    global_probs = global_cal.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

    regimes = build_calibration_regimes(calib_context, market_key)
    if market_key != "full_game" or len(regimes) != len(y_arr):
        return global_cal, np.clip(global_probs, 1e-6, 1.0 - 1e-6), 0

    regime_models: Dict[str, LogisticRegression] = {}
    min_rows_per_regime = 45

    for regime_name in sorted(set(regimes.tolist())):
        mask = regimes == regime_name
        if int(mask.sum()) < min_rows_per_regime:
            continue
        y_reg = y_arr[mask]
        if len(np.unique(y_reg)) < 2:
            continue
        reg_cal = LogisticRegression(C=1.0, solver="lbfgs")
        reg_cal.fit(raw_probs[mask].reshape(-1, 1), y_reg)
        regime_models[regime_name] = reg_cal

    if not regime_models:
        return global_cal, np.clip(global_probs, 1e-6, 1.0 - 1e-6), 0

    calibrator = RegimeAwareLogisticCalibrator(global_calibrator=global_cal, regime_calibrators=regime_models)
    regime_probs = calibrator.predict_proba(raw_probs.reshape(-1, 1), regimes=regimes)[:, 1]
    return calibrator, np.clip(regime_probs, 1e-6, 1.0 - 1e-6), int(len(regime_models))


def predict_calibrated_probs(
    calibrator: object,
    raw_probs: np.ndarray,
    df_context: pd.DataFrame,
    market_key: str,
) -> np.ndarray:
    raw_2d = np.asarray(raw_probs, dtype=float).reshape(-1, 1)
    if isinstance(calibrator, RegimeAwareLogisticCalibrator):
        regimes = build_calibration_regimes(df_context, market_key)
        return calibrator.predict_proba(raw_2d, regimes=regimes)[:, 1]
    return calibrator.predict_proba(raw_2d)[:, 1]


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
    if market_key == "full_game":
        base["min_threshold"] = float(min(FULL_GAME_THR_MIN, FULL_GAME_THR_MAX))
        base["max_threshold"] = float(max(FULL_GAME_THR_MIN, FULL_GAME_THR_MAX))
        base["threshold_step"] = float(FULL_GAME_THR_STEP)
    elif market_key == "totals":
        base["min_edge"] = float(np.clip(_env_float("NBA_MLB_TOTALS_MIN_EDGE", float(base.get("min_edge", 0.75))), 0.0, 10.0))
        base["publish_edge"] = float(np.clip(_env_float("NBA_MLB_TOTALS_PUBLISH_EDGE", float(base.get("publish_edge", 1.50))), 0.0, 10.0))
    elif market_key == "total_hits_event":
        base["min_edge"] = float(
            np.clip(_env_float("NBA_MLB_TOTAL_HITS_EVENT_MIN_EDGE", float(base.get("min_edge", 0.80))), 0.0, 10.0)
        )
        base["publish_edge"] = float(
            np.clip(_env_float("NBA_MLB_TOTAL_HITS_EVENT_PUBLISH_EDGE", float(base.get("publish_edge", 1.60))), 0.0, 10.0)
        )
    elif market_key == "run_line":
        base["min_edge"] = float(np.clip(_env_float("NBA_MLB_RUN_LINE_MIN_EDGE", float(base.get("min_edge", 0.60))), 0.0, 10.0))
        base["publish_edge"] = float(np.clip(_env_float("NBA_MLB_RUN_LINE_PUBLISH_EDGE", float(base.get("publish_edge", 1.60))), 0.0, 10.0))
    else:
        base["threshold_step"] = float(base.get("threshold_step", 0.01))
    return base


def shrink_probs_toward_half(probs: np.ndarray, shrink: float) -> np.ndarray:
    shrink = float(np.clip(shrink, 0.0, 0.49))
    return np.clip(0.5 + ((probs - 0.5) * (1.0 - shrink)), 1e-6, 1.0 - 1e-6)


def apply_bayesian_probability_blend(
    probs: np.ndarray,
    prior: float,
    strength: float,
    sample_size: int,
) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)
    s = max(0.0, float(strength))
    n = max(1.0, float(sample_size))
    if s <= 0.0:
        return p
    prior_p = float(np.clip(prior, 0.05, 0.95))
    blend = s / (n + s)
    return np.clip(((1.0 - blend) * p) + (blend * prior_p), 1e-6, 1.0 - 1e-6)


def apply_probability_shift(probs: np.ndarray, shift: float) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    return np.clip(p + float(shift), 1e-6, 1.0 - 1e-6)


def full_game_global_score_from_probs(
    probs: np.ndarray,
    y_true: np.ndarray,
    brier_weight: float = FULL_GAME_GLOBAL_BRIER_WEIGHT,
) -> Tuple[float, float, float]:
    p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)
    y = np.asarray(y_true, dtype=int)
    pred = (p >= 0.5).astype(int)
    acc = float((pred == y).mean())
    brier = float(np.mean((p - y) ** 2))
    score = float(acc - (float(brier_weight) * brier))
    return acc, brier, score


def choose_best_bayesian_blend(
    probs: np.ndarray,
    y_true: np.ndarray,
    prior: float,
    sample_size: int,
    strength_grid: List[float],
) -> Tuple[float, np.ndarray, float]:
    y = np.asarray(y_true, dtype=int)
    base = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)

    best_strength = 0.0
    best_probs = base
    best_score = -1e9

    for strength in strength_grid:
        cand = apply_bayesian_probability_blend(
            probs=base,
            prior=prior,
            strength=float(strength),
            sample_size=sample_size,
        )
        _, _, score = full_game_global_score_from_probs(cand, y, brier_weight=FULL_GAME_GLOBAL_BRIER_WEIGHT)

        if score > best_score:
            best_strength = float(strength)
            best_probs = cand
            best_score = score

    return best_strength, best_probs, best_score


def choose_best_probability_shift(
    probs: np.ndarray,
    y_true: np.ndarray,
    shift_grid: List[float],
) -> Tuple[float, np.ndarray, float]:
    y = np.asarray(y_true, dtype=int)
    base = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)

    best_pred = (base >= 0.5).astype(int)
    best_acc = float((best_pred == y).mean())
    best_shift = 0.0
    best_probs = base

    best_brier = float(np.mean((base - y) ** 2))
    best_score = float(best_acc - (FULL_GAME_GLOBAL_BRIER_WEIGHT * best_brier))

    for shift in shift_grid:
        cand = apply_probability_shift(base, float(shift))
        pred = (cand >= 0.5).astype(int)
        acc = float((pred == y).mean())
        brier = float(np.mean((cand - y) ** 2))
        score = float(acc - (FULL_GAME_GLOBAL_BRIER_WEIGHT * brier))

        improves_acc = acc > (best_acc + 1e-9)
        ties_acc_improves_score = abs(acc - best_acc) <= 1e-9 and score > (best_score + 1e-9)
        if improves_acc or ties_acc_improves_score:
            best_shift = float(shift)
            best_probs = cand
            best_acc = acc
            best_score = score

    return best_shift, best_probs, best_score


def choose_best_decision_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
    threshold_grid: List[float],
    center: float = 0.5,
    drift_penalty: float = 0.01,
) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probs, dtype=float)

    best_t = float(center)
    best_acc = float(((p >= best_t).astype(int) == y).mean())
    best_score = best_acc

    for thr in threshold_grid:
        t = float(thr)
        pred = (p >= t).astype(int)
        acc = float((pred == y).mean())
        score = float(acc - (abs(t - float(center)) * float(drift_penalty)))
        if score > (best_score + 1e-9):
            best_t = t
            best_acc = acc
            best_score = score

    return float(best_t)


def build_dynamic_decision_threshold_from_vol_scores(
    vol_scores: np.ndarray,
    beta: float,
    center: float = FULL_GAME_VOL_DECISION_CENTER,
    max_shift: float = FULL_GAME_VOL_DECISION_MAX_SHIFT,
) -> np.ndarray:
    vol = np.asarray(vol_scores, dtype=float)
    if len(vol) == 0:
        return np.zeros(0, dtype=float)

    shift = np.clip(
        float(beta) * (vol - float(center)),
        -float(max_shift),
        float(max_shift),
    )
    return np.clip(0.5 + shift, 0.35, 0.65)


def choose_best_volatility_decision_beta(
    probs: np.ndarray,
    y_true: np.ndarray,
    vol_scores: np.ndarray,
    beta_grid: List[float],
    center: float = FULL_GAME_VOL_DECISION_CENTER,
    max_shift: float = FULL_GAME_VOL_DECISION_MAX_SHIFT,
    beta_penalty: float = FULL_GAME_VOL_DECISION_BETA_PENALTY,
) -> Tuple[float, np.ndarray, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)
    vol = np.asarray(vol_scores, dtype=float)

    base_threshold = np.full(len(p), 0.5, dtype=float)
    if len(p) == 0:
        return 0.0, base_threshold, 0.0
    if len(vol) != len(p):
        base_acc = float(((p >= 0.5).astype(int) == y).mean())
        return 0.0, base_threshold, base_acc

    best_beta = 0.0
    best_threshold = base_threshold
    best_pred = (p >= 0.5).astype(int)
    best_acc = float((best_pred == y).mean())
    best_score = float(best_acc)

    for beta in beta_grid:
        thr = build_dynamic_decision_threshold_from_vol_scores(
            vol_scores=vol,
            beta=float(beta),
            center=float(center),
            max_shift=float(max_shift),
        )
        pred = (p >= thr).astype(int)
        acc = float((pred == y).mean())
        score = float(acc - (abs(float(beta)) * float(beta_penalty)))

        if score > (best_score + 1e-9):
            best_beta = float(beta)
            best_threshold = thr
            best_pred = pred
            best_acc = acc
            best_score = score

    return float(best_beta), best_threshold, float(best_acc)


def _full_game_context_signed_feature(
    df_context: pd.DataFrame,
    col: str,
    scale: float,
    invert: bool = False,
    default: float = 0.0,
) -> np.ndarray:
    arr = _safe_numeric_array(df_context, col)
    if arr is None:
        arr = np.full(len(df_context) if isinstance(df_context, pd.DataFrame) else 0, float(default), dtype=float)
    else:
        arr = np.nan_to_num(arr, nan=float(default))
    if invert:
        arr = -arr
    denom = max(float(scale), 1e-6)
    return np.clip(arr / denom, -1.0, 1.0)


def compute_full_game_reliability_meta(
    probs: np.ndarray,
    df_context: pd.DataFrame,
    vol_scores: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)
    n = len(p)
    if n == 0:
        return {
            "reliability_score": np.zeros(0, dtype=float),
            "unreliability": np.zeros(0, dtype=float),
            "side_hint": np.zeros(0, dtype=float),
            "conflict_flag": np.zeros(0, dtype=float),
            "context_strength": np.zeros(0, dtype=float),
            "volatility_proxy": np.zeros(0, dtype=float),
        }

    if not isinstance(df_context, pd.DataFrame) or len(df_context) != n:
        edge = np.clip(np.abs(p - 0.5) * 2.0, 0.0, 1.0)
        reliability = np.clip(edge, 0.0, 1.0)
        return {
            "reliability_score": reliability,
            "unreliability": 1.0 - reliability,
            "side_hint": np.zeros(n, dtype=float),
            "conflict_flag": np.zeros(n, dtype=float),
            "context_strength": np.zeros(n, dtype=float),
            "volatility_proxy": np.zeros(n, dtype=float),
        }

    quality_signal = _full_game_context_signed_feature(df_context, "diff_pitcher_recent_quality_score", scale=4.0)
    quality_start_signal = _full_game_context_signed_feature(df_context, "diff_pitcher_quality_start_rate_L10", scale=0.30)
    blowup_signal = _full_game_context_signed_feature(df_context, "diff_pitcher_blowup_rate_L10", scale=0.35, invert=True)
    era_trend_signal = _full_game_context_signed_feature(df_context, "diff_pitcher_era_trend", scale=0.40, invert=True)
    whip_trend_signal = _full_game_context_signed_feature(df_context, "diff_pitcher_whip_trend", scale=0.08, invert=True)
    form_signal = _full_game_context_signed_feature(df_context, "diff_form_power", scale=3.0)
    elo_signal = _full_game_context_signed_feature(df_context, "diff_elo", scale=120.0)

    home_fav_raw = _safe_numeric_array(df_context, "home_is_favorite")
    if home_fav_raw is None:
        favorite_signal = np.zeros(n, dtype=float)
    else:
        favorite_signal = np.where(np.nan_to_num(home_fav_raw, nan=0.0) >= 0.5, 1.0, -1.0)

    signal_weights = np.array([1.20, 1.00, 1.00, 0.80, 0.80, 0.70, 0.55, 0.35], dtype=float)
    signal_matrix = np.vstack(
        [
            quality_signal,
            quality_start_signal,
            blowup_signal,
            era_trend_signal,
            whip_trend_signal,
            form_signal,
            elo_signal,
            favorite_signal,
        ]
    )
    weighted_signal = np.sum(signal_matrix * signal_weights[:, None], axis=0)
    side_hint = np.clip(weighted_signal / max(float(signal_weights.sum()), 1e-6), -1.0, 1.0)
    context_strength = np.abs(side_hint)

    model_side = np.where(p >= 0.5, 1.0, -1.0)
    conflict_flag = ((np.sign(side_hint) * model_side) < 0).astype(float)
    conflict_flag = conflict_flag * (context_strength >= 0.20).astype(float)

    edge = np.clip(np.abs(p - 0.5) * 2.0, 0.0, 1.0)

    if isinstance(vol_scores, np.ndarray) and len(vol_scores) == n:
        volatility_proxy = np.clip(np.nan_to_num(vol_scores, nan=0.0), 0.0, 1.0)
    else:
        volatility_proxy = np.clip(
            0.35 * np.abs(_full_game_context_signed_feature(df_context, "diff_runs_scored_std_L10", scale=2.5))
            + 0.35 * np.abs(_full_game_context_signed_feature(df_context, "diff_runs_allowed_std_L10", scale=2.5))
            + 0.30 * np.abs(_full_game_context_signed_feature(df_context, "diff_pitcher_blowup_rate_L10", scale=0.35)),
            0.0,
            1.0,
        )

    missing_pitchers = get_missing_pitcher_flag(df_context).astype(float)

    base_reliability = np.clip((0.55 * edge) + (0.45 * context_strength), 0.0, 1.0)
    risk = 1.0 - base_reliability
    risk += 0.22 * conflict_flag * context_strength
    risk += 0.18 * volatility_proxy
    risk += 0.24 * missing_pitchers
    risk = np.clip(risk, 0.0, 1.0)
    reliability = np.clip(1.0 - risk, 0.0, 1.0)

    return {
        "reliability_score": reliability,
        "unreliability": 1.0 - reliability,
        "side_hint": side_hint,
        "conflict_flag": conflict_flag,
        "context_strength": context_strength,
        "volatility_proxy": volatility_proxy,
    }


def apply_full_game_reliability_adjustment(
    probs: np.ndarray,
    df_context: pd.DataFrame,
    vol_scores: np.ndarray | None = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)
    n = len(p)
    meta = {
        "reliability_score": np.ones(n, dtype=float),
        "unreliability": np.zeros(n, dtype=float),
        "side_hint": np.zeros(n, dtype=float),
        "conflict_flag": np.zeros(n, dtype=float),
        "context_strength": np.zeros(n, dtype=float),
        "volatility_proxy": np.zeros(n, dtype=float),
        "shrink_factor": np.ones(n, dtype=float),
        "side_shift": np.zeros(n, dtype=float),
    }
    if n == 0 or FULL_GAME_RELIABILITY_ENABLED <= 0:
        return p, meta

    rel_meta = compute_full_game_reliability_meta(
        probs=p,
        df_context=df_context,
        vol_scores=vol_scores,
    )
    unreliability = np.clip(rel_meta.get("unreliability", np.zeros(n, dtype=float)), 0.0, 1.0)
    side_hint = np.clip(rel_meta.get("side_hint", np.zeros(n, dtype=float)), -1.0, 1.0)
    conflict_flag = np.clip(rel_meta.get("conflict_flag", np.zeros(n, dtype=float)), 0.0, 1.0)

    shrink_factor = np.clip(
        1.0 - (FULL_GAME_RELIABILITY_SHRINK_ALPHA * unreliability),
        0.35,
        1.0,
    )
    adjusted = 0.5 + ((p - 0.5) * shrink_factor)

    side_shift = FULL_GAME_RELIABILITY_SIDE_SHIFT * unreliability * side_hint
    side_shift += FULL_GAME_RELIABILITY_CONFLICT_SHIFT * conflict_flag * np.sign(side_hint)
    adjusted = np.clip(adjusted + side_shift, 1e-6, 1.0 - 1e-6)

    meta = {
        **rel_meta,
        "shrink_factor": shrink_factor,
        "side_shift": side_shift,
    }
    return adjusted, meta


def _odds_values_to_decimal(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    out = np.full(len(raw), np.nan, dtype=float)

    finite = np.isfinite(raw)
    if not finite.any():
        return out

    # Heuristic: American odds are generally far from 0; decimal odds are usually between 1 and 20.
    american = finite & (np.abs(raw) >= 20.0)
    pos = american & (raw > 0)
    neg = american & (raw < 0)
    out[pos] = 1.0 + (raw[pos] / 100.0)
    out[neg] = 1.0 + (100.0 / np.abs(raw[neg]))

    decimal_like = finite & (~american) & (raw > 1.0)
    out[decimal_like] = raw[decimal_like]

    out[(~np.isfinite(out)) | (out <= 1.0)] = np.nan
    return out


def resolve_full_game_pick_odds_and_edge(
    probs_calibrated: np.ndarray,
    df_context: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probs = np.clip(np.asarray(probs_calibrated, dtype=float), 1e-6, 1.0 - 1e-6)
    n = len(probs)

    pick_decimal_odds = np.full(n, np.nan, dtype=float)
    pick_odds_source = np.array(["none"] * n, dtype=object)

    if not isinstance(df_context, pd.DataFrame) or len(df_context) != n:
        pick_prob = np.maximum(probs, 1.0 - probs)
        pick_implied_prob = np.full(n, np.nan, dtype=float)
        pick_ev_edge = np.full(n, np.nan, dtype=float)
        return pick_decimal_odds, pick_implied_prob, pick_ev_edge, pick_odds_source

    pred_side = (probs >= 0.5).astype(int)
    home_mask = pred_side == 1
    away_mask = ~home_mask

    def _fill_side(mask: np.ndarray, cols: List[str]) -> None:
        for col in cols:
            if col not in df_context.columns:
                continue
            raw = pd.to_numeric(df_context[col], errors="coerce").to_numpy(dtype=float)
            dec = _odds_values_to_decimal(raw)
            fill = mask & (~np.isfinite(pick_decimal_odds)) & np.isfinite(dec)
            if fill.any():
                pick_decimal_odds[fill] = dec[fill]
                pick_odds_source[fill] = col

    _fill_side(home_mask, ["current_home_moneyline", "home_moneyline_odds"])
    _fill_side(away_mask, ["current_away_moneyline", "away_moneyline_odds"])

    # Fallback for feeds exposing only one generic moneyline odds column.
    for col in ["closing_moneyline_odds", "moneyline_odds"]:
        if col not in df_context.columns:
            continue
        raw = pd.to_numeric(df_context[col], errors="coerce").to_numpy(dtype=float)
        dec = _odds_values_to_decimal(raw)
        fill = (~np.isfinite(pick_decimal_odds)) & np.isfinite(dec)
        if fill.any():
            pick_decimal_odds[fill] = dec[fill]
            pick_odds_source[fill] = col

    pick_prob = np.where(pred_side == 1, probs, 1.0 - probs)
    pick_implied_prob = np.where(
        np.isfinite(pick_decimal_odds) & (pick_decimal_odds > 1.0),
        1.0 / pick_decimal_odds,
        np.nan,
    )
    pick_ev_edge = pick_prob - pick_implied_prob
    return pick_decimal_odds, pick_implied_prob, pick_ev_edge, pick_odds_source


def choose_full_game_roi_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    df_context: pd.DataFrame,
    *,
    min_threshold: float,
    max_threshold: float,
    step: float,
    min_coverage: float,
    target_coverage: float | None,
    coverage_tolerance: float,
    min_published_rows: int,
) -> Dict[str, float] | None:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)
    if len(y) == 0 or len(p) == 0 or len(y) != len(p):
        return None

    pred_side = (p >= 0.5).astype(int)
    conf = np.maximum(p, 1.0 - p)
    pick_decimal_odds, _, pick_ev_edge, _ = resolve_full_game_pick_odds_and_edge(
        probs_calibrated=p,
        df_context=df_context,
    )
    if not np.isfinite(pick_decimal_odds).any():
        return None

    candidates = np.arange(float(min_threshold), float(max_threshold) + 1e-9, float(step))

    best: Dict[str, float] | None = None
    best_score = -1e9

    required_priced_rows = int(max(5, FULL_GAME_ROI_MIN_PRICED_ROWS))
    pass_requirements = [required_priced_rows]
    relaxed_required = int(max(5, required_priced_rows // 2))
    if relaxed_required < required_priced_rows:
        pass_requirements.append(relaxed_required)

    for required_rows in pass_requirements:
        for thr in candidates:
            take = conf >= float(thr)
            if FULL_GAME_ROI_MIN_EDGE > 0:
                take = take & np.isfinite(pick_ev_edge) & (pick_ev_edge >= FULL_GAME_ROI_MIN_EDGE)

            published_rows = int(take.sum())
            coverage = float(take.mean()) if len(take) else 0.0
            if coverage < float(min_coverage) or published_rows < int(max(1, min_published_rows)):
                continue

            acc = float((pred_side[take] == y[take]).mean()) if published_rows else 0.0
            if acc < float(FULL_GAME_ROI_MIN_ACCURACY):
                continue

            positive_rate = float(pred_side[take].mean()) if published_rows else 0.0
            priced_take = take & np.isfinite(pick_decimal_odds)
            priced_rows = int(priced_take.sum())
            if priced_rows < int(required_rows):
                continue

            hits = pred_side[priced_take] == y[priced_take]
            returns = np.where(hits, pick_decimal_odds[priced_take] - 1.0, -1.0)
            roi_per_bet = float(np.mean(returns)) if len(returns) else 0.0
            total_return_units = float(np.sum(returns)) if len(returns) else 0.0
            priced_coverage = float(priced_rows / len(y)) if len(y) else 0.0
            mean_ev_edge = float(np.nanmean(pick_ev_edge[priced_take])) if priced_rows else 0.0

            target_penalty = 0.0
            if target_coverage is not None and float(target_coverage) > 0.0:
                target_cov = float(target_coverage)
                cov_shortfall = max(0.0, target_cov - coverage)
                cov_excess = max(0.0, coverage - (target_cov + float(max(0.0, coverage_tolerance))))
                target_penalty = (cov_shortfall * 0.25) + (cov_excess * 0.20)

            score = (
                (float(FULL_GAME_ROI_SCORE_ROI_WEIGHT) * roi_per_bet)
                + (float(FULL_GAME_ROI_SCORE_ACC_WEIGHT) * acc)
                + (float(FULL_GAME_ROI_SCORE_COV_WEIGHT) * coverage)
                - float(target_penalty)
            )

            if score > (best_score + 1e-9):
                best_score = float(score)
                best = {
                    "threshold": float(round(float(thr), 2)),
                    "score": float(score),
                    "accuracy": float(acc),
                    "coverage": float(coverage),
                    "positive_rate": float(positive_rate),
                    "published_rows": int(published_rows),
                    "target_penalty": float(target_penalty),
                    "threshold_objective": "roi",
                    "roi_per_bet": float(roi_per_bet),
                    "total_return_units": float(total_return_units),
                    "priced_rows": int(priced_rows),
                    "priced_coverage": float(priced_coverage),
                    "mean_ev_edge": float(mean_ev_edge),
                    "roi_min_edge": float(FULL_GAME_ROI_MIN_EDGE),
                    "roi_min_accuracy": float(FULL_GAME_ROI_MIN_ACCURACY),
                }

        if best is not None:
            break

    return best


def _normalize_game_id_series(values: pd.Series) -> pd.Series:
    s = values.astype(str).str.strip()
    numeric = pd.to_numeric(s, errors="coerce")
    mask = np.isfinite(numeric)
    if mask.any():
        s = s.copy()
        s.loc[mask] = numeric.loc[mask].astype(np.int64).astype(str)
    return s


def load_full_game_odds_lookup() -> pd.DataFrame:
    global _FULL_GAME_ODDS_LOOKUP_CACHE

    if _FULL_GAME_ODDS_LOOKUP_CACHE is not None:
        return _FULL_GAME_ODDS_LOOKUP_CACHE

    if not RAW_ADVANCED_HISTORY_PATH.exists():
        _FULL_GAME_ODDS_LOOKUP_CACHE = pd.DataFrame()
        return _FULL_GAME_ODDS_LOOKUP_CACHE

    use_cols = [
        "game_id",
        "date",
        "home_team",
        "away_team",
        "home_moneyline_odds",
        "away_moneyline_odds",
        "closing_moneyline_odds",
    ]
    try:
        raw = pd.read_csv(RAW_ADVANCED_HISTORY_PATH, usecols=use_cols, low_memory=False)
    except Exception:
        _FULL_GAME_ODDS_LOOKUP_CACHE = pd.DataFrame()
        return _FULL_GAME_ODDS_LOOKUP_CACHE

    if raw.empty:
        _FULL_GAME_ODDS_LOOKUP_CACHE = pd.DataFrame()
        return _FULL_GAME_ODDS_LOOKUP_CACHE

    raw = raw.copy()
    raw["game_id"] = _normalize_game_id_series(raw["game_id"])
    raw["date"] = raw["date"].astype(str).str.strip()
    raw["home_team"] = raw["home_team"].astype(str).str.strip().str.upper()
    raw["away_team"] = raw["away_team"].astype(str).str.strip().str.upper()
    for col in ["home_moneyline_odds", "away_moneyline_odds", "closing_moneyline_odds"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.sort_values(["date", "game_id"]).drop_duplicates(subset=["game_id"], keep="last")
    _FULL_GAME_ODDS_LOOKUP_CACHE = raw
    return _FULL_GAME_ODDS_LOOKUP_CACHE


def enrich_full_game_context_with_odds(df_context: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df_context, pd.DataFrame) or df_context.empty:
        return df_context

    odds_lookup = load_full_game_odds_lookup()
    if odds_lookup.empty:
        return df_context

    out = df_context.copy()
    out["game_id"] = _normalize_game_id_series(out["game_id"]) if "game_id" in out.columns else ""
    out["date"] = out["date"].astype(str).str.strip() if "date" in out.columns else ""
    out["home_team"] = out["home_team"].astype(str).str.strip().str.upper() if "home_team" in out.columns else ""
    out["away_team"] = out["away_team"].astype(str).str.strip().str.upper() if "away_team" in out.columns else ""

    for col in ["home_moneyline_odds", "away_moneyline_odds", "closing_moneyline_odds"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    gid_cols = ["game_id", "home_moneyline_odds", "away_moneyline_odds", "closing_moneyline_odds"]
    gid_lookup = odds_lookup[gid_cols].dropna(subset=["game_id"]).copy()
    gid_lookup = gid_lookup.drop_duplicates(subset=["game_id"], keep="last")
    out = out.merge(gid_lookup, on="game_id", how="left", suffixes=("", "_gid"))
    for col in ["home_moneyline_odds", "away_moneyline_odds", "closing_moneyline_odds"]:
        src = f"{col}_gid"
        if src not in out.columns:
            continue
        fill_mask = (~np.isfinite(out[col])) | (out[col] == 0)
        src_valid = np.isfinite(out[src]) & (out[src] != 0)
        out.loc[fill_mask & src_valid, col] = out.loc[fill_mask & src_valid, src]
        out = out.drop(columns=[src])

    team_cols = ["date", "home_team", "away_team", "home_moneyline_odds", "away_moneyline_odds", "closing_moneyline_odds"]
    team_lookup = odds_lookup[team_cols].drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    out = out.merge(team_lookup, on=["date", "home_team", "away_team"], how="left", suffixes=("", "_team"))
    for col in ["home_moneyline_odds", "away_moneyline_odds", "closing_moneyline_odds"]:
        src = f"{col}_team"
        if src not in out.columns:
            continue
        fill_mask = (~np.isfinite(out[col])) | (out[col] == 0)
        src_valid = np.isfinite(out[src]) & (out[src] != 0)
        out.loc[fill_mask & src_valid, col] = out.loc[fill_mask & src_valid, src]
        out = out.drop(columns=[src])

    return out


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


class FullGameVolatilityNormalizer:
    def __init__(
        self,
        component_stats: Dict[str, Tuple[float, float]],
        component_weights: Dict[str, float],
        high_threshold: float,
    ):
        self.component_stats = component_stats
        self.component_weights = component_weights
        self.high_threshold = float(np.clip(high_threshold, 0.0, 1.0))

    def score(self, df_context: pd.DataFrame) -> np.ndarray:
        if not isinstance(df_context, pd.DataFrame) or df_context.empty:
            return np.zeros(len(df_context) if isinstance(df_context, pd.DataFrame) else 0, dtype=float)

        out = np.zeros(len(df_context), dtype=float)
        total_weight = 0.0

        for col, weight in self.component_weights.items():
            if col not in self.component_stats:
                continue
            arr = _safe_numeric_array(df_context, col)
            if arr is None:
                continue
            arr = np.abs(np.nan_to_num(arr, nan=0.0))
            lo, hi = self.component_stats[col]
            span = max(float(hi - lo), 1e-6)
            norm = np.clip((arr - float(lo)) / span, 0.0, 1.0)
            out += float(weight) * norm
            total_weight += float(weight)

        # Missing pitchers is a direct volatility source in MLB.
        missing_pitcher_flag = get_missing_pitcher_flag(df_context).astype(float)
        out += 1.25 * missing_pitcher_flag
        total_weight += 1.25

        if total_weight <= 0:
            return np.zeros(len(df_context), dtype=float)
        return np.clip(out / total_weight, 0.0, 1.0)

    def apply_to_probs(
        self,
        probs: np.ndarray,
        df_context: pd.DataFrame,
        alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)
        if len(p) == 0:
            return p, np.zeros(0, dtype=float)

        vol_score = self.score(df_context)
        if len(vol_score) != len(p):
            vol_score = np.zeros(len(p), dtype=float)

        shrink = np.clip(1.0 - (float(alpha) * vol_score), 0.35, 1.0)
        adjusted = 0.5 + ((p - 0.5) * shrink)
        return np.clip(adjusted, 1e-6, 1.0 - 1e-6), vol_score

    def compute_threshold_bonus(
        self,
        df_context: pd.DataFrame,
        bonus_cap: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        vol_score = self.score(df_context)
        bonus = np.clip(float(bonus_cap) * vol_score, 0.0, float(bonus_cap))
        return bonus, vol_score


def fit_full_game_volatility_normalizer(df_context: pd.DataFrame) -> FullGameVolatilityNormalizer | None:
    if FULL_GAME_VOL_NORM_ENABLED <= 0:
        return None
    if not isinstance(df_context, pd.DataFrame) or df_context.empty:
        return None
    if len(df_context) < FULL_GAME_VOL_NORM_MIN_ROWS:
        return None

    component_stats: Dict[str, Tuple[float, float]] = {}
    active_weights: Dict[str, float] = {}

    for col, weight in FULL_GAME_VOLATILITY_COMPONENT_WEIGHTS.items():
        arr = _safe_numeric_array(df_context, col)
        if arr is None:
            continue
        vals = np.abs(np.nan_to_num(arr, nan=0.0))
        finite_vals = vals[np.isfinite(vals)]
        if len(finite_vals) < 40:
            continue

        lo = float(np.nanquantile(finite_vals, 0.10))
        hi = float(np.nanquantile(finite_vals, 0.90))
        if (not np.isfinite(lo)) or (not np.isfinite(hi)):
            continue
        if hi <= (lo + 1e-6):
            continue

        component_stats[col] = (lo, hi)
        active_weights[col] = float(weight)

    if not component_stats:
        return None

    probe = FullGameVolatilityNormalizer(
        component_stats=component_stats,
        component_weights=active_weights,
        high_threshold=0.80,
    )
    calib_scores = probe.score(df_context)
    if len(calib_scores) == 0:
        return None

    high_thr = float(np.nanquantile(calib_scores, FULL_GAME_VOL_NORM_HIGH_QUANTILE))
    high_thr = float(np.clip(high_thr, 0.05, 0.95))

    return FullGameVolatilityNormalizer(
        component_stats=component_stats,
        component_weights=active_weights,
        high_threshold=high_thr,
    )


FULL_GAME_GATE_FEATURE_COLUMNS = [
    "fg_confidence",
    "fg_edge",
    "both_pitchers_available",
    "diff_pitcher_recent_quality_score",
    "diff_pitcher_quality_start_rate_L10",
    "diff_pitcher_blowup_rate_L10",
    "diff_pitcher_era_trend",
    "diff_pitcher_whip_trend",
    "diff_form_power",
    "diff_elo",
    "home_is_favorite",
]


class FullGamePlayabilityGate:
    def __init__(self, model: LogisticRegression, threshold: float):
        self.model = model
        self.threshold = float(threshold)

    def predict_mask(self, df_context: pd.DataFrame, probs_calibrated: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_gate = build_full_game_gate_feature_frame(df_context, probs_calibrated)
        score = self.model.predict_proba(X_gate)[:, 1]
        keep = (score >= self.threshold).astype(int)
        return keep, np.clip(score, 1e-6, 1.0 - 1e-6)


def build_full_game_gate_feature_frame(df_context: pd.DataFrame, probs_calibrated: np.ndarray) -> pd.DataFrame:
    probs = np.asarray(probs_calibrated, dtype=float)
    conf = np.maximum(probs, 1.0 - probs)
    edge = np.abs(probs - 0.5)

    out = pd.DataFrame({
        "fg_confidence": conf,
        "fg_edge": edge,
    })

    def _col_or_default(col: str, default: float = 0.0) -> np.ndarray:
        if not isinstance(df_context, pd.DataFrame) or col not in df_context.columns:
            return np.full(len(out), float(default), dtype=float)
        vals = pd.to_numeric(df_context[col], errors="coerce").fillna(default).to_numpy(dtype=float)
        if len(vals) != len(out):
            return np.full(len(out), float(default), dtype=float)
        return vals

    out["both_pitchers_available"] = (_col_or_default("both_pitchers_available", 1.0) >= 1.0).astype(float)
    out["diff_pitcher_recent_quality_score"] = _col_or_default("diff_pitcher_recent_quality_score", 0.0)
    out["diff_pitcher_quality_start_rate_L10"] = _col_or_default("diff_pitcher_quality_start_rate_L10", 0.0)
    out["diff_pitcher_blowup_rate_L10"] = _col_or_default("diff_pitcher_blowup_rate_L10", 0.0)
    out["diff_pitcher_era_trend"] = _col_or_default("diff_pitcher_era_trend", 0.0)
    out["diff_pitcher_whip_trend"] = _col_or_default("diff_pitcher_whip_trend", 0.0)
    out["diff_form_power"] = _col_or_default("diff_form_power", 0.0)
    out["diff_elo"] = _col_or_default("diff_elo", 0.0)
    out["home_is_favorite"] = (_col_or_default("home_is_favorite", 0.0) > 0).astype(float)

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out[FULL_GAME_GATE_FEATURE_COLUMNS]


def fit_full_game_playability_gate(
    df_context: pd.DataFrame,
    probs_calibrated: np.ndarray,
    y_true: np.ndarray,
    pred_label: np.ndarray,
    base_publish_pick: np.ndarray,
) -> Tuple[FullGamePlayabilityGate | None, Dict[str, float]]:
    meta = {
        "meta_gate_enabled": 0,
        "meta_gate_threshold": 0.0,
        "meta_gate_base_accuracy": 0.0,
        "meta_gate_final_accuracy": 0.0,
        "meta_gate_base_coverage": 0.0,
        "meta_gate_final_coverage": 0.0,
    }

    y_arr = np.asarray(y_true, dtype=int)
    pred_arr = np.asarray(pred_label, dtype=int)
    publish_base = np.asarray(base_publish_pick, dtype=int)

    if FULL_GAME_META_GATE_ENABLED <= 0:
        return None, meta

    if len(y_arr) < FULL_GAME_META_GATE_MIN_CALIB_ROWS:
        return None, meta

    base_mask = publish_base == 1
    base_rows = int(base_mask.sum())
    if base_rows < FULL_GAME_META_GATE_MIN_BASE_ROWS:
        return None, meta

    base_acc = float((pred_arr[base_mask] == y_arr[base_mask]).mean())
    base_cov = float(base_mask.mean())

    target_correct = (pred_arr == y_arr).astype(int)
    if len(np.unique(target_correct)) < 2:
        return None, meta

    X_gate = build_full_game_gate_feature_frame(df_context, probs_calibrated)
    gate_model = LogisticRegression(C=FULL_GAME_META_GATE_MODEL_C, solver="lbfgs", class_weight="balanced")
    gate_model.fit(X_gate, target_correct)
    gate_score = gate_model.predict_proba(X_gate)[:, 1]

    thr_min = min(FULL_GAME_META_GATE_THRESHOLD_MIN, FULL_GAME_META_GATE_THRESHOLD_MAX)
    thr_max = max(FULL_GAME_META_GATE_THRESHOLD_MIN, FULL_GAME_META_GATE_THRESHOLD_MAX)
    thr_step = max(0.005, FULL_GAME_META_GATE_THRESHOLD_STEP)

    best = None
    for thr in np.arange(thr_min, thr_max + 1e-9, thr_step):
        keep = base_mask & (gate_score >= thr)
        rows = int(keep.sum())
        if rows < FULL_GAME_META_GATE_MIN_KEEP_ROWS:
            continue

        acc = float((pred_arr[keep] == y_arr[keep]).mean())
        cov = float(keep.mean())
        retention = rows / max(1, base_rows)
        score = (
            acc
            + (cov * FULL_GAME_META_GATE_COVERAGE_BONUS)
            - (max(0.0, FULL_GAME_META_GATE_RETENTION_TARGET - retention) * FULL_GAME_META_GATE_RETENTION_PENALTY)
        )

        if (best is None) or (score > best["score"]):
            best = {
                "thr": float(thr),
                "acc": acc,
                "cov": cov,
                "score": float(score),
            }

    if best is None:
        return None, meta

    improves_acc = best["acc"] >= (base_acc + FULL_GAME_META_GATE_MIN_ACC_GAIN)
    keeps_volume = best["cov"] >= (base_cov * FULL_GAME_META_GATE_MIN_COVERAGE_RETENTION)
    if not (improves_acc and keeps_volume):
        return None, meta

    gate = FullGamePlayabilityGate(model=gate_model, threshold=best["thr"])
    meta.update(
        {
            "meta_gate_enabled": 1,
            "meta_gate_threshold": float(best["thr"]),
            "meta_gate_base_accuracy": float(base_acc),
            "meta_gate_final_accuracy": float(best["acc"]),
            "meta_gate_base_coverage": float(base_cov),
            "meta_gate_final_coverage": float(best["cov"]),
        }
    )
    return gate, meta


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
    volatility_normalizer: FullGameVolatilityNormalizer | None = None,
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

    vol_norm_bonus = np.zeros(len(df_context), dtype=float)
    vol_norm_scores = np.zeros(len(df_context), dtype=float)
    if market_key == "full_game" and volatility_normalizer is not None:
        vol_norm_bonus, vol_norm_scores = volatility_normalizer.compute_threshold_bonus(
            df_context=df_context,
            bonus_cap=float(FULL_GAME_VOL_NORM_THRESHOLD_BONUS),
        )
        effective_threshold += vol_norm_bonus

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

    roi_priced_coverage = 0.0
    roi_mean_ev_edge_published = 0.0
    roi_edge_filter_pass_rate = 1.0
    if market_key == "full_game" and FULL_GAME_THRESHOLD_OBJECTIVE == "roi":
        pick_decimal_odds, _, pick_ev_edge, _ = resolve_full_game_pick_odds_and_edge(
            probs_calibrated=probs_calibrated,
            df_context=df_context,
        )
        edge_mask = np.ones(len(publish_pick), dtype=bool)
        if FULL_GAME_ROI_MIN_EDGE > 0:
            edge_mask = np.isfinite(pick_ev_edge) & (pick_ev_edge >= FULL_GAME_ROI_MIN_EDGE)
            publish_pick = (publish_pick.astype(bool) & edge_mask).astype(int)

        priced_mask = np.isfinite(pick_decimal_odds) & (pick_decimal_odds > 1.0)
        pub_mask = publish_pick.astype(int) == 1
        if len(pub_mask):
            roi_priced_coverage = float(np.mean(pub_mask & priced_mask))
        if pub_mask.any():
            pub_edges = pick_ev_edge[pub_mask]
            if np.isfinite(pub_edges).any():
                roi_mean_ev_edge_published = float(np.nanmean(pub_edges))
            else:
                roi_mean_ev_edge_published = 0.0
        if len(edge_mask):
            roi_edge_filter_pass_rate = float(np.mean(edge_mask))

    meta = {
        "fallback_penalty": float(cfg.get("fallback_penalty", 0.0)) if int(used_fallback) == 1 else 0.0,
        "missing_pitcher_penalty": float(cfg.get("missing_pitcher_penalty", 0.0)),
        "prob_shrink": float(cfg.get("prob_shrink", 0.0)),
        "missing_pitchers_rate": float(missing_pitcher_flag.mean()) if len(missing_pitcher_flag) else 0.0,
        "momentum_penalty_mean": float(momentum_meta.get("momentum_penalty_mean", 0.0)),
        "momentum_penalty_max": float(momentum_meta.get("momentum_penalty_max", 0.0)),
        "momentum_penalty_active_rate": float(momentum_meta.get("momentum_penalty_active_rate", 0.0)),
        "threshold_objective": "roi" if (market_key == "full_game" and FULL_GAME_THRESHOLD_OBJECTIVE == "roi") else "accuracy_cov",
        "roi_min_edge": float(FULL_GAME_ROI_MIN_EDGE) if (market_key == "full_game" and FULL_GAME_THRESHOLD_OBJECTIVE == "roi") else 0.0,
        "roi_edge_filter_pass_rate": float(roi_edge_filter_pass_rate),
        "roi_priced_coverage": float(roi_priced_coverage),
        "roi_mean_ev_edge_published": float(roi_mean_ev_edge_published),
        "vol_norm_enabled": int(market_key == "full_game" and volatility_normalizer is not None),
        "vol_norm_alpha": float(FULL_GAME_VOL_NORM_ALPHA) if (market_key == "full_game" and volatility_normalizer is not None) else 0.0,
        "vol_norm_threshold_bonus_cap": float(FULL_GAME_VOL_NORM_THRESHOLD_BONUS) if (market_key == "full_game" and volatility_normalizer is not None) else 0.0,
        "vol_norm_threshold_bonus_mean": float(np.mean(vol_norm_bonus)) if len(vol_norm_bonus) else 0.0,
        "vol_norm_mean": float(np.mean(vol_norm_scores)) if len(vol_norm_scores) else 0.0,
        "vol_norm_high_rate": float(np.mean(vol_norm_scores >= float(volatility_normalizer.high_threshold))) if (market_key == "full_game" and volatility_normalizer is not None and len(vol_norm_scores)) else 0.0,
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


def summarize_published_roi(detail_df: pd.DataFrame) -> Dict[str, float]:
    base = {
        "published_priced_picks": 0,
        "published_total_return_units": 0.0,
        "published_roi_per_bet": 0.0,
        "published_yield_pct": 0.0,
        "published_priced_coverage": 0.0,
        "published_mean_ev_edge": 0.0,
    }
    if detail_df.empty or "publish_pick" not in detail_df.columns or "return_per_unit" not in detail_df.columns:
        return base

    pub = detail_df[detail_df["publish_pick"].fillna(0).astype(int) == 1].copy()
    if pub.empty:
        return base

    ret = pd.to_numeric(pub["return_per_unit"], errors="coerce")
    priced = ret[np.isfinite(ret)]
    if priced.empty:
        return base

    total_return_units = float(priced.sum())
    priced_picks = int(len(priced))
    roi_per_bet = float(total_return_units / priced_picks) if priced_picks > 0 else 0.0
    yield_pct = float(roi_per_bet * 100.0)
    priced_coverage = float(priced_picks / len(detail_df)) if len(detail_df) else 0.0

    mean_ev_edge = 0.0
    if "pick_ev_edge" in pub.columns:
        ev = pd.to_numeric(pub["pick_ev_edge"], errors="coerce")
        ev = ev[np.isfinite(ev)]
        if not ev.empty:
            mean_ev_edge = float(ev.mean())

    return {
        "published_priced_picks": int(priced_picks),
        "published_total_return_units": float(total_return_units),
        "published_roi_per_bet": float(roi_per_bet),
        "published_yield_pct": float(yield_pct),
        "published_priced_coverage": float(priced_coverage),
        "published_mean_ev_edge": float(mean_ev_edge),
    }


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
    full_game_ablation_mode = str(os.getenv("NBA_MLB_FULL_GAME_ABLATION", "baseball_only") or "baseball_only").strip().lower()

    def _dedupe_keep_order(items: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for item in items:
            key = str(item).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

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
        "diff_elo", "home_elo_pre", "away_elo_pre", "diff_rest_days", "diff_games_last_5_days", "diff_win_pct_L10",
        "diff_run_diff_L10", "diff_runs_scored_L5", "diff_runs_allowed_L5",
        "diff_games_in_season_before", "diff_prev_win_pct", "diff_prev_run_diff_pg",
        "prev_season_data_available",
        "diff_prev_runs_scored_pg", "diff_prev_runs_allowed_pg", "diff_win_pct_L10_blend",
        "diff_run_diff_L10_blend", "diff_runs_scored_L5_blend", "diff_runs_allowed_L5_blend",
        "diff_runs_scored_std_L10", "diff_runs_allowed_std_L10", "diff_surface_win_pct_L5",
        "diff_surface_run_diff_L5", "diff_surface_edge", "diff_win_pct_L10_vs_league",
        "diff_run_diff_L10_vs_league", "diff_fatigue_index", "diff_form_power",
        "diff_pitcher_data_available", "diff_pitcher_rest_days", "diff_pitcher_bb_allowed_L5",
        "diff_pitcher_baserunners_allowed_L5", "diff_pitcher_total_bases_allowed_L5", "diff_pitcher_runs_allowed_L5",
        "diff_pitcher_runs_allowed_L10", "diff_pitcher_start_win_rate_L10", "diff_bullpen_runs_allowed_L5",
        "diff_bullpen_runs_allowed_L10", "diff_bullpen_load_L3", "diff_offense_vs_pitcher",
        "diff_baserunners_L10", "diff_baserunners_allowed_L10", "diff_runs_per_baserunner_L10",
        "diff_player_hits_L10", "diff_player_hits_allowed_L10", "diff_player_walks_L10",
        "diff_player_walks_allowed_L10", "diff_player_total_bases_L10", "diff_player_total_bases_allowed_L10",
        "diff_player_obp_proxy_L10", "diff_player_slg_proxy_L10", "diff_player_k_rate_L10",
        "diff_top4_hits_share_L10",
        "park_factor_delta", "park_offense_pressure", "park_bullpen_pressure", "park_form_pressure",
        "umpire_zone_delta", "umpire_sample_log",
        "home_is_favorite", "odds_over_under", "open_line", "current_line", "line_movement",
        "open_total", "current_total", "total_movement", "current_home_moneyline",
        "current_away_moneyline", "current_total_line", "bookmakers_count", "snapshot_count",
        "market_moneyline_gap", "market_total_delta", "market_line_velocity", "market_missing",
        "market_micro_missing", "both_pitchers_available",
    ]

    full_game_baseball_only = [
        "diff_elo", "home_elo_pre", "away_elo_pre",
        "diff_rest_days", "diff_games_last_5_days",
        "diff_win_pct_L10", "diff_run_diff_L10",
        "diff_runs_scored_L5", "diff_runs_allowed_L5",
        "diff_win_pct_L10_blend", "diff_run_diff_L10_blend",
        "diff_runs_scored_L5_blend", "diff_runs_allowed_L5_blend",
        "diff_runs_scored_std_L10", "diff_runs_allowed_std_L10",
        "diff_surface_win_pct_L5", "diff_surface_run_diff_L5", "diff_surface_edge",
        "diff_win_pct_L10_vs_league", "diff_run_diff_L10_vs_league",
        "diff_fatigue_index", "diff_form_power",
        "diff_pitcher_data_available", "diff_pitcher_rest_days",
        "diff_pitcher_bb_allowed_L5", "diff_pitcher_baserunners_allowed_L5",
        "diff_pitcher_total_bases_allowed_L5",
        "diff_pitcher_runs_allowed_L5", "diff_pitcher_runs_allowed_L10",
        "diff_pitcher_start_win_rate_L10",
        "diff_bullpen_runs_allowed_L5", "diff_bullpen_runs_allowed_L10", "diff_bullpen_load_L3",
        "diff_offense_vs_pitcher",
        "diff_baserunners_L10", "diff_baserunners_allowed_L10", "diff_runs_per_baserunner_L10",
        "diff_player_hits_L10", "diff_player_hits_allowed_L10", "diff_player_walks_L10",
        "diff_player_walks_allowed_L10", "diff_player_total_bases_L10", "diff_player_total_bases_allowed_L10",
        "diff_player_obp_proxy_L10", "diff_player_slg_proxy_L10", "diff_player_k_rate_L10",
        "diff_top4_hits_share_L10",
        "both_pitchers_available",
    ]

    full_game_market_basic = full_game_baseball_only + [
        "home_is_favorite",
        "odds_over_under",
        "market_missing",
    ]

    full_game_market_full = full_game_baseball_only + [
        "home_is_favorite", "odds_over_under",
        "open_line", "current_line", "line_movement",
        "open_total", "current_total", "total_movement",
        "current_home_moneyline", "current_away_moneyline", "current_total_line",
        "bookmakers_count", "snapshot_count",
        "market_moneyline_gap", "market_total_delta", "market_line_velocity",
        "market_missing", "market_micro_missing",
    ]

    full_game_ablation_map = {
        "everything": full_game_keep,
        "baseball_only": full_game_baseball_only,
        "market_basic": full_game_market_basic,
        "market_full": full_game_market_full,
    }
    full_game_keep = full_game_ablation_map.get(full_game_ablation_mode, full_game_keep)
    if full_game_ablation_mode not in full_game_ablation_map:
        print(f"   WARNING: NBA_MLB_FULL_GAME_ABLATION desconocido: {full_game_ablation_mode}; usando baseball_only")
    else:
        print(f"   INFO: Full-game ablation mode: {full_game_ablation_mode} ({len(full_game_keep)} columnas objetivo)")

    # Optional controlled overrides for full_game experiments without changing defaults.
    full_game_keep = _dedupe_keep_order(full_game_keep)

    full_game_extras_raw = str(os.getenv("NBA_MLB_FULL_GAME_EXTRA_FEATURES", "") or "").strip()
    if full_game_extras_raw:
        requested = _dedupe_keep_order(full_game_extras_raw.split(","))
        valid = [feat for feat in requested if feat in all_feature_cols]
        missing = [feat for feat in requested if feat not in all_feature_cols]
        if missing:
            print(f"   WARNING: Full-game extra features ignoradas (no existen en dataset): {missing}")
        if valid:
            full_game_keep.extend([feat for feat in valid if feat not in full_game_keep])
            print(f"   INFO: Full-game extra features ON: {valid}")

    full_game_drop_raw = str(os.getenv("NBA_MLB_FULL_GAME_DROP_FEATURES", "") or "").strip()
    if full_game_drop_raw:
        to_drop = set(_dedupe_keep_order(full_game_drop_raw.split(",")))
        if to_drop:
            before = len(full_game_keep)
            full_game_keep = [feat for feat in full_game_keep if feat not in to_drop]
            dropped = before - len(full_game_keep)
            print(f"   INFO: Full-game drop features ON: {sorted(to_drop)} | removidas={dropped}")

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
    out = df_test[
        [
            c
            for c in [
                "game_id",
                "date",
                "home_team",
                "away_team",
                "odds_over_under",
                "odds_total_hits_event",
                "odds_total_hits",
                "home_is_favorite",
            ]
            if c in df_test.columns
        ]
    ].copy()
    out["market_key"] = market_key
    out["split_id"] = int(split_id)
    out["pred_value"] = np.asarray(preds, dtype=float)
    market_cfg = get_market_runtime_config(market_key)

    if market_key == "totals":
        line = pd.to_numeric(df_test.get("odds_over_under", np.nan), errors="coerce")
        actual_total = pd.to_numeric(df_test["TARGET_total_runs"], errors="coerce")
        valid_mask = np.isfinite(line) & (line > 0) & np.isfinite(actual_total)
        out["line_value"] = line
        out["actual_value"] = actual_total
        out["pred_label"] = np.where(valid_mask, (out["pred_value"] > line).astype(int), np.nan)
        out["y_true"] = np.where(valid_mask, (actual_total > line).astype(int), np.nan)
        out["edge_value"] = np.where(valid_mask, np.abs(out["pred_value"] - line), np.nan)
        publish_edge = float(market_cfg.get("publish_edge", 1.5))
        out["publish_pick"] = np.where(valid_mask, (out["edge_value"] >= publish_edge).astype(int), 0)
    elif market_key == "total_hits_event":
        line = pd.to_numeric(df_test.get("odds_total_hits_event", np.nan), errors="coerce")
        for fallback_col in [
            "odds_total_hits",
            "closing_total_hits_line",
            "current_total_hits_line",
            "open_total_hits",
        ]:
            candidate = pd.to_numeric(df_test.get(fallback_col, np.nan), errors="coerce")
            line = line.where(np.isfinite(line) & (line > 0), candidate)

        actual_total_hits = pd.to_numeric(df_test.get("TARGET_total_hits_event", np.nan), errors="coerce")
        valid_mask = np.isfinite(line) & (line > 0) & np.isfinite(actual_total_hits)
        out["line_value"] = line
        out["actual_value"] = actual_total_hits
        out["pred_label"] = np.where(valid_mask, (out["pred_value"] > line).astype(int), np.nan)
        out["y_true"] = np.where(valid_mask, (actual_total_hits > line).astype(int), np.nan)
        out["edge_value"] = np.where(valid_mask, np.abs(out["pred_value"] - line), np.nan)
        publish_edge = float(market_cfg.get("publish_edge", 1.60))
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
        publish_edge = float(market_cfg.get("publish_edge", 1.25))
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
) -> Tuple[WeightedEnsembleModel, object, Dict[str, float], pd.DataFrame, FullGamePlayabilityGate | None, FullGameVolatilityNormalizer | None]:
    xgb_probs = xgb_model.predict_proba(X_calib)[:, 1]
    lgbm_probs = lgbm_model.predict_proba(X_calib)[:, 1]
    y_calib_arr = y_calib.to_numpy().astype(int)

    rf_model = None
    rf_probs = None

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
        "regime_calibrators": 0,
        "meta_gate_enabled": 0,
        "meta_gate_threshold": 0.0,
        "meta_gate_base_accuracy": 0.0,
        "meta_gate_final_accuracy": 0.0,
        "meta_gate_base_coverage": 0.0,
        "meta_gate_final_coverage": 0.0,
        "selection_objective": "publish_quality",
        "global_selection_accuracy": 0.0,
        "global_selection_brier": 0.0,
        "global_selection_logloss": 0.0,
        "calibrator_mode": "global_lr",
        "rf_weight": 0.0,
        "stacking_mode": "none",
        "bayes_prior": float(np.clip(y_calib_arr.mean(), 0.05, 0.95)),
        "bayes_strength": 0.0,
        "bayes_sample_size": int(len(y_calib_arr)),
        "prob_shift": 0.0,
        "decision_threshold": 0.5,
        "vol_decision_enabled": 0,
        "vol_decision_beta": 0.0,
        "vol_decision_center": float(FULL_GAME_VOL_DECISION_CENTER),
        "vol_decision_max_shift": 0.0,
        "vol_decision_acc_calib": 0.0,
        "threshold_objective": "accuracy_cov",
        "roi_per_bet": 0.0,
        "total_return_units": 0.0,
        "priced_rows": 0,
        "priced_coverage": 0.0,
        "mean_ev_edge": 0.0,
        "roi_min_edge": 0.0,
        "roi_min_accuracy": 0.0,
        "vol_norm_enabled": 0,
        "vol_norm_alpha": 0.0,
        "vol_norm_threshold_bonus_cap": 0.0,
        "vol_norm_mean_calib": 0.0,
        "vol_norm_high_rate_calib": 0.0,
        "vol_norm_threshold_bonus_mean_calib": 0.0,
        "reliability_enabled": 0,
        "reliability_mean_calib": 0.0,
        "reliability_low_rate_calib": 0.0,
        "reliability_conflict_rate_calib": 0.0,
        "reliability_side_shift_mean_calib": 0.0,
    }
    best_calib_probs = None
    best_calibrator = None
    full_game_gate: FullGamePlayabilityGate | None = None
    volatility_normalizer: FullGameVolatilityNormalizer | None = None
    best_rf_weight = 0.0
    best_stack_model: LogisticRegression | None = None
    calib_reliability_meta: Dict[str, np.ndarray] | None = None

    if market_key == "full_game" and FULL_GAME_VOL_NORM_ENABLED > 0:
        volatility_normalizer = fit_full_game_volatility_normalizer(calib_context if isinstance(calib_context, pd.DataFrame) else pd.DataFrame())
        if volatility_normalizer is not None:
            best["vol_norm_enabled"] = 1
            best["vol_norm_alpha"] = float(FULL_GAME_VOL_NORM_ALPHA)
            best["vol_norm_threshold_bonus_cap"] = float(FULL_GAME_VOL_NORM_THRESHOLD_BONUS)
            best["vol_decision_enabled"] = int(FULL_GAME_VOL_DECISION_ENABLED > 0)
            best["vol_decision_center"] = float(FULL_GAME_VOL_DECISION_CENTER)
            best["vol_decision_max_shift"] = float(FULL_GAME_VOL_DECISION_MAX_SHIFT)

    rf_weight_grid = [0.0]

    for rf_weight in rf_weight_grid:
        remaining = max(1e-6, 1.0 - rf_weight)
        xgb_weight_grid = FULL_GAME_XGB_WEIGHT_GRID if market_key == "full_game" else [0.20, 0.35, 0.50, 0.65, 0.80]
        for xgb_weight in xgb_weight_grid:
            lgbm_weight = 1.0 - xgb_weight
            blend = (xgb_weight * xgb_probs) + (lgbm_weight * lgbm_probs)
            if rf_probs is not None and rf_weight > 0:
                blend = (remaining * blend) + (rf_weight * rf_probs)
            raw_probs = np.clip(blend, 1e-6, 1.0 - 1e-6)

            lr_calibrator = LogisticRegression(C=1.0, solver='lbfgs')
            lr_calibrator.fit(raw_probs.reshape(-1, 1), y_calib_arr)
            calib_probs = lr_calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

            oof_calib_probs = None
            if len(y_calib) >= 120 and y_calib.nunique() >= 2:
                n_splits = 3 if len(y_calib_arr) >= 240 else 2
                oof_preds = np.full(len(y_calib_arr), np.nan, dtype=float)
                folds_used = 0
                for tr_idx, val_idx in TimeSeriesSplit(n_splits=n_splits).split(raw_probs):
                    y_tr = y_calib_arr[tr_idx]
                    if len(np.unique(y_tr)) < 2 or len(val_idx) == 0:
                        continue
                    fold_calibrator = LogisticRegression(C=1.0, solver='lbfgs')
                    fold_calibrator.fit(raw_probs[tr_idx].reshape(-1, 1), y_tr)
                    oof_preds[val_idx] = fold_calibrator.predict_proba(raw_probs[val_idx].reshape(-1, 1))[:, 1]
                    folds_used += 1

                if folds_used > 0:
                    missing_mask = np.isnan(oof_preds)
                    if missing_mask.any():
                        oof_preds[missing_mask] = lr_calibrator.predict_proba(raw_probs[missing_mask].reshape(-1, 1))[:, 1]
                    oof_calib_probs = np.clip(oof_preds, 1e-6, 1.0 - 1e-6)

            selection_probs = oof_calib_probs if oof_calib_probs is not None else calib_probs
            selection_mode = "oof_timeseries" if oof_calib_probs is not None else "in_sample"

            candidate_stack_model = None
            if cfg.get("prob_shrink", 0.0) > 0:
                calib_probs = shrink_probs_toward_half(calib_probs, cfg["prob_shrink"])
                selection_probs = shrink_probs_toward_half(selection_probs, cfg["prob_shrink"])

            bayes_prior = float(np.clip(y_calib_arr.mean(), 0.05, 0.95))
            bayes_strength = 0.0
            if market_key == "full_game":
                bayes_strength, selection_probs, _ = choose_best_bayesian_blend(
                    probs=selection_probs,
                    y_true=y_calib_arr,
                    prior=bayes_prior,
                    sample_size=len(y_calib_arr),
                    strength_grid=[0.0, 8.0, 16.0, 24.0],
                )

            vol_scores_selection = np.zeros(len(selection_probs), dtype=float)
            if market_key == "full_game" and volatility_normalizer is not None:
                calib_probs, _ = volatility_normalizer.apply_to_probs(
                    probs=calib_probs,
                    df_context=calib_context if isinstance(calib_context, pd.DataFrame) else pd.DataFrame(index=range(len(calib_probs))),
                    alpha=float(FULL_GAME_VOL_NORM_ALPHA),
                )
                selection_probs, vol_scores_selection = volatility_normalizer.apply_to_probs(
                    probs=selection_probs,
                    df_context=calib_context if isinstance(calib_context, pd.DataFrame) else pd.DataFrame(index=range(len(selection_probs))),
                    alpha=float(FULL_GAME_VOL_NORM_ALPHA),
                )

            reliability_meta_selection = None
            if market_key == "full_game" and isinstance(calib_context, pd.DataFrame):
                reliability_context = calib_context if len(calib_context) == len(selection_probs) else pd.DataFrame(index=range(len(selection_probs)))
                reliability_vol_scores = vol_scores_selection if len(vol_scores_selection) == len(selection_probs) else None
                calib_probs, _ = apply_full_game_reliability_adjustment(
                    probs=calib_probs,
                    df_context=reliability_context,
                    vol_scores=reliability_vol_scores,
                )
                selection_probs, reliability_meta_selection = apply_full_game_reliability_adjustment(
                    probs=selection_probs,
                    df_context=reliability_context,
                    vol_scores=reliability_vol_scores,
                )

            target_cov = float(cfg.get("target_coverage", 0.0))
            if target_cov <= 0:
                target_cov = None

            min_published_rows = int(cfg.get("min_published_rows", max(10, int(len(y_calib) * float(cfg.get("min_coverage", 0.02)) * 0.5))))

            threshold_info = choose_threshold_from_calibration(
                y_true=y_calib.to_numpy(),
                probs=selection_probs,
                min_threshold=cfg["min_threshold"],
                max_threshold=cfg["max_threshold"],
                step=float(cfg.get("threshold_step", 0.01)),
                min_coverage=cfg["min_coverage"],
                target_coverage=target_cov,
                min_published_rows=min_published_rows,
            )
            threshold_info["threshold_objective"] = "accuracy_cov"
            threshold_info["roi_per_bet"] = 0.0
            threshold_info["total_return_units"] = 0.0
            threshold_info["priced_rows"] = 0
            threshold_info["priced_coverage"] = 0.0
            threshold_info["mean_ev_edge"] = 0.0
            threshold_info["roi_min_edge"] = 0.0
            threshold_info["roi_min_accuracy"] = 0.0
            threshold_info["vol_norm_enabled"] = int(market_key == "full_game" and volatility_normalizer is not None)
            threshold_info["vol_norm_alpha"] = float(FULL_GAME_VOL_NORM_ALPHA) if (market_key == "full_game" and volatility_normalizer is not None) else 0.0
            threshold_info["vol_norm_threshold_bonus_cap"] = float(FULL_GAME_VOL_NORM_THRESHOLD_BONUS) if (market_key == "full_game" and volatility_normalizer is not None) else 0.0
            threshold_info["vol_norm_mean"] = float(np.mean(vol_scores_selection)) if len(vol_scores_selection) else 0.0
            if market_key == "full_game" and volatility_normalizer is not None and len(vol_scores_selection):
                threshold_info["vol_norm_high_rate"] = float(np.mean(vol_scores_selection >= float(volatility_normalizer.high_threshold)))
            else:
                threshold_info["vol_norm_high_rate"] = 0.0

            if (
                market_key == "full_game"
                and FULL_GAME_THRESHOLD_OBJECTIVE == "roi"
                and isinstance(calib_context, pd.DataFrame)
                and len(calib_context) == len(selection_probs)
            ):
                roi_threshold_info = choose_full_game_roi_threshold(
                    y_true=y_calib_arr,
                    probs=selection_probs,
                    df_context=calib_context,
                    min_threshold=cfg["min_threshold"],
                    max_threshold=cfg["max_threshold"],
                    step=float(cfg.get("threshold_step", 0.01)),
                    min_coverage=cfg["min_coverage"],
                    target_coverage=target_cov,
                    coverage_tolerance=0.02,
                    min_published_rows=min_published_rows,
                )
                if roi_threshold_info is not None:
                    threshold_info = roi_threshold_info

            if market_key == "full_game" and reliability_meta_selection is not None:
                rel_scores = np.clip(reliability_meta_selection.get("reliability_score", np.zeros(len(selection_probs), dtype=float)), 0.0, 1.0)
                rel_conflict = np.clip(reliability_meta_selection.get("conflict_flag", np.zeros(len(selection_probs), dtype=float)), 0.0, 1.0)
                rel_shift = np.asarray(reliability_meta_selection.get("side_shift", np.zeros(len(selection_probs), dtype=float)), dtype=float)
                threshold_info["reliability_enabled"] = int(FULL_GAME_RELIABILITY_ENABLED > 0)
                threshold_info["reliability_mean"] = float(np.mean(rel_scores)) if len(rel_scores) else 0.0
                threshold_info["reliability_low_rate"] = float(np.mean(rel_scores < 0.45)) if len(rel_scores) else 0.0
                threshold_info["reliability_conflict_rate"] = float(np.mean(rel_conflict)) if len(rel_conflict) else 0.0
                threshold_info["reliability_side_shift_mean"] = float(np.mean(np.abs(rel_shift))) if len(rel_shift) else 0.0
            else:
                threshold_info["reliability_enabled"] = 0
                threshold_info["reliability_mean"] = 0.0
                threshold_info["reliability_low_rate"] = 0.0
                threshold_info["reliability_conflict_rate"] = 0.0
                threshold_info["reliability_side_shift_mean"] = 0.0

            y_sel = y_calib_arr
            sel_probs = np.clip(selection_probs, 1e-6, 1.0 - 1e-6)
            decision_acc_for_score = float(((sel_probs >= 0.5).astype(int) == y_sel).mean())
            vol_decision_beta = 0.0
            if (
                market_key == "full_game"
                and volatility_normalizer is not None
                and FULL_GAME_VOL_DECISION_ENABLED > 0
                and len(vol_scores_selection) == len(sel_probs)
            ):
                vol_decision_beta, _, decision_acc_for_score = choose_best_volatility_decision_beta(
                    probs=sel_probs,
                    y_true=y_sel,
                    vol_scores=vol_scores_selection,
                    beta_grid=FULL_GAME_VOL_DECISION_BETA_GRID,
                    center=float(FULL_GAME_VOL_DECISION_CENTER),
                    max_shift=float(FULL_GAME_VOL_DECISION_MAX_SHIFT),
                    beta_penalty=float(FULL_GAME_VOL_DECISION_BETA_PENALTY),
                )
            global_acc, global_brier, global_score = full_game_global_score_from_probs(
                sel_probs,
                y_sel,
                brier_weight=FULL_GAME_GLOBAL_BRIER_WEIGHT,
            )
            if market_key == "full_game":
                global_score = float(decision_acc_for_score - (FULL_GAME_GLOBAL_BRIER_WEIGHT * global_brier))
            global_logloss = float(
                -np.mean((y_sel * np.log(sel_probs)) + ((1 - y_sel) * np.log(1.0 - sel_probs)))
            )

            if market_key == "f5":
                print(
                    f"[F5 CALIB] xgb_w={xgb_weight:.2f} | "
                    f"thr={threshold_info.get('threshold', -1):.3f} | "
                    f"acc={threshold_info.get('accuracy', 0.0):.4f} | "
                    f"cov={threshold_info.get('coverage', 0.0):.4f} | "
                    f"rows={threshold_info.get('published_rows', 0)} | "
                    f"score={threshold_info.get('score', -999):.4f}"
                )

            if market_key == "full_game":
                score = global_score
                invalid_candidate = not np.isfinite(score)
                selection_objective = f"global_acc050_brier_soft|thr_{threshold_info.get('threshold_objective', 'accuracy_cov')}"
            else:
                invalid_candidate = (
                    threshold_info.get("coverage", 0.0) <= 0.0
                    or threshold_info.get("accuracy", 0.0) <= 0.0
                    or threshold_info.get("score", -1e9) <= 0.0
                )
                score = float(threshold_info["score"])
                selection_objective = "publish_quality"

            if invalid_candidate:
                continue

            if score > best["score"]:
                best = {
                    "xgb_weight": float(round(xgb_weight * remaining, 2)),
                    "lgbm_weight": float(round(lgbm_weight * remaining, 2)),
                    "rf_weight": 0.0,
                    **threshold_info,
                    "score": float(score),
                    "used_fallback": 0,
                    "prob_shrink": float(cfg.get("prob_shrink", 0.0)),
                    "selection_mode": selection_mode,
                    "regime_calibrators": 0,
                    "selection_objective": selection_objective,
                    "global_selection_accuracy": float(decision_acc_for_score),
                    "global_selection_brier": float(global_brier),
                    "global_selection_logloss": float(global_logloss),
                    "calibrator_mode": "global_lr",
                    "stacking_mode": "none",
                    "bayes_prior": bayes_prior,
                    "bayes_strength": float(bayes_strength),
                    "bayes_sample_size": int(len(y_calib_arr)),
                    "prob_shift": 0.0,
                    "decision_threshold": 0.5,
                    "vol_decision_enabled": int(
                        market_key == "full_game"
                        and volatility_normalizer is not None
                        and FULL_GAME_VOL_DECISION_ENABLED > 0
                    ),
                    "vol_decision_beta": float(vol_decision_beta),
                    "vol_decision_center": float(FULL_GAME_VOL_DECISION_CENTER),
                    "vol_decision_max_shift": float(FULL_GAME_VOL_DECISION_MAX_SHIFT) if (
                        market_key == "full_game"
                        and volatility_normalizer is not None
                        and FULL_GAME_VOL_DECISION_ENABLED > 0
                    ) else 0.0,
                    "vol_decision_acc_calib": float(decision_acc_for_score),
                    "reliability_enabled": int(threshold_info.get("reliability_enabled", 0)),
                    "reliability_mean_calib": float(threshold_info.get("reliability_mean", 0.0)),
                    "reliability_low_rate_calib": float(threshold_info.get("reliability_low_rate", 0.0)),
                    "reliability_conflict_rate_calib": float(threshold_info.get("reliability_conflict_rate", 0.0)),
                    "reliability_side_shift_mean_calib": float(threshold_info.get("reliability_side_shift_mean", 0.0)),
                }
                best_rf_weight = 0.0
                best_calib_probs = selection_probs
                best_calibrator = lr_calibrator
                best_stack_model = candidate_stack_model

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
            "selection_mode": "fallback_default",
            "published_rows": 0,
            "regime_calibrators": 0,
            "meta_gate_enabled": 0,
            "meta_gate_threshold": 0.0,
            "meta_gate_base_accuracy": 0.0,
            "meta_gate_final_accuracy": 0.0,
            "meta_gate_base_coverage": 0.0,
            "meta_gate_final_coverage": 0.0,
            "selection_objective": "fallback_default",
            "global_selection_accuracy": 0.0,
            "global_selection_brier": 0.0,
            "global_selection_logloss": 0.0,
            "calibrator_mode": "global_lr",
            "rf_weight": 0.0,
            "stacking_mode": "none",
            "bayes_prior": float(np.clip(y_calib_arr.mean(), 0.05, 0.95)),
            "bayes_strength": 0.0,
            "bayes_sample_size": int(len(y_calib_arr)),
            "prob_shift": 0.0,
            "decision_threshold": 0.5,
            "vol_decision_enabled": 0,
            "vol_decision_beta": 0.0,
            "vol_decision_center": float(FULL_GAME_VOL_DECISION_CENTER),
            "vol_decision_max_shift": 0.0,
            "vol_decision_acc_calib": 0.0,
            "threshold_objective": "fallback_default",
            "roi_per_bet": 0.0,
            "total_return_units": 0.0,
            "priced_rows": 0,
            "priced_coverage": 0.0,
            "mean_ev_edge": 0.0,
            "roi_min_edge": 0.0,
            "roi_min_accuracy": 0.0,
            "reliability_enabled": 0,
            "reliability_mean_calib": 0.0,
            "reliability_low_rate_calib": 0.0,
            "reliability_conflict_rate_calib": 0.0,
            "reliability_side_shift_mean_calib": 0.0,
        }
        best_rf_weight = 0.0
        best_stack_model = None

    if market_key == "full_game" and isinstance(calib_context, pd.DataFrame) and not calib_context.empty:
        best_raw_probs = np.clip(
            (float(best["xgb_weight"]) * xgb_probs) + (float(best["lgbm_weight"]) * lgbm_probs),
            1e-6,
            1.0 - 1e-6,
        )
        base_probs = np.clip(best_calibrator.predict_proba(best_raw_probs.reshape(-1, 1))[:, 1], 1e-6, 1.0 - 1e-6)
        if cfg.get("prob_shrink", 0.0) > 0:
            base_probs = shrink_probs_toward_half(base_probs, cfg["prob_shrink"])

        y_eval = y_calib.to_numpy().astype(int)
        base_acc, base_brier, base_global_score = full_game_global_score_from_probs(
            base_probs,
            y_eval,
            brier_weight=FULL_GAME_GLOBAL_BRIER_WEIGHT,
        )
        base_logloss = float(
            -np.mean((y_eval * np.log(base_probs)) + ((1 - y_eval) * np.log(1.0 - base_probs)))
        )

        regime_calibrator, regime_probs, regime_count = fit_regime_aware_calibrator(
            raw_probs=best_raw_probs,
            y_calib=y_calib,
            calib_context=calib_context,
            market_key=market_key,
        )
        if cfg.get("prob_shrink", 0.0) > 0:
            regime_probs = shrink_probs_toward_half(regime_probs, cfg["prob_shrink"])

        regime_acc, regime_brier, regime_global_score = full_game_global_score_from_probs(
            regime_probs,
            y_eval,
            brier_weight=FULL_GAME_GLOBAL_BRIER_WEIGHT,
        )
        regime_logloss = float(
            -np.mean((y_eval * np.log(regime_probs)) + ((1 - y_eval) * np.log(1.0 - regime_probs)))
        )

        prefer_regime = regime_global_score > (base_global_score + 1e-4)
        if FULL_GAME_CALIBRATOR_MODE == "regime_aware":
            prefer_regime = True
        elif FULL_GAME_CALIBRATOR_MODE == "global_lr":
            prefer_regime = False

        if prefer_regime:
            best_calibrator = regime_calibrator
            best_calib_probs = regime_probs
            best["regime_calibrators"] = int(regime_count)
            best["calibrator_mode"] = "regime_aware"
        else:
            best_calib_probs = base_probs
            best["regime_calibrators"] = 0
            best["calibrator_mode"] = "global_lr"

        bayes_prior_final = float(np.clip(best.get("bayes_prior", y_eval.mean()), 0.05, 0.95))
        bayes_strength_final, best_calib_probs, _ = choose_best_bayesian_blend(
            probs=best_calib_probs,
            y_true=y_eval,
            prior=bayes_prior_final,
            sample_size=len(y_eval),
            strength_grid=[0.0, 8.0, 16.0, 24.0],
        )
        best["bayes_prior"] = float(bayes_prior_final)
        best["bayes_strength"] = float(bayes_strength_final)
        best["bayes_sample_size"] = int(len(y_eval))

        if FULL_GAME_PROB_SHIFT_ENABLED > 0:
            shift_min = min(FULL_GAME_PROB_SHIFT_MIN, FULL_GAME_PROB_SHIFT_MAX)
            shift_max = max(FULL_GAME_PROB_SHIFT_MIN, FULL_GAME_PROB_SHIFT_MAX)
            shift_step = max(0.001, FULL_GAME_PROB_SHIFT_STEP)
            shift_grid = [float(s) for s in np.arange(shift_min, shift_max + 1e-9, shift_step)]
            if not shift_grid:
                shift_grid = [0.0]
            if not any(abs(s) <= 1e-12 for s in shift_grid):
                shift_grid.append(0.0)
            shift_grid = sorted(set([float(round(s, 6)) for s in shift_grid]))
        else:
            shift_grid = [0.0]

        prob_shift, best_calib_probs, _ = choose_best_probability_shift(
            probs=best_calib_probs,
            y_true=y_eval,
            shift_grid=shift_grid,
        )
        best["prob_shift"] = float(prob_shift)

        if FULL_GAME_RELIABILITY_ENABLED > 0 and isinstance(calib_context, pd.DataFrame) and len(calib_context) == len(best_calib_probs):
            reliability_vol_scores = None
            if volatility_normalizer is not None:
                candidate_vol_scores = volatility_normalizer.score(calib_context)
                if len(candidate_vol_scores) == len(best_calib_probs):
                    reliability_vol_scores = candidate_vol_scores
            best_calib_probs, calib_reliability_meta = apply_full_game_reliability_adjustment(
                probs=best_calib_probs,
                df_context=calib_context,
                vol_scores=reliability_vol_scores,
            )
            rel_scores = np.clip(calib_reliability_meta.get("reliability_score", np.zeros(len(best_calib_probs), dtype=float)), 0.0, 1.0)
            rel_conflict = np.clip(calib_reliability_meta.get("conflict_flag", np.zeros(len(best_calib_probs), dtype=float)), 0.0, 1.0)
            rel_shift = np.asarray(calib_reliability_meta.get("side_shift", np.zeros(len(best_calib_probs), dtype=float)), dtype=float)
            best["reliability_enabled"] = 1
            best["reliability_mean_calib"] = float(np.mean(rel_scores)) if len(rel_scores) else 0.0
            best["reliability_low_rate_calib"] = float(np.mean(rel_scores < 0.45)) if len(rel_scores) else 0.0
            best["reliability_conflict_rate_calib"] = float(np.mean(rel_conflict)) if len(rel_conflict) else 0.0
            best["reliability_side_shift_mean_calib"] = float(np.mean(np.abs(rel_shift))) if len(rel_shift) else 0.0
        else:
            calib_reliability_meta = None
            best["reliability_enabled"] = 0
            best["reliability_mean_calib"] = 0.0
            best["reliability_low_rate_calib"] = 0.0
            best["reliability_conflict_rate_calib"] = 0.0
            best["reliability_side_shift_mean_calib"] = 0.0

        best["decision_threshold"] = 0.5
        best["vol_decision_enabled"] = 0
        best["vol_decision_beta"] = 0.0
        best["vol_decision_center"] = float(FULL_GAME_VOL_DECISION_CENTER)
        best["vol_decision_max_shift"] = 0.0
        best["vol_decision_acc_calib"] = float(((best_calib_probs >= 0.5).astype(int) == y_eval).mean())
        best["global_selection_accuracy"] = float(best["vol_decision_acc_calib"])

        if (
            volatility_normalizer is not None
            and FULL_GAME_VOL_DECISION_ENABLED > 0
            and isinstance(calib_context, pd.DataFrame)
            and len(calib_context) == len(best_calib_probs)
        ):
            calib_vol_scores = volatility_normalizer.score(calib_context)
            if len(calib_vol_scores) == len(best_calib_probs):
                best_beta, _, best_decision_acc = choose_best_volatility_decision_beta(
                    probs=best_calib_probs,
                    y_true=y_eval,
                    vol_scores=calib_vol_scores,
                    beta_grid=FULL_GAME_VOL_DECISION_BETA_GRID,
                    center=float(FULL_GAME_VOL_DECISION_CENTER),
                    max_shift=float(FULL_GAME_VOL_DECISION_MAX_SHIFT),
                    beta_penalty=float(FULL_GAME_VOL_DECISION_BETA_PENALTY),
                )
                best["vol_decision_enabled"] = 1
                best["vol_decision_beta"] = float(best_beta)
                best["vol_decision_center"] = float(FULL_GAME_VOL_DECISION_CENTER)
                best["vol_decision_max_shift"] = float(FULL_GAME_VOL_DECISION_MAX_SHIFT)
                best["vol_decision_acc_calib"] = float(best_decision_acc)
                best["global_selection_accuracy"] = float(best_decision_acc)

    ensemble_model = WeightedEnsembleModel(
        xgb_model=xgb_model,
        lgbm_model=lgbm_model,
        xgb_weight=best["xgb_weight"],
        lgbm_weight=best["lgbm_weight"],
        rf_model=rf_model,
        rf_weight=best_rf_weight,
        stacking_model=best_stack_model,
    )

    calib_publish_pick, calib_effective_threshold, calib_policy_meta = apply_publish_policy(
        probs_calibrated=best_calib_probs,
        df_context=calib_context if isinstance(calib_context, pd.DataFrame) else (X_calib if isinstance(X_calib, pd.DataFrame) else pd.DataFrame(index=range(len(best_calib_probs)))),
        base_threshold=float(best["threshold"]),
        market_key=market_key,
        used_fallback=int(best.get("used_fallback", 0)),
        volatility_normalizer=volatility_normalizer if market_key == "full_game" else None,
    )

    calib_decision_threshold = np.full(
        len(best_calib_probs),
        float(best.get("decision_threshold", 0.5)),
        dtype=float,
    )
    if (
        market_key == "full_game"
        and volatility_normalizer is not None
        and int(best.get("vol_decision_enabled", 0)) == 1
    ):
        calib_context_for_decision = (
            calib_context
            if isinstance(calib_context, pd.DataFrame)
            else pd.DataFrame(index=range(len(best_calib_probs)))
        )
        calib_vol_scores = volatility_normalizer.score(calib_context_for_decision)
        if len(calib_vol_scores) == len(calib_decision_threshold):
            calib_decision_threshold = build_dynamic_decision_threshold_from_vol_scores(
                vol_scores=calib_vol_scores,
                beta=float(best.get("vol_decision_beta", 0.0)),
                center=float(best.get("vol_decision_center", FULL_GAME_VOL_DECISION_CENTER)),
                max_shift=float(best.get("vol_decision_max_shift", FULL_GAME_VOL_DECISION_MAX_SHIFT)),
            )
    calib_pred_label = (best_calib_probs >= calib_decision_threshold).astype(int)
    gate_keep = np.ones(len(best_calib_probs), dtype=int)
    gate_score = np.ones(len(best_calib_probs), dtype=float)

    if market_key == "full_game" and isinstance(calib_context, pd.DataFrame) and not calib_context.empty:
        gate_obj, gate_meta = fit_full_game_playability_gate(
            df_context=calib_context,
            probs_calibrated=best_calib_probs,
            y_true=y_calib.to_numpy(),
            pred_label=calib_pred_label,
            base_publish_pick=calib_publish_pick,
        )
        best.update(gate_meta)
        if gate_obj is not None:
            full_game_gate = gate_obj
            gate_keep, gate_score = full_game_gate.predict_mask(calib_context, best_calib_probs)
            calib_publish_pick = (calib_publish_pick.astype(int) * gate_keep.astype(int)).astype(int)

    calib_detail = pd.DataFrame(
        {
            "y_true": y_calib.to_numpy().astype(int),
            "ensemble_prob_calibrated": best_calib_probs,
            "pred_label": calib_pred_label,
            "decision_threshold": calib_decision_threshold.astype(float),
            "publish_pick": calib_publish_pick.astype(int),
            "publish_threshold_effective": calib_effective_threshold.astype(float),
            "used_fallback": int(best.get("used_fallback", 0)),
        }
    )
    if market_key == "full_game":
        calib_detail["full_game_gate_keep"] = gate_keep.astype(int)
        calib_detail["full_game_gate_score"] = gate_score.astype(float)
        if calib_reliability_meta is not None:
            calib_detail["full_game_reliability_score"] = np.asarray(calib_reliability_meta.get("reliability_score", np.ones(len(calib_detail), dtype=float)), dtype=float)
            calib_detail["full_game_reliability_side_hint"] = np.asarray(calib_reliability_meta.get("side_hint", np.zeros(len(calib_detail), dtype=float)), dtype=float)
            calib_detail["full_game_reliability_conflict_flag"] = np.asarray(calib_reliability_meta.get("conflict_flag", np.zeros(len(calib_detail), dtype=float)), dtype=float)
            calib_detail["full_game_reliability_side_shift"] = np.asarray(calib_reliability_meta.get("side_shift", np.zeros(len(calib_detail), dtype=float)), dtype=float)
            calib_detail["full_game_reliability_shrink"] = np.asarray(calib_reliability_meta.get("shrink_factor", np.ones(len(calib_detail), dtype=float)), dtype=float)
        else:
            calib_detail["full_game_reliability_score"] = 1.0
            calib_detail["full_game_reliability_side_hint"] = 0.0
            calib_detail["full_game_reliability_conflict_flag"] = 0.0
            calib_detail["full_game_reliability_side_shift"] = 0.0
            calib_detail["full_game_reliability_shrink"] = 1.0

    published_calib = calib_detail[calib_detail["publish_pick"] == 1].copy()
    best["published_accuracy"] = float((published_calib["pred_label"].astype(int) == published_calib["y_true"].astype(int)).mean()) if len(published_calib) else 0.0
    best["published_coverage"] = float(calib_detail["publish_pick"].mean()) if len(calib_detail) else 0.0
    best["fallback_penalty"] = float(calib_policy_meta.get("fallback_penalty", 0.0))
    best["missing_pitcher_penalty"] = float(calib_policy_meta.get("missing_pitcher_penalty", 0.0))
    best["missing_pitchers_rate_calib"] = float(calib_policy_meta.get("missing_pitchers_rate", 0.0))
    best["momentum_penalty_mean_calib"] = float(calib_policy_meta.get("momentum_penalty_mean", 0.0))
    best["momentum_penalty_active_rate_calib"] = float(calib_policy_meta.get("momentum_penalty_active_rate", 0.0))
    best["vol_norm_mean_calib"] = float(calib_policy_meta.get("vol_norm_mean", 0.0))
    best["vol_norm_high_rate_calib"] = float(calib_policy_meta.get("vol_norm_high_rate", 0.0))
    best["vol_norm_threshold_bonus_mean_calib"] = float(calib_policy_meta.get("vol_norm_threshold_bonus_mean", 0.0))
    best["vol_norm_enabled"] = int(calib_policy_meta.get("vol_norm_enabled", best.get("vol_norm_enabled", 0)))
    best["vol_norm_alpha"] = float(calib_policy_meta.get("vol_norm_alpha", best.get("vol_norm_alpha", 0.0)))
    best["vol_norm_threshold_bonus_cap"] = float(calib_policy_meta.get("vol_norm_threshold_bonus_cap", best.get("vol_norm_threshold_bonus_cap", 0.0)))

    return ensemble_model, best_calibrator, best, calib_detail, full_game_gate, volatility_normalizer


def build_prediction_rows(
    df_test: pd.DataFrame,
    market_key: str,
    probs_raw: np.ndarray,
    probs_calibrated: np.ndarray,
    threshold: float,
    decision_threshold: float | np.ndarray,
    split_id: int,
    used_fallback: int,
    prob_shrink: float,
    full_game_gate: FullGamePlayabilityGate | None = None,
    volatility_normalizer: FullGameVolatilityNormalizer | None = None,
    full_game_reliability_meta: Dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    candidate_cols = [
        "game_id", "date", "home_team", "away_team",
        "current_home_moneyline", "current_away_moneyline",
        "home_moneyline_odds", "away_moneyline_odds", "closing_moneyline_odds", "moneyline_odds",
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

    if np.isscalar(decision_threshold):
        decision_threshold_arr = np.full(len(out), float(decision_threshold), dtype=float)
    else:
        decision_threshold_arr = np.asarray(decision_threshold, dtype=float).reshape(-1)
        if len(decision_threshold_arr) != len(out):
            decision_threshold_arr = np.full(len(out), 0.5, dtype=float)
    decision_threshold_arr = np.clip(decision_threshold_arr, 0.30, 0.70)

    out["decision_threshold"] = decision_threshold_arr
    out["pred_label"] = (probs_calibrated >= decision_threshold_arr).astype(int)
    out["confidence"] = np.maximum(probs_calibrated, 1.0 - probs_calibrated)
    out["threshold"] = float(threshold)
    out["used_fallback"] = int(used_fallback)
    out["prob_shrink"] = float(prob_shrink)
    out["y_true"] = df_test["target_col_runtime"].astype(int).to_numpy()

    if market_key == "full_game":
        vol_score = np.zeros(len(out), dtype=float)
        pick_decimal_odds, pick_implied_prob, pick_ev_edge, pick_odds_source = resolve_full_game_pick_odds_and_edge(
            probs_calibrated=probs_calibrated,
            df_context=df_test,
        )
        out["pick_decimal_odds"] = pick_decimal_odds
        out["pick_implied_prob"] = pick_implied_prob
        out["pick_ev_edge"] = pick_ev_edge
        out["pick_odds_source"] = pick_odds_source

        priced_pick = np.isfinite(pick_decimal_odds) & (pick_decimal_odds > 1.0)
        hit_mask = out["pred_label"].astype(int).to_numpy() == out["y_true"].astype(int).to_numpy()
        returns = np.full(len(out), np.nan, dtype=float)
        returns[priced_pick & hit_mask] = pick_decimal_odds[priced_pick & hit_mask] - 1.0
        returns[priced_pick & (~hit_mask)] = -1.0
        out["priced_pick"] = priced_pick.astype(int)
        out["return_per_unit"] = returns

        if volatility_normalizer is not None:
            vol_score = volatility_normalizer.score(df_test)
            out["full_game_volatility_score"] = vol_score
            out["full_game_high_volatility_flag"] = (vol_score >= float(volatility_normalizer.high_threshold)).astype(int)
        else:
            out["full_game_volatility_score"] = 0.0
            out["full_game_high_volatility_flag"] = 0

        reliability_meta = full_game_reliability_meta
        if reliability_meta is None:
            reliability_meta = compute_full_game_reliability_meta(
                probs=probs_calibrated,
                df_context=df_test,
                vol_scores=vol_score if len(vol_score) == len(out) else None,
            )

        reliability_score = np.asarray(reliability_meta.get("reliability_score", np.ones(len(out), dtype=float)), dtype=float)
        reliability_side_hint = np.asarray(reliability_meta.get("side_hint", np.zeros(len(out), dtype=float)), dtype=float)
        reliability_conflict = np.asarray(reliability_meta.get("conflict_flag", np.zeros(len(out), dtype=float)), dtype=float)
        reliability_shift = np.asarray(reliability_meta.get("side_shift", np.zeros(len(out), dtype=float)), dtype=float)
        reliability_shrink = np.asarray(reliability_meta.get("shrink_factor", np.ones(len(out), dtype=float)), dtype=float)

        if len(reliability_score) != len(out):
            reliability_score = np.ones(len(out), dtype=float)
        if len(reliability_side_hint) != len(out):
            reliability_side_hint = np.zeros(len(out), dtype=float)
        if len(reliability_conflict) != len(out):
            reliability_conflict = np.zeros(len(out), dtype=float)
        if len(reliability_shift) != len(out):
            reliability_shift = np.zeros(len(out), dtype=float)
        if len(reliability_shrink) != len(out):
            reliability_shrink = np.ones(len(out), dtype=float)

        out["full_game_reliability_enabled"] = int(FULL_GAME_RELIABILITY_ENABLED > 0)
        out["full_game_reliability_score"] = reliability_score
        out["full_game_reliability_side_hint"] = reliability_side_hint
        out["full_game_reliability_conflict_flag"] = reliability_conflict
        out["full_game_reliability_side_shift"] = reliability_shift
        out["full_game_reliability_shrink"] = reliability_shrink

    publish_pick, effective_threshold, policy_meta = apply_publish_policy(
        probs_calibrated=probs_calibrated,
        df_context=df_test,
        base_threshold=threshold,
        market_key=market_key,
        used_fallback=used_fallback,
        volatility_normalizer=volatility_normalizer if market_key == "full_game" else None,
    )

    if market_key == "full_game" and full_game_gate is not None:
        gate_keep, gate_score = full_game_gate.predict_mask(df_test, probs_calibrated)
        publish_pick = (publish_pick.astype(int) * gate_keep.astype(int)).astype(int)
        out["full_game_gate_keep"] = gate_keep.astype(int)
        out["full_game_gate_score"] = gate_score.astype(float)

    out["publish_pick"] = publish_pick.astype(int)
    out["publish_threshold_effective"] = effective_threshold.astype(float)
    out["missing_pitchers_rate_test_split"] = float(policy_meta.get("missing_pitchers_rate", 0.0))
    out["threshold_objective"] = str(policy_meta.get("threshold_objective", "accuracy_cov"))
    out["roi_min_edge"] = float(policy_meta.get("roi_min_edge", 0.0))
    out["roi_edge_filter_pass_rate"] = float(policy_meta.get("roi_edge_filter_pass_rate", 1.0))
    out["roi_priced_coverage"] = float(policy_meta.get("roi_priced_coverage", 0.0))
    out["roi_mean_ev_edge_published"] = float(policy_meta.get("roi_mean_ev_edge_published", 0.0))
    return out


def run_market_walkforward(
    df: pd.DataFrame,
    market_key: str,
    target_col: str,
    feature_cols: List[str],
) -> Dict[str, pd.DataFrame]:
    market_df = df.dropna(subset=[target_col]).copy()
    market_df = market_df.sort_values(["date_dt", "game_id"]).reset_index(drop=True)
    if market_key == "full_game":
        market_df = enrich_full_game_context_with_odds(market_df)
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
            print("      split omitido por tamano insuficiente")
            continue

        y_train = train_df[target_col].astype(int)
        y_calib = calib_df[target_col].astype(int)
        y_test = test_df[target_col].astype(int)

        if y_train.nunique() < 2:
            print("      split omitido: y_train con una sola clase")
            continue
        if y_calib.nunique() < 2:
            print("      split omitido: y_calib con una sola clase")
            continue

        X_train = sanitize_feature_frame(train_df, feature_cols)
        X_calib = sanitize_feature_frame(calib_df, feature_cols)
        X_test = sanitize_feature_frame(test_df, feature_cols)

        print("      entrenando XGB + LGBM...")
        xgb_model, lgbm_model = fit_market_models(X_train, y_train, market_key)
        print("      modelos entrenados, calibrando ensemble...")

        ensemble_model, calibrator, threshold_info, calib_detail, full_game_gate, volatility_normalizer = choose_ensemble_and_threshold(
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
            f"global_sel_acc={threshold_info.get('global_selection_accuracy', 0.0):.4f} | "
            f"global_sel_brier={threshold_info.get('global_selection_brier', 0.0):.4f} | "
            f"bayes_s={threshold_info.get('bayes_strength', 0.0):.1f} | "
            f"p_shift={threshold_info.get('prob_shift', 0.0):+.3f} | "
            f"published_acc={threshold_info.get('published_accuracy', 0.0):.4f} | "
            f"published_cov={threshold_info.get('published_coverage', 0.0):.4f} | "
            f"published_rows={threshold_info.get('published_rows', 0)} | "
            f"thr_obj={threshold_info.get('threshold_objective', 'accuracy_cov')} | "
            f"roi={threshold_info.get('roi_per_bet', 0.0):+.4f} | "
            f"priced_rows={threshold_info.get('priced_rows', 0)} | "
            f"xgb_w={threshold_info['xgb_weight']:.2f} | "
            f"lgbm_w={threshold_info['lgbm_weight']:.2f} | "
            f"rf_w={threshold_info.get('rf_weight', 0.0):.2f} | "
            f"fallback={threshold_info.get('used_fallback', 0)} | "
            f"shrink={threshold_info.get('prob_shrink', 0.0):.2f} | "
            f"objective={threshold_info.get('selection_objective', 'n/a')} | "
            f"selection={threshold_info.get('selection_mode', 'n/a')} | "
            f"calibrator={threshold_info.get('calibrator_mode', 'n/a')} | "
            f"stack={threshold_info.get('stacking_mode', 'none')} | "
            f"regime_cal={threshold_info.get('regime_calibrators', 0)} | "
            f"meta_gate={threshold_info.get('meta_gate_enabled', 0)} | "
            f"vol_norm={threshold_info.get('vol_norm_enabled', 0)} | "
            f"vol_dec_beta={threshold_info.get('vol_decision_beta', 0.0):+.3f} | "
            f"rel_mean={threshold_info.get('reliability_mean_calib', 0.0):.3f}"
        )

        raw_test_probs = ensemble_model.predict_proba(X_test)[:, 1]
        calibrated_test_probs = predict_calibrated_probs(
            calibrator=calibrator,
            raw_probs=raw_test_probs,
            df_context=test_df,
            market_key=market_key,
        )
        
        if float(threshold_info.get("prob_shrink", 0.0)) > 0:
            calibrated_test_probs = shrink_probs_toward_half(
                calibrated_test_probs,
                float(threshold_info["prob_shrink"]),
            )
        if market_key == "full_game" and float(threshold_info.get("bayes_strength", 0.0)) > 0:
            calibrated_test_probs = apply_bayesian_probability_blend(
                probs=calibrated_test_probs,
                prior=float(threshold_info.get("bayes_prior", 0.5)),
                strength=float(threshold_info.get("bayes_strength", 0.0)),
                sample_size=int(threshold_info.get("bayes_sample_size", len(y_calib))),
            )
        if market_key == "full_game" and abs(float(threshold_info.get("prob_shift", 0.0))) > 1e-12:
            calibrated_test_probs = apply_probability_shift(
                probs=calibrated_test_probs,
                shift=float(threshold_info.get("prob_shift", 0.0)),
            )

        test_volatility_scores = np.zeros(len(calibrated_test_probs), dtype=float)
        if market_key == "full_game" and volatility_normalizer is not None:
            calibrated_test_probs, test_volatility_scores = volatility_normalizer.apply_to_probs(
                probs=calibrated_test_probs,
                df_context=test_df,
                alpha=float(threshold_info.get("vol_norm_alpha", FULL_GAME_VOL_NORM_ALPHA)),
            )

        test_reliability_meta = None
        if market_key == "full_game":
            reliability_vol_scores = test_volatility_scores if len(test_volatility_scores) == len(calibrated_test_probs) else None
            calibrated_test_probs, test_reliability_meta = apply_full_game_reliability_adjustment(
                probs=calibrated_test_probs,
                df_context=test_df,
                vol_scores=reliability_vol_scores,
            )

        test_decision_threshold = np.full(
            len(calibrated_test_probs),
            float(threshold_info.get("decision_threshold", 0.5)),
            dtype=float,
        )
        if (
            market_key == "full_game"
            and volatility_normalizer is not None
            and int(threshold_info.get("vol_decision_enabled", 0)) == 1
            and len(test_volatility_scores) == len(test_decision_threshold)
        ):
            test_decision_threshold = build_dynamic_decision_threshold_from_vol_scores(
                vol_scores=test_volatility_scores,
                beta=float(threshold_info.get("vol_decision_beta", 0.0)),
                center=float(threshold_info.get("vol_decision_center", FULL_GAME_VOL_DECISION_CENTER)),
                max_shift=float(threshold_info.get("vol_decision_max_shift", FULL_GAME_VOL_DECISION_MAX_SHIFT)),
            )

        test_pred_label = (calibrated_test_probs >= test_decision_threshold).astype(int)
        test_metrics = safe_binary_metrics(y_test, calibrated_test_probs, threshold=0.5)
        test_metrics["accuracy"] = float((test_pred_label == y_test.to_numpy(dtype=int)).mean()) if len(y_test) else 0.0

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
            decision_threshold=test_decision_threshold if market_key == "full_game" else float(threshold_info.get("decision_threshold", 0.5)),
            split_id=split.split_id,
            used_fallback=int(threshold_info.get("used_fallback", 0)),
            prob_shrink=float(threshold_info.get("prob_shrink", 0.0)),
            full_game_gate=full_game_gate,
            volatility_normalizer=volatility_normalizer if market_key == "full_game" else None,
            full_game_reliability_meta=test_reliability_meta if market_key == "full_game" else None,
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
                "ensemble_rf_weight": float(threshold_info.get("rf_weight", 0.0)),
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
                "regime_calibrators": int(threshold_info.get("regime_calibrators", 0)),
                "selection_objective": str(threshold_info.get("selection_objective", "n/a")),
                "calibrator_mode": str(threshold_info.get("calibrator_mode", "n/a")),
                "stacking_mode": str(threshold_info.get("stacking_mode", "none")),
                "bayes_strength": float(threshold_info.get("bayes_strength", 0.0)),
                "bayes_prior": float(threshold_info.get("bayes_prior", 0.5)),
                "prob_shift": float(threshold_info.get("prob_shift", 0.0)),
                "global_selection_accuracy": float(threshold_info.get("global_selection_accuracy", 0.0)),
                "global_selection_brier": float(threshold_info.get("global_selection_brier", 0.0)),
                "global_selection_logloss": float(threshold_info.get("global_selection_logloss", 0.0)),
                "threshold_objective": str(threshold_info.get("threshold_objective", "accuracy_cov")),
                "calib_roi_per_bet": float(threshold_info.get("roi_per_bet", 0.0)),
                "calib_total_return_units": float(threshold_info.get("total_return_units", 0.0)),
                "calib_priced_rows": int(threshold_info.get("priced_rows", 0)),
                "calib_priced_coverage": float(threshold_info.get("priced_coverage", 0.0)),
                "calib_mean_ev_edge": float(threshold_info.get("mean_ev_edge", 0.0)),
                "roi_min_edge": float(threshold_info.get("roi_min_edge", 0.0)),
                "meta_gate_enabled": int(threshold_info.get("meta_gate_enabled", 0)),
                "meta_gate_threshold": float(threshold_info.get("meta_gate_threshold", 0.0)),
                "meta_gate_base_accuracy": float(threshold_info.get("meta_gate_base_accuracy", 0.0)),
                "meta_gate_final_accuracy": float(threshold_info.get("meta_gate_final_accuracy", 0.0)),
                "meta_gate_base_coverage": float(threshold_info.get("meta_gate_base_coverage", 0.0)),
                "meta_gate_final_coverage": float(threshold_info.get("meta_gate_final_coverage", 0.0)),
                "momentum_penalty_mean_calib": float(threshold_info.get("momentum_penalty_mean_calib", 0.0)),
                "momentum_penalty_active_rate_calib": float(threshold_info.get("momentum_penalty_active_rate_calib", 0.0)),
                "vol_norm_enabled": int(threshold_info.get("vol_norm_enabled", 0)),
                "vol_norm_alpha": float(threshold_info.get("vol_norm_alpha", 0.0)),
                "vol_norm_threshold_bonus_cap": float(threshold_info.get("vol_norm_threshold_bonus_cap", 0.0)),
                "vol_norm_mean_calib": float(threshold_info.get("vol_norm_mean_calib", 0.0)),
                "vol_norm_high_rate_calib": float(threshold_info.get("vol_norm_high_rate_calib", 0.0)),
                "vol_norm_threshold_bonus_mean_calib": float(threshold_info.get("vol_norm_threshold_bonus_mean_calib", 0.0)),
                "vol_norm_mean_test": float(np.mean(test_volatility_scores)) if len(test_volatility_scores) else 0.0,
                "vol_norm_high_rate_test": float(np.mean(test_volatility_scores >= float(volatility_normalizer.high_threshold))) if (market_key == "full_game" and volatility_normalizer is not None and len(test_volatility_scores)) else 0.0,
                "vol_decision_enabled": int(threshold_info.get("vol_decision_enabled", 0)),
                "vol_decision_beta": float(threshold_info.get("vol_decision_beta", 0.0)),
                "vol_decision_center": float(threshold_info.get("vol_decision_center", FULL_GAME_VOL_DECISION_CENTER)),
                "vol_decision_max_shift": float(threshold_info.get("vol_decision_max_shift", 0.0)),
                "vol_decision_acc_calib": float(threshold_info.get("vol_decision_acc_calib", 0.0)),
                "vol_decision_mean_test_threshold": float(np.mean(test_decision_threshold)) if len(test_decision_threshold) else 0.5,
                "reliability_enabled": int(threshold_info.get("reliability_enabled", 0)),
                "reliability_mean_calib": float(threshold_info.get("reliability_mean_calib", 0.0)),
                "reliability_low_rate_calib": float(threshold_info.get("reliability_low_rate_calib", 0.0)),
                "reliability_conflict_rate_calib": float(threshold_info.get("reliability_conflict_rate_calib", 0.0)),
                "reliability_side_shift_mean_calib": float(threshold_info.get("reliability_side_shift_mean_calib", 0.0)),
                "reliability_mean_test": float(np.mean(test_reliability_meta.get("reliability_score", np.ones(len(test_df), dtype=float)))) if (market_key == "full_game" and isinstance(test_reliability_meta, dict)) else 0.0,
                "reliability_low_rate_test": float(np.mean(np.asarray(test_reliability_meta.get("reliability_score", np.ones(len(test_df), dtype=float)), dtype=float) < 0.45)) if (market_key == "full_game" and isinstance(test_reliability_meta, dict)) else 0.0,
                "reliability_conflict_rate_test": float(np.mean(test_reliability_meta.get("conflict_flag", np.zeros(len(test_df), dtype=float)))) if (market_key == "full_game" and isinstance(test_reliability_meta, dict)) else 0.0,
                "reliability_side_shift_mean_test": float(np.mean(np.abs(np.asarray(test_reliability_meta.get("side_shift", np.zeros(len(test_df), dtype=float)), dtype=float)))) if (market_key == "full_game" and isinstance(test_reliability_meta, dict)) else 0.0,
                "missing_pitchers_rate_test": float(missing_pitchers_rate_test),
                "test_accuracy_at_050": float(test_metrics["accuracy"]),
                "test_accuracy_dynamic_decision": float(test_metrics["accuracy"]),
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
            print("      split omitido por tamano insuficiente")
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
        metrics["reliability_enabled_split_rate"] = float(split_df["reliability_enabled"].fillna(0).mean()) if (not split_df.empty and "reliability_enabled" in split_df.columns) else 0.0
        metrics["avg_reliability_mean_calib"] = float(split_df["reliability_mean_calib"].fillna(0).mean()) if (not split_df.empty and "reliability_mean_calib" in split_df.columns) else 0.0
        metrics["avg_reliability_mean_test"] = float(split_df["reliability_mean_test"].fillna(0).mean()) if (not split_df.empty and "reliability_mean_test" in split_df.columns) else 0.0
        metrics["avg_reliability_low_rate_test"] = float(split_df["reliability_low_rate_test"].fillna(0).mean()) if (not split_df.empty and "reliability_low_rate_test" in split_df.columns) else 0.0
        metrics["avg_reliability_conflict_rate_test"] = float(split_df["reliability_conflict_rate_test"].fillna(0).mean()) if (not split_df.empty and "reliability_conflict_rate_test" in split_df.columns) else 0.0
        metrics["avg_momentum_penalty_mean_calib"] = float(split_df["momentum_penalty_mean_calib"].fillna(0).mean()) if (not split_df.empty and "momentum_penalty_mean_calib" in split_df.columns) else 0.0
        metrics["avg_momentum_penalty_active_rate_calib"] = float(split_df["momentum_penalty_active_rate_calib"].fillna(0).mean()) if (not split_df.empty and "momentum_penalty_active_rate_calib" in split_df.columns) else 0.0
        metrics["avg_calib_roi_per_bet"] = float(split_df["calib_roi_per_bet"].fillna(0).mean()) if (not split_df.empty and "calib_roi_per_bet" in split_df.columns) else 0.0
        metrics["avg_calib_priced_coverage"] = float(split_df["calib_priced_coverage"].fillna(0).mean()) if (not split_df.empty and "calib_priced_coverage" in split_df.columns) else 0.0
        metrics["avg_calib_mean_ev_edge"] = float(split_df["calib_mean_ev_edge"].fillna(0).mean()) if (not split_df.empty and "calib_mean_ev_edge" in split_df.columns) else 0.0
        metrics["threshold_objective_mode"] = str(split_df["threshold_objective"].mode().iloc[0]) if (not split_df.empty and "threshold_objective" in split_df.columns and not split_df["threshold_objective"].dropna().empty) else "accuracy_cov"
        metrics["roi_threshold_splits"] = int((split_df["threshold_objective"].astype(str) == "roi").sum()) if (not split_df.empty and "threshold_objective" in split_df.columns) else 0
        metrics["roi_threshold_split_rate"] = float((split_df["threshold_objective"].astype(str) == "roi").mean()) if (not split_df.empty and "threshold_objective" in split_df.columns) else 0.0
        metrics.update(summarize_published_roi(detail_df))
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
    selected_markets = _env_market_filter()

    global_summary = {}

    if selected_markets:
        print(f"Filtro de mercados activo ({len(selected_markets)}): {sorted(selected_markets)}")

    for market_key, cfg in TARGET_CONFIG.items():
        if selected_markets and market_key.lower() not in selected_markets:
            continue

        target_col = cfg["target_col"]
        feature_cols = get_market_feature_columns(df, market_key)

        print(f"\nWalk-forward MLB | mercado={market_key} | features={len(feature_cols)}")
        outputs = run_market_walkforward(df, market_key, target_col, feature_cols)
        saved = save_market_outputs(market_key, outputs["detail"], outputs["splits"])
        global_summary[market_key] = saved["metrics"]
        print(f"OK {market_key}: {saved['metrics']}")

    for market_key, cfg in REGRESSION_TARGET_CONFIG.items():
        if selected_markets and market_key.lower() not in selected_markets:
            continue

        target_col = cfg["target_col"]
        feature_cols = get_market_feature_columns(df, market_key)

        print(f"\nWalk-forward MLB | mercado={market_key} | features={len(feature_cols)}")
        outputs = run_regression_market_walkforward(df, market_key, target_col, feature_cols)
        saved = save_market_outputs(market_key, outputs["detail"], outputs["splits"])
        global_summary[market_key] = saved["metrics"]
        print(f"OK {market_key}: {saved['metrics']}")

    if not global_summary:
        raise RuntimeError("No markets selected for walk-forward run. Check NBA_MLB_MARKETS value.")

    summary_path = OUTPUT_DIR / "walkforward_summary_mlb.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)

    print(f"\nResumen global guardado en: {summary_path}")


if __name__ == "__main__":
    main()
