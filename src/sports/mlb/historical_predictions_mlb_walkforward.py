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
        "min_threshold": 0.56,
        "max_threshold": 0.66,
        "min_coverage": 0.02,
        "target_coverage": 0.075,
        "min_published_rows": 14,
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
        for v in _env_float_list("NBA_MLB_FULL_GAME_XGB_WEIGHT_GRID", [0.20, 0.35, 0.50, 0.65, 0.80])
    }
)
if len(FULL_GAME_XGB_WEIGHT_GRID) == 0:
    FULL_GAME_XGB_WEIGHT_GRID = [0.20, 0.35, 0.50, 0.65, 0.80]

_cal_mode_raw = _env_str("NBA_MLB_FULL_GAME_CALIBRATOR_MODE", "auto").lower()
if _cal_mode_raw not in {"auto", "global_lr", "regime_aware"}:
    _cal_mode_raw = "auto"
FULL_GAME_CALIBRATOR_MODE = _cal_mode_raw

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

    regime = np.full(len(df_context), "fg_normal", dtype=object)
    regime[~has_pitchers] = "fg_missing_pitchers"
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
        "diff_elo", "home_elo_pre", "away_elo_pre", "diff_rest_days", "diff_games_last_5_days", "diff_win_pct_L10",
        "diff_run_diff_L10", "diff_runs_scored_L5", "diff_runs_allowed_L5",
        "diff_games_in_season_before", "diff_prev_win_pct", "diff_prev_run_diff_pg",
        "prev_season_data_available",
        "diff_prev_runs_scored_pg", "diff_prev_runs_allowed_pg", "diff_win_pct_L10_blend",
        "diff_run_diff_L10_blend", "diff_runs_scored_L5_blend", "diff_runs_allowed_L5_blend",
        "diff_runs_scored_std_L10", "diff_runs_allowed_std_L10", "diff_surface_win_pct_L5",
        "diff_surface_run_diff_L5", "diff_surface_edge", "diff_win_pct_L10_vs_league",
        "diff_run_diff_L10_vs_league", "diff_fatigue_index", "diff_form_power",
        "diff_pitcher_data_available", "diff_pitcher_rest_days", "diff_pitcher_runs_allowed_L5",
        "diff_pitcher_runs_allowed_L10", "diff_pitcher_start_win_rate_L10", "diff_bullpen_runs_allowed_L5",
        "diff_bullpen_runs_allowed_L10", "diff_bullpen_load_L3", "diff_offense_vs_pitcher",
        "park_factor_delta", "park_offense_pressure", "park_bullpen_pressure", "park_form_pressure",
        "umpire_zone_delta", "umpire_sample_log",
        "home_is_favorite", "odds_over_under", "open_line", "current_line", "line_movement",
        "open_total", "current_total", "total_movement", "current_home_moneyline",
        "current_away_moneyline", "current_total_line", "bookmakers_count", "snapshot_count",
        "market_moneyline_gap", "market_total_delta", "market_line_velocity", "market_missing",
        "market_micro_missing", "both_pitchers_available",
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
) -> Tuple[WeightedEnsembleModel, object, Dict[str, float], pd.DataFrame, FullGamePlayabilityGate | None]:
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
    }
    best_calib_probs = None
    best_calibrator = None
    full_game_gate: FullGamePlayabilityGate | None = None
    best_rf_weight = 0.0
    best_stack_model: LogisticRegression | None = None

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

            y_sel = y_calib_arr
            sel_probs = np.clip(selection_probs, 1e-6, 1.0 - 1e-6)
            sel_pred = (sel_probs >= 0.5).astype(int)
            global_acc, global_brier, global_score = full_game_global_score_from_probs(
                sel_probs,
                y_sel,
                brier_weight=FULL_GAME_GLOBAL_BRIER_WEIGHT,
            )
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
                selection_objective = "global_acc050_brier_soft"
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
                    "global_selection_accuracy": float(global_acc),
                    "global_selection_brier": float(global_brier),
                    "global_selection_logloss": float(global_logloss),
                    "calibrator_mode": "global_lr",
                    "stacking_mode": "none",
                    "bayes_prior": bayes_prior,
                    "bayes_strength": float(bayes_strength),
                    "bayes_sample_size": int(len(y_calib_arr)),
                    "prob_shift": 0.0,
                    "decision_threshold": 0.5,
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

        best["decision_threshold"] = 0.5

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
    )

    calib_decision_threshold = float(best.get("decision_threshold", 0.5))
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
            "publish_pick": calib_publish_pick.astype(int),
            "publish_threshold_effective": calib_effective_threshold.astype(float),
            "used_fallback": int(best.get("used_fallback", 0)),
        }
    )
    if market_key == "full_game":
        calib_detail["full_game_gate_keep"] = gate_keep.astype(int)
        calib_detail["full_game_gate_score"] = gate_score.astype(float)

    published_calib = calib_detail[calib_detail["publish_pick"] == 1].copy()
    best["published_accuracy"] = float((published_calib["pred_label"].astype(int) == published_calib["y_true"].astype(int)).mean()) if len(published_calib) else 0.0
    best["published_coverage"] = float(calib_detail["publish_pick"].mean()) if len(calib_detail) else 0.0
    best["fallback_penalty"] = float(calib_policy_meta.get("fallback_penalty", 0.0))
    best["missing_pitcher_penalty"] = float(calib_policy_meta.get("missing_pitcher_penalty", 0.0))
    best["missing_pitchers_rate_calib"] = float(calib_policy_meta.get("missing_pitchers_rate", 0.0))
    best["momentum_penalty_mean_calib"] = float(calib_policy_meta.get("momentum_penalty_mean", 0.0))
    best["momentum_penalty_active_rate_calib"] = float(calib_policy_meta.get("momentum_penalty_active_rate", 0.0))

    return ensemble_model, best_calibrator, best, calib_detail, full_game_gate


def build_prediction_rows(
    df_test: pd.DataFrame,
    market_key: str,
    probs_raw: np.ndarray,
    probs_calibrated: np.ndarray,
    threshold: float,
    split_id: int,
    used_fallback: int,
    prob_shrink: float,
    full_game_gate: FullGamePlayabilityGate | None = None,
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

    if market_key == "full_game" and full_game_gate is not None:
        gate_keep, gate_score = full_game_gate.predict_mask(df_test, probs_calibrated)
        publish_pick = (publish_pick.astype(int) * gate_keep.astype(int)).astype(int)
        out["full_game_gate_keep"] = gate_keep.astype(int)
        out["full_game_gate_score"] = gate_score.astype(float)

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

        ensemble_model, calibrator, threshold_info, calib_detail, full_game_gate = choose_ensemble_and_threshold(
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
            f"meta_gate={threshold_info.get('meta_gate_enabled', 0)}"
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
            full_game_gate=full_game_gate,
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
                "meta_gate_enabled": int(threshold_info.get("meta_gate_enabled", 0)),
                "meta_gate_threshold": float(threshold_info.get("meta_gate_threshold", 0.0)),
                "meta_gate_base_accuracy": float(threshold_info.get("meta_gate_base_accuracy", 0.0)),
                "meta_gate_final_accuracy": float(threshold_info.get("meta_gate_final_accuracy", 0.0)),
                "meta_gate_base_coverage": float(threshold_info.get("meta_gate_base_coverage", 0.0)),
                "meta_gate_final_coverage": float(threshold_info.get("meta_gate_final_coverage", 0.0)),
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
