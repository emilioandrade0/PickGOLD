import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project `src` root is on sys.path so imports of shared modules work
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from xgboost import XGBRegressor
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor

# Base directory should be src root so models live in src/data or src/models consistently
BASE_DIR = SRC_ROOT
INPUT_FILE = BASE_DIR / "data" / "mlb" / "processed" / "model_ready_features_mlb.csv"
RAW_HISTORY_FILE = BASE_DIR / "data" / "mlb" / "raw" / "mlb_advanced_history.csv"

MODELS_DIR = BASE_DIR / "data" / "mlb" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = BASE_DIR / "data" / "mlb" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CONFIG = {
    "full_game": {
        "target_col": "TARGET_home_win",
        "description": "Ganador full game",
    },
    "yrfi": {
        "target_col": "TARGET_yrfi",
        "description": "YRFI / NRFI",
    },
    "f5": {
        "target_col": "TARGET_home_win_f5",
        "description": "Ganador first 5 innings",
    },
}

REGRESSION_TARGET_CONFIG = {
    "totals": {
        "target_col": "TARGET_total_runs",
        "description": "Total de carreras del juego",
    },
    "run_line": {
        "target_col": "TARGET_home_run_margin",
        "description": "Margen de carreras del local",
    },
}

NON_FEATURE_COLUMNS = {
    "game_id",
    "date",
    "time",
    "season",
    "home_team",
    "away_team",
    "TARGET_home_win",
    "TARGET_yrfi",
    "TARGET_home_win_f5",
    "TARGET_total_runs",
    "TARGET_home_run_margin",
}

MARKET_FEATURE_PRIORITY = {
    "full_game": [
        "diff_elo",
        "diff_rest_days",
        "diff_games_last_5_days",
        "diff_win_pct_L10",
        "diff_run_diff_L10",
        "diff_runs_scored_L5",
        "diff_runs_allowed_L5",
        "diff_runs_scored_std_L10",
        "diff_runs_allowed_std_L10",
        "diff_surface_win_pct_L5",
        "diff_surface_run_diff_L5",
        "diff_surface_edge",
        "diff_win_pct_L10_vs_league",
        "diff_run_diff_L10_vs_league",
        "diff_fatigue_index",
        "diff_form_power",
        "diff_pitcher_rest_days",
        "diff_pitcher_era_L5",
        "diff_pitcher_whip_L5",
        "diff_pitcher_k_bb_L5",
        "diff_pitcher_quality_start_rate_L10",
        "diff_pitcher_blowup_rate_L10",
        "diff_pitcher_era_trend",
        "diff_pitcher_whip_trend",
        "diff_pitcher_recent_quality_score",
        "diff_bullpen_runs_allowed_L5",
        "diff_bullpen_runs_allowed_L10",
        "diff_bullpen_load_L3",
        "diff_offense_vs_pitcher",
        "home_is_favorite",
    ],
    "yrfi": [
        # Núcleo duro: tasas R1 y matchup vs pitcher
        "home_r1_scored_rate_L5",
        "home_r1_scored_rate_L10",
        "away_r1_scored_rate_L5",
        "away_r1_scored_rate_L10",

        "home_r1_allowed_rate_L5",
        "home_r1_allowed_rate_L10",
        "away_r1_allowed_rate_L5",
        "away_r1_allowed_rate_L10",

        "home_pitcher_r1_allowed_rate_L5",
        "home_pitcher_r1_allowed_rate_L10",
        "away_pitcher_r1_allowed_rate_L5",
        "away_pitcher_r1_allowed_rate_L10",
        "diff_pitcher_quality_start_rate_L10",
        "diff_pitcher_blowup_rate_L10",
        "diff_pitcher_era_trend",
        "diff_pitcher_whip_trend",
        "diff_pitcher_recent_quality_score",

        "home_r1_vs_away_pitcher",
        "away_r1_vs_home_pitcher",
        "diff_r1_vs_pitcher",
        "diff_r1_vs_pitcher_L5",

        # YRFI pressure
        "yrfi_pressure_home",
        "yrfi_pressure_away",
        "diff_yrfi_pressure",
        "total_yrfi_pressure",
        "yrfi_pressure_home_L5",
        "yrfi_pressure_away_L5",
        "diff_yrfi_pressure_L5",
        "total_yrfi_pressure_L5",

        # Disponibilidad de pitchers
        # Mantener versiones de tasa global como respaldo (baja prioridad)
        "home_yrfi_rate_L10",
        "away_yrfi_rate_L10",
        "diff_yrfi_rate_L10",
    ],
    "f5": [
        "diff_elo",
        "diff_rest_days",
        "diff_win_pct_L5",
        "diff_run_diff_L5",
        "diff_f5_win_pct_L5",
        "diff_f5_diff_L5",
        "diff_surface_f5_win_pct_L5",
        "diff_f5_win_pct_L5_vs_league",
        "diff_pitcher_rest_days",
        "diff_pitcher_f5_runs_allowed_L5",
        "diff_pitcher_quality_start_rate_L10",
        "diff_pitcher_blowup_rate_L10",
        "diff_pitcher_era_trend",
        "diff_pitcher_whip_trend",
        "diff_pitcher_recent_quality_score",
        "diff_f5_vs_pitcher",
        "diff_offense_vs_pitcher",
        "diff_bullpen_runs_allowed_L5",
        "diff_bullpen_runs_allowed_L10",
        "diff_bullpen_load_L3",
        "home_is_favorite",
    ],
    "totals": [
        "home_runs_scored_L5",
        "away_runs_scored_L5",
        "diff_runs_scored_L5",
        "home_runs_allowed_L5",
        "away_runs_allowed_L5",
        "diff_runs_allowed_L5",
        "home_runs_scored_std_L10",
        "away_runs_scored_std_L10",
        "home_runs_allowed_std_L10",
        "away_runs_allowed_std_L10",
        "home_hits_L5",
        "away_hits_L5",
        "home_hits_allowed_L5",
        "away_hits_allowed_L5",
        "home_pitcher_era_L5",
        "away_pitcher_era_L5",
        "home_pitcher_whip_L5",
        "away_pitcher_whip_L5",
        "home_pitcher_hr9_L5",
        "away_pitcher_hr9_L5",
        "home_pitcher_ip_L5",
        "away_pitcher_ip_L5",
        "home_pitcher_f5_runs_allowed_L5",
        "away_pitcher_f5_runs_allowed_L5",
        "home_bullpen_runs_allowed_L5",
        "away_bullpen_runs_allowed_L5",
        "home_bullpen_load_L3",
        "away_bullpen_load_L3",
        "home_offense_vs_away_pitcher",
        "away_offense_vs_home_pitcher",
        "avg_park_factor",
        "weather_temp",
        "weather_wind",
        "weather_humidity",
        "odds_over_under",
        "market_missing",
        "both_pitchers_available",
    ],
    "run_line": [
        "diff_elo",
        "diff_rest_days",
        "diff_games_last_5_days",
        "diff_win_pct_L5",
        "diff_win_pct_L10",
        "diff_run_diff_L5",
        "diff_run_diff_L10",
        "diff_runs_scored_L5",
        "diff_runs_allowed_L5",
        "diff_surface_win_pct_L5",
        "diff_surface_run_diff_L5",
        "diff_form_power",
        "diff_regression_risk",
        "diff_bounce_back_signal",
        "favorite_trap_signal",
        "underdog_upset_signal",
        "diff_pitcher_rest_days",
        "diff_pitcher_era_L5",
        "diff_pitcher_whip_L5",
        "diff_pitcher_k_bb_L5",
        "diff_pitcher_hr9_L5",
        "diff_pitcher_ip_L5",
        "diff_pitcher_runs_allowed_L5",
        "diff_pitcher_runs_allowed_L10",
        "diff_pitcher_start_win_rate_L10",
        "diff_bullpen_runs_allowed_L5",
        "diff_bullpen_runs_allowed_L10",
        "diff_bullpen_load_L3",
        "diff_offense_vs_pitcher",
        "home_is_favorite",
        "odds_over_under",
        "market_missing",
        "both_pitchers_available",
    ],
}

MARKET_THRESHOLD_RANGES = {
    "full_game": np.arange(0.45, 0.621, 0.01),
    "yrfi": np.arange(0.35, 0.601, 0.01),
    "f5": np.arange(0.45, 0.671, 0.01),
}

MARKET_WEIGHT_GRID = {
    "full_game": np.arange(0.20, 0.81, 0.05),
    "yrfi": np.arange(0.25, 0.76, 0.05),
    "f5": np.arange(0.20, 0.81, 0.05),
}

# Optional positive-rate constraints per market to avoid degenerate thresholds
MARKET_POS_RATE_LIMITS = {
    # stricter positive-rate limits for yrfi to avoid degenerate thresholds
    # use 30%..70% as requested
    "yrfi": (0.30, 0.70),
}


class ConstantBinaryModel:
    def __init__(self, prob_class_1: float):
        self.prob_class_1 = float(min(max(prob_class_1, 0.0), 1.0))

    def predict_proba(self, X):
        n = len(X)
        p1 = np.full(n, self.prob_class_1, dtype=float)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


def load_dataset() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"No existe el archivo de features MLB: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    if RAW_HISTORY_FILE.exists():
        raw_cols = [
            "game_id",
            "date",
            "home_runs_total",
            "away_runs_total",
        ]
        raw_df = pd.read_csv(RAW_HISTORY_FILE, usecols=raw_cols, dtype={"game_id": str})
        raw_df["date"] = raw_df["date"].astype(str)
        raw_df["TARGET_total_runs"] = (
            pd.to_numeric(raw_df["home_runs_total"], errors="coerce").fillna(0)
            + pd.to_numeric(raw_df["away_runs_total"], errors="coerce").fillna(0)
        )
        raw_df["TARGET_home_run_margin"] = (
            pd.to_numeric(raw_df["home_runs_total"], errors="coerce").fillna(0)
            - pd.to_numeric(raw_df["away_runs_total"], errors="coerce").fillna(0)
        )
        df["game_id"] = df["game_id"].astype(str)
        df["date"] = df["date"].astype(str)
        df = df.merge(
            raw_df[["game_id", "date", "TARGET_total_runs", "TARGET_home_run_margin"]],
            on=["game_id", "date"],
            how="left",
        )
    df["date"] = df["date"].astype(str)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt"]).sort_values(["date_dt", "game_id"]).reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS and c != "date_dt"]


def _drop_constant_feature_columns(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    keep: List[str] = []
    removed: List[str] = []

    for col in feature_cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.nunique(dropna=True) <= 1:
            removed.append(col)
            continue
        keep.append(col)

    if removed:
        preview = removed[:8]
        suffix = "..." if len(removed) > 8 else ""
        print(f"   ⚠️ Features constantes omitidas ({len(removed)}): {preview}{suffix}")

    return keep


def get_market_feature_columns(df: pd.DataFrame, market_key: str) -> List[str]:
    all_feature_cols = get_feature_columns(df)
    all_cols = set(all_feature_cols)
    preferred = MARKET_FEATURE_PRIORITY.get(market_key, [])
    selected = [c for c in preferred if c in all_cols]

    # fallback de seguridad
    if len(selected) < 8:
        selected = all_feature_cols

    selected = _drop_constant_feature_columns(df, selected)

    if len(selected) < 8:
        selected = _drop_constant_feature_columns(df, all_feature_cols)

    return selected


def time_based_split(df: pd.DataFrame, valid_fraction: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 100:
        train_df, valid_df = train_test_split(df, test_size=valid_fraction, shuffle=False)
        return train_df.copy(), valid_df.copy()

    split_idx = int(len(df) * (1 - valid_fraction))
    split_idx = max(50, min(split_idx, len(df) - 20))

    train_df = df.iloc[:split_idx].copy()
    valid_df = df.iloc[split_idx:].copy()
    return train_df, valid_df


def get_scale_pos_weight(y: pd.Series) -> float:
    positives = float((y == 1).sum())
    negatives = float((y == 0).sum())
    if positives == 0:
        return 1.0
    return max(1.0, negatives / positives)


def build_xgb(scale_pos_weight: float, market_key: str) -> XGBClassifier:
    if market_key == "full_game":
        return XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=2,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
        )

    if market_key == "yrfi":
        return XGBClassifier(
            n_estimators=180,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.78,
            colsample_bytree=0.70,
            reg_alpha=0.50,
            reg_lambda=2.0,
            min_child_weight=6,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
        )

    return XGBClassifier(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.2,
        min_child_weight=3,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
    )


def build_lgbm(scale_pos_weight: float, market_key: str) -> LGBMClassifier:
    if market_key == "full_game":
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.8,
            objective="binary",
            random_state=42,
            n_jobs=-1,
            class_weight=None if scale_pos_weight <= 1 else {0: 1.0, 1: scale_pos_weight},
            verbose=-1,
        )

    if market_key == "yrfi":
        return LGBMClassifier(
            n_estimators=220,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=-1,
            min_child_samples=40,
            subsample=0.78,
            colsample_bytree=0.70,
            reg_alpha=0.45,
            reg_lambda=1.8,
            objective="binary",
            random_state=42,
            n_jobs=-1,
            class_weight=None if scale_pos_weight <= 1 else {0: 1.0, 1: scale_pos_weight},
            verbose=-1,
        )

    return LGBMClassifier(
        n_estimators=400,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.15,
        reg_lambda=1.0,
        objective="binary",
        random_state=42,
        n_jobs=-1,
        class_weight=None if scale_pos_weight <= 1 else {0: 1.0, 1: scale_pos_weight},
        verbose=-1,
    )


def build_xgb_regressor(market_key: str) -> XGBRegressor:
    if market_key == "totals":
        return XGBRegressor(
            n_estimators=320,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.88,
            colsample_bytree=0.88,
            reg_alpha=0.15,
            reg_lambda=1.0,
            min_child_weight=2,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )

    return XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=2,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )


def build_lgbm_regressor(market_key: str) -> LGBMRegressor:
    if market_key == "totals":
        return LGBMRegressor(
            n_estimators=320,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.88,
            colsample_bytree=0.88,
            reg_alpha=0.1,
            reg_lambda=0.8,
            objective="regression",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    return LGBMRegressor(
        n_estimators=320,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.8,
        objective="regression",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def evaluate_probs(y_true: pd.Series, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "logloss": float(log_loss(y_true, probs, labels=[0, 1])),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
    except Exception:
        metrics["roc_auc"] = 0.5
    return metrics


def evaluate_regression(y_true: pd.Series, preds: np.ndarray) -> Dict[str, float]:
    y_true_numeric = pd.to_numeric(y_true, errors="coerce").fillna(0.0)
    preds_arr = np.asarray(preds, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true_numeric, preds_arr)))
    return {
        "mae": float(mean_absolute_error(y_true_numeric, preds_arr)),
        "rmse": rmse,
        "r2": float(r2_score(y_true_numeric, preds_arr)),
    }


def choose_best_threshold(y_true: pd.Series, probs: np.ndarray, market_key: str) -> Dict[str, float]:
    candidates = MARKET_THRESHOLD_RANGES.get(market_key, np.arange(0.35, 0.651, 0.01))

    best = {
        "threshold": 0.50,
        "score": -999.0,
        "accuracy": 0.0,
        "positive_rate": 0.0,
    }

    for thr in candidates:
        preds = (probs >= thr).astype(int)
        acc = float(accuracy_score(y_true, preds))
        positive_rate = float(preds.mean())

        # enforce market-specific positive-rate constraints (skip degenerate thresholds)
        pr_limits = MARKET_POS_RATE_LIMITS.get(market_key)
        if pr_limits is not None:
            low, high = pr_limits
            if positive_rate < float(low) or positive_rate > float(high):
                continue

        if market_key == "full_game":
            balance_penalty = abs(positive_rate - 0.5) * 0.02
        elif market_key == "yrfi":
            balance_penalty = abs(positive_rate - 0.5) * 0.04
        else:  # f5
            balance_penalty = abs(positive_rate - 0.5) * 0.025

        score = acc - balance_penalty

        if score > best["score"]:
            best = {
                "threshold": float(round(thr, 2)),
                "score": score,
                "accuracy": acc,
                "positive_rate": positive_rate,
            }

    return best


def choose_best_ensemble_params(
    y_true: pd.Series,
    xgb_probs: np.ndarray,
    lgbm_probs: np.ndarray,
    market_key: str,
) -> Dict[str, float]:
    best = {
        "xgb_weight": 0.5,
        "lgbm_weight": 0.5,
        "threshold": 0.5,
        "score": -999.0,
        "accuracy": 0.0,
        "positive_rate": 0.0,
    }

    weight_grid = MARKET_WEIGHT_GRID.get(market_key, np.arange(0.2, 0.81, 0.05))

    for w in weight_grid:
        probs = w * xgb_probs + (1 - w) * lgbm_probs
        th_info = choose_best_threshold(y_true, probs, market_key)
        score = th_info["score"]

        if score > best["score"]:
            best = {
                "xgb_weight": float(round(w, 2)),
                "lgbm_weight": float(round(1 - w, 2)),
                "threshold": float(th_info["threshold"]),
                "score": float(score),
                "accuracy": float(th_info["accuracy"]),
                "positive_rate": float(th_info["positive_rate"]),
            }

    return best


def train_single_market(df: pd.DataFrame, market_key: str, target_col: str, feature_cols: List[str]) -> Dict:
    market_dir = MODELS_DIR / market_key
    market_dir.mkdir(parents=True, exist_ok=True)

    market_df = df.dropna(subset=[target_col]).copy()

    X = market_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = market_df[target_col].astype(int)

    split_df = market_df.copy()
    split_df[feature_cols] = X

    train_df, valid_df = time_based_split(split_df, valid_fraction=0.2)

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train_df[target_col].astype(int)

    X_valid = valid_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_valid = valid_df[target_col].astype(int)

    scale_pos_weight = get_scale_pos_weight(y_train)

    xgb_model = build_xgb(scale_pos_weight, market_key)
    lgbm_model = build_lgbm(scale_pos_weight, market_key)

    print(f"\n🚀 Entrenando mercado MLB: {market_key}")
    print(f"   Target        : {target_col}")
    print(f"   Train rows    : {len(X_train)}")
    print(f"   Valid rows    : {len(X_valid)}")
    print(f"   Features      : {len(feature_cols)}")
    print(f"   Pos weight    : {scale_pos_weight:.3f}")

    if y_train.nunique() < 2:
        fixed_prob = 1.0 if int(y_train.iloc[0]) == 1 else 0.0
        print(f"   ⚠️ Target single-class en train ({int(y_train.iloc[0])}). Usando modelo constante.")
        xgb_model = ConstantBinaryModel(fixed_prob)
        lgbm_model = ConstantBinaryModel(fixed_prob)
    else:
        xgb_model.fit(X_train, y_train)
        lgbm_model.fit(X_train, y_train)

    xgb_valid_probs = xgb_model.predict_proba(X_valid)[:, 1]
    lgbm_valid_probs = lgbm_model.predict_proba(X_valid)[:, 1]
    ensemble_params = choose_best_ensemble_params(
        y_valid,
        xgb_valid_probs,
        lgbm_valid_probs,
        market_key,
    )
    best_threshold = ensemble_params["threshold"]
    ensemble_valid_probs = (
        ensemble_params["xgb_weight"] * xgb_valid_probs
        + ensemble_params["lgbm_weight"] * lgbm_valid_probs
    )

    xgb_metrics = evaluate_probs(y_valid, xgb_valid_probs, 0.50)
    lgbm_metrics = evaluate_probs(y_valid, lgbm_valid_probs, 0.50)
    ensemble_metrics = evaluate_probs(y_valid, ensemble_valid_probs, best_threshold)

    xgb_path = market_dir / "xgb_model.pkl"
    lgbm_path = market_dir / "lgbm_model.pkl"
    feature_path = market_dir / "feature_columns.json"
    metadata_path = market_dir / "metadata.json"

    joblib.dump(xgb_model, xgb_path)
    joblib.dump(lgbm_model, lgbm_path)

    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2, ensure_ascii=False)

    xgb_importance_arr = getattr(xgb_model, "feature_importances_", np.zeros(len(feature_cols)))
    lgbm_importance_arr = getattr(lgbm_model, "feature_importances_", np.zeros(len(feature_cols)))

    xgb_importance = dict(zip(feature_cols, map(float, xgb_importance_arr)))
    lgbm_importance = dict(zip(feature_cols, map(float, lgbm_importance_arr)))

    combined_importance = []
    for col in feature_cols:
        xi = float(xgb_importance.get(col, 0.0))
        li = float(lgbm_importance.get(col, 0.0))
        combined_importance.append((col, (xi + li) / 2.0))

    combined_importance = sorted(combined_importance, key=lambda x: x[1], reverse=True)[:25]

    metadata = {
        "market_key": market_key,
        "target_col": target_col,
        "train_rows": int(len(X_train)),
        "valid_rows": int(len(X_valid)),
        "feature_count": int(len(feature_cols)),
        "ensemble_threshold": float(best_threshold),
        "ensemble_weights": {
            "xgboost": float(ensemble_params["xgb_weight"]),
            "lightgbm": float(ensemble_params["lgbm_weight"]),
        },
        "metrics": {
            "xgboost_valid": xgb_metrics,
            "lightgbm_valid": lgbm_metrics,
            "ensemble_valid": ensemble_metrics,
        },
        "top_features": [
            {"feature": col, "importance": float(score)}
            for col, score in combined_importance
        ],
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"   ✅ Guardado XGB   : {xgb_path}")
    print(f"   ✅ Guardado LGBM  : {lgbm_path}")
    print(f"   ✅ Threshold ens. : {best_threshold}")
    print(f"   ✅ Accuracy ens.  : {ensemble_metrics['accuracy']:.4f}")
    print(f"   ✅ LogLoss ens.   : {ensemble_metrics['logloss']:.4f}")
    print(f"   ✅ ROC AUC ens.   : {ensemble_metrics['roc_auc']:.4f}")

    return {
        "market_key": market_key,
        "target_col": target_col,
        "threshold": best_threshold,
        "xgb_metrics": xgb_metrics,
        "lgbm_metrics": lgbm_metrics,
        "ensemble_metrics": ensemble_metrics,
        "top_features": metadata["top_features"][:10],
    }


def train_single_regression_market(
    df: pd.DataFrame,
    market_key: str,
    target_col: str,
    feature_cols: List[str],
) -> Dict:
    market_dir = MODELS_DIR / market_key
    market_dir.mkdir(parents=True, exist_ok=True)

    market_df = df.dropna(subset=[target_col]).copy()
    X = market_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = pd.to_numeric(market_df[target_col], errors="coerce").fillna(0.0)

    split_df = market_df.copy()
    split_df[feature_cols] = X
    split_df[target_col] = y

    train_df, valid_df = time_based_split(split_df, valid_fraction=0.2)
    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = pd.to_numeric(train_df[target_col], errors="coerce").fillna(0.0)
    X_valid = valid_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_valid = pd.to_numeric(valid_df[target_col], errors="coerce").fillna(0.0)

    xgb_model = build_xgb_regressor(market_key)
    lgbm_model = build_lgbm_regressor(market_key)

    print(f"\n🚀 Entrenando mercado MLB: {market_key}")
    print(f"   Target        : {target_col}")
    print(f"   Train rows    : {len(X_train)}")
    print(f"   Valid rows    : {len(X_valid)}")
    print(f"   Features      : {len(feature_cols)}")

    xgb_model.fit(X_train, y_train)
    lgbm_model.fit(X_train, y_train)

    xgb_valid_preds = xgb_model.predict(X_valid)
    lgbm_valid_preds = lgbm_model.predict(X_valid)
    ensemble_valid_preds = (0.5 * xgb_valid_preds) + (0.5 * lgbm_valid_preds)

    xgb_metrics = evaluate_regression(y_valid, xgb_valid_preds)
    lgbm_metrics = evaluate_regression(y_valid, lgbm_valid_preds)
    ensemble_metrics = evaluate_regression(y_valid, ensemble_valid_preds)

    xgb_path = market_dir / "xgb_model.pkl"
    lgbm_path = market_dir / "lgbm_model.pkl"
    feature_path = market_dir / "feature_columns.json"
    metadata_path = market_dir / "metadata.json"

    joblib.dump(xgb_model, xgb_path)
    joblib.dump(lgbm_model, lgbm_path)

    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2, ensure_ascii=False)

    xgb_importance_arr = getattr(xgb_model, "feature_importances_", np.zeros(len(feature_cols)))
    lgbm_importance_arr = getattr(lgbm_model, "feature_importances_", np.zeros(len(feature_cols)))

    combined_importance = []
    for col, xi, li in zip(feature_cols, xgb_importance_arr, lgbm_importance_arr):
        combined_importance.append((col, (float(xi) + float(li)) / 2.0))
    combined_importance = sorted(combined_importance, key=lambda x: x[1], reverse=True)[:25]

    metadata = {
        "market_key": market_key,
        "target_col": target_col,
        "task_type": "regression",
        "train_rows": int(len(X_train)),
        "valid_rows": int(len(X_valid)),
        "feature_count": int(len(feature_cols)),
        "ensemble_weights": {
            "xgboost": 0.5,
            "lightgbm": 0.5,
        },
        "metrics": {
            "xgboost_valid": xgb_metrics,
            "lightgbm_valid": lgbm_metrics,
            "ensemble_valid": ensemble_metrics,
        },
        "top_features": [
            {"feature": col, "importance": float(score)}
            for col, score in combined_importance
        ],
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"   ✅ Guardado XGB   : {xgb_path}")
    print(f"   ✅ Guardado LGBM  : {lgbm_path}")
    print(f"   ✅ MAE ens.       : {ensemble_metrics['mae']:.4f}")
    print(f"   ✅ RMSE ens.      : {ensemble_metrics['rmse']:.4f}")
    print(f"   ✅ R2 ens.        : {ensemble_metrics['r2']:.4f}")

    return {
        "market_key": market_key,
        "target_col": target_col,
        "xgb_metrics": xgb_metrics,
        "lgbm_metrics": lgbm_metrics,
        "ensemble_metrics": ensemble_metrics,
        "top_features": metadata["top_features"][:10],
        "task_type": "regression",
    }


def train_all_models() -> Dict:
    df = load_dataset()
    feature_cols = get_feature_columns(df)

    print("📦 Dataset MLB cargado")
    print(f"   Filas totales : {len(df)}")
    print(f"   Features      : {len(feature_cols)}")
    print(f"   Archivo       : {INPUT_FILE}")

    results = {}

    for market_key, config in TARGET_CONFIG.items():
        market_feature_cols = get_market_feature_columns(df, market_key)

        print(f"\n🧩 Features seleccionadas para {market_key}: {len(market_feature_cols)}")

        result = train_single_market(
            df=df,
            market_key=market_key,
            target_col=config["target_col"],
            feature_cols=market_feature_cols,
        )
        results[market_key] = result

    for market_key, config in REGRESSION_TARGET_CONFIG.items():
        market_feature_cols = get_market_feature_columns(df, market_key)

        print(f"\n🧩 Features seleccionadas para {market_key}: {len(market_feature_cols)}")

        result = train_single_regression_market(
            df=df,
            market_key=market_key,
            target_col=config["target_col"],
            feature_cols=market_feature_cols,
        )
        results[market_key] = result

    summary_path = REPORTS_DIR / "training_summary_mlb.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n📊 RESUMEN FINAL MLB")
    for market_key, result in results.items():
        metrics = result["ensemble_metrics"]
        if result.get("task_type") == "regression":
            print(
                f"   {market_key.upper():<10} | "
                f"MAE: {metrics['mae']:.4f} | "
                f"RMSE: {metrics['rmse']:.4f} | "
                f"R2: {metrics['r2']:.4f}"
            )
        else:
            print(
                f"   {market_key.upper():<10} | "
                f"ACC: {metrics['accuracy']:.4f} | "
                f"LOGLOSS: {metrics['logloss']:.4f} | "
                f"AUC: {metrics['roc_auc']:.4f} | "
                f"THR: {result['threshold']:.2f}"
            )

    print(f"\n💾 Resumen guardado en: {summary_path}")
    return results


if __name__ == "__main__":
    train_all_models()
