import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import sys
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = SRC_ROOT
INPUT_FILE = BASE_DIR / "data" / "liga_mx" / "processed" / "model_ready_features_liga_mx.csv"

MODELS_DIR = BASE_DIR / "data" / "liga_mx" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = BASE_DIR / "data" / "liga_mx" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CONFIG = {
    "full_game": {
        "target_col": "TARGET_full_game",
        "description": "Ganador/Empate (win/draw/loss)",
        "problem_type": "multiclass",
        "num_classes": 3,
    },
    "over_25": {
        "target_col": "TARGET_over_25",
        "description": "Over/Under 2.5 goles",
        "problem_type": "binary",
        "num_classes": 2,
    },
    "btts": {
        "target_col": "TARGET_btts",
        "description": "Both Teams To Score",
        "problem_type": "binary",
        "num_classes": 2,
    },
    "corners_over_95": {
        "target_col": "TARGET_corners_over_95",
        "description": "Corners Over/Under 9.5",
        "problem_type": "binary",
        "num_classes": 2,
    },
}

NON_FEATURE_COLUMNS = {
    "game_id",
    "date",
    "season",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "is_draw",
    "total_goals",
    "TARGET_full_game",
    "TARGET_over_25",
    "TARGET_btts",
    "TARGET_corners_over_95",
}

ADVANCED_EXPERIMENTAL_FEATURES = {
    "home_games_last_10_days", "away_games_last_10_days", "diff_games_last_10_days",
    "home_games_last_14_days", "away_games_last_14_days", "diff_games_last_14_days",
    "home_goal_diff_L3", "away_goal_diff_L3", "diff_goal_diff_L3",
    "home_goals_scored_L3", "away_goals_scored_L3", "diff_goals_scored_L3",
    "home_goals_scored_L10", "away_goals_scored_L10", "diff_goals_scored_L10",
    "home_goals_allowed_L3", "away_goals_allowed_L3", "diff_goals_allowed_L3",
    "home_goals_allowed_L10", "away_goals_allowed_L10", "diff_goals_allowed_L10",
    "home_goal_diff_std_L10", "away_goal_diff_std_L10", "diff_goal_diff_std_L10",
    "home_goals_scored_std_L10", "away_goals_scored_std_L10", "diff_goals_scored_std_L10",
    "home_goals_allowed_std_L10", "away_goals_allowed_std_L10", "diff_goals_allowed_std_L10",
    "home_schedule_load_exp", "away_schedule_load_exp", "diff_schedule_load_exp",
    "home_attack_closing_trend", "away_attack_closing_trend", "diff_attack_closing_trend",
    "home_defense_closing_trend", "away_defense_closing_trend", "diff_defense_closing_trend",
    "home_match_stability_L10", "away_match_stability_L10", "diff_match_stability_L10",
    "draw_equilibrium_index",
}


class ConstantBinaryModel:
    """Fallback model for single-class binary targets."""

    def __init__(self, constant_class: int):
        self.constant_class = int(constant_class)

    def predict_proba(self, X):
        n = len(X)
        probs = np.zeros((n, 2), dtype=float)
        probs[:, self.constant_class] = 1.0
        return probs


def _binary_positive_proba(model, X: pd.DataFrame) -> np.ndarray:
    probs = model.predict_proba(X)
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 1:
        return probs
    if probs.shape[1] == 1:
        return probs[:, 0]
    return probs[:, 1]


def _full_game_two_stage_predict_proba(model_pack: dict, X: pd.DataFrame) -> np.ndarray:
    p_draw = _binary_positive_proba(model_pack["draw_model"], X)
    p_home_cond = _binary_positive_proba(model_pack["winner_model"], X)
    p_home_cond = np.clip(p_home_cond, 1e-9, 1 - 1e-9)
    p_away_cond = 1.0 - p_home_cond

    p_no_draw = np.clip(1.0 - p_draw, 0.0, 1.0)
    away = p_no_draw * p_away_cond
    home = p_no_draw * p_home_cond
    draw = np.clip(p_draw, 0.0, 1.0)

    probs = np.column_stack([away, home, draw]).astype(float)
    probs = np.clip(probs, 1e-9, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def _safe_feature_importance(model, feature_cols: List[str]):
    values = getattr(model, "feature_importances_", None)
    if values is None:
        values = np.zeros(len(feature_cols), dtype=float)
    return dict(zip(feature_cols, map(float, values)))


def load_dataset() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"No existe el archivo de features Liga MX: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, dtype={"game_id": str})
    df["date"] = df["date"].astype(str)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt"]).sort_values(["date_dt", "game_id"]).reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS and c != "date_dt"]


def time_based_split(df: pd.DataFrame, valid_fraction: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporal: Ãºltimos N% para validaciÃ³n."""
    if len(df) < 100:
        train_df, valid_df = train_test_split(df, test_size=valid_fraction, shuffle=False)
        return train_df.copy(), valid_df.copy()

    split_idx = int(len(df) * (1 - valid_fraction))
    split_idx = max(50, min(split_idx, len(df) - 20))

    train_df = df.iloc[:split_idx].copy()
    valid_df = df.iloc[split_idx:].copy()
    return train_df, valid_df


def get_scale_pos_weight(y: pd.Series) -> float:
    """Calcula scale_pos_weight para dataset desbalanceado."""
    positives = float((y == 1).sum())
    negatives = float((y == 0).sum())
    if positives == 0:
        return 1.0
    return max(1.0, negatives / positives)


def build_xgb_binary(scale_pos_weight: float) -> XGBClassifier:
    """XGBoost para problemas binarios (Over/Under, BTTS)."""
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=2,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
    )


def build_xgb_multiclass() -> XGBClassifier:
    """XGBoost para problemas multiclase (Full Game con 3 clases)."""
    return XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=3,
        n_estimators=450,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.2,
        min_child_weight=2,
        random_state=7,
        n_jobs=-1,
    )


def build_lgbm_binary(scale_pos_weight: float) -> LGBMClassifier:
    """LightGBM para problemas binarios."""
    return LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=15,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def build_lgbm_multiclass() -> LGBMClassifier:
    """LightGBM para problemas multiclase."""
    return LGBMClassifier(
        objective="multiclass",
        metric="multi_logloss",
        num_class=3,
        n_estimators=520,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=16,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.08,
        reg_lambda=1.2,
        random_state=7,
        n_jobs=-1,
        verbose=-1,
    )


def build_lgbm_secondary_binary(scale_pos_weight: float) -> LGBMClassifier:
    """LightGBM secundario para problemas binarios (hiperparÃ¡metros alternativos)."""
    return LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=280,
        learning_rate=0.08,
        num_leaves=25,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=123,
        n_jobs=-1,
        verbose=-1,
    )


def build_lgbm_secondary_multiclass() -> LGBMClassifier:
    """LightGBM secundario para problemas multiclase (hiperparÃ¡metros alternativos)."""
    return LGBMClassifier(
        objective="multiclass",
        metric="multi_logloss",
        num_class=3,
        n_estimators=320,
        learning_rate=0.06,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=10,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.25,
        reg_lambda=1.2,
        random_state=7,
        n_jobs=-1,
        verbose=-1,
    )


def evaluate_probs_binary(y_true: pd.Series, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    """EvalÃºa predicciones binarias."""
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


def evaluate_probs_multiclass(y_true: pd.Series, probs: np.ndarray) -> Dict[str, float]:
    """EvalÃºa predicciones multiclase."""
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 1e-12, None)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs = probs / row_sums

    preds = np.argmax(probs, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "logloss": float(log_loss(y_true, probs, labels=[0, 1, 2])),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs, multi_class="ovr", labels=[0, 1, 2]))
    except Exception:
        metrics["roc_auc"] = 0.0
    return metrics


def choose_best_threshold_binary(y_true: pd.Series, probs: np.ndarray) -> Dict[str, float]:
    """Elige threshold Ã³ptimo para binario."""
    candidates = np.arange(0.35, 0.651, 0.01)

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
        score = acc

        if score > best["score"]:
            best = {
                "threshold": float(round(thr, 2)),
                "score": score,
                "accuracy": acc,
                "positive_rate": positive_rate,
            }

    return best


def choose_best_ensemble_weights_multiclass(
    y_true: pd.Series,
    xgb_probs: np.ndarray,
    lgbm_probs: np.ndarray,
    lgbm_sec_probs: np.ndarray,
) -> Dict[str, float]:
    """Busca pesos de ensemble para maximizar accuracy multiclass."""
    best = {
        "xgb_weight": 1 / 3,
        "lgbm_weight": 1 / 3,
        "lgbm_secondary_weight": 1 / 3,
        "accuracy": -1.0,
        "logloss": 999.0,
    }

    for wx in np.arange(0.0, 1.01, 0.05):
        for wl in np.arange(0.0, 1.01 - wx, 0.05):
            ws = 1.0 - wx - wl
            if ws < 0:
                continue

            probs = wx * xgb_probs + wl * lgbm_probs + ws * lgbm_sec_probs
            metrics = evaluate_probs_multiclass(y_true, probs)
            acc = metrics["accuracy"]
            ll = metrics["logloss"]

            if (acc > best["accuracy"]) or (acc == best["accuracy"] and ll < best["logloss"]):
                best = {
                    "xgb_weight": float(round(wx, 3)),
                    "lgbm_weight": float(round(wl, 3)),
                    "lgbm_secondary_weight": float(round(ws, 3)),
                    "accuracy": float(acc),
                    "logloss": float(ll),
                }

    return best


def choose_best_ensemble_weights_binary(
    y_true: pd.Series,
    xgb_probs: np.ndarray,
    lgbm_probs: np.ndarray,
    lgbm_sec_probs: np.ndarray,
) -> Dict[str, float]:
    """Busca pesos + threshold para binario maximizando score de validaciÃ³n."""
    best = {
        "xgb_weight": 1 / 3,
        "lgbm_weight": 1 / 3,
        "lgbm_secondary_weight": 1 / 3,
        "threshold": 0.50,
        "accuracy": -1.0,
        "logloss": 999.0,
        "positive_rate": 0.0,
        "score": -999.0,
    }

    for wx in np.arange(0.0, 1.01, 0.05):
        for wl in np.arange(0.0, 1.01 - wx, 0.05):
            ws = 1.0 - wx - wl
            if ws < 0:
                continue

            probs = wx * xgb_probs + wl * lgbm_probs + ws * lgbm_sec_probs
            threshold_info = choose_best_threshold_binary(y_true, probs)
            metrics = evaluate_probs_binary(y_true, probs, threshold_info["threshold"])

            score = float(threshold_info["score"])
            acc = float(metrics["accuracy"])
            ll = float(metrics["logloss"])

            if (acc > best["accuracy"]) or (
                acc == best["accuracy"] and (ll < best["logloss"] or (ll == best["logloss"] and score > best["score"]))
            ):
                best = {
                    "xgb_weight": float(round(wx, 3)),
                    "lgbm_weight": float(round(wl, 3)),
                    "lgbm_secondary_weight": float(round(ws, 3)),
                    "threshold": float(threshold_info["threshold"]),
                    "accuracy": acc,
                    "logloss": ll,
                    "positive_rate": float(threshold_info["positive_rate"]),
                    "score": score,
                }

    return best


def train_single_market(df: pd.DataFrame, market_key: str, target_col: str, feature_cols: List[str], problem_type: str) -> Dict:
    """Entrena un modelo individual para un mercado."""
    market_dir = MODELS_DIR / market_key
    market_dir.mkdir(parents=True, exist_ok=True)

    market_df = df.dropna(subset=[target_col]).copy()

    model_feature_cols = [c for c in feature_cols if c not in ADVANCED_EXPERIMENTAL_FEATURES]
    if len(model_feature_cols) < 20:
        model_feature_cols = feature_cols.copy()

    X = market_df[model_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = market_df[target_col].astype(int)

    split_df = market_df.copy()
    split_df[model_feature_cols] = X

    train_df, valid_df = time_based_split(split_df, valid_fraction=0.2)

    X_train = train_df[model_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train_df[target_col].astype(int)

    X_valid = valid_df[model_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_valid = valid_df[target_col].astype(int)

    print(f"\nðŸš€ Entrenando mercado Liga MX: {market_key}")
    print(f"   Target        : {target_col} ({problem_type})")
    print(f"   Train rows    : {len(X_train)}")
    print(f"   Valid rows    : {len(X_valid)}")
    print(f"   Features      : {len(model_feature_cols)}")

    if problem_type == "binary":
        ensemble_weights = {
            "xgb_weight": 1 / 3,
            "lgbm_weight": 1 / 3,
            "lgbm_secondary_weight": 1 / 3,
        }
        if y_train.nunique() < 2:
            constant_class = int(y_train.iloc[0]) if len(y_train) else 0
            print(f"   âš ï¸ Single-class detectado ({constant_class}). Se usa fallback constante.")

            xgb_model = ConstantBinaryModel(constant_class)
            lgbm_model = ConstantBinaryModel(constant_class)
            lgbm_secondary = ConstantBinaryModel(constant_class)

            xgb_valid_probs = xgb_model.predict_proba(X_valid)[:, 1]
            lgbm_valid_probs = lgbm_model.predict_proba(X_valid)[:, 1]
            lgbm_sec_valid_probs = lgbm_secondary.predict_proba(X_valid)[:, 1]
            ensemble_valid_probs = (xgb_valid_probs + lgbm_valid_probs + lgbm_sec_valid_probs) / 3.0
            best_threshold = 0.50

            xgb_metrics = evaluate_probs_binary(y_valid, xgb_valid_probs, best_threshold)
            lgbm_metrics = evaluate_probs_binary(y_valid, lgbm_valid_probs, best_threshold)
            lgbm_sec_metrics = evaluate_probs_binary(y_valid, lgbm_sec_valid_probs, best_threshold)
            ensemble_metrics = evaluate_probs_binary(y_valid, ensemble_valid_probs, best_threshold)
        else:
            scale_pos_weight = get_scale_pos_weight(y_train)
            print(f"   Pos weight    : {scale_pos_weight:.3f}")

            xgb_model = build_xgb_binary(scale_pos_weight)
            lgbm_model = build_lgbm_binary(scale_pos_weight)
            lgbm_secondary = build_lgbm_secondary_binary(scale_pos_weight)

            xgb_model.fit(X_train, y_train)
            lgbm_model.fit(X_train, y_train)
            lgbm_secondary.fit(X_train, y_train)

            xgb_valid_probs = xgb_model.predict_proba(X_valid)[:, 1]
            lgbm_valid_probs = lgbm_model.predict_proba(X_valid)[:, 1]
            lgbm_sec_valid_probs = lgbm_secondary.predict_proba(X_valid)[:, 1]

            # Ensemble: bÃºsqueda de pesos + threshold sobre validaciÃ³n temporal
            ensemble_weights = choose_best_ensemble_weights_binary(
                y_valid, xgb_valid_probs, lgbm_valid_probs, lgbm_sec_valid_probs
            )
            ensemble_valid_probs = (
                ensemble_weights["xgb_weight"] * xgb_valid_probs
                + ensemble_weights["lgbm_weight"] * lgbm_valid_probs
                + ensemble_weights["lgbm_secondary_weight"] * lgbm_sec_valid_probs
            )
            best_threshold = ensemble_weights["threshold"]

            xgb_metrics = evaluate_probs_binary(y_valid, xgb_valid_probs, 0.50)
            lgbm_metrics = evaluate_probs_binary(y_valid, lgbm_valid_probs, 0.50)
            lgbm_sec_metrics = evaluate_probs_binary(y_valid, lgbm_sec_valid_probs, 0.50)
            ensemble_metrics = evaluate_probs_binary(y_valid, ensemble_valid_probs, best_threshold)

            print(f"   âœ… Threshold ens. : {best_threshold}")
            print(f"   âœ… XGB Accuracy   : {xgb_metrics['accuracy']:.4f}")
            print(f"   âœ… LGBM Accuracy  : {lgbm_metrics['accuracy']:.4f}")
            print(f"   âœ… LGBM-Sec Acc   : {lgbm_sec_metrics['accuracy']:.4f}")
            print(f"   âœ… Ensemble Acc   : {ensemble_metrics['accuracy']:.4f}")
            print(f"   âœ… LogLoss ens.   : {ensemble_metrics['logloss']:.4f}")
            print(f"   âœ… ROC AUC ens.   : {ensemble_metrics['roc_auc']:.4f}")

    else:  # multiclass (full_game)
        seed_candidates = [7, 11, 23, 42, 77]
        best_pack = None

        y_draw_train = (y_train == 2).astype(int)
        winner_mask = y_train != 2
        X_winner_train = X_train.loc[winner_mask]
        y_winner_train = (y_train.loc[winner_mask] == 1).astype(int)

        draw_pos_weight = get_scale_pos_weight(y_draw_train)
        winner_pos_weight = get_scale_pos_weight(y_winner_train) if len(y_winner_train) else 1.0

        for seed in seed_candidates:
            # XGB two-stage
            if y_draw_train.nunique() < 2:
                xgb_draw = ConstantBinaryModel(int(y_draw_train.iloc[0]))
            else:
                xgb_draw = build_xgb_binary(draw_pos_weight)
                xgb_draw.set_params(random_state=seed)
                xgb_draw.fit(X_train, y_draw_train)

            if y_winner_train.nunique() < 2:
                xgb_winner = ConstantBinaryModel(int(y_winner_train.iloc[0]) if len(y_winner_train) else 0)
            else:
                xgb_winner = build_xgb_binary(winner_pos_weight)
                xgb_winner.set_params(random_state=seed)
                xgb_winner.fit(X_winner_train, y_winner_train)

            xgb_model_try = {
                "model_type": "two_stage_full_game",
                "draw_model": xgb_draw,
                "winner_model": xgb_winner,
                "seed": seed,
                "family": "xgboost",
            }

            # LGBM two-stage
            if y_draw_train.nunique() < 2:
                lgbm_draw = ConstantBinaryModel(int(y_draw_train.iloc[0]))
            else:
                lgbm_draw = build_lgbm_binary(draw_pos_weight)
                lgbm_draw.set_params(random_state=seed)
                lgbm_draw.fit(X_train, y_draw_train)

            if y_winner_train.nunique() < 2:
                lgbm_winner = ConstantBinaryModel(int(y_winner_train.iloc[0]) if len(y_winner_train) else 0)
            else:
                lgbm_winner = build_lgbm_binary(winner_pos_weight)
                lgbm_winner.set_params(random_state=seed)
                lgbm_winner.fit(X_winner_train, y_winner_train)

            lgbm_model_try = {
                "model_type": "two_stage_full_game",
                "draw_model": lgbm_draw,
                "winner_model": lgbm_winner,
                "seed": seed,
                "family": "lightgbm_primary",
            }

            # LGBM secondary two-stage
            if y_draw_train.nunique() < 2:
                lgbm_sec_draw = ConstantBinaryModel(int(y_draw_train.iloc[0]))
            else:
                lgbm_sec_draw = build_lgbm_secondary_binary(draw_pos_weight)
                lgbm_sec_draw.set_params(random_state=seed)
                lgbm_sec_draw.fit(X_train, y_draw_train)

            if y_winner_train.nunique() < 2:
                lgbm_sec_winner = ConstantBinaryModel(int(y_winner_train.iloc[0]) if len(y_winner_train) else 0)
            else:
                lgbm_sec_winner = build_lgbm_secondary_binary(winner_pos_weight)
                lgbm_sec_winner.set_params(random_state=seed)
                lgbm_sec_winner.fit(X_winner_train, y_winner_train)

            lgbm_secondary_try = {
                "model_type": "two_stage_full_game",
                "draw_model": lgbm_sec_draw,
                "winner_model": lgbm_sec_winner,
                "seed": seed,
                "family": "lightgbm_secondary",
            }

            xgb_valid_probs_try = _full_game_two_stage_predict_proba(xgb_model_try, X_valid)
            lgbm_valid_probs_try = _full_game_two_stage_predict_proba(lgbm_model_try, X_valid)
            lgbm_sec_valid_probs_try = _full_game_two_stage_predict_proba(lgbm_secondary_try, X_valid)

            ensemble_weights_try = choose_best_ensemble_weights_multiclass(
                y_valid, xgb_valid_probs_try, lgbm_valid_probs_try, lgbm_sec_valid_probs_try
            )
            ensemble_valid_probs_try = (
                ensemble_weights_try["xgb_weight"] * xgb_valid_probs_try
                + ensemble_weights_try["lgbm_weight"] * lgbm_valid_probs_try
                + ensemble_weights_try["lgbm_secondary_weight"] * lgbm_sec_valid_probs_try
            )
            ensemble_metrics_try = evaluate_probs_multiclass(y_valid, ensemble_valid_probs_try)

            candidate = {
                "seed": seed,
                "xgb_model": xgb_model_try,
                "lgbm_model": lgbm_model_try,
                "lgbm_secondary": lgbm_secondary_try,
                "xgb_valid_probs": xgb_valid_probs_try,
                "lgbm_valid_probs": lgbm_valid_probs_try,
                "lgbm_sec_valid_probs": lgbm_sec_valid_probs_try,
                "ensemble_weights": ensemble_weights_try,
                "ensemble_metrics": ensemble_metrics_try,
            }
            if best_pack is None:
                best_pack = candidate
            else:
                curr = best_pack["ensemble_metrics"]
                new = candidate["ensemble_metrics"]
                if (new["accuracy"] > curr["accuracy"]) or (
                    new["accuracy"] == curr["accuracy"] and new["logloss"] < curr["logloss"]
                ):
                    best_pack = candidate

        xgb_model = best_pack["xgb_model"]
        lgbm_model = best_pack["lgbm_model"]
        lgbm_secondary = best_pack["lgbm_secondary"]
        xgb_valid_probs = best_pack["xgb_valid_probs"]
        lgbm_valid_probs = best_pack["lgbm_valid_probs"]
        lgbm_sec_valid_probs = best_pack["lgbm_sec_valid_probs"]
        ensemble_weights = best_pack["ensemble_weights"]

        ensemble_valid_probs = (
            ensemble_weights["xgb_weight"] * xgb_valid_probs
            + ensemble_weights["lgbm_weight"] * lgbm_valid_probs
            + ensemble_weights["lgbm_secondary_weight"] * lgbm_sec_valid_probs
        )

        best_threshold = 0.0  # No threshold para multiclass
        xgb_metrics = evaluate_probs_multiclass(y_valid, xgb_valid_probs)
        lgbm_metrics = evaluate_probs_multiclass(y_valid, lgbm_valid_probs)
        lgbm_sec_metrics = evaluate_probs_multiclass(y_valid, lgbm_sec_valid_probs)
        ensemble_metrics = evaluate_probs_multiclass(y_valid, ensemble_valid_probs)

        print(f"   ✅ XGB Accuracy   : {xgb_metrics['accuracy']:.4f}")
        print(f"   ✅ LGBM Accuracy  : {lgbm_metrics['accuracy']:.4f}")
        print(f"   ✅ LGBM-Sec Acc   : {lgbm_sec_metrics['accuracy']:.4f}")
        print(f"   ✅ Ensemble Acc   : {ensemble_metrics['accuracy']:.4f}")
        print(f"   ✅ LogLoss ens.   : {ensemble_metrics['logloss']:.4f}")
        print(f"   ✅ ROC AUC ens.   : {ensemble_metrics['roc_auc']:.4f}")
        print(f"   ✅ Best seed      : {best_pack['seed']}")
        print(
            f"   ✅ Pesos ens.     : XGB={ensemble_weights['xgb_weight']:.2f}, "
            f"LGBM={ensemble_weights['lgbm_weight']:.2f}, "
            f"LGBM-Sec={ensemble_weights['lgbm_secondary_weight']:.2f}"
        )

    xgb_path = market_dir / "xgb_model.pkl"
    lgbm_path = market_dir / "lgbm_model.pkl"
    lgbm_sec_path = market_dir / "lgbm_secondary_model.pkl"
    feature_path = market_dir / "feature_columns.json"
    metadata_path = market_dir / "metadata.json"

    joblib.dump(xgb_model, xgb_path)
    joblib.dump(lgbm_model, lgbm_path)
    joblib.dump(lgbm_secondary, lgbm_sec_path)

    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump(model_feature_cols, f, indent=2, ensure_ascii=False)

    xgb_importance = _safe_feature_importance(xgb_model, model_feature_cols)
    lgbm_importance = _safe_feature_importance(lgbm_model, model_feature_cols)
    lgbm_sec_importance = _safe_feature_importance(lgbm_secondary, model_feature_cols)

    # Combinar importancias de los 3 modelos
    combined_importance = []
    for col in model_feature_cols:
        xi = float(xgb_importance.get(col, 0.0))
        li = float(lgbm_importance.get(col, 0.0))
        lsi = float(lgbm_sec_importance.get(col, 0.0))
        combined_importance.append((col, (xi + li + lsi) / 3.0))

    combined_importance = sorted(combined_importance, key=lambda x: x[1], reverse=True)[:25]

    if problem_type == "multiclass":
        ensemble_weight_payload = {
            "xgboost": float(ensemble_weights["xgb_weight"]),
            "lightgbm_primary": float(ensemble_weights["lgbm_weight"]),
            "lightgbm_secondary": float(ensemble_weights["lgbm_secondary_weight"]),
        }
    else:
        ensemble_weight_payload = {
            "xgboost": float(ensemble_weights["xgb_weight"]),
            "lightgbm_primary": float(ensemble_weights["lgbm_weight"]),
            "lightgbm_secondary": float(ensemble_weights["lgbm_secondary_weight"]),
        }

    metadata = {
        "market_key": market_key,
        "target_col": target_col,
        "problem_type": problem_type,
        "train_rows": int(len(X_train)),
        "valid_rows": int(len(X_valid)),
        "feature_count": int(len(model_feature_cols)),
        "ensemble_threshold": float(best_threshold) if problem_type == "binary" else None,
        "ensemble_weights": ensemble_weight_payload,
        "metrics": {
            "xgboost_valid": xgb_metrics,
            "lightgbm_valid": lgbm_metrics,
            "lightgbm_secondary_valid": lgbm_sec_metrics,
            "ensemble_valid": ensemble_metrics,
        },
        "top_features": [
            {"feature": col, "importance": float(score)}
            for col, score in combined_importance
        ],
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"   âœ… Guardado XGB   : {xgb_path}")
    print(f"   âœ… Guardado LGBM  : {lgbm_path}")
    print(f"   âœ… Guardado LGBM-Sec : {lgbm_sec_path}")

    return {
        "market_key": market_key,
        "target_col": target_col,
        "problem_type": problem_type,
        "threshold": best_threshold if problem_type == "binary" else None,
        "xgb_metrics": xgb_metrics,
        "lgbm_metrics": lgbm_metrics,
        "ensemble_metrics": ensemble_metrics,
        "top_features": metadata["top_features"][:10],
    }


def train_all_models() -> Dict:
    """Entrena todos los modelos para Liga MX."""
    print("=" * 70)
    print("ðŸŽ¯ ENTRENAMIENTO DE MODELOS PARA LIGA MX")
    print("=" * 70)

    df = load_dataset()
    feature_cols = get_feature_columns(df)

    print("\nðŸ“¦ Dataset Liga MX cargado")
    print(f"   Filas totales : {len(df)}")
    print(f"   Features      : {len(feature_cols)}")
    print(f"   Archivo       : {INPUT_FILE}")

    results = {}

    for market_key, config in TARGET_CONFIG.items():
        result = train_single_market(
            df=df,
            market_key=market_key,
            target_col=config["target_col"],
            feature_cols=feature_cols,
            problem_type=config["problem_type"],
        )
        results[market_key] = result

    summary_path = REPORTS_DIR / "training_summary_liga_mx.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("ðŸ“Š RESUMEN FINAL LIGA MX")
    print("=" * 70)
    for market_key, result in results.items():
        metrics = result["ensemble_metrics"]
        prob_type = result["problem_type"]
        thr_str = f"THR: {result['threshold']:.2f}" if result["threshold"] is not None else "N/A"
        print(
            f"\n{market_key.upper():<12} ({prob_type})"
            f"\n   ACC: {metrics['accuracy']:.4f} | LOGLOSS: {metrics['logloss']:.4f} | AUC: {metrics['roc_auc']:.4f}"
            f"\n   {thr_str}"
        )

    print(f"\nðŸ’¾ Resumen guardado en: {summary_path}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    train_all_models()

