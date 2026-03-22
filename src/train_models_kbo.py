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

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "kbo" / "processed" / "model_ready_features_kbo.csv"

MODELS_DIR = BASE_DIR / "data" / "kbo" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = BASE_DIR / "data" / "kbo" / "reports"
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

NON_FEATURE_COLUMNS = {
    "game_id",
    "date",
    "season",
    "home_team",
    "away_team",
    "TARGET_home_win",
    "TARGET_yrfi",
    "TARGET_home_win_f5",
}


def load_dataset() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"No existe el archivo de features kbo: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    df["date"] = df["date"].astype(str)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt"]).sort_values(["date_dt", "game_id"]).reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS and c != "date_dt"]


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


def choose_best_threshold(y_true: pd.Series, probs: np.ndarray) -> Dict[str, float]:
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
        balance_penalty = abs(positive_rate - 0.5) * 0.03
        score = acc - balance_penalty

        if score > best["score"]:
            best = {
                "threshold": float(round(thr, 2)),
                "score": score,
                "accuracy": acc,
                "positive_rate": positive_rate,
            }

    return best


def choose_best_ensemble_params(y_true: pd.Series, xgb_probs: np.ndarray, lgbm_probs: np.ndarray) -> Dict[str, float]:
    best = {
        "xgb_weight": 0.5,
        "lgbm_weight": 0.5,
        "threshold": 0.5,
        "score": -999.0,
        "accuracy": 0.0,
        "positive_rate": 0.0,
    }

    for w in np.arange(0.2, 0.81, 0.05):
        probs = w * xgb_probs + (1 - w) * lgbm_probs
        th_info = choose_best_threshold(y_true, probs)
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

    print(f"\nðŸš€ Entrenando mercado kbo: {market_key}")
    print(f"   Target        : {target_col}")
    print(f"   Train rows    : {len(X_train)}")
    print(f"   Valid rows    : {len(X_valid)}")
    print(f"   Features      : {len(feature_cols)}")
    print(f"   Pos weight    : {scale_pos_weight:.3f}")

    xgb_model.fit(X_train, y_train)
    lgbm_model.fit(X_train, y_train)

    xgb_valid_probs = xgb_model.predict_proba(X_valid)[:, 1]
    lgbm_valid_probs = lgbm_model.predict_proba(X_valid)[:, 1]
    ensemble_params = choose_best_ensemble_params(y_valid, xgb_valid_probs, lgbm_valid_probs)
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

    xgb_importance = dict(zip(feature_cols, map(float, xgb_model.feature_importances_)))
    lgbm_importance = dict(zip(feature_cols, map(float, lgbm_model.feature_importances_)))

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

    print(f"   âœ… Guardado XGB   : {xgb_path}")
    print(f"   âœ… Guardado LGBM  : {lgbm_path}")
    print(f"   âœ… Threshold ens. : {best_threshold}")
    print(f"   âœ… Accuracy ens.  : {ensemble_metrics['accuracy']:.4f}")
    print(f"   âœ… LogLoss ens.   : {ensemble_metrics['logloss']:.4f}")
    print(f"   âœ… ROC AUC ens.   : {ensemble_metrics['roc_auc']:.4f}")

    return {
        "market_key": market_key,
        "target_col": target_col,
        "threshold": best_threshold,
        "xgb_metrics": xgb_metrics,
        "lgbm_metrics": lgbm_metrics,
        "ensemble_metrics": ensemble_metrics,
        "top_features": metadata["top_features"][:10],
    }


def train_all_models() -> Dict:
    df = load_dataset()
    feature_cols = get_feature_columns(df)

    print("ðŸ“¦ Dataset kbo cargado")
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
        )
        results[market_key] = result

    summary_path = REPORTS_DIR / "training_summary_kbo.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nðŸ“Š RESUMEN FINAL kbo")
    for market_key, result in results.items():
        metrics = result["ensemble_metrics"]
        print(
            f"   {market_key.upper():<10} | "
            f"ACC: {metrics['accuracy']:.4f} | "
            f"LOGLOSS: {metrics['logloss']:.4f} | "
            f"AUC: {metrics['roc_auc']:.4f} | "
            f"THR: {result['threshold']:.2f}"
        )

    print(f"\nðŸ’¾ Resumen guardado en: {summary_path}")
    return results


if __name__ == "__main__":
    train_all_models()
