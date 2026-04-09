import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import sys
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
INPUT_FILE = BASE_DIR / "data" / "nhl" / "processed" / "model_ready_features_nhl.csv"

MODELS_DIR = BASE_DIR / "data" / "nhl" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = BASE_DIR / "data" / "nhl" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CONFIG = {
    "full_game": {
        "target_col": "TARGET_full_game",
        "description": "Home Win (1) vs Away Win (0)",
        "problem_type": "binary",
        "num_classes": 2,
        "threshold": 0.50,
    },
    "handicap_1_5": {
        "target_col": "TARGET_spread_1_5",
        "description": "Home -1.5 covers (1) vs Away +1.5 covers (0)",
        "problem_type": "binary",
        "num_classes": 2,
        "threshold": 0.50,
    },
    "spread_2_5": {
        "target_col": "TARGET_over_55",
        "description": "Over/Under 5.5 goals",
        "problem_type": "binary",
        "num_classes": 2,
        "threshold": 0.50,
    },
    "q1_over_15": {
        "target_col": "TARGET_p1_over_15",
        "description": "1P Goals Over/Under 1.5",
        "problem_type": "binary",
        "num_classes": 2,
        "threshold": 0.50,
    },
    "home_over_2_5": {
        "target_col": "TARGET_home_over_25",
        "description": "Home Team Over/Under 2.5 goals",
        "problem_type": "binary",
        "num_classes": 2,
        "threshold": 0.50,
    },
}

NON_FEATURE_COLUMNS = {
    "game_id", "date", "date_dt", "time", "season", "home_team", "away_team",
    "home_score", "away_score", "is_draw", "total_goals",
    "home_p1_goals", "away_p1_goals", "total_p1_goals",
    "completed", "venue_name", "odds_details", "odds_over_under",
    "TARGET_full_game", "TARGET_over_55", "TARGET_home_over_25",
    "TARGET_spread_1_5", "TARGET_p1_over_15",
}


def load_dataset() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, dtype={"game_id": str})
    df["date"] = df["date"].astype(str)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt"]).sort_values(["date_dt", "game_id"]).reset_index(drop=True)
    return df


def get_all_numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS and c != "date_dt"]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    banned_keywords = ["correct", "actual", "winner", "result", "target"]
    return [c for c in cols if not any(k in c.lower() for k in banned_keywords)]


def get_nhl_market_feature_columns(df: pd.DataFrame, market: str) -> List[str]:
    all_numeric = get_all_numeric_feature_columns(df)

    if market == "full_game":
        keep = []
        keep_prefixes = ("fg_",)
        keep_exact = {
            "elo_diff",
            "home_elo_pre", "away_elo_pre",
            "h2h_home_win_rate",
            "home_goals_scored_l5", "away_goals_scored_l5",
            "home_goals_allowed_l5", "away_goals_allowed_l5",
            "home_goals_scored_l10", "away_goals_scored_l10",
            "home_goals_allowed_l10", "away_goals_allowed_l10",
            "home_win_rate_l5", "away_win_rate_l5",
            "home_win_rate_l10", "away_win_rate_l10",
            "home_goal_diff_l5", "away_goal_diff_l5",
            "home_goal_diff_l10", "away_goal_diff_l10",
            "home_rest_days", "away_rest_days",
            "home_is_b2b", "away_is_b2b",
            "home_games_last_3", "away_games_last_3",
            "home_games_last_5", "away_games_last_5",
            "home_goalie_found", "away_goalie_found",
            "home_goalie_confirmed", "away_goalie_confirmed",
            "home_goalie_games_started_before", "away_goalie_games_started_before",
            "both_goalies_found", "both_goalies_confirmed",
        }

        drop_keywords = [
            "over_55", "home_over_25", "total", "under", "over",
            "team_total", "q1_", "1p_", "spread_pick", "correct"
        ]

        for c in all_numeric:
            c_low = c.lower()
            if any(k in c_low for k in drop_keywords):
                continue
            if c in keep_exact or c.startswith(keep_prefixes):
                keep.append(c)

        return sorted(set(keep))

    return sorted(all_numeric)


def time_based_split(df: pd.DataFrame, valid_fraction: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - valid_fraction))
    split_idx = max(50, min(split_idx, len(df) - 20))
    train_df = df.iloc[:split_idx].copy()
    valid_df = df.iloc[split_idx:].copy()
    return train_df, valid_df


def build_xgb_binary() -> XGBClassifier:
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
        n_jobs=-1,
    )


def build_lgbm_binary() -> LGBMClassifier:
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


def build_lgbm_secondary_binary() -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=300,
        learning_rate=0.06,
        num_leaves=25,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=43,
        n_jobs=-1,
        verbose=-1,
    )


def build_catboost_binary() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5.0,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        random_state=44,
        verbose=0,
        allow_writing_files=False,
    )


def calculate_optimal_threshold(y_valid: np.ndarray, probs: np.ndarray, market: str) -> float:
    if market == "full_game":
        search_space = np.arange(0.45, 0.61, 0.01)
    else:
        search_space = np.arange(0.35, 0.66, 0.01)

    best_threshold = 0.5
    best_accuracy = -1

    for threshold in search_space:
        preds = (probs >= threshold).astype(int)
        acc = accuracy_score(y_valid, preds)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = float(threshold)

    return best_threshold


def train_single_market(df: pd.DataFrame, market: str, config: Dict):
    target_col = config["target_col"]
    problem_type = config["problem_type"]

    print(f"\n🎯 Entrenando mercado: {market} ({problem_type})")
    print(f"   Descripción: {config['description']}")

    feature_cols = get_nhl_market_feature_columns(df, market)
    X = df[feature_cols].fillna(0)
    y = pd.to_numeric(df[target_col], errors="coerce")

    valid_idx = ~y.isna()
    X = X.loc[valid_idx].copy()
    y = y.loc[valid_idx].copy()

    if market == "full_game":
        # Ya viene binario desde feature engineering: 1=home, 0=away
        y = y.astype(int)

    print(f"   Juegos: {len(X)}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Classes: {sorted(y.unique().tolist())}")

    if len(X) < 80 or len(np.unique(y)) < 2:
        print(f"   [WARN] No hay suficiente data útil para {market}")
        return None

    split_idx = int(len(X) * 0.8)
    split_idx = max(50, min(split_idx, len(X) - 20))
    X_train, X_valid = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_valid = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

    print("   Entrenando 4 modelos...")

    xgb = build_xgb_binary()
    lgbm = build_lgbm_binary()
    lgbm_sec = build_lgbm_secondary_binary()
    catboost = build_catboost_binary()

    xgb.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)
    lgbm_sec.fit(X_train, y_train)
    catboost.fit(X_train, y_train)

    xgb_probs = xgb.predict_proba(X_valid)[:, 1]
    lgbm_probs = lgbm.predict_proba(X_valid)[:, 1]
    lgbm_sec_probs = lgbm_sec.predict_proba(X_valid)[:, 1]
    catboost_probs = catboost.predict_proba(X_valid)[:, 1]

    ensemble_probs = (xgb_probs + lgbm_probs + lgbm_sec_probs + catboost_probs) / 4.0
    threshold = calculate_optimal_threshold(y_valid.values, ensemble_probs, market)
    ensemble_preds = (ensemble_probs >= threshold).astype(int)

    accuracy = accuracy_score(y_valid, ensemble_preds)
    try:
        logloss = log_loss(y_valid, np.clip(ensemble_probs, 1e-6, 1 - 1e-6))
    except Exception:
        logloss = None

    try:
        auc = roc_auc_score(y_valid, ensemble_probs)
    except Exception:
        auc = None

    print(f"   Accuracy valid: {accuracy:.4f}")
    if logloss is not None:
        print(f"   LogLoss valid: {logloss:.4f}")
    if auc is not None:
        print(f"   ROC AUC valid: {auc:.4f}")
    print(f"   Best threshold: {threshold:.2f}")

    market_dir = MODELS_DIR / market
    market_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(xgb, market_dir / "xgboost_model.pkl")
    joblib.dump(lgbm, market_dir / "lgbm_model.pkl")
    joblib.dump(lgbm_sec, market_dir / "lgbm_secondary_model.pkl")
    joblib.dump(catboost, market_dir / "catboost_model.pkl")

    metadata = {
        "market": market,
        "target_col": target_col,
        "problem_type": problem_type,
        "threshold": float(threshold),
        "feature_columns": feature_cols,
        "n_rows": int(len(X)),
        "n_features": int(len(feature_cols)),
        "valid_accuracy": float(accuracy),
        "valid_logloss": None if logloss is None else float(logloss),
        "valid_auc": None if auc is None else float(auc),
        "classes": [int(x) for x in sorted(np.unique(y).tolist())],
    }

    with open(market_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


def train_nhl_models():
    print("🏒 NHL MODEL TRAINING")
    print("=" * 60)

    df = load_dataset()
    print(f"📂 Dataset cargado: {len(df)} juegos")

    results = {}
    for market, config in TARGET_CONFIG.items():
        result = train_single_market(df, market, config)
        if result is not None:
            results[market] = result

    summary_file = REPORTS_DIR / "training_summary_nhl.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Summary guardado en: {summary_file}")


if __name__ == "__main__":
    train_nhl_models()
