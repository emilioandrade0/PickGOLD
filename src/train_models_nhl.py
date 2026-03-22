"""
Train 4-model ensemble for NHL predictions (XGBoost + LightGBM Primary + LightGBM Secondary + CatBoost).
Mirrors the structure of the Liga MX enhanced training pipeline.
"""
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

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "nhl" / "processed" / "model_ready_features_nhl.csv"

MODELS_DIR = BASE_DIR / "data" / "nhl" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = BASE_DIR / "data" / "nhl" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CONFIG = {
    "full_game": {
        "target_col": "TARGET_full_game",
        "description": "Home Win (1) vs Away Win (2)",
        "problem_type": "binary",
        "num_classes": 2,
        "threshold": 0.30,
    },
    "spread_2_5": {
        "target_col": "TARGET_over_55",
        "description": "Over/Under 5.5 goals",
        "problem_type": "binary",
        "num_classes": 2,
        "threshold": 0.30,
    },
    "home_over_2_5": {
        "target_col": "TARGET_home_over_25",
        "description": "Home Team Over/Under 2.5 goals",
        "problem_type": "binary",
        "num_classes": 2,
        "threshold": 0.35,
    },
}

NON_FEATURE_COLUMNS = {
    "game_id", "date", "date_dt", "time", "season", "home_team", "away_team",
    "home_score", "away_score", "is_draw", "total_goals",
    "completed", "venue_name", "odds_details", "odds_over_under",
    "TARGET_full_game", "TARGET_over_55", "TARGET_home_over_25",
}


def load_dataset() -> pd.DataFrame:
    """Load and prepare NHL dataset."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {INPUT_FILE}")
    
    df = pd.read_csv(INPUT_FILE, dtype={"game_id": str})
    df["date"] = df["date"].astype(str)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt"]).sort_values(["date_dt", "game_id"]).reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns (excluding targets and metadata)."""
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS and c != "date_dt"]


def time_based_split(df: pd.DataFrame, valid_fraction: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based train/validation split."""
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


def build_xgb_multiclass() -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softmax",
        eval_metric="mlogloss",
        num_class=3,
        n_estimators=350,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.2,
        reg_lambda=1.2,
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


def build_lgbm_multiclass() -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        metric="multi_logloss",
        num_class=3,
        n_estimators=400,
        learning_rate=0.04,
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


def build_lgbm_secondary_multiclass() -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        metric="multi_logloss",
        num_class=3,
        n_estimators=350,
        learning_rate=0.05,
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
        bootstrap_type='Bernoulli',
        subsample=0.8,
        random_state=44,
        verbose=0,
        allow_writing_files=False,
    )


def build_catboost_multiclass() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=350,
        learning_rate=0.04,
        depth=7,
        l2_leaf_reg=6.0,
        bootstrap_type='Bernoulli',
        subsample=0.8,
        random_state=44,
        verbose=0,
        allow_writing_files=False,
    )


def calculate_optimal_threshold(y_valid: np.ndarray, probs: np.ndarray) -> float:
    """Find optimal threshold to maximize accuracy on validation set.
    Search in [0.2, 0.8] to avoid extreme thresholds that cause model bias."""
    best_threshold = 0.5
    best_accuracy = 0
    
    for threshold in np.arange(0.2, 0.8, 0.05):
        preds = (probs >= threshold).astype(int)
        acc = accuracy_score(y_valid, preds)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = threshold
    
    return best_threshold


def train_single_market(df: pd.DataFrame, market: str, config: Dict):
    """Train 4-model ensemble for a single market."""
    
    target_col = config["target_col"]
    problem_type = config["problem_type"]
    
    print(f"\n🎯 Entrenando mercado: {market} ({problem_type})")
    print(f"   Descripción: {config['description']}")
    
    # Prepare data
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].fillna(0)
    y = df[target_col].copy()
    
    # Remove rows with missing targets
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # For full_game binary with classes [1,2], remap to [0,1]
    if problem_type == "binary" and market == "full_game":
        y = y.map({1: 0, 2: 1})  # Remap: Home Win (1) -> 0, Away Win (2) -> 1
    
    print(f"   Juegos: {len(X)}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Classes: {sorted(y.unique())}")
    
    # Temporal split (same criterion as the rest of the pipeline)
    split_idx = int(len(X) * 0.8)
    split_idx = max(50, min(split_idx, len(X) - 20))
    X_train, X_valid = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_valid = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()
    
    # Train models
    print(f"   Entrenando 4 modelos...")
    
    if problem_type == "multiclass":
        xgb = build_xgb_multiclass()
        lgbm = build_lgbm_multiclass()
        lgbm_sec = build_lgbm_secondary_multiclass()
        catboost = build_catboost_multiclass()
    else:
        xgb = build_xgb_binary()
        lgbm = build_lgbm_binary()
        lgbm_sec = build_lgbm_secondary_binary()
        catboost = build_catboost_binary()
    
    xgb.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)
    lgbm_sec.fit(X_train, y_train)
    catboost.fit(X_train, y_train)
    
    # Get predictions
    xgb_probs = xgb.predict_proba(X_valid)
    lgbm_probs = lgbm.predict_proba(X_valid)
    lgbm_sec_probs = lgbm_sec.predict_proba(X_valid)
    catboost_probs = catboost.predict_proba(X_valid)
    
    # Ensemble
    ensemble_probs = (xgb_probs + lgbm_probs + lgbm_sec_probs + catboost_probs) / 4.0
    
    # Calculate metrics
    if problem_type == "multiclass":
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        xgb_preds = xgb.predict(X_valid)
        lgbm_preds = lgbm.predict(X_valid)
        catboost_preds = catboost.predict(X_valid)
        
        accuracy = accuracy_score(y_valid, ensemble_preds)
        threshold = None
    else:
        threshold = calculate_optimal_threshold(y_valid, ensemble_probs[:, 1])
        ensemble_preds = (ensemble_probs[:, 1] >= threshold).astype(int)
        accuracy = accuracy_score(y_valid, ensemble_preds)
    
    print(f"   ✅ Ensemble Accuracy: {accuracy:.4f}")
    if threshold:
        print(f"   ✅ Optimal Threshold: {threshold:.2f}")
    
    # Save models
    market_dir = MODELS_DIR / market
    market_dir.mkdir(exist_ok=True)
    
    joblib.dump(xgb, market_dir / "xgboost_model.pkl")
    joblib.dump(lgbm, market_dir / "lgbm_model.pkl")
    joblib.dump(lgbm_sec, market_dir / "lgbm_secondary_model.pkl")
    joblib.dump(catboost, market_dir / "catboost_model.pkl")
    
    # Save metadata
    metadata = {
        "market": market,
        "problem_type": problem_type,
        "num_classes": config.get("num_classes"),
        "accuracy": float(accuracy),
        "threshold": float(threshold) if threshold else None,
        "feature_columns": feature_cols,
        "training_samples": len(X_train),
        "validation_samples": len(X_valid),
    }
    
    with open(market_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return accuracy, threshold


def main():
    """Train all markets."""
    print("🏒 NHL Model Training - 4-Model Ensemble")
    print("=" * 60)
    
    try:
        df = load_dataset()
        print(f"✅ Dataset loaded: {len(df)} games")
        
        results = {}
        for market, config in TARGET_CONFIG.items():
            try:
                accuracy, threshold = train_single_market(df, market, config)
                results[market] = {
                    "accuracy": accuracy,
                    "threshold": threshold,
                }
            except Exception as e:
                print(f"❌ Error training {market}: {e}")
        
        # Save summary report
        summary_file = REPORTS_DIR / "training_summary.json"
        with open(summary_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "=" * 60)
        print("✅ Training completed!")
        print(f"   Results saved to {summary_file.name}")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
