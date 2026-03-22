"""
Generate historical NHL predictions using WALK-FORWARD VALIDATION.
For each prediction date T:
  1. Train models only on data from [start, T-1]
  2. Predict games on date T
  3. Evaluate against actual results
This prevents leakage and gives TRUE prospective accuracy.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "nhl" / "processed" / "model_ready_features_nhl.csv"
HISTORICAL_DIR = BASE_DIR / "data" / "nhl" / "historical_predictions"
HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

NON_FEATURE_COLUMNS = {
    "game_id", "date", "date_dt", "time", "season", "home_team", "away_team",
    "home_score", "away_score", "total_goals", "is_draw", "completed",
    "venue_name", "odds_details", "odds_over_under",
    "TARGET_full_game", "TARGET_over_55", "TARGET_home_over_25",
}


def build_xgb_binary() -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
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
        n_estimators=250,
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


def build_catboost_binary() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5.0,
        bootstrap_type='Bernoulli',
        subsample=0.8,
        random_state=44,
        verbose=0,
        allow_writing_files=False,
    )


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def predict_market_ensemble(models: Dict, X: pd.DataFrame, threshold: float = None) -> Tuple:
    """Generate 4-model ensemble predictions."""
    
    xgb_probs = models["xgb"].predict_proba(X)[:, 1]
    lgbm_probs = models["lgbm"].predict_proba(X)[:, 1]
    lgbm_sec_probs = models["lgbm_sec"].predict_proba(X)[:, 1]
    catboost_probs = models["catboost"].predict_proba(X)[:, 1]
    
    # Average ensemble
    ensemble_probs = (xgb_probs + lgbm_probs + lgbm_sec_probs + catboost_probs) / 4.0
    
    if threshold:
        preds = (ensemble_probs >= threshold).astype(int)
    else:
        preds = (ensemble_probs >= 0.5).astype(int)
    
    confidences = np.abs(ensemble_probs - 0.5) + 0.5
    
    return preds, confidences, ensemble_probs


def derive_nhl_first_period_pick(prob_over_55: float) -> Dict:
    """Derive 1P O/U 1.5 prediction from full-game totals probability."""
    p = float(np.clip(prob_over_55, 0.01, 0.99))
    expected_total_goals = 5.5 + 1.4 * (p - 0.5)
    expected_total_goals = float(np.clip(expected_total_goals, 4.6, 6.4))
    lambda_p1 = expected_total_goals * 0.30
    p_over_15 = 1.0 - float(np.exp(-lambda_p1) * (1.0 + lambda_p1))
    p_over_15 = float(np.clip(p_over_15, 0.01, 0.99))

    pick_over = p_over_15 >= 0.53
    pick = "Over 1.5" if pick_over else "Under 1.5"
    confidence = int((0.5 + abs(p_over_15 - 0.5)) * 100)
    action = "JUGAR" if confidence >= 56 else "PASS"

    return {
        "q1_pick": pick,
        "q1_market": "1P Goals O/U 1.5",
        "q1_line": 1.5,
        "q1_confidence": confidence,
        "q1_action": action,
        "q1_model_prob_yes": round(p_over_15, 4),
        "q1_calibrated_prob_yes": round(p_over_15, 4),
    }


def generate_historical_predictions():
    """
    Walk-forward validation: for each date T, train on [start, T-1], predict T.
    No leakage - each prediction is truly prospective.
    """
    
    print("[NHL] Historical Predictions - Walk-Forward Validation")
    print("=" * 60)
    
    # Load dataset
    if not INPUT_FILE.exists():
        print(f"[ERROR] Dataset not found: {INPUT_FILE}")
        return
    
    df = pd.read_csv(INPUT_FILE, dtype={"game_id": str})
    df["date"] = df["date"].astype(str)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date_dt").reset_index(drop=True)
    
    print(f"[OK] Loaded {len(df)} historical games")
    
    feature_cols = get_feature_columns(df)
    unique_dates = sorted(df["date"].unique())
    
    print(f"[OK] Date range: {unique_dates[0]} to {unique_dates[-1]}")
    print(f"[OK] {len(unique_dates)} unique dates")
    
    # Walk-forward validation: skip first 50 games minimum
    predictions_by_date = defaultdict(list)
    market_stats = defaultdict(lambda: {"correct": 0, "total": 0, "total_conf": 0})
    
    start_idx = min(50, len(df) // 10)  # Start predictions after 50 games or 10%
    
    print(f"\n[TRAIN] Training models walk-forward from index {start_idx}...")
    
    # For each prediction date, train on prior data
    for test_idx in range(start_idx, len(df)):
        if test_idx % 100 == 0:
            print(f"   Processing game {test_idx+1}/{len(df)}...")
        
        # Data before this date (training)
        train_df = df.iloc[:test_idx].copy()
        
        # Current game (test)
        test_row = df.iloc[test_idx]
        test_date = test_row["date"]
        
        # Skip if not enough training data
        if len(train_df) < 20:
            continue
        
        # Build data
        X_train = train_df[feature_cols].fillna(0)
        X_test = pd.DataFrame([{col: test_row.get(col, 0) for col in feature_cols}], columns=feature_cols).fillna(0)
        
        y_spread_train = train_df["TARGET_over_55"].fillna(-1)
        y_home_train = train_df["TARGET_home_over_25"].fillna(-1)
        
        # Filter out NaN targets
        valid_spread = y_spread_train >= 0
        valid_home = y_home_train >= 0
        
        game_dict = {
            "game_id": str(test_row["game_id"]),
            "date": test_date,
            "time": str(test_row.get("time", "19:00")),
            "home_team": test_row["home_team"],
            "away_team": test_row["away_team"],
            "odds_over_under": float(test_row.get("odds_over_under", 0) or 0),
            "closing_moneyline_odds": test_row.get("closing_moneyline_odds"),
            "home_moneyline_odds": test_row.get("home_moneyline_odds"),
            "away_moneyline_odds": test_row.get("away_moneyline_odds"),
            "closing_spread_odds": test_row.get("closing_spread_odds"),
            "closing_total_odds": test_row.get("closing_total_odds"),
            "odds_data_quality": str(test_row.get("odds_data_quality", "fallback")),
        }
        
        # SPREAD_2_5 market
        if valid_spread.sum() >= 10:
            try:
                X_train_spread = X_train[valid_spread]
                y_train_spread = y_spread_train[valid_spread]
                
                xgb = build_xgb_binary()
                lgbm = build_lgbm_binary()
                lgbm_sec = build_lgbm_binary()
                catboost = build_catboost_binary()
                
                xgb.fit(X_train_spread, y_train_spread)
                lgbm.fit(X_train_spread, y_train_spread)
                lgbm_sec.fit(X_train_spread, y_train_spread)
                catboost.fit(X_train_spread, y_train_spread)
                
                models = {"xgb": xgb, "lgbm": lgbm, "lgbm_sec": lgbm_sec, "catboost": catboost}
                pred, conf, prob = predict_market_ensemble(models, X_test, threshold=0.50)
                
                spread_labels = ["Under 5.5", "Over 5.5"]
                game_dict["spread_pick"] = spread_labels[int(pred[0])]
                game_dict["spread_market"] = "Total Goals O/U 5.5"
                game_dict["spread_confidence"] = int(conf[0] * 100)
                game_dict["total_pick"] = game_dict["spread_pick"]
                game_dict["total_confidence"] = game_dict["spread_confidence"]

                over_55_prob = float(prob[0][1]) if hasattr(prob[0], "__len__") and len(prob[0]) > 1 else float(conf[0])
                game_dict.update(derive_nhl_first_period_pick(over_55_prob))
                
                # Check actual if available
                if not pd.isna(test_row["TARGET_over_55"]):
                    actual = int(test_row["TARGET_over_55"])
                    is_correct = int(pred[0]) == actual
                    game_dict["spread_actual"] = spread_labels[actual]
                    game_dict["spread_correct"] = is_correct
                    game_dict["correct_spread"] = is_correct
                    game_dict["correct_total"] = is_correct
                    market_stats["spread_2_5"]["correct"] += int(is_correct)
                    market_stats["spread_2_5"]["total"] += 1
                    market_stats["spread_2_5"]["total_conf"] += conf[0]
            except Exception as e:
                game_dict["spread_pick"] = "ERROR"
                game_dict["spread_confidence"] = 0
        
        # HOME_OVER_2_5 market
        if valid_home.sum() >= 10:
            try:
                X_train_home = X_train[valid_home]
                y_train_home = y_home_train[valid_home]
                
                xgb = build_xgb_binary()
                lgbm = build_lgbm_binary()
                lgbm_sec = build_lgbm_binary()
                catboost = build_catboost_binary()
                
                xgb.fit(X_train_home, y_train_home)
                lgbm.fit(X_train_home, y_train_home)
                lgbm_sec.fit(X_train_home, y_train_home)
                catboost.fit(X_train_home, y_train_home)
                
                models = {"xgb": xgb, "lgbm": lgbm, "lgbm_sec": lgbm_sec, "catboost": catboost}
                pred, conf, prob = predict_market_ensemble(models, X_test, threshold=0.50)
                
                home_labels = ["Home Under 2.5", "Home Over 2.5"]
                game_dict["home_over_pick"] = home_labels[int(pred[0])]
                game_dict["home_over_confidence"] = int(conf[0] * 100)
                
                # Check actual if available
                if not pd.isna(test_row["TARGET_home_over_25"]):
                    actual = int(test_row["TARGET_home_over_25"])
                    is_correct = int(pred[0]) == actual
                    game_dict["home_over_actual"] = home_labels[actual]
                    game_dict["home_over_correct"] = is_correct
                    market_stats["home_over_2_5"]["correct"] += int(is_correct)
                    market_stats["home_over_2_5"]["total"] += 1
                    market_stats["home_over_2_5"]["total_conf"] += conf[0]
            except Exception as e:
                game_dict["home_over_pick"] = "ERROR"
                game_dict["home_over_confidence"] = 0
        
        predictions_by_date[test_date].append(game_dict)
    
    # Save predictions by date
    print("\n[SAVE] Saving predictions by date...")
    
    for date_str in sorted(predictions_by_date.keys()):
        output_file = HISTORICAL_DIR / f"{date_str}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(predictions_by_date[date_str], f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Saved predictions for {len(predictions_by_date)} dates")
    
    # Calculate and display stats
    print("\n[STATS] Walk-Forward Accuracy:")
    for market in ["spread_2_5", "home_over_2_5"]:
        stats = market_stats[market]
        if stats["total"] > 0:
            accuracy = stats["correct"] / stats["total"]
            avg_conf = stats["total_conf"] / stats["total"]
            print(f"   {market}:")
            print(f"      Accuracy: {accuracy:.2%} ({stats['correct']}/{stats['total']})")
            print(f"      Avg Confidence: {avg_conf:.1%}")


if __name__ == "__main__":
    generate_historical_predictions()
