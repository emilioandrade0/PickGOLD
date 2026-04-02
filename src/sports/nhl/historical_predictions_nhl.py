import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

import sys
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
INPUT_FILE = BASE_DIR / "data" / "nhl" / "processed" / "model_ready_features_nhl.csv"
RAW_FILE = BASE_DIR / "data" / "nhl" / "raw" / "nhl_advanced_history.csv"
HISTORICAL_DIR = BASE_DIR / "data" / "nhl" / "historical_predictions"
HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

NON_FEATURE_COLUMNS = {
    "game_id", "date", "date_dt", "time", "season", "home_team", "away_team",
    "home_score", "away_score", "total_goals", "is_draw", "completed",
    "venue_name", "odds_details", "odds_over_under", "odds_data_quality",
    "home_goalie_name", "away_goalie_name", "home_goalie_id", "away_goalie_id",
    "goalie_data_quality",
    "TARGET_full_game", "TARGET_over_55", "TARGET_home_over_25", "TARGET_spread_1_5",
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
        bootstrap_type="Bernoulli",
        subsample=0.8,
        random_state=44,
        verbose=0,
        allow_writing_files=False,
        thread_count=-1,
    )


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    candidate_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]

    banned_keywords = ["correct", "actual", "winner", "result", "target"]
    safe_cols = []
    dropped_leaks = []

    for c in numeric_cols:
        if any(banned in c.lower() for banned in banned_keywords):
            dropped_leaks.append(c)
        else:
            safe_cols.append(c)

    if dropped_leaks:
        print(f"\n🛡️ Escudo activado: se bloquearon {len(dropped_leaks)} columnas sospechosas.")
        for c in dropped_leaks:
            print(f"   - {c}")

    return safe_cols


def derive_nhl_first_period_pick(prob_over_55: float) -> Dict:
    p = float(np.clip(prob_over_55, 0.01, 0.99))
    expected_total_goals = float(np.clip(5.5 + 1.4 * (p - 0.5), 4.6, 6.4))
    lambda_p1 = expected_total_goals * 0.30
    p_over_15 = float(np.clip(1.0 - float(np.exp(-lambda_p1) * (1.0 + lambda_p1)), 0.01, 0.99))

    pick_over = p_over_15 >= 0.53
    confidence = int((0.5 + abs(p_over_15 - 0.5)) * 100)

    return {
        "q1_pick": "Over 1.5" if pick_over else "Under 1.5",
        "q1_market": "1P Goals O/U 1.5",
        "q1_line": 1.5,
        "q1_confidence": confidence,
        "q1_action": "JUGAR" if confidence >= 56 else "PASS",
        "q1_model_prob_yes": round(p_over_15, 4),
        "q1_calibrated_prob_yes": round(p_over_15, 4),
    }


def choose_optimal_binary_threshold(y_true: np.ndarray, probs: np.ndarray, market: str = "full_game") -> float:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)

    if market == "full_game":
        search_space = np.arange(0.45, 0.61, 0.01)
    else:
        search_space = np.arange(0.35, 0.66, 0.01)

    best_threshold = 0.50
    best_accuracy = -1.0

    for threshold in search_space:
        preds = (probs >= threshold).astype(int)
        acc = float((preds == y_true).mean())
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = float(threshold)

    return best_threshold


def fit_calibrated_ensemble(X_train: pd.DataFrame, y_train: pd.Series, market: str = "full_game"):
    n = len(X_train)
    if n < 40:
        base_X = X_train
        base_y = y_train
        calib_X = None
        calib_y = None
    else:
        split_idx = int(n * 0.8)
        split_idx = max(20, min(split_idx, n - 10))
        base_X = X_train.iloc[:split_idx]
        base_y = y_train.iloc[:split_idx]
        calib_X = X_train.iloc[split_idx:]
        calib_y = y_train.iloc[split_idx:]

    models = {
        "xgb": build_xgb_binary(),
        "lgbm": build_lgbm_binary(),
        "lgbm_sec": build_lgbm_binary(),
        "catboost": build_catboost_binary(),
    }

    for model in models.values():
        model.fit(base_X, base_y)

    calibrator = None
    chosen_threshold = 0.50

    if calib_X is not None and calib_y is not None and len(np.unique(calib_y)) > 1:
        calib_raw_probs = (
            models["xgb"].predict_proba(calib_X)[:, 1]
            + models["lgbm"].predict_proba(calib_X)[:, 1]
            + models["lgbm_sec"].predict_proba(calib_X)[:, 1]
            + models["catboost"].predict_proba(calib_X)[:, 1]
        ) / 4.0

        try:
            calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            calibrator.fit(calib_raw_probs.reshape(-1, 1), calib_y)
            calib_final_probs = calibrator.predict_proba(calib_raw_probs.reshape(-1, 1))[:, 1]
        except Exception:
            calibrator = None
            calib_final_probs = calib_raw_probs

        chosen_threshold = choose_optimal_binary_threshold(
            y_true=np.asarray(calib_y).astype(int),
            probs=calib_final_probs,
            market=market,
        )

    return models, calibrator, chosen_threshold


def predict_calibrated_ensemble(models, calibrator, X_test: pd.DataFrame, threshold: float = 0.50):
    raw_probs = (
        models["xgb"].predict_proba(X_test)[:, 1]
        + models["lgbm"].predict_proba(X_test)[:, 1]
        + models["lgbm_sec"].predict_proba(X_test)[:, 1]
        + models["catboost"].predict_proba(X_test)[:, 1]
    ) / 4.0

    if calibrator is not None:
        final_probs = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    else:
        final_probs = raw_probs

    preds = (final_probs >= threshold).astype(int)
    confidences = np.maximum(final_probs, 1.0 - final_probs)
    return preds, confidences, final_probs


def _bad_team_value(v) -> bool:
    if pd.isna(v):
        return True
    s = str(v).strip().upper()
    return s in {"", "0", "0.0", "NAN", "NONE", "NULL"}


def recover_team_names_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    if not RAW_FILE.exists():
        return df

    try:
        raw_df = pd.read_csv(RAW_FILE, dtype={"game_id": str})
    except Exception:
        return df

    if "game_id" not in raw_df.columns:
        return df

    needed_cols = {"game_id", "home_team", "away_team"}
    if not needed_cols.issubset(set(raw_df.columns)):
        return df

    raw_map = raw_df[["game_id", "home_team", "away_team"]].drop_duplicates("game_id", keep="last")
    raw_map = raw_map.rename(
        columns={
            "home_team": "raw_home_team",
            "away_team": "raw_away_team",
        }
    )

    df = df.merge(raw_map, on="game_id", how="left")

    bad_home = df["home_team"].apply(_bad_team_value)
    bad_away = df["away_team"].apply(_bad_team_value)

    recovered_home = int(bad_home.sum())
    recovered_away = int(bad_away.sum())

    df["home_team"] = df["home_team"].astype("object")
    df["away_team"] = df["away_team"].astype("object")

    df.loc[bad_home, "home_team"] = df.loc[bad_home, "raw_home_team"]
    df.loc[bad_away, "away_team"] = df.loc[bad_away, "raw_away_team"]

    df["home_team"] = df["home_team"].fillna("UNK").astype(str)
    df["away_team"] = df["away_team"].fillna("UNK").astype(str)

    df = df.drop(columns=["raw_home_team", "raw_away_team"], errors="ignore")

    if recovered_home or recovered_away:
        print(
            f"[FIX] Team labels recovered from raw file | "
            f"home_team: {recovered_home}, away_team: {recovered_away}"
        )

    return df


def confidence_tier_from_pct(conf_pct: float) -> str:
    if conf_pct >= 70:
        return "ELITE"
    if conf_pct >= 62:
        return "PREMIUM"
    if conf_pct >= 55:
        return "STRONG"
    if conf_pct >= 52:
        return "NORMAL"
    return "PASS"


def generate_historical_predictions():
    print("[NHL] Historical Predictions - Block Walk-Forward Validation")
    print("=" * 60)

    if not INPUT_FILE.exists():
        print(f"[ERROR] Dataset not found: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE, dtype={"game_id": str})
    df["date"] = df["date"].astype(str)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt"]).sort_values("date_dt").reset_index(drop=True)

    df = recover_team_names_from_raw(df)

    feature_cols = get_feature_columns(df)
    unique_dates = sorted(df["date"].unique())

    print(f"[OK] Loaded {len(df)} games | {len(feature_cols)} features | {len(unique_dates)} dates")

    predictions_by_date = defaultdict(list)
    overall_stats = defaultdict(lambda: {"correct": 0, "total": 0, "total_conf": 0.0})
    premium_stats = defaultdict(lambda: {"correct": 0, "total": 0, "total_conf": 0.0})

    STEP_DAYS = 7
    start_date_idx = min(20, len(unique_dates) // 10)

    print(f"\n[TRAIN] Executing Block Walk-Forward (Step: {STEP_DAYS} days)...")

    for i in range(start_date_idx, len(unique_dates), STEP_DAYS):
        test_start_date = unique_dates[i]
        test_end_date = unique_dates[min(i + STEP_DAYS - 1, len(unique_dates) - 1)]

        train_df = df[df["date"] < test_start_date].copy()
        test_df = df[(df["date"] >= test_start_date) & (df["date"] <= test_end_date)].copy()

        if len(train_df) < 20 or test_df.empty:
            continue

        print(
            f"   Training on {len(train_df)} games -> Predicting "
            f"{test_start_date} to {test_end_date} ({len(test_df)} games)"
        )

        X_train = train_df[feature_cols].fillna(0)
        X_test = test_df[feature_cols].fillna(0)

        # FULL GAME
        y_full_raw = pd.to_numeric(train_df["TARGET_full_game"], errors="coerce")
        valid_full = y_full_raw.isin([0, 1])
        y_full_train = y_full_raw.loc[valid_full].astype(int)

        pred_full, conf_full, prob_full = None, None, None
        full_threshold = 0.50

        if valid_full.sum() >= 10 and len(np.unique(y_full_train)) > 1:
            models_full, calib_full, full_threshold = fit_calibrated_ensemble(
                X_train.loc[valid_full],
                y_full_train,
                market="full_game",
            )
            pred_full, conf_full, prob_full = predict_calibrated_ensemble(
                models_full,
                calib_full,
                X_test,
                threshold=full_threshold,
            )

        # TOTALS 5.5
        y_totals_train = pd.to_numeric(train_df["TARGET_over_55"], errors="coerce").fillna(-1)
        valid_totals = y_totals_train >= 0
        pred_totals, conf_totals, prob_totals = None, None, None
        totals_threshold = 0.50

        if valid_totals.sum() >= 10 and len(np.unique(y_totals_train.loc[valid_totals])) > 1:
            models_totals, calib_totals, totals_threshold = fit_calibrated_ensemble(
                X_train.loc[valid_totals],
                y_totals_train.loc[valid_totals].astype(int),
                market="totals_5_5",
            )
            pred_totals, conf_totals, prob_totals = predict_calibrated_ensemble(
                models_totals,
                calib_totals,
                X_test,
                threshold=totals_threshold,
            )

        # HOME OVER 2.5
        y_home_train = pd.to_numeric(train_df["TARGET_home_over_25"], errors="coerce").fillna(-1)
        valid_home = y_home_train >= 0
        pred_home, conf_home, prob_home = None, None, None
        home_threshold = 0.50

        if valid_home.sum() >= 10 and len(np.unique(y_home_train.loc[valid_home])) > 1:
            models_home, calib_home, home_threshold = fit_calibrated_ensemble(
                X_train.loc[valid_home],
                y_home_train.loc[valid_home].astype(int),
                market="home_over_2_5",
            )
            pred_home, conf_home, prob_home = predict_calibrated_ensemble(
                models_home,
                calib_home,
                X_test,
                threshold=home_threshold,
            )

        for idx, test_row in test_df.reset_index(drop=True).iterrows():
            game_dict = {
                "game_id": str(test_row["game_id"]),
                "date": test_row["date"],
                "time": str(test_row.get("time", "19:00")),
                "home_team": str(test_row["home_team"]),
                "away_team": str(test_row["away_team"]),
                "home_score": None if pd.isna(test_row.get("home_score")) else int(test_row.get("home_score")),
                "away_score": None if pd.isna(test_row.get("away_score")) else int(test_row.get("away_score")),
                "odds_over_under": float(test_row.get("odds_over_under", 0) or 0),
                "closing_moneyline_odds": test_row.get("closing_moneyline_odds"),
                "home_moneyline_odds": test_row.get("home_moneyline_odds"),
                "away_moneyline_odds": test_row.get("away_moneyline_odds"),
                "closing_spread_odds": test_row.get("closing_spread_odds"),
                "closing_total_odds": test_row.get("closing_total_odds"),
                "odds_data_quality": str(test_row.get("odds_data_quality", "fallback")),
                "spread_pick": None,
                "spread_market": None,
                "spread_line": None,
                "spread_confidence": None,
                "correct_spread": None,
            }

            # FULL GAME
            if pred_full is not None:
                actual_target = test_row.get("TARGET_full_game", np.nan)

                if pd.notna(actual_target) and int(actual_target) in (0, 1):
                    confidence = float(conf_full[idx])
                    prob_home_win = float(prob_full[idx])
                    pred_bin = int(pred_full[idx])  # 1=home, 0=away

                    pick_team = test_row["home_team"] if pred_bin == 1 else test_row["away_team"]
                    actual_bin = int(actual_target)
                    actual_winner = test_row["home_team"] if actual_bin == 1 else test_row["away_team"]
                    is_correct = int(pred_bin == actual_bin)

                    conf_pct = int(round(confidence * 100))
                    tier = confidence_tier_from_pct(conf_pct)

                    game_dict["moneyline_pick"] = str(pick_team)
                    game_dict["moneyline_confidence"] = conf_pct
                    game_dict["moneyline_recommended_score"] = round(confidence * 100, 1)
                    game_dict["moneyline_prob_home"] = round(prob_home_win, 4)
                    game_dict["moneyline_threshold_used"] = round(float(full_threshold), 4)
                    game_dict["moneyline_actual"] = str(actual_winner)
                    game_dict["moneyline_correct"] = is_correct

                    game_dict["market"] = "FULL_GAME"
                    game_dict["pick"] = str(pick_team)
                    game_dict["pick_team"] = str(pick_team)
                    game_dict["probability"] = round(
                        prob_home_win if pred_bin == 1 else (1.0 - prob_home_win),
                        4
                    )
                    game_dict["threshold_used"] = round(float(full_threshold), 4)
                    game_dict["confidence"] = conf_pct
                    game_dict["tier"] = tier
                    game_dict["actual_winner"] = str(actual_winner)
                    game_dict["correct"] = is_correct

                    overall_stats["full_game"]["correct"] += is_correct
                    overall_stats["full_game"]["total"] += 1
                    overall_stats["full_game"]["total_conf"] += confidence

                    if confidence >= 0.55:
                        premium_stats["full_game"]["correct"] += is_correct
                        premium_stats["full_game"]["total"] += 1
                        premium_stats["full_game"]["total_conf"] += confidence

            # TOTALS 5.5
            if pred_totals is not None:
                confidence = float(conf_totals[idx])
                total_labels = ["Under 5.5", "Over 5.5"]
                actual = int(test_row["TARGET_over_55"])
                pred_val = int(pred_totals[idx])
                total_pick = total_labels[pred_val]
                is_correct = int(pred_val == actual)

                game_dict["total_pick"] = total_pick
                game_dict["total_market"] = "Total Goals O/U 5.5"
                game_dict["total_line"] = 5.5
                game_dict["total_confidence"] = int(round(confidence * 100))
                game_dict["total_recommended_pick"] = total_pick
                game_dict["total_recommended_score"] = round(confidence * 100, 1)
                game_dict["total_prob_over"] = round(float(prob_totals[idx]), 4)
                game_dict["total_threshold_used"] = round(float(totals_threshold), 4)
                game_dict["total_actual"] = total_labels[actual]
                game_dict["total_correct"] = is_correct
                game_dict["correct_total"] = is_correct
                game_dict.update(derive_nhl_first_period_pick(float(prob_totals[idx])))

                overall_stats["totals_5_5"]["correct"] += is_correct
                overall_stats["totals_5_5"]["total"] += 1
                overall_stats["totals_5_5"]["total_conf"] += confidence

                if confidence >= 0.55:
                    premium_stats["totals_5_5"]["correct"] += is_correct
                    premium_stats["totals_5_5"]["total"] += 1
                    premium_stats["totals_5_5"]["total_conf"] += confidence

            # HOME OVER 2.5
            if pred_home is not None:
                confidence = float(conf_home[idx])
                home_labels = ["Home Under 2.5", "Home Over 2.5"]
                actual = int(test_row["TARGET_home_over_25"])
                pred_val = int(pred_home[idx])
                is_correct = int(pred_val == actual)

                game_dict["home_over_pick"] = home_labels[pred_val]
                game_dict["home_over_confidence"] = int(round(confidence * 100))
                game_dict["home_over_recommended_score"] = round(confidence * 100, 1)
                game_dict["home_over_prob_yes"] = round(float(prob_home[idx]), 4)
                game_dict["home_over_threshold_used"] = round(float(home_threshold), 4)
                game_dict["home_over_actual"] = home_labels[actual]
                game_dict["home_over_correct"] = is_correct

                overall_stats["home_over_2_5"]["correct"] += is_correct
                overall_stats["home_over_2_5"]["total"] += 1
                overall_stats["home_over_2_5"]["total_conf"] += confidence

                if confidence >= 0.55:
                    premium_stats["home_over_2_5"]["correct"] += is_correct
                    premium_stats["home_over_2_5"]["total"] += 1
                    premium_stats["home_over_2_5"]["total_conf"] += confidence

            predictions_by_date[test_row["date"]].append(game_dict)

    print("\n[SAVE] Saving predictions by date...")
    for date_str in sorted(predictions_by_date.keys()):
        output_file = HISTORICAL_DIR / f"{date_str}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(predictions_by_date[date_str], f, indent=2, ensure_ascii=False)

    print("\n[STATS] OVERALL ACCURACY (Todos los picks):")
    for market in ["full_game", "totals_5_5", "home_over_2_5"]:
        stats = overall_stats[market]
        if stats["total"] > 0:
            accuracy = stats["correct"] / stats["total"]
            avg_conf = stats["total_conf"] / stats["total"]
            print(f"   {market.upper()}:")
            print(f"      Accuracy: {accuracy:.2%} ({stats['correct']}/{stats['total']})")
            print(f"      Avg Confidence: {avg_conf:.1%}")

    print("\n[STATS] PREMIUM PICKS ACCURACY (Confianza >= 55%):")
    for market in ["full_game", "totals_5_5", "home_over_2_5"]:
        stats = premium_stats[market]
        if stats["total"] > 0:
            accuracy = stats["correct"] / stats["total"]
            avg_conf = stats["total_conf"] / stats["total"]
            print(f"   {market.upper()}:")
            print(f"      Accuracy: {accuracy:.2%} ({stats['correct']}/{stats['total']})")
            print(f"      Avg Confidence: {avg_conf:.1%}")


if __name__ == "__main__":
    generate_historical_predictions()