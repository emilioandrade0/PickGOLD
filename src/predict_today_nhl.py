"""
Predict today's upcoming NHL games using 4-model ensemble.
Loads real data from nhl_upcoming_schedule.csv and builds features from historical data.
Mirrors the structure of MLB's predict_today_mlb.py
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from calibration import calibrate_probability, load_calibration_config
from pattern_engine import aggregate_pattern_edge
from pattern_engine_nhl import generate_nhl_patterns
from pick_selector import fuse_with_pattern_score, recommendation_score

BASE_DIR = Path(__file__).resolve().parent
HISTORY_FILE = BASE_DIR / "data" / "nhl" / "processed" / "model_ready_features_nhl.csv"
UPCOMING_FILE = BASE_DIR / "data" / "nhl" / "raw" / "nhl_upcoming_schedule.csv"
MODELS_DIR = BASE_DIR / "data" / "nhl" / "models"
PREDICTIONS_DIR = BASE_DIR / "data" / "nhl" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_FILE = MODELS_DIR / "calibration_params.json"

# Threshold behavior for live binary markets:
# - trained: use threshold from each market metadata.json
# - fixed: use values in LIVE_FIXED_THRESHOLDS
NHL_LIVE_THRESHOLD_MODE = os.getenv("NHL_LIVE_THRESHOLD_MODE", "trained").strip().lower()
LIVE_FIXED_THRESHOLDS = {
    "spread_2_5": 0.5,
    "home_over_2_5": 0.5,
}

NON_FEATURE_COLUMNS = {
    "game_id", "date", "date_dt", "time", "season", "home_team", "away_team",
    "home_score", "away_score", "total_goals", "is_draw", "completed",
    "venue_name", "odds_details", "odds_over_under",
    "TARGET_full_game", "TARGET_over_55", "TARGET_home_over_25",
}


def load_history() -> pd.DataFrame:
    """Load historical features for building snapshots."""
    if not HISTORY_FILE.exists():
        raise FileNotFoundError(f"History file not found: {HISTORY_FILE}")

    df = pd.read_csv(HISTORY_FILE, dtype={"game_id": str})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").astype(str)
    return df.sort_values(["date", "game_id"]).reset_index(drop=True)


def get_team_snapshot(history_df: pd.DataFrame, team: str, cutoff_date: str) -> Optional[Dict]:
    """Get the last game features snapshot for a team before cutoff_date."""
    prior = history_df[history_df["date"] < cutoff_date].copy()
    if prior.empty:
        return None

    home_rows = prior[prior["home_team"] == team]
    away_rows = prior[prior["away_team"] == team]

    last_home = home_rows.iloc[-1] if not home_rows.empty else None
    last_away = away_rows.iloc[-1] if not away_rows.empty else None

    last_game = None
    if last_home is not None and last_away is not None:
        last_game = last_home if last_home["date"] >= last_away["date"] else last_away
    elif last_home is not None:
        last_game = last_home
    elif last_away is not None:
        last_game = last_away
    else:
        return None

    prefix = "home_" if last_game["home_team"] == team else "away_"

    snapshot = {}
    for col in history_df.columns:
        if col.startswith(prefix) and col not in NON_FEATURE_COLUMNS:
            feature_name = col.replace(prefix, "")
            snapshot[feature_name] = last_game[col]

    return snapshot


def build_pregame_features(history_df: pd.DataFrame, upcoming_game: Dict) -> Optional[Dict]:
    """Build feature vector for an upcoming game."""
    cutoff_date = upcoming_game.get("date")
    home_team = upcoming_game.get("home_team")
    away_team = upcoming_game.get("away_team")

    if not cutoff_date or not home_team or not away_team:
        return None

    home_snap = get_team_snapshot(history_df, home_team, cutoff_date)
    away_snap = get_team_snapshot(history_df, away_team, cutoff_date)

    if not home_snap or not away_snap:
        return None

    features = {}
    for key, val in home_snap.items():
        features[f"home_{key}"] = val
    for key, val in away_snap.items():
        features[f"away_{key}"] = val

    return features


def load_model_assets(market: str) -> Dict:
    """Load trained models and metadata for a market."""
    market_dir = MODELS_DIR / market

    if not market_dir.exists():
        raise FileNotFoundError(f"Models not found for {market}")

    try:
        assets = {
            "xgb": joblib.load(market_dir / "xgboost_model.pkl"),
            "lgbm": joblib.load(market_dir / "lgbm_model.pkl"),
            "lgbm_secondary": joblib.load(market_dir / "lgbm_secondary_model.pkl"),
            "catboost": joblib.load(market_dir / "catboost_model.pkl"),
        }

        with open(market_dir / "metadata.json", "r") as f:
            assets["metadata"] = json.load(f)

        return assets
    except Exception as e:
        raise FileNotFoundError(f"Could not load models for {market}: {e}")


def predict_market(assets: Dict, features_dict: Dict, market: str = None) -> Tuple[int, float, np.ndarray]:
    """Generate 4-model ensemble predictions for a single game."""
    metadata = assets["metadata"]
    feature_names = metadata["feature_columns"]

    X = pd.DataFrame(
        [{f: features_dict.get(f, 0.0) for f in feature_names}],
        columns=feature_names,
    )

    xgb_probs = assets["xgb"].predict_proba(X)
    lgbm_probs = assets["lgbm"].predict_proba(X)
    lgbm_sec_probs = assets["lgbm_secondary"].predict_proba(X)
    catboost_probs = assets["catboost"].predict_proba(X)

    ensemble_probs = (xgb_probs + lgbm_probs + lgbm_sec_probs + catboost_probs) / 4.0

    problem_type = metadata["problem_type"]
    threshold = metadata.get("threshold", 0.5)

    if problem_type != "multiclass" and market:
        if NHL_LIVE_THRESHOLD_MODE == "fixed":
            threshold = float(LIVE_FIXED_THRESHOLDS.get(market, threshold))
        else:
            threshold = float(threshold)

    if problem_type == "multiclass":
        preds = np.argmax(ensemble_probs, axis=1)
        confidences = np.max(ensemble_probs, axis=1)
    else:
        if threshold:
            preds = (ensemble_probs[:, 1] >= threshold).astype(int)
        else:
            preds = np.argmax(ensemble_probs, axis=1)
        confidences = np.abs(ensemble_probs[:, 1] - 0.5) + 0.5

    if market == "full_game":
        preds = preds + 1

    return int(preds[0]), float(confidences[0]), ensemble_probs[0]


def derive_nhl_first_period_pick(prob_over_55: float) -> Dict:
    """Derive 1P O/U 1.5 prediction from full-game totals probability.

    This is a calibrated proxy while first-period training labels are unavailable.
    """
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


def predict_today_nhl():
    """Generate REAL predictions for upcoming NHL games."""
    print("[NHL] Live Predictions")
    print("=" * 60)
    print(f"[CFG] threshold_mode={NHL_LIVE_THRESHOLD_MODE}")

    try:
        history_df = load_history()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    if not UPCOMING_FILE.exists():
        print(f"⚠️  No upcoming schedule found: {UPCOMING_FILE}")
        output_file = PREDICTIONS_DIR / datetime.now().strftime("%Y-%m-%d.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        return

    upcoming_df = pd.read_csv(UPCOMING_FILE, dtype={"game_id": str})
    if upcoming_df.empty:
        print("⚠️  No upcoming games today")
        output_file = PREDICTIONS_DIR / datetime.now().strftime("%Y-%m-%d.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        return

    print(f"[OK] Loaded {len(upcoming_df)} upcoming games")

    upcoming_df["date"] = pd.to_datetime(upcoming_df["date"], errors="coerce").astype(str)

    markets_available = {}
    for market in ["full_game", "spread_2_5", "home_over_2_5"]:
        try:
            markets_available[market] = load_model_assets(market)
        except Exception as e:
            print(f"   [WARN] {market} models not available: {e}")

    if not markets_available:
        print("   [ERROR] No models available for prediction")
        return

    calibration_cfg = load_calibration_config(CALIBRATION_FILE)

    for date_str, group_df in upcoming_df.groupby("date"):
        print(f"\n[{date_str}] Predicting: {len(group_df)} games")

        games_predictions = []

        for _, row in group_df.iterrows():
            game_id = str(row.get("game_id", ""))
            home_team = row.get("home_team")
            away_team = row.get("away_team")
            time_str = row.get("time", "19:00")

            features = build_pregame_features(
                history_df,
                {"date": date_str, "home_team": home_team, "away_team": away_team},
            )

            if not features:
                print(f"   [WARN] Could not build features for game {game_id}: {home_team} vs {away_team}")
                continue

            game_dict = {
                "game_id": game_id,
                "date": date_str,
                "time": time_str,
                "home_team": home_team,
                "away_team": away_team,
            }

            nhl_patterns = generate_nhl_patterns(features)
            pattern_edge = aggregate_pattern_edge(nhl_patterns)
            game_dict["detected_patterns"] = nhl_patterns
            game_dict["pattern_edge"] = round(pattern_edge, 4)

            confidences_list = []

            if "full_game" in markets_available:
                try:
                    pred, conf, probs = predict_market(
                        markets_available["full_game"],
                        features,
                        market="full_game",
                    )
                    full_labels = ["N/A", "Home Win", "Away Win"]
                    game_dict["full_game_pick"] = full_labels[int(pred)]

                    model_prob_pick = float(conf)
                    calibrated_prob_pick = calibrate_probability(
                        model_prob_pick,
                        "nhl",
                        "full_game",
                        calibration_cfg,
                    )
                    score = fuse_with_pattern_score(
                        recommendation_score(calibrated_prob_pick),
                        pattern_edge,
                    )

                    game_dict["full_game_confidence"] = int(calibrated_prob_pick * 100)
                    game_dict["full_game_model_prob_pick"] = round(model_prob_pick, 4)
                    game_dict["full_game_calibrated_prob_pick"] = round(calibrated_prob_pick, 4)
                    game_dict["full_game_recommended_score"] = round(score, 1)

                    confidences_list.append(game_dict["full_game_confidence"])
                except Exception as e:
                    print(f"   [WARN] full_game error for {game_id}: {e}")
                    game_dict["full_game_pick"] = "N/A"
                    game_dict["full_game_confidence"] = 0
                    game_dict["full_game_recommended_score"] = 0.0

            # TOTAL predictions (NHL usa totals, no spread real)
            if "spread_2_5" in markets_available:
                try:
                    pred, conf, probs = predict_market(
                        markets_available["spread_2_5"],
                        features,
                        market="spread_2_5",
                    )
                    total_labels = ["Under 5.5", "Over 5.5"]
                    total_pick = total_labels[int(pred)]

                    model_prob_pick = float(conf)
                    calibrated_prob_pick = calibrate_probability(
                        model_prob_pick,
                        "nhl",
                        "total",
                        calibration_cfg,
                    )
                    score = fuse_with_pattern_score(
                        recommendation_score(calibrated_prob_pick),
                        pattern_edge,
                    )

                    game_dict["total_pick"] = total_pick
                    game_dict["total_market"] = "Total Goals O/U 5.5"
                    game_dict["total_line"] = 5.5
                    game_dict["total_confidence"] = int(calibrated_prob_pick * 100)
                    game_dict["total_model_prob_pick"] = round(model_prob_pick, 4)
                    game_dict["total_calibrated_prob_pick"] = round(calibrated_prob_pick, 4)
                    game_dict["total_recommended_pick"] = total_pick
                    game_dict["total_recommended_score"] = round(score, 1)

                    # No contaminar spread_* con totals en NHL
                    game_dict["spread_pick"] = None
                    game_dict["spread_market"] = None
                    game_dict["spread_line"] = None
                    game_dict["spread_confidence"] = None
                    game_dict["spread_model_prob_pick"] = None
                    game_dict["spread_calibrated_prob_pick"] = None
                    game_dict["spread_recommended_score"] = None

                    over_55_prob = float(probs[1]) if hasattr(probs, "__len__") and len(probs) > 1 else float(conf)
                    game_dict.update(derive_nhl_first_period_pick(over_55_prob))

                    confidences_list.append(game_dict["total_confidence"])
                except Exception as e:
                    print(f"   [WARN] total error for {game_id}: {e}")

                    game_dict["total_pick"] = "N/A"
                    game_dict["total_market"] = "Total Goals O/U 5.5"
                    game_dict["total_line"] = 5.5
                    game_dict["total_confidence"] = 0
                    game_dict["total_recommended_pick"] = "N/A"
                    game_dict["total_recommended_score"] = 0.0

                    game_dict["spread_pick"] = None
                    game_dict["spread_market"] = None
                    game_dict["spread_line"] = None
                    game_dict["spread_confidence"] = None
                    game_dict["spread_model_prob_pick"] = None
                    game_dict["spread_calibrated_prob_pick"] = None
                    game_dict["spread_recommended_score"] = None

            if "home_over_2_5" in markets_available:
                try:
                    pred, conf, probs = predict_market(
                        markets_available["home_over_2_5"],
                        features,
                        market="home_over_2_5",
                    )
                    home_labels = ["Home Under 2.5", "Home Over 2.5"]
                    game_dict["home_over_pick"] = home_labels[int(pred)]

                    model_prob_pick = float(conf)
                    calibrated_prob_pick = calibrate_probability(
                        model_prob_pick,
                        "nhl",
                        "home_over_25",
                        calibration_cfg,
                    )
                    score = fuse_with_pattern_score(
                        recommendation_score(calibrated_prob_pick),
                        pattern_edge,
                    )

                    game_dict["home_over_confidence"] = int(calibrated_prob_pick * 100)
                    game_dict["home_over_model_prob_pick"] = round(model_prob_pick, 4)
                    game_dict["home_over_calibrated_prob_pick"] = round(calibrated_prob_pick, 4)
                    game_dict["home_over_recommended_score"] = round(score, 1)

                    confidences_list.append(game_dict["home_over_confidence"])
                except Exception as e:
                    print(f"   [WARN] home_over error for {game_id}: {e}")
                    game_dict["home_over_pick"] = "N/A"
                    game_dict["home_over_confidence"] = 0
                    game_dict["home_over_recommended_score"] = 0.0

            avg_conf = sum(confidences_list) / len(confidences_list) if confidences_list else 0
            game_dict["recommended"] = avg_conf >= 55

            score_candidates = [
                float(game_dict.get("full_game_recommended_score", 0.0) or 0.0),
                float(game_dict.get("total_recommended_score", 0.0) or 0.0),
                float(game_dict.get("home_over_recommended_score", 0.0) or 0.0),
            ]
            best_score = max(score_candidates) if score_candidates else 0.0
            game_dict["recommended_score"] = round(best_score, 1)
            game_dict["recommended"] = best_score >= 56.0

            games_predictions.append(game_dict)

        output_file = PREDICTIONS_DIR / f"{date_str}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(games_predictions, f, indent=2, ensure_ascii=False)

        print(f"   [OK] Saved {len(games_predictions)} predictions to {output_file.name}")


if __name__ == "__main__":
    predict_today_nhl()