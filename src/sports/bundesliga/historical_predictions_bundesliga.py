import json
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

import event_adjustments_bundesliga as adj_engine
from event_adjustments_bundesliga import (
    apply_probability_adjustment,
    detect_pre_match_events,
    get_h2h_features,
    get_recent_team_form_features,
    probability_to_confidence,
)

BASE_DIR = SRC_ROOT
FEATURES_FILE = BASE_DIR / "data" / "bundesliga" / "processed" / "model_ready_features_bundesliga.csv"
RAW_HISTORY_FILE = BASE_DIR / "data" / "bundesliga" / "raw" / "bundesliga_advanced_history.csv"
MODELS_DIR = BASE_DIR / "data" / "bundesliga" / "models"
PREDICTIONS_DIR = BASE_DIR / "data" / "bundesliga" / "historical_predictions"
REPORTS_DIR = BASE_DIR / "data" / "bundesliga" / "reports"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Quick on/off switch requested by user.
USE_EVENT_ADJUSTMENTS = True


class ConstantBinaryModel:
    """Fallback model placeholder for joblib deserialization."""

    def __init__(self, constant_class: int = 0):
        self.constant_class = int(constant_class)

    def predict_proba(self, X):
        n = len(X)
        probs = np.zeros((n, 2), dtype=float)
        probs[:, self.constant_class] = 1.0
        return probs


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_market_assets(market_key: str):
    market_dir = MODELS_DIR / market_key
    xgb = joblib.load(market_dir / "xgb_model.pkl")
    lgbm = joblib.load(market_dir / "lgbm_model.pkl")
    lgbm_secondary = joblib.load(market_dir / "lgbm_secondary_model.pkl")
    catboost_path = market_dir / "catboost_model.pkl"
    catboost = joblib.load(catboost_path) if catboost_path.exists() else None
    feature_columns = load_json(market_dir / "feature_columns.json")
    metadata = load_json(market_dir / "metadata.json")
    threshold = float(metadata.get("ensemble_threshold", 0.5) or 0.5)

    if isinstance(feature_columns, dict):
        feat_list = feature_columns.get("features", [])
    else:
        feat_list = feature_columns

    return {
        "xgb": xgb,
        "lgbm": lgbm,
        "lgbm_secondary": lgbm_secondary,
        "catboost": catboost,
        "feature_columns": feat_list,
        "threshold": threshold,
    }


def predict_market_probabilities(row_df: pd.DataFrame, market_key: str, assets: dict):
    feat_list = assets[market_key]["feature_columns"]
    X = row_df[feat_list].replace([np.inf, -np.inf], np.nan).fillna(0)

    xgb_probs = assets[market_key]["xgb"].predict_proba(X)
    lgbm_probs = assets[market_key]["lgbm"].predict_proba(X)
    lgbm_sec_probs = assets[market_key]["lgbm_secondary"].predict_proba(X)
    probs = xgb_probs + lgbm_probs + lgbm_sec_probs
    model_count = 3
    if assets[market_key]["catboost"] is not None:
        probs = probs + assets[market_key]["catboost"].predict_proba(X)
        model_count += 1

    probs = probs / float(model_count)
    return probs[0]


def normalize_multiclass_probs(home_p: float, away_p: float, draw_p: float):
    arr = np.array([away_p, home_p, draw_p], dtype=float)
    arr = np.clip(arr, 1e-9, None)
    arr = arr / arr.sum()
    return float(arr[1]), float(arr[0]), float(arr[2])


def _is_constant_binary_asset(asset: dict) -> bool:
    for key in ["xgb", "lgbm", "lgbm_secondary", "catboost"]:
        model = asset.get(key)
        if model is not None and model.__class__.__name__ == "ConstantBinaryModel":
            return True
    return False


def _team_recent_corners_avg(raw_history_df: pd.DataFrame, team: str, match_date: str, lookback: int = 8) -> float:
    prior = raw_history_df[raw_history_df["date"] < match_date]
    if prior.empty:
        return 5.0

    team_rows = prior[(prior["home_team"] == team) | (prior["away_team"] == team)].tail(lookback)
    if team_rows.empty:
        return 5.0

    corners = []
    for _, r in team_rows.iterrows():
        if "home_corners" in team_rows.columns and "away_corners" in team_rows.columns:
            if r["home_team"] == team:
                value = r.get("home_corners", np.nan)
            else:
                value = r.get("away_corners", np.nan)
            if pd.notna(value):
                corners.append(float(value))

    if len(corners) >= 3:
        return float(np.mean(corners))

    # If corners history is sparse, approximate team corners from recent game pace.
    goals_totals = []
    for _, r in team_rows.iterrows():
        hs = r.get("home_score", np.nan)
        aw = r.get("away_score", np.nan)
        if pd.notna(hs) and pd.notna(aw):
            goals_totals.append(float(hs) + float(aw))

    if goals_totals:
        avg_total_goals = float(np.mean(goals_totals))
        return float(np.clip(3.2 + 0.95 * avg_total_goals, 3.0, 7.5))

    return 5.0


def estimate_corners_over95_probability(raw_history_df: pd.DataFrame, home_team: str, away_team: str, match_date: str) -> float:
    home_avg = _team_recent_corners_avg(raw_history_df, home_team, match_date)
    away_avg = _team_recent_corners_avg(raw_history_df, away_team, match_date)
    expected_total = home_avg + away_avg

    # Smooth logistic mapping around the 9.5 corners line.
    prob = 1.0 / (1.0 + np.exp(-(expected_total - 9.5) / 1.35))
    return float(np.clip(prob, 0.05, 0.95))


def parse_actual_result(row: pd.Series):
    home_score = int(row.get("home_score", 0) or 0)
    away_score = int(row.get("away_score", 0) or 0)

    if home_score > away_score:
        actual_class = 1
        actual_result = str(row["home_team"])
    elif away_score > home_score:
        actual_class = 0
        actual_result = str(row["away_team"])
    else:
        actual_class = 2
        actual_result = "DRAW"

    total_goals = home_score + away_score
    actual_over = 1 if total_goals > 2.5 else 0
    actual_btts = 1 if (home_score > 0 and away_score > 0) else 0
    corners_raw = row.get("TARGET_corners_over_95", 0)
    actual_corners = 0 if pd.isna(corners_raw) else int(corners_raw)

    return {
        "actual_home_score": home_score,
        "actual_away_score": away_score,
        "actual_result": actual_result,
        "actual_full_game_class": actual_class,
        "actual_over": actual_over,
        "actual_btts": actual_btts,
        "actual_corners": actual_corners,
    }


def pick_from_multiclass_probs(probs: np.ndarray, home_team: str, away_team: str):
    idx = int(np.argmax(probs))
    if idx == 1:
        return home_team, float(probs[1]), idx
    if idx == 2:
        return "DRAW", float(probs[2]), idx
    return away_team, float(probs[0]), idx


def save_predictions_by_date(predictions: list):
    by_date = {}
    for pred in predictions:
        d = pred["date"]
        by_date.setdefault(d, []).append(pred)

    for date_str, rows in sorted(by_date.items()):
        out_path = PREDICTIONS_DIR / f"{date_str}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"   OK {date_str}: {len(rows)} predicciones")

    return by_date


def build_walk_forward_predictions(df_features: pd.DataFrame, raw_history_df: pd.DataFrame):
    assets = {
        "full_game": load_market_assets("full_game"),
        "over_25": load_market_assets("over_25"),
        "btts": load_market_assets("btts"),
        "corners_over_95": load_market_assets("corners_over_95"),
    }

    adj_engine.USE_EVENT_ADJUSTMENTS = USE_EVENT_ADJUSTMENTS
    corners_constant_fallback = _is_constant_binary_asset(assets["corners_over_95"])

    rows = []
    for i, (_, row) in enumerate(df_features.iterrows(), start=1):
        row_df = pd.DataFrame([row])

        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        match_date = str(row["date"])

        fg_probs_base = predict_market_probabilities(row_df, "full_game", assets)
        over_prob_base = float(predict_market_probabilities(row_df, "over_25", assets)[1])
        btts_prob_base = float(predict_market_probabilities(row_df, "btts", assets)[1])
        corners_prob_base = float(predict_market_probabilities(row_df, "corners_over_95", assets)[1])
        if corners_constant_fallback:
            corners_prob_base = estimate_corners_over95_probability(raw_history_df, home_team, away_team, match_date)

        fg_base_pick, fg_base_pick_prob, fg_base_class = pick_from_multiclass_probs(
            fg_probs_base, home_team, away_team
        )

        over_thr = assets["over_25"]["threshold"]
        btts_thr = assets["btts"]["threshold"]
        corners_thr = assets["corners_over_95"]["threshold"]
        if corners_constant_fallback:
            corners_thr = max(corners_thr, 0.86)

        over_base_pick = "OVER 2.5" if over_prob_base >= over_thr else "UNDER 2.5"
        btts_base_pick = "BTTS YES" if btts_prob_base >= btts_thr else "BTTS NO"
        corners_base_pick = "OVER 9.5" if corners_prob_base >= corners_thr else "UNDER 9.5"

        fg_probs_adj = np.array(fg_probs_base, dtype=float)
        over_prob_adj = over_prob_base
        btts_prob_adj = btts_prob_base
        corners_prob_adj = corners_prob_base

        detected_events = []
        full_game_breakdown = []
        over_breakdown = []
        btts_breakdown = []
        corners_breakdown = []

        if USE_EVENT_ADJUSTMENTS:
            recent_features = get_recent_team_form_features(
                raw_history_df,
                home_team=home_team,
                away_team=away_team,
                match_date=match_date,
                lookback=5,
            )
            h2h_features = get_h2h_features(
                raw_history_df,
                home_team=home_team,
                away_team=away_team,
                match_date=match_date,
            )

            events_full = detect_pre_match_events(recent_features, h2h_features, market_type="full_game")
            events_over = detect_pre_match_events(recent_features, h2h_features, market_type="over_25")
            events_btts = detect_pre_match_events(recent_features, h2h_features, market_type="btts")
            detected_events = events_full

            fg_home_adj = apply_probability_adjustment(
                float(fg_probs_base[1]), events_full, h2h_features, market_type="full_game_home"
            )
            fg_away_adj = apply_probability_adjustment(
                float(fg_probs_base[0]), events_full, h2h_features, market_type="full_game_away"
            )
            fg_draw_adj = apply_probability_adjustment(
                float(fg_probs_base[2]), events_full, h2h_features, market_type="full_game_draw"
            )

            home_adj, away_adj, draw_adj = normalize_multiclass_probs(
                fg_home_adj["adjusted_prob"], fg_away_adj["adjusted_prob"], fg_draw_adj["adjusted_prob"]
            )
            fg_probs_adj = np.array([away_adj, home_adj, draw_adj], dtype=float)

            over_adj_info = apply_probability_adjustment(
                over_prob_base, events_over, h2h_features, market_type="over_25"
            )
            btts_adj_info = apply_probability_adjustment(
                btts_prob_base, events_btts, h2h_features, market_type="btts"
            )

            over_prob_adj = float(over_adj_info["adjusted_prob"])
            btts_prob_adj = float(btts_adj_info["adjusted_prob"])

            if fg_base_pick == home_team:
                full_game_breakdown = fg_home_adj["adjustment_breakdown"]
            elif fg_base_pick == away_team:
                full_game_breakdown = fg_away_adj["adjustment_breakdown"]
            else:
                full_game_breakdown = fg_draw_adj["adjustment_breakdown"]

            over_breakdown = over_adj_info["adjustment_breakdown"]
            btts_breakdown = btts_adj_info["adjustment_breakdown"]

        fg_adj_pick, fg_adj_pick_prob, fg_adj_class = pick_from_multiclass_probs(
            fg_probs_adj, home_team, away_team
        )
        over_adj_pick = "OVER 2.5" if over_prob_adj >= over_thr else "UNDER 2.5"
        btts_adj_pick = "BTTS YES" if btts_prob_adj >= btts_thr else "BTTS NO"
        corners_adj_pick = "OVER 9.5" if corners_prob_adj >= corners_thr else "UNDER 9.5"

        actual = parse_actual_result(row)

        rows.append(
            {
                "game_id": str(row["game_id"]),
                "date": match_date,
                "home_team": home_team,
                "away_team": away_team,
                "odds_over_under": float(row.get("odds_over_under", 0) or 0),
                "closing_moneyline_odds": row.get("closing_moneyline_odds"),
                "home_moneyline_odds": row.get("home_moneyline_odds"),
                "away_moneyline_odds": row.get("away_moneyline_odds"),
                "closing_total_odds": row.get("closing_total_odds"),
                "odds_data_quality": str(row.get("odds_data_quality", "fallback")),

                "full_game_pick": fg_base_pick,
                "full_game_confidence": probability_to_confidence(fg_base_pick_prob),
                "full_game_tier": "ELITE" if probability_to_confidence(fg_base_pick_prob) >= 72 else "NORMAL",
                "recommended_pick": fg_adj_pick,
                "recommended_confidence": probability_to_confidence(fg_adj_pick_prob),

                "base_probability": round(float(fg_base_pick_prob), 6),
                "adjusted_probability": round(float(fg_adj_pick_prob), 6),
                "adjustment_amount": round(float(fg_adj_pick_prob - fg_base_pick_prob), 6),
                "adjustment_breakdown": full_game_breakdown,
                "detected_events": detected_events,

                "total_pick": over_base_pick,
                "total_recommended_pick": over_adj_pick,
                "total_base_probability": round(float(over_prob_base), 6),
                "total_adjusted_probability": round(float(over_prob_adj), 6),
                "total_adjustment_amount": round(float(over_prob_adj - over_prob_base), 6),
                "total_adjustment_breakdown": over_breakdown,

                "btts_pick": btts_base_pick,
                "btts_recommended_pick": btts_adj_pick,
                "btts_base_probability": round(float(btts_prob_base), 6),
                "btts_adjusted_probability": round(float(btts_prob_adj), 6),
                "btts_adjustment_amount": round(float(btts_prob_adj - btts_prob_base), 6),
                "btts_adjustment_breakdown": btts_breakdown,

                "corners_pick": corners_base_pick,
                "corners_recommended_pick": corners_adj_pick,
                "corners_base_probability": round(float(corners_prob_base), 6),
                "corners_adjusted_probability": round(float(corners_prob_adj), 6),
                "corners_adjustment_amount": round(float(corners_prob_adj - corners_prob_base), 6),
                "corners_adjustment_breakdown": corners_breakdown,

                "actual_home_score": actual["actual_home_score"],
                "actual_away_score": actual["actual_away_score"],
                "actual_result": actual["actual_result"],

                "correct_full_game_base": 1 if fg_base_class == actual["actual_full_game_class"] else 0,
                "correct_full_game_adjusted": 1 if fg_adj_class == actual["actual_full_game_class"] else 0,
                "correct_total_base": 1 if (1 if "OVER" in over_base_pick else 0) == actual["actual_over"] else 0,
                "correct_total_adjusted": 1 if (1 if "OVER" in over_adj_pick else 0) == actual["actual_over"] else 0,
                "correct_btts_base": 1 if (1 if "YES" in btts_base_pick else 0) == actual["actual_btts"] else 0,
                "correct_btts_adjusted": 1 if (1 if "YES" in btts_adj_pick else 0) == actual["actual_btts"] else 0,
                "correct_corners_base": 1 if (1 if "OVER" in corners_base_pick else 0) == actual["actual_corners"] else 0,
                "correct_corners_adjusted": 1 if (1 if "OVER" in corners_adj_pick else 0) == actual["actual_corners"] else 0,

                "actual_full_game_class": actual["actual_full_game_class"],
                "fg_prob_away_base": float(fg_probs_base[0]),
                "fg_prob_home_base": float(fg_probs_base[1]),
                "fg_prob_draw_base": float(fg_probs_base[2]),
                "fg_prob_away_adjusted": float(fg_probs_adj[0]),
                "fg_prob_home_adjusted": float(fg_probs_adj[1]),
                "fg_prob_draw_adjusted": float(fg_probs_adj[2]),
                "over_actual": actual["actual_over"],
                "btts_actual": actual["actual_btts"],
                "corners_actual": actual["actual_corners"],
            }
        )

        if i % 50 == 0:
            print(f"   ... {i} juegos procesados (walk-forward)...")

    return rows


def compute_comparative_metrics(predictions: list):
    if not predictions:
        return {}

    df = pd.DataFrame(predictions)

    y_fg = df["actual_full_game_class"].astype(int).to_numpy()
    y_over = df["over_actual"].astype(int).to_numpy()
    y_btts = df["btts_actual"].astype(int).to_numpy()
    y_corners = df["corners_actual"].astype(int).to_numpy()

    fg_base_probs = df[["fg_prob_away_base", "fg_prob_home_base", "fg_prob_draw_base"]].to_numpy()
    fg_adj_probs = df[["fg_prob_away_adjusted", "fg_prob_home_adjusted", "fg_prob_draw_adjusted"]].to_numpy()

    over_base_probs = df["total_base_probability"].astype(float).to_numpy()
    over_adj_probs = df["total_adjusted_probability"].astype(float).to_numpy()
    btts_base_probs = df["btts_base_probability"].astype(float).to_numpy()
    btts_adj_probs = df["btts_adjusted_probability"].astype(float).to_numpy()
    corners_base_probs = df["corners_base_probability"].astype(float).to_numpy()
    corners_adj_probs = df["corners_adjusted_probability"].astype(float).to_numpy()

    metrics = {
        "full_game": {
            "base_accuracy": float(accuracy_score(y_fg, np.argmax(fg_base_probs, axis=1))),
            "adjusted_accuracy": float(accuracy_score(y_fg, np.argmax(fg_adj_probs, axis=1))),
            "base_log_loss": float(log_loss(y_fg, fg_base_probs, labels=[0, 1, 2])),
            "adjusted_log_loss": float(log_loss(y_fg, fg_adj_probs, labels=[0, 1, 2])),
        },
        "over_25": {
            "base_accuracy": float(df["correct_total_base"].mean()),
            "adjusted_accuracy": float(df["correct_total_adjusted"].mean()),
            "base_log_loss": float(log_loss(y_over, over_base_probs, labels=[0, 1])),
            "adjusted_log_loss": float(log_loss(y_over, over_adj_probs, labels=[0, 1])),
        },
        "btts": {
            "base_accuracy": float(df["correct_btts_base"].mean()),
            "adjusted_accuracy": float(df["correct_btts_adjusted"].mean()),
            "base_log_loss": float(log_loss(y_btts, btts_base_probs, labels=[0, 1])),
            "adjusted_log_loss": float(log_loss(y_btts, btts_adj_probs, labels=[0, 1])),
        },
        "corners_over_95": {
            "base_accuracy": float(df["correct_corners_base"].mean()),
            "adjusted_accuracy": float(df["correct_corners_adjusted"].mean()),
            "base_log_loss": float(log_loss(y_corners, corners_base_probs, labels=[0, 1])),
            "adjusted_log_loss": float(log_loss(y_corners, corners_adj_probs, labels=[0, 1])),
        },
    }

    return metrics


def main():
    print("=" * 70)
    print("[*] HISTORICAL PREDICTIONS Bundesliga (WALK-FORWARD + EVENT ADJUSTMENTS)")
    print("=" * 70)

    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"No existe: {FEATURES_FILE}")
    if not RAW_HISTORY_FILE.exists():
        raise FileNotFoundError(f"No existe: {RAW_HISTORY_FILE}")

    print("\n[DATA] Cargando features procesados...")
    df_features = pd.read_csv(FEATURES_FILE, dtype={"game_id": str})
    df_features["date"] = df_features["date"].astype(str)
    df_features = df_features.sort_values(["date", "game_id"]).reset_index(drop=True)
    print(f"   OK {len(df_features)} juegos cargados")

    print("\n[DATA] Cargando histÃ³rico raw...")
    raw_history_df = pd.read_csv(RAW_HISTORY_FILE, dtype={"game_id": str})
    raw_history_df["date"] = raw_history_df["date"].astype(str)
    raw_history_df = raw_history_df.sort_values(["date", "game_id"]).reset_index(drop=True)
    print(f"   OK {len(raw_history_df)} juegos en histÃ³rico raw")

    print("\n[ML] Generando predicciones walk-forward...")
    predictions = build_walk_forward_predictions(df_features, raw_history_df)

    if not predictions:
        print("[ERROR] No se pudieron construir predicciones.")
        return

    print(f"\n[OK] Total de predicciones generadas: {len(predictions)}")

    print("\n[SAVE] Guardando predicciones histÃ³ricas por fecha...")
    predictions_by_date = save_predictions_by_date(predictions)

    metrics = compute_comparative_metrics(predictions)

    summary = {
        "use_event_adjustments": bool(USE_EVENT_ADJUSTMENTS),
        "games_evaluated": len(predictions),
        "metrics": metrics,
    }

    summary_path = REPORTS_DIR / "historical_adjustment_comparison_bundesliga.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[STATS] COMPARATIVA BASE VS ADJUSTED")
    print("=" * 70)
    for market in ["full_game", "over_25", "btts", "corners_over_95"]:
        m = metrics.get(market, {})
        print(
            f"{market.upper():<10} | "
            f"ACC base: {m.get('base_accuracy', 0.0):.4f} -> adj: {m.get('adjusted_accuracy', 0.0):.4f} | "
            f"LL base: {m.get('base_log_loss', 0.0):.4f} -> adj: {m.get('adjusted_log_loss', 0.0):.4f}"
        )

    print(f"\n[INFO] Fechas con predicciones: {len(predictions_by_date)}")
    print(f"[INFO] JSON por fecha: {PREDICTIONS_DIR}")
    print(f"[INFO] Comparativa guardada en: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

