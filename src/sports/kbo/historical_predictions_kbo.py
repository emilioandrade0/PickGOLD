import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import sys
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT

FEATURES_FILE = BASE_DIR / "data" / "kbo" / "processed" / "model_ready_features_kbo.csv"
MODELS_DIR = BASE_DIR / "data" / "kbo" / "models"
HIST_DIR = BASE_DIR / "data" / "kbo" / "historical_predictions"
HIST_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_market_assets(market_key: str):
    market_dir = MODELS_DIR / market_key
    xgb = joblib.load(market_dir / "xgb_model.pkl")
    lgbm = joblib.load(market_dir / "lgbm_model.pkl")
    feature_columns = load_json(market_dir / "feature_columns.json")
    metadata = load_json(market_dir / "metadata.json")
    threshold = metadata.get("ensemble_threshold", 0.5)
    return xgb, lgbm, feature_columns, threshold, metadata


def predict_market(df: pd.DataFrame, market_key: str):
    xgb, lgbm, feature_columns, threshold, metadata = load_market_assets(market_key)

    X = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)

    xgb_probs = xgb.predict_proba(X)[:, 1]
    lgbm_probs = lgbm.predict_proba(X)[:, 1]
    weights = metadata.get("ensemble_weights", {"xgboost": 0.5, "lightgbm": 0.5})
    wx = float(weights.get("xgboost", 0.5))
    wl = float(weights.get("lightgbm", 0.5))
    probs = wx * xgb_probs + wl * lgbm_probs
    preds = (probs >= threshold).astype(int)
    return probs, preds


def confidence_from_prob(prob: float) -> int:
    return int(round(max(prob, 1 - prob) * 100))


def tier_from_conf(conf: int) -> str:
    if conf >= 72:
        return "ELITE"
    if conf >= 66:
        return "PREMIUM"
    if conf >= 60:
        return "STRONG"
    if conf >= 54:
        return "NORMAL"
    return "PASS"


def build_rows(df_day: pd.DataFrame):
    fg_probs, fg_preds = predict_market(df_day, "full_game")
    yrfi_probs, yrfi_preds = predict_market(df_day, "yrfi")
    f5_probs, f5_preds = predict_market(df_day, "f5")

    rows = []

    for i, row in df_day.reset_index(drop=True).iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]

        fg_prob_home = float(fg_probs[i])
        f5_prob_home = float(f5_probs[i])
        yrfi_prob_yes = float(yrfi_probs[i])

        full_game_pick = home_team if fg_preds[i] == 1 else away_team
        full_game_conf = confidence_from_prob(fg_prob_home)

        f5_pick = f"{home_team} F5" if f5_preds[i] == 1 else f"{away_team} F5"
        f5_conf = confidence_from_prob(f5_prob_home)

        yrfi_pick = "YRFI" if yrfi_preds[i] == 1 else "NRFI"
        yrfi_conf = confidence_from_prob(yrfi_prob_yes)

        total_line = float(row.get("odds_over_under", 0) or 0)
        total_pick = "OVER" if total_line > 0 and yrfi_preds[i] == 1 else "UNDER"

        rows.append(
            {
                "game_id": str(row["game_id"]),
                "date": str(row["date"]),
                "time": "",
                "game_name": f"{away_team} @ {home_team}",
                "home_team": home_team,
                "away_team": away_team,
                "full_game_pick": full_game_pick,
                "full_game_confidence": full_game_conf,
                "full_game_tier": tier_from_conf(full_game_conf),
                "q1_pick": yrfi_pick,
                "q1_confidence": yrfi_conf,
                "q1_action": "JUGAR" if yrfi_conf >= 56 else "PASS",
                "spread_pick": f"{full_game_pick} ML",
                "spread_market": "ML",
                "total_pick": total_pick,
                "odds_over_under": total_line,
                "closing_moneyline_odds": row.get("closing_moneyline_odds"),
                "home_moneyline_odds": row.get("home_moneyline_odds"),
                "away_moneyline_odds": row.get("away_moneyline_odds"),
                "closing_total_odds": row.get("closing_total_odds"),
                "odds_data_quality": str(row.get("odds_data_quality", "fallback")),
                "assists_pick": f5_pick,
                "extra_f5_confidence": f5_conf,
                "extra_f5_tier": tier_from_conf(f5_conf),
            }
        )

    return rows


def main():
    df = pd.read_csv(FEATURES_FILE)
    df["date"] = df["date"].astype(str)

    all_dates = sorted(df["date"].unique())
    total_files = 0

    for date_str in all_dates:
        df_day = df[df["date"] == date_str].copy()
        if df_day.empty:
            continue

        rows = build_rows(df_day)

        output_path = HIST_DIR / f"{date_str}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

        total_files += 1
        print(f"âœ… {date_str} -> {len(rows)} juegos")

    print(f"\nðŸ’¾ Archivos histÃ³ricos generados: {total_files}")


if __name__ == "__main__":
    main()
