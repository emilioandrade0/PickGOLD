from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd


SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
RAW_HISTORY_FILE = BASE_DIR / "data" / "tennis" / "raw" / "tennis_advanced_history.csv"
UPCOMING_FILE = BASE_DIR / "data" / "tennis" / "raw" / "tennis_upcoming_schedule.csv"
PREDICTIONS_DIR = BASE_DIR / "data" / "tennis" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

from sports.tennis.tennis_features import build_upcoming_features
from sports.tennis.tennis_inference import load_model, predict_match, tier_from_conf


def main() -> None:
    if not UPCOMING_FILE.exists():
        print(f"Tennis predict_today: no existe {UPCOMING_FILE}")
        return

    upcoming_df = pd.read_csv(UPCOMING_FILE, dtype={"match_id": str})
    if upcoming_df.empty:
        print("Tennis predict_today: agenda vacia.")
        return

    raw_history_df = pd.read_csv(RAW_HISTORY_FILE, dtype={"match_id": str}) if RAW_HISTORY_FILE.exists() else pd.DataFrame()
    feature_df = build_upcoming_features(raw_history_df, upcoming_df)
    feature_by_id = {str(row["match_id"]): row for _, row in feature_df.iterrows()}
    model, feature_columns = load_model()

    for date_str, day_df in upcoming_df.groupby("date"):
        rows = []
        for _, row in day_df.sort_values(["time", "match_id"], kind="stable").iterrows():
            match_id = str(row.get("match_id") or "")
            player_a = str(row.get("player_a") or "PLAYER A")
            player_b = str(row.get("player_b") or "PLAYER B")
            home_ml = row.get("home_moneyline_odds")
            away_ml = row.get("away_moneyline_odds")

            feature_row = feature_by_id.get(match_id)
            pick, conf, model_source, predicted_prob = predict_match(row, feature_row, model=model, feature_columns=feature_columns)
            pick_odds = home_ml if pick == player_a else away_ml if pick == player_b else None

            rows.append(
                {
                    "game_id": match_id,
                    "date": str(row.get("date") or ""),
                    "time": str(row.get("time") or ""),
                    "game_name": f"{player_a} vs {player_b}",
                    "home_team": player_a,
                    "away_team": player_b,
                    "tournament": str(row.get("tournament") or ""),
                    "tour": str(row.get("tour") or ""),
                    "surface": str(row.get("surface") or ""),
                    "round": str(row.get("round") or ""),
                    "full_game_pick": pick,
                    "full_game_confidence": conf,
                    "full_game_tier": tier_from_conf(conf),
                    "full_game_model_prob_pick": predicted_prob,
                    "moneyline_odds": pick_odds,
                    "recommended_tier": tier_from_conf(conf),
                    "home_moneyline_odds": home_ml,
                    "away_moneyline_odds": away_ml,
                    "status_completed": int(pd.to_numeric(row.get("status_completed"), errors="coerce") or 0),
                    "status_state": str(row.get("status_state") or "pre"),
                    "status_description": str(row.get("status_description") or "Scheduled"),
                    "status_detail": str(row.get("status_detail") or "Scheduled"),
                    "result_available": False,
                    "model_source": model_source,
                    "model_probability": predicted_prob,
                }
            )

        output_file = PREDICTIONS_DIR / f"{date_str}.json"
        output_file.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Tennis today: {len(rows)} eventos -> {output_file.name}")


if __name__ == "__main__":
    main()
