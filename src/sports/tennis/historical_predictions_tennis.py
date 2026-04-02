from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
RAW_HISTORY_FILE = BASE_DIR / "data" / "tennis" / "raw" / "tennis_advanced_history.csv"
HIST_DIR = BASE_DIR / "data" / "tennis" / "historical_predictions"
REPORTS_DIR = BASE_DIR / "data" / "tennis" / "reports"
HIST_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

from sports.tennis.tennis_features import FEATURE_COLUMNS, build_history_features
from sports.tennis.tennis_inference import predict_match, tier_from_conf

MIN_TRAIN_ROWS = 25


def _result_label(pred_pick: str, row: pd.Series) -> tuple[str, int | None]:
    winner = str(row.get("winner") or "").strip()
    player_a = str(row.get("player_a") or "").strip()
    player_b = str(row.get("player_b") or "").strip()
    if winner not in {player_a, player_b}:
        return "PENDIENTE", None
    hit = int(pred_pick == winner)
    return ("ACIERTO" if hit else "FALLO"), hit


def main() -> None:
    if not RAW_HISTORY_FILE.exists():
        summary = {"status": "skipped", "reason": "missing_history"}
        output = REPORTS_DIR / "walkforward_summary_tennis.json"
        output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Tennis historical: no existe historico -> {output}")
        return

    raw_history_df = pd.read_csv(RAW_HISTORY_FILE, dtype={"match_id": str})
    feature_df = build_history_features(raw_history_df)
    if feature_df.empty:
        summary = {"status": "skipped", "reason": "empty_feature_history", "rows": 0}
        output = REPORTS_DIR / "walkforward_summary_tennis.json"
        output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Tennis historical: sin filas utilizables -> {output}")
        return

    raw_by_id = {str(row["match_id"]): row for _, row in raw_history_df.iterrows()}
    feature_df = feature_df.sort_values(["date", "time", "match_id"], kind="stable").reset_index(drop=True)

    split_idx = max(int(len(feature_df) * 0.8), MIN_TRAIN_ROWS)
    train_df = feature_df.iloc[:split_idx].copy()
    test_df = feature_df.iloc[split_idx:].copy()
    usable_features = [col for col in FEATURE_COLUMNS if col in train_df.columns and not train_df[col].isna().all()]

    model = None
    if len(train_df) >= MIN_TRAIN_ROWS and train_df["TARGET_player_a_win"].nunique() >= 2 and usable_features:
        X_train = train_df[usable_features].apply(pd.to_numeric, errors="coerce")
        y_train = train_df["TARGET_player_a_win"].astype(int)
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=4000, solver="lbfgs")),
            ]
        )
        model.fit(X_train, y_train)

    rows = []
    for _, feat_row in test_df.iterrows():
        match_id = str(feat_row.get("match_id") or "")
        raw_row = raw_by_id.get(match_id)
        if raw_row is None:
            continue

        pick, conf, model_source, prob = predict_match(raw_row, feat_row, model=model, feature_columns=usable_features or FEATURE_COLUMNS)
        result_label, result_hit = _result_label(pick, raw_row)
        home_ml = raw_row.get("player_a_odds")
        away_ml = raw_row.get("player_b_odds")
        pick_odds = home_ml if pick == str(raw_row.get("player_a") or "") else away_ml if pick == str(raw_row.get("player_b") or "") else None
        rows.append(
            {
                "game_id": match_id,
                "date": str(raw_row.get("date") or ""),
                "time": str(raw_row.get("time") or ""),
                "game_name": f"{raw_row.get('player_a', '')} vs {raw_row.get('player_b', '')}",
                "home_team": str(raw_row.get("player_a") or ""),
                "away_team": str(raw_row.get("player_b") or ""),
                "tour": str(raw_row.get("tour") or ""),
                "tournament": str(raw_row.get("tournament") or ""),
                "surface": str(raw_row.get("surface") or ""),
                "round": str(raw_row.get("round") or ""),
                "full_game_pick": pick,
                "full_game_confidence": conf,
                "full_game_tier": tier_from_conf(conf),
                "full_game_model_prob_pick": prob,
                "moneyline_odds": pick_odds,
                "recommended_tier": tier_from_conf(conf),
                "home_moneyline_odds": home_ml,
                "away_moneyline_odds": away_ml,
                "status_completed": int(pd.to_numeric(raw_row.get("status_completed"), errors="coerce") or 1),
                "status_state": str(raw_row.get("status_state") or "post"),
                "status_description": str(raw_row.get("status_description") or "Final"),
                "status_detail": str(raw_row.get("status_detail") or "FINAL"),
                "result_available": True,
                "result_label": result_label,
                "result_hit": result_hit,
                "model_source": model_source,
                "model_probability": prob,
            }
        )

    pred_df = pd.DataFrame(rows)
    for date_str, day_df in pred_df.groupby("date"):
        out_file = HIST_DIR / f"{date_str}.json"
        out_file.write_text(day_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    resolved = pred_df[pred_df["result_hit"].notna()].copy()
    accuracy = float(resolved["result_hit"].mean()) if not resolved.empty else None
    by_tier = {}
    if not resolved.empty:
        for tier, part in resolved.groupby("recommended_tier"):
            by_tier[str(tier)] = {
                "rows": int(len(part)),
                "accuracy": float(part["result_hit"].mean()),
            }

    summary = {
        "status": "completed",
        "rows": int(len(pred_df)),
        "resolved_rows": int(len(resolved)),
        "accuracy": accuracy,
        "tiers": by_tier,
        "unique_dates": int(pred_df["date"].nunique()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }
    output = REPORTS_DIR / "walkforward_summary_tennis.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Tennis historical listo: {output}")


if __name__ == "__main__":
    main()
