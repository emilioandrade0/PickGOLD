from __future__ import annotations

from pathlib import Path
import json
import sys

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
INPUT_FILE = BASE_DIR / "data" / "tennis" / "processed" / "model_ready_features_tennis.csv"
MODELS_DIR = BASE_DIR / "data" / "tennis" / "models"
REPORTS_DIR = BASE_DIR / "data" / "tennis" / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILE = MODELS_DIR / "match_winner_model.pkl"
META_FILE = MODELS_DIR / "match_winner_model_meta.json"

from sports.tennis.tennis_features import FEATURE_COLUMNS


MIN_ROWS = 25


def main() -> None:
    if not INPUT_FILE.exists():
        summary = {"status": "skipped", "reason": "missing_features", "input_file": str(INPUT_FILE)}
    else:
        df = pd.read_csv(INPUT_FILE, dtype={"match_id": str})
        df = df.dropna(subset=["TARGET_player_a_win"]).copy()
        for col in FEATURE_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        usable_features = [col for col in FEATURE_COLUMNS if col in df.columns and not df[col].isna().all()]

        if len(df) < MIN_ROWS or df["TARGET_player_a_win"].nunique() < 2 or not usable_features:
            summary = {
                "status": "skipped",
                "reason": "insufficient_training_rows",
                "rows": int(len(df)),
                "required_min_rows": MIN_ROWS,
            }
        else:
            df = df.sort_values(["date", "time", "match_id"], kind="stable")
            split_idx = max(int(len(df) * 0.8), 1)
            train_df = df.iloc[:split_idx].copy()
            valid_df = df.iloc[split_idx:].copy()
            if valid_df.empty or valid_df["TARGET_player_a_win"].nunique() < 2:
                valid_df = train_df.tail(min(len(train_df), max(5, len(train_df) // 5))).copy()
                train_df = train_df.iloc[:-len(valid_df)].copy() if len(train_df) > len(valid_df) else train_df.copy()

            X_train = train_df[usable_features]
            y_train = train_df["TARGET_player_a_win"].astype(int)
            X_valid = valid_df[usable_features]
            y_valid = valid_df["TARGET_player_a_win"].astype(int)

            model = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=4000, solver="lbfgs")),
                ]
            )
            model.fit(X_train, y_train)

            valid_prob = model.predict_proba(X_valid)[:, 1]
            valid_pred = (valid_prob >= 0.5).astype(int)
            try:
                auc = float(roc_auc_score(y_valid, valid_prob))
            except Exception:
                auc = None

            joblib.dump(model, MODEL_FILE)
            META_FILE.write_text(
                json.dumps(
                    {
                        "feature_columns": usable_features,
                        "usable_feature_columns": usable_features,
                        "target": "TARGET_player_a_win",
                        "model_type": "logistic_regression",
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            summary = {
                "status": "trained",
                "rows": int(len(df)),
                "train_rows": int(len(train_df)),
                "valid_rows": int(len(valid_df)),
                "features": len(usable_features),
                "accuracy": float(accuracy_score(y_valid, valid_pred)),
                "logloss": float(log_loss(y_valid, valid_prob)),
                "roc_auc": auc,
                "model_file": str(MODEL_FILE),
            }

    summary_path = REPORTS_DIR / "training_summary_tennis.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Tennis training summary guardado en: {summary_path}")


if __name__ == "__main__":
    main()
