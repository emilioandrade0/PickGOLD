"""Prototype: train a ranking / ordinal-style model for full_game.

Approach (simple, dependency-free prototype):
- Train a regression model to predict run differential (home - away) using diff features.
- Convert predicted margin to win probability via logistic calibration (sklearn LogisticRegression on predicted margin).
- Save two artifacts: regressor and calibrator.

This is an experimental prototype — use it to compare against existing binary models.
"""
from pathlib import Path
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
INPUT_FILE = BASE_DIR / "data" / "mlb" / "processed" / "model_ready_features_mlb.csv"
MODELS_DIR = BASE_DIR / "data" / "mlb" / "models" / "ordinal_full_game"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_diff_features(df: pd.DataFrame):
    # conservative diff feature list — prefer existing diff columns
    diff_cols = [c for c in df.columns if c.startswith("diff_")]
    # ensure at least some core diffs
    core = ["diff_elo", "diff_rest_days", "diff_win_pct_L10", "diff_run_diff_L10"]
    for c in core:
        if c in df.columns and c not in diff_cols:
            diff_cols.append(c)
    return diff_cols


def main(test_size=0.2, random_state=42):
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Features missing: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    df = df.fillna(0)

    # target: run differential (home_runs_total - away_runs_total) if available, else approximate from targets
    if {"home_runs_total", "away_runs_total"}.issubset(df.columns):
        y_margin = df["home_runs_total"] - df["away_runs_total"]
    else:
        # fallback: map binary target to small margin
        y_margin = df.get("TARGET_home_win", pd.Series(0, index=df.index)) * 1.0

    X = df[get_diff_features(df)].replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y_margin, test_size=test_size, random_state=random_state)

    reg = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=random_state)
    reg.fit(X_train, y_train)

    pred_train = reg.predict(X_train)
    pred_test = reg.predict(X_test)

    # calibrate to win prob: train logistic regressor on predicted margin -> actual binary outcome
    y_binary = (y_margin > 0).astype(int)
    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(pred_train.reshape(-1, 1), y_binary.loc[X_train.index])

    prob_test = lr.predict_proba(pred_test.reshape(-1, 1))[:, 1]
    pred_label = (prob_test >= 0.5).astype(int)

    metrics = {
        "mse_test": float(mean_squared_error(y_test, pred_test)),
        "logloss_test": float(log_loss(y_binary.loc[X_test.index], prob_test, labels=[0, 1])),
        "accuracy_test": float(accuracy_score(y_binary.loc[X_test.index], pred_label)),
    }

    # save models
    joblib.dump(reg, MODELS_DIR / "ordinal_margin_regressor.pkl")
    joblib.dump(lr, MODELS_DIR / "ordinal_margin_calibrator.pkl")

    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "features": X.columns.tolist()}, f, indent=2)

    print("✅ Ordinal/ranking prototype trained and saved")
    print(metrics)


if __name__ == "__main__":
    main()
