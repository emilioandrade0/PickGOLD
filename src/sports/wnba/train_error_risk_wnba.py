from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sports.wnba.error_risk_utils import (
    ERROR_RISK_FEATURES,
    build_error_risk_feature_dict,
    market_pick_from_row,
)


RAW_DATA = SRC_ROOT / "data" / "wnba" / "raw" / "wnba_advanced_history.csv"
HIST_PRED_DIR = SRC_ROOT / "data" / "wnba" / "historical_predictions"
MODELS_DIR = SRC_ROOT / "data" / "wnba" / "models"
REPORT_DIR = SRC_ROOT / "reports" / "wnba_error_risk"
MODEL_OUT = MODELS_DIR / "error_risk_fullgame.pkl"


def _load_dataset(base_threshold: float = 0.51, market_threshold: float = 0.495):
    raw = pd.read_csv(RAW_DATA, dtype={"game_id": str})
    raw["game_id"] = raw["game_id"].astype(str)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["real_winner"] = np.where(raw["home_pts_total"] > raw["away_pts_total"], raw["home_team"], raw["away_team"])
    actual = raw[["game_id", "date", "real_winner"]].copy()

    rows = []
    for fp in HIST_PRED_DIR.glob("*.json"):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, list):
            rows.extend(data)

    pred = pd.DataFrame(rows)
    if pred.empty:
        return pred

    pred["game_id"] = pred["game_id"].astype(str)
    pred["date"] = pd.to_datetime(pred["date"], errors="coerce")
    pred["home_spread"] = pd.to_numeric(pred.get("home_spread"), errors="coerce").fillna(0.0)
    pred["full_game_calibrated_prob_home"] = pd.to_numeric(pred.get("full_game_calibrated_prob_home"), errors="coerce")
    pred = pred.dropna(subset=["date", "full_game_calibrated_prob_home"]).copy()
    pred = pred.merge(actual, on=["game_id"], how="inner", suffixes=("", "_act"))
    pred = pred.sort_values(["date", "game_id"]).reset_index(drop=True)

    feature_rows = []
    for _, r in pred.iterrows():
        cur_pick = str(r.get("full_game_pick", ""))
        if not cur_pick:
            continue
        threshold_used = float(market_threshold if abs(float(r.get("home_spread", 0.0))) > 0 else base_threshold)
        feat = build_error_risk_feature_dict(
            row=r,
            calibrated_prob_home=float(r.get("full_game_calibrated_prob_home")),
            threshold_used=threshold_used,
            current_pick=cur_pick,
        )
        hit = 1 if cur_pick == str(r.get("real_winner", "")) else 0
        feature_rows.append(
            {
                "date": r["date"],
                "game_id": r["game_id"],
                "home_team": r.get("home_team"),
                "away_team": r.get("away_team"),
                "current_pick": cur_pick,
                "real_winner": str(r.get("real_winner", "")),
                "error_target": int(1 - hit),
                "home_spread": float(r.get("home_spread", 0.0)),
                **feat,
            }
        )

    out = pd.DataFrame(feature_rows).sort_values(["date", "game_id"]).reset_index(drop=True)
    return out


def _evaluate_policy(df: pd.DataFrame, risk_prob: np.ndarray, risk_threshold: float, min_spread_abs: float):
    df = df.copy()
    df["risk_prob"] = risk_prob

    final_pick = []
    for _, r in df.iterrows():
        cur = str(r["current_pick"])
        mk = market_pick_from_row(r)
        if mk is not None and abs(float(r.get("home_spread", 0.0))) >= float(min_spread_abs):
            if cur != str(mk) and float(r["risk_prob"]) >= float(risk_threshold):
                cur = str(mk)
        final_pick.append(cur)

    final_pick = np.array(final_pick, dtype=object)
    acc = float((final_pick == df["real_winner"].to_numpy(dtype=object)).mean()) if len(df) else 0.0
    return acc


def run():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    data = _load_dataset()
    if data.empty:
        raise RuntimeError("No rows found to train error-risk model.")

    n = len(data)
    split_idx = max(1, min(n - 1, int(n * 0.8)))
    train = data.iloc[:split_idx].copy()
    valid = data.iloc[split_idx:].copy()

    X_train = train[ERROR_RISK_FEATURES].to_numpy(dtype=float)
    y_train = train["error_target"].to_numpy(dtype=int)
    X_valid = valid[ERROR_RISK_FEATURES].to_numpy(dtype=float)
    y_valid = valid["error_target"].to_numpy(dtype=int)

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train, y_train)

    p_train = model.predict_proba(X_train)[:, 1]
    p_valid = model.predict_proba(X_valid)[:, 1]
    p_full = model.predict_proba(data[ERROR_RISK_FEATURES].to_numpy(dtype=float))[:, 1]

    base_valid_acc = float((valid["current_pick"].to_numpy(dtype=object) == valid["real_winner"].to_numpy(dtype=object)).mean())
    base_full_acc = float((data["current_pick"].to_numpy(dtype=object) == data["real_winner"].to_numpy(dtype=object)).mean())

    best = {
        "enabled": False,
        "risk_threshold": 0.62,
        "min_spread_abs": 0.0,
        "only_when_disagree": True,
        "valid_acc": base_valid_acc,
        "full_acc": base_full_acc,
    }

    for thr in np.arange(0.55, 0.81, 0.01):
        for min_sp in [0, 1, 2, 3, 4, 5, 6, 7]:
            acc_valid = _evaluate_policy(valid, p_valid, float(thr), float(min_sp))
            acc_full = _evaluate_policy(data, p_full, float(thr), float(min_sp))
            if acc_valid > best["valid_acc"] or (acc_valid == best["valid_acc"] and acc_full > best["full_acc"]):
                best = {
                    "enabled": True,
                    "risk_threshold": float(round(thr, 4)),
                    "min_spread_abs": float(min_sp),
                    "only_when_disagree": True,
                    "valid_acc": float(acc_valid),
                    "full_acc": float(acc_full),
                }

    bundle = {
        "model": model,
        "feature_names": list(ERROR_RISK_FEATURES),
        "policy": {
            "enabled": bool(best["enabled"]),
            "risk_threshold": float(best["risk_threshold"]),
            "min_spread_abs": float(best["min_spread_abs"]),
            "only_when_disagree": True,
        },
        "metrics": {
            "rows_total": int(n),
            "rows_train": int(len(train)),
            "rows_valid": int(len(valid)),
            "base_valid_acc": float(base_valid_acc),
            "base_full_acc": float(base_full_acc),
            "best_valid_acc": float(best["valid_acc"]),
            "best_full_acc": float(best["full_acc"]),
            "auc_train": float(roc_auc_score(y_train, p_train)) if len(np.unique(y_train)) > 1 else None,
            "auc_valid": float(roc_auc_score(y_valid, p_valid)) if len(np.unique(y_valid)) > 1 else None,
        },
    }

    joblib.dump(bundle, MODEL_OUT)
    (REPORT_DIR / "error_risk_summary.json").write_text(json.dumps(bundle["metrics"] | {"policy": bundle["policy"]}, indent=2), encoding="utf-8")

    report_df = valid.copy()
    report_df["risk_prob"] = p_valid
    report_df.to_csv(REPORT_DIR / "valid_rows_with_risk.csv", index=False)

    print("Error-risk model trained.")
    print(f"Model: {MODEL_OUT}")
    print(f"Policy: {bundle['policy']}")
    print(f"Base valid acc: {base_valid_acc:.4f} | Best valid acc: {best['valid_acc']:.4f}")


if __name__ == "__main__":
    run()
