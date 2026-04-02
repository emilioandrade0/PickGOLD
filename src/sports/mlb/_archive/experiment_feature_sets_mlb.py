from pathlib import Path
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

import sys
from pathlib import Path

# Ensure project `src` root is on sys.path so imports of shared modules work
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Base directory should point to src root
BASE_DIR = SRC_ROOT
DEFAULT_PROCESSED = BASE_DIR / "data" / "mlb" / "processed" / "model_ready_features_mlb.csv"
REPORTS_DIR = BASE_DIR / "data" / "mlb" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def detect_feature_groups(df: pd.DataFrame):
    cols = set(df.columns)
    groups = {}

    def pick(keys):
        out = [c for c in cols if any(k in c.lower() for k in keys)]
        return sorted(out)

    groups["baseline"] = sorted(set(pick(["elo", "team_rating", "rating", "recent", "rolling"])) )
    groups["park_weather"] = sorted(set(pick(["park", "weather", "temp", "wind", "humidity"])) )
    groups["pitcher_stats"] = sorted(set(pick(["pitcher", "ip", "era", "whip", "k_per", "pitch_count"])) )
    groups["umpire_lineup"] = sorted(set(pick(["umpire", "lineup", "batting_order", "lineup_strength"])) )
    groups["line_movement"] = sorted(set(pick(["line_movement", "line_move", "open_line", "current_line", "line"])) )
    # exclude obvious ID/label columns and any column that looks like a target
    excluded = set(["date", "game_id", "home", "away", "y_true", "publish_pick", "pred_label"])  
    groups["all"] = sorted([c for c in cols if c.lower() not in excluded and "target" not in c.lower()])

    # Filter empty groups and ensure baseline fallback
    if not groups["baseline"]:
        # pick some numeric predictors as fallback
        groups["baseline"] = sorted([c for c in cols if df[c].dtype.kind in "fi" and c not in ("y_true")][:10])

    return groups


def walkforward_eval(df: pd.DataFrame, feature_cols, label_col="y_true", date_col="date", min_train_days=90, test_window_days=30):
    df = df.copy()
    # drop rows where label is missing to avoid casting errors
    if label_col in df.columns:
        df = df.dropna(subset=[label_col])
    if date_col not in df.columns:
        raise RuntimeError("Data must contain a date column for walkforward evaluation")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    unique_dates = sorted(df[date_col].dt.date.unique())
    if len(unique_dates) < 2:
        return {}

    results = []
    # define sliding train/test windows by days count
    for i in range(min_train_days, len(unique_dates) - test_window_days + 1, test_window_days):
        train_dates = unique_dates[:i]
        test_dates = unique_dates[i:i + test_window_days]
        tr = df[df[date_col].dt.date.isin(train_dates)]
        te = df[df[date_col].dt.date.isin(test_dates)]
        if tr.empty or te.empty:
            continue

        # ensure features are numeric; skip non-numeric columns
        numeric_cols = [c for c in feature_cols if c in tr.columns and tr[c].dtype.kind in "fi"]
        if not numeric_cols:
            continue
        Xtr = tr[numeric_cols].fillna(0).values
        ytr = tr[label_col].astype(int).values
        Xte = te[numeric_cols].fillna(0).values
        yte = te[label_col].astype(int).values

        if len(np.unique(ytr)) < 2:
            continue

        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(Xtr, ytr)
        prob = model.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)

        try:
            auc = float(roc_auc_score(yte, prob))
        except Exception:
            auc = None
        try:
            ll = float(log_loss(yte, prob, labels=[0, 1]))
        except Exception:
            ll = None

        acc = float(accuracy_score(yte, pred))
        results.append({
            "train_start": str(train_dates[0]),
            "train_end": str(train_dates[-1]),
            "test_start": str(test_dates[0]),
            "test_end": str(test_dates[-1]),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "accuracy": acc,
            "auc": auc,
            "logloss": ll,
        })

    if not results:
        return {}

    dfres = pd.DataFrame(results)
    summary = {
        "n_folds": int(len(dfres)),
        "accuracy_mean": float(dfres["accuracy"].mean()),
        "accuracy_std": float(dfres["accuracy"].std(ddof=0)),
        "auc_mean": float(dfres["auc"].mean()) if dfres["auc"].notna().any() else None,
        "logloss_mean": float(dfres["logloss"].mean()) if dfres["logloss"].notna().any() else None,
    }
    return {"summary": summary, "folds": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=str, default=str(DEFAULT_PROCESSED))
    parser.add_argument("--min-train-days", type=int, default=90)
    parser.add_argument("--test-window-days", type=int, default=30)
    parser.add_argument("--label-col", type=str, default="y_true")
    parser.add_argument("--date-col", type=str, default="date")
    parser.add_argument("--out", type=str, default=str(REPORTS_DIR / "experiment_results.json"))
    args = parser.parse_args()

    p = Path(args.processed)
    if not p.exists():
        print(f"Processed features file not found: {p}")
        return

    df = pd.read_csv(p)
    if args.label_col not in df.columns:
        print(f"Label column '{args.label_col}' not found in processed features")
        return

    groups = detect_feature_groups(df)
    out = {}
    csv_rows = []
    for name, cols in groups.items():
        if not cols:
            continue
        print(f"Running walkforward for group: {name} (n_features={len(cols)})")
        res = walkforward_eval(df, cols, label_col=args.label_col, date_col=args.date_col, min_train_days=args.min_train_days, test_window_days=args.test_window_days)
        out[name] = res.get("summary") if res else None
        if res:
            csv_rows.append({"group": name, **res["summary"]})

    # save JSON and CSV
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(str(Path(args.out).with_suffix(".csv")), index=False)

    print(f"Experiments complete. Results saved to: {args.out}")


if __name__ == "__main__":
    main()
