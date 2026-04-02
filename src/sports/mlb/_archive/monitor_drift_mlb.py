from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import json
import pandas as pd
import numpy as np

BASE_DIR = SRC_ROOT
WALK_DIR = BASE_DIR / "data" / "mlb" / "walkforward"
REPORTS_DIR = BASE_DIR / "data" / "mlb" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def gather_walkforward_details():
    rows = []
    if not WALK_DIR.exists():
        return pd.DataFrame()
    for market_dir in WALK_DIR.iterdir():
        if not market_dir.is_dir():
            continue
        detail = market_dir / "walkforward_predictions_detail.csv"
        if not detail.exists():
            continue
        try:
            df = pd.read_csv(detail)
            df["market_key"] = market_dir.name
            rows.append(df)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out


def compute_metrics(detail_df: pd.DataFrame):
    if detail_df.empty:
        return {}

    detail_df["date"] = detail_df["date"].astype(str)
    metrics = {}
    for market, grp in detail_df.groupby("market_key"):
        grp = grp.sort_values("date")
        # compute published accuracy per date
        per_date = grp.groupby("date").apply(lambda g: pd.Series({
            "published_accuracy": float((g[g["publish_pick"] == 1]["pred_label"].astype(int) == g[g["publish_pick"] == 1]["y_true"].astype(int)).mean()) if len(g[g["publish_pick"] == 1]) else np.nan,
            "published_coverage": float(g["publish_pick"].fillna(0).astype(int).mean()),
            "n_games": int(len(g)),
        }))
        per_date = per_date.reset_index()

        # rolling windows
        per_date["pub_acc_7d"] = per_date["published_accuracy"].rolling(window=7, min_periods=1).mean()
        per_date["pub_acc_30d"] = per_date["published_accuracy"].rolling(window=30, min_periods=1).mean()
        per_date["cov_7d"] = per_date["published_coverage"].rolling(window=7, min_periods=1).mean()

        metrics[market] = per_date.to_dict(orient="records")

    return metrics


def save_report(metrics: dict):
    out_path = REPORTS_DIR / "drift_report.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Drift report saved to: {out_path}")
    except Exception as e:
        print(f"Failed to save report: {e}")


def main():
    df = gather_walkforward_details()
    if df.empty:
        print("No walkforward detail data found.")
        return
    metrics = compute_metrics(df)
    save_report(metrics)


if __name__ == "__main__":
    main()
