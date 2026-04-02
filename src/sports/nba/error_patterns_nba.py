import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SRC_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA = SRC_ROOT / "data" / "raw" / "nba_advanced_history.csv"
HIST_PRED_DIR = SRC_ROOT / "data" / "historical_predictions"
REPORT_DIR = SRC_ROOT / "reports" / "nba_error_patterns"


def _safe_float(value):
    try:
        x = float(value)
        if np.isnan(x):
            return np.nan
        return x
    except Exception:
        return np.nan


def _safe_int(value, default=0):
    try:
        if pd.isna(value):
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _group_accuracy(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame(columns=[group_col, "games", "hits", "accuracy"])
    out = (
        df.groupby(group_col, dropna=False)
        .agg(games=("correct_full", "size"), hits=("correct_full", "sum"))
        .reset_index()
    )
    out["accuracy"] = np.where(out["games"] > 0, out["hits"] / out["games"], np.nan)
    return out.sort_values(["accuracy", "games"], ascending=[False, False]).reset_index(drop=True)


def _market_favorite_from_spread(row: pd.Series):
    hs = _safe_float(row.get("home_spread"))
    if np.isnan(hs) or hs == 0:
        return None
    return row["home_team"] if hs < 0 else row["away_team"]


def _selected_pick_prob(row: pd.Series):
    p_home = _safe_float(row.get("full_game_calibrated_prob_home"))
    if np.isnan(p_home):
        return np.nan
    if str(row.get("full_game_pick", "")) == str(row.get("home_team", "")):
        return p_home
    return 1.0 - p_home


def _load_predictions() -> pd.DataFrame:
    rows = []
    for fp in HIST_PRED_DIR.glob("*.json"):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, list):
            rows.extend(data)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["game_id"] = df["game_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _load_actuals() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA, dtype={"game_id": str})
    use_cols = [
        "game_id",
        "date",
        "home_team",
        "away_team",
        "home_pts_total",
        "away_pts_total",
        "home_q1",
        "away_q1",
        "home_spread",
        "spread_abs",
        "home_is_favorite",
        "odds_data_quality",
    ]
    existing = [c for c in use_cols if c in df.columns]
    df = df[existing].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _prepare_merged(date_from: str | None = None, date_to: str | None = None) -> pd.DataFrame:
    preds = _load_predictions()
    if preds.empty:
        return preds

    actuals = _load_actuals()
    df = preds.merge(actuals, on="game_id", how="inner", suffixes=("", "_raw"))
    if df.empty:
        return df

    if date_from:
        df = df[df["date"] >= pd.to_datetime(date_from)]
    if date_to:
        df = df[df["date"] <= pd.to_datetime(date_to)]
    if df.empty:
        return df

    df["real_full_winner"] = np.where(df["home_pts_total"] > df["away_pts_total"], df["home_team"], df["away_team"])
    df["real_q1_winner"] = np.where(
        df["home_q1"] > df["away_q1"],
        df["home_team"],
        np.where(df["away_q1"] > df["home_q1"], df["away_team"], "TIE"),
    )
    df["correct_full"] = (df["full_game_pick"] == df["real_full_winner"]).astype(int)
    df["q1_evaluable"] = (df["real_q1_winner"] != "TIE").astype(int)
    df["correct_q1"] = np.where(df["q1_evaluable"] == 1, (df["q1_pick"] == df["real_q1_winner"]).astype(int), np.nan)

    df["full_game_confidence"] = pd.to_numeric(df.get("full_game_confidence"), errors="coerce")
    df["q1_confidence"] = pd.to_numeric(df.get("q1_confidence"), errors="coerce")
    df["market_missing"] = pd.to_numeric(df.get("market_missing"), errors="coerce").fillna(1).astype(int)

    df["spread_abs"] = pd.to_numeric(df.get("spread_abs"), errors="coerce")
    if "home_spread" in df.columns:
        df["home_spread"] = pd.to_numeric(df["home_spread"], errors="coerce")

    df["pick_prob_selected"] = df.apply(_selected_pick_prob, axis=1)
    df["market_favorite"] = df.apply(_market_favorite_from_spread, axis=1)
    df["model_vs_market"] = np.where(
        df["market_favorite"].notna(),
        np.where(df["full_game_pick"] == df["market_favorite"], "agree", "against"),
        "no_market_side",
    )

    df["confidence_bin"] = pd.cut(
        df["full_game_confidence"],
        bins=[0, 55, 60, 65, 70, 75, 80, 100],
        labels=["<=55", "55-60", "60-65", "65-70", "70-75", "75-80", "80+"],
        include_lowest=True,
    )
    df["spread_bin"] = pd.cut(
        df["spread_abs"],
        bins=[-0.001, 0.001, 3, 6, 9, 100],
        labels=["PK", "0-3", "3-6", "6-9", "9+"],
        include_lowest=True,
    )
    return df


def run(date_from: str | None = None, date_to: str | None = None):
    if not RAW_DATA.exists():
        raise FileNotFoundError(f"RAW missing: {RAW_DATA}")
    if not HIST_PRED_DIR.exists():
        raise FileNotFoundError(f"Historical predictions folder missing: {HIST_PRED_DIR}")

    df = _prepare_merged(date_from=date_from, date_to=date_to)
    if df.empty:
        print("No rows available after merge/filter. Check files or date range.")
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    n_games = int(len(df))
    full_hits = int(df["correct_full"].sum())
    full_acc = float(full_hits / n_games) if n_games else np.nan

    q1_eval = int(df["q1_evaluable"].sum())
    q1_hits = int(df["correct_q1"].fillna(0).sum())
    q1_acc = float(q1_hits / q1_eval) if q1_eval else np.nan

    summary = {
        "date_from": str(df["date"].min().date()) if df["date"].notna().any() else None,
        "date_to": str(df["date"].max().date()) if df["date"].notna().any() else None,
        "games": n_games,
        "full_game_accuracy": round(full_acc, 6) if pd.notna(full_acc) else None,
        "full_game_hits": full_hits,
        "q1_accuracy": round(q1_acc, 6) if pd.notna(q1_acc) else None,
        "q1_hits": q1_hits,
        "q1_evaluable_games": q1_eval,
    }

    by_tier = _group_accuracy(df, "full_game_tier")
    by_rule = _group_accuracy(df, "full_game_pick_rule")
    by_market_missing = _group_accuracy(df, "market_missing")
    by_conf_bin = _group_accuracy(df, "confidence_bin")
    by_spread_bin = _group_accuracy(df, "spread_bin")
    by_model_vs_market = _group_accuracy(df, "model_vs_market")

    calib = df.dropna(subset=["pick_prob_selected"]).copy()
    calib["prob_bin"] = pd.cut(
        calib["pick_prob_selected"],
        bins=np.linspace(0.0, 1.0, 11),
        include_lowest=True,
    )
    calib_table = (
        calib.groupby("prob_bin", dropna=False)
        .agg(
            games=("correct_full", "size"),
            avg_pred_prob=("pick_prob_selected", "mean"),
            observed_win_rate=("correct_full", "mean"),
        )
        .reset_index()
    )
    calib_table["calibration_gap"] = calib_table["avg_pred_prob"] - calib_table["observed_win_rate"]

    # Save tables
    by_tier.to_csv(REPORT_DIR / "by_tier.csv", index=False)
    by_rule.to_csv(REPORT_DIR / "by_pick_rule.csv", index=False)
    by_market_missing.to_csv(REPORT_DIR / "by_market_missing.csv", index=False)
    by_conf_bin.to_csv(REPORT_DIR / "by_confidence_bin.csv", index=False)
    by_spread_bin.to_csv(REPORT_DIR / "by_spread_bin.csv", index=False)
    by_model_vs_market.to_csv(REPORT_DIR / "by_model_vs_market.csv", index=False)
    calib_table.to_csv(REPORT_DIR / "calibration_bins.csv", index=False)

    preview_cols = [
        "date",
        "game_id",
        "away_team",
        "home_team",
        "full_game_pick",
        "real_full_winner",
        "correct_full",
        "full_game_confidence",
        "full_game_tier",
        "full_game_pick_rule",
        "market_missing",
        "spread_abs",
        "model_vs_market",
        "pick_prob_selected",
    ]
    existing_preview = [c for c in preview_cols if c in df.columns]
    df[existing_preview].sort_values("date").to_csv(REPORT_DIR / "detailed_rows.csv", index=False)

    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("NBA error patterns report generated.")
    print(f"Summary: games={n_games} full_acc={full_acc:.4f} q1_acc={q1_acc:.4f}")
    print(f"Output folder: {REPORT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze NBA historical prediction error patterns.")
    parser.add_argument("--date-from", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--date-to", type=str, default=None, help="YYYY-MM-DD")
    args = parser.parse_args()

    run(date_from=args.date_from, date_to=args.date_to)
