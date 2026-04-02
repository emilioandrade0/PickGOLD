import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Paths
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA = SRC_ROOT / "data" / "raw" / "nba_advanced_history.csv"
HIST_PRED_DIR = SRC_ROOT / "data" / "historical_predictions"


def load_actuals():
    if not RAW_DATA.exists():
        raise FileNotFoundError("RAW data CSV missing")
    df = pd.read_csv(RAW_DATA, dtype={"game_id": str})
    df = df[["game_id", "date", "home_team", "away_team", "home_pts_total", "away_pts_total", "home_q1", "away_q1", "home_spread"]]
    df["date"] = pd.to_datetime(df["date"]) 
    df["full_winner"] = df.apply(lambda r: r["home_team"] if r["home_pts_total"] > r["away_pts_total"] else r["away_team"], axis=1)
    return df.set_index("game_id")


def load_predictions():
    files = list(HIST_PRED_DIR.glob("*.json"))
    rows = []
    for f in files:
        try:
            data = json.load(open(f, "r", encoding="utf-8"))
        except Exception:
            continue
        for p in data:
            rows.append(p)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]) 
    return df


def backtest_report():
    actuals = load_actuals()
    preds = load_predictions()
    if preds.empty:
        print("No historical predictions found.")
        return

    # Merge
    preds["game_id"] = preds["game_id"].astype(str)
    merged = preds.merge(actuals, left_on="game_id", right_index=True, how="inner", suffixes=("", "_act"))

    # Correctness
    merged["correct_full"] = merged.apply(lambda r: 1 if r["full_game_pick"] == r["full_winner"] else 0, axis=1)

    # Ensure injury columns exist
    if "home_injuries_count" not in merged.columns:
        merged["home_injuries_count"] = 0
    if "away_injuries_count" not in merged.columns:
        merged["away_injuries_count"] = 0

    # Day of week analysis
    merged["weekday"] = merged["date"].dt.day_name()
    dow = merged.groupby("weekday").agg(total=("correct_full","count"), hits=("correct_full","sum"))
    dow["accuracy"] = (dow["hits"] / dow["total"]).fillna(0)

    print("\n=== Performance por Día de la Semana ===")
    print(dow.sort_values("accuracy", ascending=False))

    # Favorites vs Underdogs (market favorite by spread sign)
    def market_favorite(r):
        try:
            s = float(r.get("home_spread", 0) or 0)
        except Exception:
            s = 0
        if s < 0:
            return r["home_team"]
        if s > 0:
            return r["away_team"]
        return None

    merged["market_favorite"] = merged.apply(market_favorite, axis=1)
    merged["model_favorite"] = merged.apply(lambda r: r["full_game_pick"], axis=1)

    fav_vs_und = merged.copy()
    fav_vs_und["is_model_vs_market"] = fav_vs_und.apply(lambda r: 1 if r["model_favorite"] != r["market_favorite"] and r["market_favorite"] is not None else 0, axis=1)

    print("\n=== Model vs Market (overall) ===")
    both = fav_vs_und[fav_vs_und["market_favorite"].notna()]
    if len(both):
        same = both[both["is_model_vs_market"] == 0]
        diff = both[both["is_model_vs_market"] == 1]
        print(f"Total with market favorite: {len(both)} | Model agrees: {len(same)} (acc={same['correct_full'].mean():.3f}) | Model vs market: {len(diff)} (acc={diff['correct_full'].mean():.3f})")

    # Underdogs >= +7.5: detect if picked team was the underdog and spread >= 7.5
    merged["spread_abs"] = merged["home_spread"].abs()
    def is_pick_underdog(r):
        if r["spread_abs"] < 7.5:
            return False
        # home_is_favorite not always present; infer from home_spread sign if possible
        home_fav = None
        try:
            home_fav = 1 if float(r.get("home_spread", 0)) < 0 else 0
        except Exception:
            home_fav = None
        if home_fav is None:
            return False
        # if home is favorite (1) then away is underdog
        if r["full_game_pick"] == r["away_team"] and home_fav == 1:
            return True
        if r["full_game_pick"] == r["home_team"] and home_fav == 0:
            return True
        return False

    merged["is_underdog_pick"] = merged.apply(is_pick_underdog, axis=1)
    und = merged[merged["is_underdog_pick"]]

    print("\n=== Underdogs >= +7.5 Backtest ===")
    if len(und) == 0:
        print("No underdog picks with spread >= 7.5 found in historical predictions.")
    else:
        print(f"Total underdog picks (>=7.5): {len(und)}")
        print(f"Accuracy on those: {und['correct_full'].mean():.3f} ({int(und['correct_full'].sum())}/{len(und)})")

    # Calibration curve: take predicted probability for the selected side
    def pick_prob(r):
        p_home = r.get("full_game_calibrated_prob_home")
        try:
            p = float(p_home)
        except Exception:
            return np.nan
        # if pick is home, prob = p, else prob = 1 - p
        return p if r["full_game_pick"] == r["home_team"] else (1 - p)

    merged["pick_prob"] = merged.apply(pick_prob, axis=1)
    calib = merged.dropna(subset=["pick_prob"]).copy()
    calib["bin"] = pd.cut(calib["pick_prob"], bins=np.linspace(0,1,11))
    calib_grp = calib.groupby("bin").agg(count=("pick_prob","count"),
                                          avg_pred=("pick_prob","mean"),
                                          obs_win=("correct_full","mean"))

    print("\n=== Calibration Curve (binned by predicted prob) ===")
    print(calib_grp)

    # --- Spread bins ---
    merged["spread_abs"] = merged["home_spread"].abs()
    def spread_bin(v):
        try:
            a = float(v)
        except Exception:
            return "unknown"
        if a == 0:
            return "PK-0"
        if a <= 3:
            return "0-3"
        if a <= 6:
            return "3-6"
        if a <= 9:
            return "6-9"
        return "9+"

    merged["spread_bin"] = merged["spread_abs"].apply(spread_bin)
    spread_stats = merged.groupby("spread_bin").agg(total=("correct_full","count"), hits=("correct_full","sum"))
    spread_stats["accuracy"] = (spread_stats["hits"] / spread_stats["total"]).fillna(0)

    # --- Back-to-backs detection (approx using historical predictions dates per team) ---
    all_games_dates = merged.sort_values("date").copy()
    # previous game date per team
    prev_date = {}
    b2b_flags = []
    for _, r in all_games_dates.iterrows():
        d = r["date"].date()
        h = r["home_team"]
        a = r["away_team"]
        h_prev = prev_date.get(h)
        a_prev = prev_date.get(a)
        h_b2b = (h_prev is not None and (d - h_prev).days == 1)
        a_b2b = (a_prev is not None and (d - a_prev).days == 1)
        b2b_flags.append(1 if (h_b2b or a_b2b) else 0)
        prev_date[h] = d
        prev_date[a] = d

    merged["is_b2b"] = b2b_flags
    b2b_stats = merged.groupby("is_b2b").agg(total=("correct_full","count"), hits=("correct_full","sum"))
    b2b_stats["accuracy"] = (b2b_stats["hits"] / b2b_stats["total"]).fillna(0)

    # --- Injuries impact ---
    merged["injury_diff"] = merged["home_injuries_count"].fillna(0) - merged["away_injuries_count"].fillna(0)
    merged["injury_imbalance"] = merged["injury_diff"].abs()
    merged["high_injury_imbalance"] = (merged["injury_imbalance"] >= 3).astype(int)
    injury_stats = merged.groupby("high_injury_imbalance").agg(total=("correct_full","count"), hits=("correct_full","sum"))
    injury_stats["accuracy"] = (injury_stats["hits"] / injury_stats["total"]).fillna(0)

    # --- Model vs Market accuracy ---
    def market_pick(r):
        try:
            s = float(r.get("home_spread", 0) or 0)
        except Exception:
            s = 0
        if s < 0:
            return r["home_team"]
        if s > 0:
            return r["away_team"]
        return None

    merged["market_pick"] = merged.apply(market_pick, axis=1)
    merged["model_against_market"] = merged.apply(lambda r: 1 if (r["full_game_pick"] != r["market_pick"] and r["market_pick"] is not None) else 0, axis=1)
    mam = merged[merged["market_pick"].notna()]
    mam_stats = {}
    if len(mam):
        ag = mam.groupby("model_against_market").agg(total=("correct_full","count"), hits=("correct_full","sum"))
        ag["accuracy"] = (ag["hits"] / ag["total"]).fillna(0)
        mam_stats = ag.reset_index().to_dict(orient="records")

    # Save a small report
    out = {
        "dow": dow.reset_index().to_dict(orient="records"),
        "underdogs": {
            "total": int(len(und)),
            "hits": int(und["correct_full"].sum()),
            "accuracy": float(und["correct_full"].mean()) if len(und) else None,
        },
        "calibration_bins": calib_grp.reset_index().to_dict(orient="records"),
        "spread_bins": spread_stats.reset_index().to_dict(orient="records"),
        "b2b": b2b_stats.reset_index().to_dict(orient="records"),
        "injury_impact": injury_stats.reset_index().to_dict(orient="records"),
        "model_vs_market": mam_stats,
    }

    out_file = HIST_PRED_DIR.parent / "backtest_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\n✅ Report saved: {out_file}")


if __name__ == '__main__':
    backtest_report()
