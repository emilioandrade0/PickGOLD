import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


SRC_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA = SRC_ROOT / "data" / "raw" / "nba_advanced_history.csv"
HIST_PRED_DIR = SRC_ROOT / "data" / "historical_predictions"
PICK_PARAMS_FILE = SRC_ROOT / "models" / "pick_params.pkl"
REPORT_DIR = SRC_ROOT / "reports" / "nba_param_search"


def _load_rows() -> pd.DataFrame:
    raw = pd.read_csv(RAW_DATA, dtype={"game_id": str})
    raw["game_id"] = raw["game_id"].astype(str)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["real_winner"] = np.where(raw["home_pts_total"] > raw["away_pts_total"], raw["home_team"], raw["away_team"])
    actual = raw[["game_id", "date", "real_winner"]].dropna(subset=["date"]).copy()

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
    pred["full_game_calibrated_prob_home"] = pd.to_numeric(pred.get("full_game_calibrated_prob_home"), errors="coerce")
    pred["home_spread"] = pd.to_numeric(pred.get("home_spread"), errors="coerce").fillna(0.0)

    keep_cols = ["game_id", "date", "home_team", "away_team", "full_game_calibrated_prob_home", "home_spread"]
    pred = pred[keep_cols].dropna(subset=["date", "full_game_calibrated_prob_home"]).copy()
    merged = pred.merge(actual, on=["game_id"], how="inner", suffixes=("", "_act"))
    merged = merged.sort_values(["date", "game_id"]).reset_index(drop=True)
    return merged


def _apply_rule(df: pd.DataFrame, threshold: float, market_threshold: float, band: float) -> pd.Series:
    p_home = df["full_game_calibrated_prob_home"].to_numpy(dtype=float)
    hs = df["home_spread"].to_numpy(dtype=float)
    home = df["home_team"].to_numpy(dtype=object)
    away = df["away_team"].to_numpy(dtype=object)

    use_market_side = np.abs(hs) > 0
    t = np.where(use_market_side, market_threshold, threshold)
    model_pick_home = p_home >= t
    picks = np.where(model_pick_home, home, away)

    market_pick = np.where(hs < 0, home, np.where(hs > 0, away, None))
    near = np.abs(p_home - t) <= band
    use_tiebreak = use_market_side & near
    picks = np.where(use_tiebreak, market_pick, picks)
    return pd.Series(picks, index=df.index)


def _score(df: pd.DataFrame, threshold: float, market_threshold: float, band: float) -> float:
    if df.empty:
        return np.nan
    picks = _apply_rule(df, threshold, market_threshold, band)
    return float((picks == df["real_winner"]).mean())


def run_search(
    train_ratio: float = 0.8,
    threshold_min: float = 0.49,
    threshold_max: float = 0.53,
    threshold_step: float = 0.005,
    market_threshold_min: float = 0.49,
    market_threshold_max: float = 0.53,
    market_threshold_step: float = 0.005,
    band_min: float = 0.0,
    band_max: float = 0.12,
    band_step: float = 0.01,
    top_k: int = 20,
):
    df = _load_rows()
    if df.empty:
        raise RuntimeError("No historical rows available to tune.")

    n = len(df)
    split_idx = max(1, min(n - 1, int(n * train_ratio)))
    train_df = df.iloc[:split_idx].copy()
    valid_df = df.iloc[split_idx:].copy()

    th_values = np.arange(threshold_min, threshold_max + 1e-12, threshold_step)
    mth_values = np.arange(market_threshold_min, market_threshold_max + 1e-12, market_threshold_step)
    band_values = np.arange(band_min, band_max + 1e-12, band_step)

    rows = []
    for th in th_values:
        for mth in mth_values:
            for band in band_values:
                train_acc = _score(train_df, float(th), float(mth), float(band))
                valid_acc = _score(valid_df, float(th), float(mth), float(band))
                full_acc = _score(df, float(th), float(mth), float(band))
                rows.append(
                    {
                        "threshold": float(round(th, 6)),
                        "market_threshold": float(round(mth, 6)),
                        "market_tiebreak_band": float(round(band, 6)),
                        "train_acc": float(train_acc),
                        "valid_acc": float(valid_acc),
                        "full_acc": float(full_acc),
                    }
                )

    res = pd.DataFrame(rows).sort_values(
        ["valid_acc", "full_acc", "train_acc"], ascending=[False, False, False]
    ).reset_index(drop=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(REPORT_DIR / "grid_results.csv", index=False)
    top = res.head(top_k).copy()
    top.to_csv(REPORT_DIR / "top_results.csv", index=False)

    best = top.iloc[0].to_dict()
    summary = {
        "games_total": int(n),
        "games_train": int(len(train_df)),
        "games_valid": int(len(valid_df)),
        "best": best,
        "top_k": int(top_k),
        "report_csv": str((REPORT_DIR / "grid_results.csv")),
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Grid search complete.")
    print(f"Total combos: {len(res)}")
    print(f"Best params: {best}")
    print(f"Reports: {REPORT_DIR}")
    return best


def apply_params(threshold: float, market_threshold: float, market_tiebreak_band: float):
    if not PICK_PARAMS_FILE.exists():
        raise FileNotFoundError(f"Missing pick params: {PICK_PARAMS_FILE}")
    obj = joblib.load(PICK_PARAMS_FILE)

    if isinstance(obj.get("game"), dict):
        obj["game"]["threshold"] = float(threshold)
        obj["game"]["market_threshold"] = float(market_threshold)
        obj["game"]["market_tiebreak_band"] = float(market_tiebreak_band)

    splits = obj.get("splits")
    if isinstance(splits, dict):
        for _, block in splits.items():
            if isinstance(block, dict) and isinstance(block.get("game"), dict):
                block["game"]["threshold"] = float(threshold)
                block["game"]["market_threshold"] = float(market_threshold)
                block["game"]["market_tiebreak_band"] = float(market_tiebreak_band)

    backup = PICK_PARAMS_FILE.with_suffix(".pkl.bak")
    joblib.dump(obj, PICK_PARAMS_FILE)
    joblib.dump(obj, backup)
    print(f"Applied params to {PICK_PARAMS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid search full-game decision params for NBA.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--threshold-min", type=float, default=0.49)
    parser.add_argument("--threshold-max", type=float, default=0.53)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument("--market-threshold-min", type=float, default=0.49)
    parser.add_argument("--market-threshold-max", type=float, default=0.53)
    parser.add_argument("--market-threshold-step", type=float, default=0.005)
    parser.add_argument("--band-min", type=float, default=0.0)
    parser.add_argument("--band-max", type=float, default=0.12)
    parser.add_argument("--band-step", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--apply-best", action="store_true")
    args = parser.parse_args()

    best = run_search(
        train_ratio=args.train_ratio,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
        market_threshold_min=args.market_threshold_min,
        market_threshold_max=args.market_threshold_max,
        market_threshold_step=args.market_threshold_step,
        band_min=args.band_min,
        band_max=args.band_max,
        band_step=args.band_step,
        top_k=args.top_k,
    )

    if args.apply_best:
        apply_params(
            threshold=best["threshold"],
            market_threshold=best["market_threshold"],
            market_tiebreak_band=best["market_tiebreak_band"],
        )
