from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import json
import argparse
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


def detect_odds_column(df: pd.DataFrame):
    candidates = [
        "published_odds",
        "odds",
        "market_odds",
        "book_odds",
        "close_odds",
        "open_odds",
        "line",
        "open_line",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def american_to_decimal(a):
    try:
        a = float(a)
    except Exception:
        s = str(a).strip()
        if s.startswith("+") or s.startswith("-"):
            a = int(s)
        else:
            try:
                return float(s)
            except Exception:
                return None
    if a == 0:
        return None
    if a > 0:
        return 1 + (a / 100.0)
    else:
        return 1 + (100.0 / abs(a))


def to_decimal_odds(val):
    if pd.isna(val):
        return None
    try:
        v = float(val)
    except Exception:
        # maybe american string
        try:
            return american_to_decimal(val)
        except Exception:
            return None
    # Heuristic: if value is between 1.01 and 50, assume decimal
    if 1.01 <= v <= 50:
        return v
    # otherwise treat as american
    return american_to_decimal(v)


def compute_backtest(detail_df: pd.DataFrame, unit=1.0, vig=0.0, include_unpublished=False, odds_col=None):
    if detail_df.empty:
        return {}

    if odds_col is None:
        odds_col = detect_odds_column(detail_df)

    picks = detail_df.copy()
    if not include_unpublished and "publish_pick" in picks.columns:
        picks = picks[picks["publish_pick"] == 1]

    picks = picks.reset_index(drop=True)
    if picks.empty:
        return {}

    results = []
    for _, r in picks.iterrows():
        odds_val = None
        if odds_col and odds_col in r.index:
            odds_val = r[odds_col]
        dec = to_decimal_odds(odds_val) if odds_val is not None else None
        if dec is None or dec <= 1.0:
            # skip if no sensible odds
            continue

        stake = float(unit)
        vig_fee = stake * float(vig)
        win = None
        if "y_true" in r.index:
            try:
                win = int(r["y_true"]) == int(r.get("pred_label", 1))
            except Exception:
                # fallback: if y_true equals pred_label
                win = int(r["y_true"]) == int(r.get("pred_label", 1))
        else:
            # can't determine result; skip
            continue

        if win:
            payout = stake * dec
            profit = payout - stake - vig_fee
        else:
            profit = -stake - vig_fee

        results.append({
            "date": r.get("date"),
            "market_key": r.get("market_key"),
            "stake": stake,
            "vig_fee": vig_fee,
            "dec_odds": dec,
            "profit": profit,
            "win": bool(win),
        })

    if not results:
        return {}

    dfr = pd.DataFrame(results)
    total_stake = dfr["stake"].sum()
    total_profit = dfr["profit"].sum()
    roi = float(total_profit / total_stake) if total_stake else None
    win_rate = float(dfr["win"].mean())

    # Daily pnl for variance estimates
    if "date" in detail_df.columns:
        daily = dfr.groupby("date")["profit"].sum().reset_index()
        mean_daily = float(daily["profit"].mean())
        std_daily = float(daily["profit"].std(ddof=0)) if len(daily) > 1 else 0.0
    else:
        mean_daily, std_daily = None, None

    summary = {
        "n_picks": int(len(dfr)),
        "total_stake": float(total_stake),
        "total_profit": float(total_profit),
        "roi": roi,
        "win_rate": win_rate,
        "mean_daily_profit": mean_daily,
        "std_daily_profit": std_daily,
        "avg_odds": float(dfr["dec_odds"].mean()),
    }

    return {"summary": summary, "detail": dfr}


def save_report(report: dict):
    out_path = REPORTS_DIR / "backtest_report.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Backtest report saved to: {out_path}")
    except Exception as e:
        print(f"Failed to save backtest report: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", type=float, default=1.0, help="Unit stake per published pick")
    parser.add_argument("--vig", type=float, default=0.0, help="Vigorish as fraction of stake (e.g. 0.02)")
    parser.add_argument("--include-unpublished", action="store_true")
    parser.add_argument("--odds-col", type=str, default=None)
    args = parser.parse_args()

    df = gather_walkforward_details()
    if df.empty:
        print("No walkforward detail data found for backtest.")
        return

    out = compute_backtest(df, unit=args.unit, vig=args.vig, include_unpublished=args.include_unpublished, odds_col=args.odds_col)
    if not out:
        print("No valid picks or odds found to run backtest.")
        return

    save_report(out["summary"])  # save only summary at top-level
    # also save detail csv of simulated bets
    detail_path = REPORTS_DIR / "backtest_detail.csv"
    out["detail"].to_csv(detail_path, index=False)
    print("Backtest complete:")
    print(json.dumps(out["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
