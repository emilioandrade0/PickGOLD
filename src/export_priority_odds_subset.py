from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
INPUT_VALIDATION = REPORTS_DIR / "odds_overrides_validation.csv"

OUTPUT_DIR = BASE_DIR / "data" / "odds_provider"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PRIORITY_CSV = OUTPUT_DIR / "closing_odds_priority_subset.csv"
OUT_PRIORITY_MLB_KBO_NCAA_CSV = OUTPUT_DIR / "closing_odds_priority_baseball_focus.csv"
OUT_PRIORITY_TODAY_CSV = OUTPUT_DIR / "closing_odds_priority_today.csv"


def _safe_date(txt: str):
    try:
        return date.fromisoformat(str(txt)[:10])
    except Exception:
        return None


def _build_missing_flags(df: pd.DataFrame):
    out = df.copy()
    out["missing_full_game"] = out["missing_expected_markets"].astype(str).str.contains("full_game", na=False)
    out["missing_spread"] = out["missing_expected_markets"].astype(str).str.contains("spread", na=False)
    out["missing_total"] = out["missing_expected_markets"].astype(str).str.contains("total", na=False)
    out["missing_q1"] = out["missing_expected_markets"].astype(str).str.contains("q1", na=False)
    out["missing_f5"] = out["missing_expected_markets"].astype(str).str.contains("f5", na=False)
    out["missing_corners"] = out["missing_expected_markets"].astype(str).str.contains("corners", na=False)
    out["missing_btts"] = out["missing_expected_markets"].astype(str).str.contains("btts", na=False)
    return out


def run_export(days_ahead: int = 10, top_n: int = 600):
    if not INPUT_VALIDATION.exists():
        raise FileNotFoundError(f"Validation file not found: {INPUT_VALIDATION}")

    df = pd.read_csv(INPUT_VALIDATION, dtype=str)
    if df.empty:
        raise ValueError("Validation file is empty")

    for col in [
        "priority_score",
        "status",
        "sport",
        "date",
        "game_id",
        "missing_expected_markets",
        "expected_markets",
    ]:
        if col not in df.columns:
            raise ValueError(f"Missing required column in validation file: {col}")

    df["priority_score"] = pd.to_numeric(df["priority_score"], errors="coerce").fillna(0).astype(int)
    df["date_obj"] = df["date"].apply(_safe_date)

    today = date.today()
    end_day = today + timedelta(days=days_ahead)

    # Keep rows that are actionable: missing price or partial coverage, within near horizon.
    actionable = df[df["status"].isin(["NO_PRICE", "PARTIAL_PRICE"])].copy()
    actionable = actionable[actionable["date_obj"].notna()]
    actionable = actionable[(actionable["date_obj"] >= today) & (actionable["date_obj"] <= end_day)]

    actionable = _build_missing_flags(actionable)

    # Market importance weighting for faster value unlock.
    actionable["market_weight"] = (
        actionable["missing_full_game"].astype(int) * 8
        + actionable["missing_spread"].astype(int) * 5
        + actionable["missing_total"].astype(int) * 5
        + actionable["missing_q1"].astype(int) * 3
        + actionable["missing_f5"].astype(int) * 3
        + actionable["missing_corners"].astype(int) * 2
        + actionable["missing_btts"].astype(int) * 2
    )

    actionable["final_priority"] = actionable["priority_score"] + actionable["market_weight"]

    cols_out = [
        "sport",
        "date",
        "game_id",
        "status",
        "missing_expected_markets",
        "expected_markets",
        "priority_score",
        "market_weight",
        "final_priority",
    ]

    priority = actionable.sort_values(
        ["final_priority", "priority_score", "date", "sport"],
        ascending=[False, False, True, True],
    ).head(max(1, int(top_n)))

    priority[cols_out].to_csv(OUT_PRIORITY_CSV, index=False)

    baseball_focus = priority[priority["sport"].isin(["mlb", "kbo", "ncaa_baseball"])].copy()
    baseball_focus[cols_out].to_csv(OUT_PRIORITY_MLB_KBO_NCAA_CSV, index=False)

    today_only = priority[priority["date"] == str(today)].copy()
    today_only[cols_out].to_csv(OUT_PRIORITY_TODAY_CSV, index=False)

    print(f"[OK] priority subset: {OUT_PRIORITY_CSV} ({len(priority)})")
    print(f"[OK] baseball focus: {OUT_PRIORITY_MLB_KBO_NCAA_CSV} ({len(baseball_focus)})")
    print(f"[OK] today subset: {OUT_PRIORITY_TODAY_CSV} ({len(today_only)})")


if __name__ == "__main__":
    run_export(days_ahead=10, top_n=600)
