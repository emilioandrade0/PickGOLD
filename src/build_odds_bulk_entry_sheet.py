from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
VALIDATION_FILE = REPORTS_DIR / "odds_overrides_validation.csv"

ODDS_DIR = BASE_DIR / "data" / "odds_provider"
ODDS_DIR.mkdir(parents=True, exist_ok=True)
OVERRIDES_FILE = ODDS_DIR / "closing_odds_overrides.csv"
OUTPUT_FILE = ODDS_DIR / "closing_odds_bulk_entry.csv"

SPORTS_PREDICTIONS_DIR = {
    "nba": BASE_DIR / "data" / "predictions",
    "mlb": BASE_DIR / "data" / "mlb" / "predictions",
    "lmb": BASE_DIR / "data" / "lmb" / "predictions",
    "kbo": BASE_DIR / "data" / "kbo" / "predictions",
    "nhl": BASE_DIR / "data" / "nhl" / "predictions",
    "liga_mx": BASE_DIR / "data" / "liga_mx" / "predictions",
    "laliga": BASE_DIR / "data" / "laliga" / "predictions",
    "euroleague": BASE_DIR / "data" / "euroleague" / "predictions",
    "ncaa_baseball": BASE_DIR / "data" / "ncaa_baseball" / "predictions",
}

ODDS_COLUMNS = [
    "closing_moneyline_odds",
    "home_moneyline_odds",
    "draw_moneyline_odds",
    "away_moneyline_odds",
    "closing_spread_odds",
    "closing_total_odds",
    "closing_q1_odds",
    "closing_f5_odds",
    "closing_home_over_odds",
    "closing_corners_odds",
    "closing_btts_odds",
]

LINE_COLUMNS = [
    "closing_spread_line",
    "closing_total_line",
]


def _read_events_json(path: Path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(payload, dict) and isinstance(payload.get("games"), list):
        return payload["games"]
    if isinstance(payload, list):
        return payload
    return []


def _events_lookup():
    rows = []
    for sport, pred_dir in SPORTS_PREDICTIONS_DIR.items():
        if not pred_dir.exists():
            continue
        for fp in pred_dir.glob("*.json"):
            date_str = str(fp.stem)
            for e in _read_events_json(fp):
                if not isinstance(e, dict):
                    continue
                gid = str(e.get("game_id") or "").strip()
                if not gid:
                    continue
                rows.append(
                    {
                        "sport": sport,
                        "date": str(e.get("date") or date_str).strip()[:10],
                        "game_id": gid,
                        "home_team": str(e.get("home_team") or "").strip(),
                        "away_team": str(e.get("away_team") or "").strip(),
                        "spread_market": str(e.get("spread_market") or "").strip(),
                        "odds_over_under": e.get("odds_over_under"),
                        "corners_line": e.get("corners_line"),
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["sport", "date", "game_id", "home_team", "away_team", "spread_market", "odds_over_under", "corners_line"])

    out["sport"] = out["sport"].astype(str).str.strip().str.lower()
    out["date"] = out["date"].astype(str).str.strip()
    out["game_id"] = out["game_id"].astype(str).str.strip()

    out = out.drop_duplicates(subset=["sport", "date", "game_id"], keep="last")
    return out


def build_bulk_entry_sheet(limit_rows: int = 2500):
    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(f"No existe: {VALIDATION_FILE}")
    if not OVERRIDES_FILE.exists():
        raise FileNotFoundError(f"No existe: {OVERRIDES_FILE}")

    validation = pd.read_csv(VALIDATION_FILE, dtype=str)
    overrides = pd.read_csv(OVERRIDES_FILE, dtype=str)

    for k in ["sport", "date", "game_id"]:
        validation[k] = validation[k].fillna("").astype(str).str.strip().str.lower() if k == "sport" else validation[k].fillna("").astype(str).str.strip()
        overrides[k] = overrides[k].fillna("").astype(str).str.strip().str.lower() if k == "sport" else overrides[k].fillna("").astype(str).str.strip()

    # Prioritize rows with missing prices for expected markets.
    validation["priority_score"] = pd.to_numeric(validation.get("priority_score"), errors="coerce").fillna(0)
    prioritized = validation.sort_values(["priority_score", "status"], ascending=[False, True]).copy()

    key_cols = ["sport", "date", "game_id"]
    info_cols = [
        "status",
        "expected_markets",
        "missing_expected_markets",
        "missing_expected_count",
        "priority_score",
    ]
    info_cols = [c for c in info_cols if c in prioritized.columns]

    rows = prioritized[key_cols + info_cols].copy()
    rows = rows.drop_duplicates(subset=key_cols, keep="first")

    # Bring current override values so user can fill only blanks.
    for c in ODDS_COLUMNS:
        if c not in overrides.columns:
            overrides[c] = pd.NA

    for c in LINE_COLUMNS:
        if c not in overrides.columns:
            overrides[c] = pd.NA

    rows = rows.merge(overrides[key_cols + ODDS_COLUMNS + LINE_COLUMNS], on=key_cols, how="left")

    # Add matchup context and line hints for easier manual capture.
    events = _events_lookup()
    rows = rows.merge(events, on=key_cols, how="left")

    rows = rows.head(max(1, int(limit_rows)))

    ordered_cols = [
        "sport",
        "date",
        "game_id",
        "home_team",
        "away_team",
        "status",
        "expected_markets",
        "missing_expected_markets",
        "missing_expected_count",
        "priority_score",
        "spread_market",
        "odds_over_under",
        "corners_line",
        *LINE_COLUMNS,
        *ODDS_COLUMNS,
        "source_book",
        "source_timestamp",
        "notes",
    ]

    for c in ordered_cols:
        if c not in rows.columns:
            rows[c] = ""

    rows = rows[ordered_cols]
    rows.to_csv(OUTPUT_FILE, index=False)

    print(f"[OK] bulk entry sheet: {OUTPUT_FILE}")
    print(f"[OK] rows: {len(rows)}")


if __name__ == "__main__":
    build_bulk_entry_sheet(limit_rows=2500)
