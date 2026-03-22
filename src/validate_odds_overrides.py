from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OVERRIDES_FILE = BASE_DIR / "data" / "odds_provider" / "closing_odds_overrides.csv"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_CSV = REPORTS_DIR / "odds_overrides_validation.csv"
VALIDATION_JSON = REPORTS_DIR / "odds_overrides_validation_summary.json"

SPORTS_PREDICTIONS_DIR = {
    "nba": BASE_DIR / "data" / "predictions",
    "mlb": BASE_DIR / "data" / "mlb" / "predictions",
    "kbo": BASE_DIR / "data" / "kbo" / "predictions",
    "nhl": BASE_DIR / "data" / "nhl" / "predictions",
    "liga_mx": BASE_DIR / "data" / "liga_mx" / "predictions",
    "laliga": BASE_DIR / "data" / "laliga" / "predictions",
    "euroleague": BASE_DIR / "data" / "euroleague" / "predictions",
    "ncaa_baseball": BASE_DIR / "data" / "ncaa_baseball" / "predictions",
}

PRICE_COLUMNS = [
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

MARKET_TO_PRICE_COLUMNS = {
    "full_game": ["closing_moneyline_odds", "home_moneyline_odds", "draw_moneyline_odds", "away_moneyline_odds"],
    "spread": ["closing_spread_odds"],
    "total": ["closing_total_odds"],
    "q1": ["closing_q1_odds"],
    "q1_yrfi": ["closing_q1_odds"],
    "f5": ["closing_f5_odds"],
    "home_over": ["closing_home_over_odds"],
    "corners": ["closing_corners_odds"],
    "btts": ["closing_btts_odds"],
}


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


def _event_markets(event: dict):
    markets = set()

    if event.get("full_game_pick") or event.get("recommended_pick"):
        markets.add("full_game")

    if event.get("spread_pick"):
        markets.add("spread")

    if event.get("total_pick") or event.get("total_recommended_pick"):
        markets.add("total")

    if event.get("q1_pick"):
        q1_pick = str(event.get("q1_pick") or "").upper()
        if q1_pick in {"YRFI", "NRFI"}:
            markets.add("q1_yrfi")
        else:
            markets.add("q1")

    if event.get("f5_pick") or event.get("assists_pick"):
        markets.add("f5")

    if event.get("home_over_pick"):
        markets.add("home_over")

    if event.get("corners_pick") or event.get("corners_recommended_pick"):
        markets.add("corners")

    if event.get("btts_pick") or event.get("btts_recommended_pick"):
        markets.add("btts")

    return markets


def _prediction_market_expectations():
    expected = {}

    for sport, pred_dir in SPORTS_PREDICTIONS_DIR.items():
        if not pred_dir.exists():
            continue

        for fp in pred_dir.glob("*.json"):
            date_str = str(fp.stem)
            events = _read_events_json(fp)
            for event in events:
                if not isinstance(event, dict):
                    continue
                game_id = str(event.get("game_id") or "").strip()
                if not game_id:
                    continue

                key = (sport, date_str, game_id)
                current = expected.setdefault(key, set())
                current.update(_event_markets(event))

    return expected


def _coerce_prices(df: pd.DataFrame):
    out = df.copy()
    for col in PRICE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _row_missing_expected_prices(row: pd.Series, expected_markets: set[str]):
    missing = []
    for market in sorted(expected_markets):
        needed_cols = MARKET_TO_PRICE_COLUMNS.get(market, [])
        if not needed_cols:
            continue
        has_any = any(pd.notna(row.get(c)) for c in needed_cols)
        if not has_any:
            missing.append(market)
    return missing


def run_validation():
    if not OVERRIDES_FILE.exists():
        raise FileNotFoundError(f"No existe overrides: {OVERRIDES_FILE}")

    df = pd.read_csv(OVERRIDES_FILE, dtype=str)
    if df.empty:
        print("[WARN] overrides vacío")
        return

    for key in ["sport", "date", "game_id"]:
        if key not in df.columns:
            raise ValueError(f"Falta columna requerida: {key}")

    df["sport"] = df["sport"].fillna("").astype(str).str.strip().str.lower()
    df["date"] = df["date"].fillna("").astype(str).str.strip()
    df["game_id"] = df["game_id"].fillna("").astype(str).str.strip()

    df = _coerce_prices(df)

    expected_map = _prediction_market_expectations()

    has_any_price = df[PRICE_COLUMNS].notna().any(axis=1)
    key_tuples = list(zip(df["sport"], df["date"], df["game_id"]))

    expected_markets_col = []
    missing_markets_col = []
    is_orphan_col = []

    for idx, key in enumerate(key_tuples):
        expected_markets = expected_map.get(key, set())
        expected_markets_col.append(",".join(sorted(expected_markets)))
        missing_markets = _row_missing_expected_prices(df.iloc[idx], expected_markets)
        missing_markets_col.append(",".join(missing_markets))
        is_orphan_col.append(len(expected_markets) == 0)

    out = df.copy()
    out["has_any_price"] = has_any_price
    out["expected_markets"] = expected_markets_col
    out["missing_expected_markets"] = missing_markets_col
    out["missing_expected_count"] = out["missing_expected_markets"].apply(lambda x: 0 if not x else len(str(x).split(",")))
    out["is_orphan_override"] = is_orphan_col

    def row_status(row):
        if not bool(row["has_any_price"]):
            return "NO_PRICE"
        if bool(row["is_orphan_override"]):
            return "ORPHAN_WITH_PRICE"
        if int(row["missing_expected_count"]) > 0:
            return "PARTIAL_PRICE"
        return "OK"

    out["status"] = out.apply(row_status, axis=1)

    # Priority: missing prices for expected markets first, then no-price rows with expected markets.
    out["priority_score"] = (
        out["missing_expected_count"].fillna(0).astype(int) * 10
        + ((~out["has_any_price"]).astype(int) * 5)
        + ((~out["is_orphan_override"]).astype(int) * 2)
    )

    out = out.sort_values(["priority_score", "status"], ascending=[False, True]).reset_index(drop=True)
    out.to_csv(VALIDATION_CSV, index=False)

    summary = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "rows_total": int(len(out)),
        "rows_with_any_price": int(out["has_any_price"].sum()),
        "rows_without_price": int((~out["has_any_price"]).sum()),
        "rows_orphan": int(out["is_orphan_override"].sum()),
        "status_counts": out["status"].value_counts(dropna=False).to_dict(),
        "top_missing_markets": (
            out.loc[out["missing_expected_markets"].astype(str) != "", "missing_expected_markets"]
            .str.split(",")
            .explode()
            .value_counts()
            .head(15)
            .to_dict()
        ),
        "validation_csv": str(VALIDATION_CSV),
    }

    VALIDATION_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] validation CSV: {VALIDATION_CSV}")
    print(f"[OK] validation JSON: {VALIDATION_JSON}")
    print(f"[OK] rows_total={summary['rows_total']} rows_with_any_price={summary['rows_with_any_price']}")
    print(f"[OK] status_counts={summary['status_counts']}")


if __name__ == "__main__":
    run_validation()
