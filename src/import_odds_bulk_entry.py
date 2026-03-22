from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ODDS_DIR = BASE_DIR / "data" / "odds_provider"
ODDS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = ODDS_DIR / "closing_odds_bulk_entry.csv"
OVERRIDES_FILE = ODDS_DIR / "closing_odds_overrides.csv"

KEY_COLUMNS = ["sport", "date", "game_id"]
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


def _valid_number(value) -> bool:
    try:
        x = float(value)
    except Exception:
        return False
    if math.isnan(x):
        return False
    return x != 0.0


def _valid_line(value) -> bool:
    try:
        x = float(value)
    except Exception:
        return False
    return not math.isnan(x)


def run_import() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"No existe hoja de captura: {INPUT_FILE}")

    incoming = pd.read_csv(INPUT_FILE, dtype=str)
    if incoming.empty:
        print("[WARN] Hoja de captura vacia")
        return

    for k in KEY_COLUMNS:
        if k not in incoming.columns:
            raise ValueError(f"Falta columna requerida en captura: {k}")

    for k in KEY_COLUMNS:
        incoming[k] = incoming[k].fillna("").astype(str).str.strip().str.lower() if k == "sport" else incoming[k].fillna("").astype(str).str.strip()

    for c in ODDS_COLUMNS:
        if c not in incoming.columns:
            incoming[c] = pd.NA
        incoming[c] = pd.to_numeric(incoming[c], errors="coerce")

    for c in LINE_COLUMNS:
        if c not in incoming.columns:
            incoming[c] = pd.NA
        incoming[c] = pd.to_numeric(incoming[c], errors="coerce")

    if OVERRIDES_FILE.exists():
        base = pd.read_csv(OVERRIDES_FILE, dtype=str)
    else:
        base = pd.DataFrame(columns=[*KEY_COLUMNS, *ODDS_COLUMNS, *LINE_COLUMNS])

    for k in KEY_COLUMNS:
        if k not in base.columns:
            base[k] = ""
        base[k] = base[k].fillna("").astype(str).str.strip().str.lower() if k == "sport" else base[k].fillna("").astype(str).str.strip()

    for c in ODDS_COLUMNS:
        if c not in base.columns:
            base[c] = pd.NA
        base[c] = pd.to_numeric(base[c], errors="coerce")

    for c in LINE_COLUMNS:
        if c not in base.columns:
            base[c] = pd.NA
        base[c] = pd.to_numeric(base[c], errors="coerce")

    base = base.drop_duplicates(subset=KEY_COLUMNS, keep="first")
    incoming = incoming.drop_duplicates(subset=KEY_COLUMNS, keep="first")

    merged = base.merge(incoming[KEY_COLUMNS + ODDS_COLUMNS + LINE_COLUMNS], on=KEY_COLUMNS, how="outer", suffixes=("", "__in"))

    cells_updated = 0
    rows_with_updates = 0

    for col in ODDS_COLUMNS:
        in_col = f"{col}__in"
        before = merged[col].copy()

        # Update only when incoming has a valid number.
        mask = merged[in_col].apply(_valid_number)
        merged.loc[mask, col] = merged.loc[mask, in_col]

        changed = (before != merged[col]) & ~(before.isna() & merged[col].isna())
        cells_updated += int(changed.sum())

    for col in LINE_COLUMNS:
        in_col = f"{col}__in"
        before = merged[col].copy()

        mask = merged[in_col].apply(_valid_line)
        merged.loc[mask, col] = merged.loc[mask, in_col]

        changed = (before != merged[col]) & ~(before.isna() & merged[col].isna())
        cells_updated += int(changed.sum())

    touched_mask = pd.Series(False, index=merged.index)
    for col in ODDS_COLUMNS:
        in_col = f"{col}__in"
        touched_mask = touched_mask | merged[in_col].apply(_valid_number)

    for col in LINE_COLUMNS:
        in_col = f"{col}__in"
        touched_mask = touched_mask | merged[in_col].apply(_valid_line)

    rows_with_updates = int(touched_mask.sum())

    for col in [f"{c}__in" for c in [*ODDS_COLUMNS, *LINE_COLUMNS]]:
        if col in merged.columns:
            merged.drop(columns=[col], inplace=True)

    out_cols = [*KEY_COLUMNS, *ODDS_COLUMNS, *LINE_COLUMNS]
    merged = merged[out_cols]
    merged = merged.sort_values(["date", "sport", "game_id"], ascending=[True, True, True]).reset_index(drop=True)

    merged.to_csv(OVERRIDES_FILE, index=False)

    print(f"[OK] overrides actualizado: {OVERRIDES_FILE}")
    print(f"[OK] rows_total={len(merged)} rows_touched={rows_with_updates} cells_updated={cells_updated}")


if __name__ == "__main__":
    run_import()
