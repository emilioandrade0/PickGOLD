from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

SPORTS = {
    "nba": {
        "raw_file": BASE_DIR / "data" / "raw" / "nba_advanced_history.csv",
        "predictions_dir": BASE_DIR / "data" / "predictions",
        "historical_dir": BASE_DIR / "data" / "historical_predictions",
    },
    "mlb": {
        "raw_file": BASE_DIR / "data" / "mlb" / "raw" / "mlb_advanced_history.csv",
        "predictions_dir": BASE_DIR / "data" / "mlb" / "predictions",
        "historical_dir": BASE_DIR / "data" / "mlb" / "historical_predictions",
    },
    "lmb": {
        "raw_file": BASE_DIR / "data" / "lmb" / "raw" / "lmb_advanced_history.csv",
        "predictions_dir": BASE_DIR / "data" / "lmb" / "predictions",
        "historical_dir": BASE_DIR / "data" / "lmb" / "historical_predictions",
    },
    "kbo": {
        "raw_file": BASE_DIR / "data" / "kbo" / "raw" / "kbo_advanced_history.csv",
        "predictions_dir": BASE_DIR / "data" / "kbo" / "predictions",
        "historical_dir": BASE_DIR / "data" / "kbo" / "historical_predictions",
    },
    "nhl": {
        "raw_file": BASE_DIR / "data" / "nhl" / "raw" / "nhl_advanced_history.csv",
        "predictions_dir": BASE_DIR / "data" / "nhl" / "predictions",
        "historical_dir": BASE_DIR / "data" / "nhl" / "historical_predictions",
    },
    "liga_mx": {
        "raw_file": BASE_DIR / "data" / "liga_mx" / "raw" / "liga_mx_advanced_history.csv",
        "predictions_dir": BASE_DIR / "data" / "liga_mx" / "predictions",
        "historical_dir": BASE_DIR / "data" / "liga_mx" / "historical_predictions",
    },
    "laliga": {
        "raw_file": BASE_DIR / "data" / "laliga" / "raw" / "laliga_advanced_history.csv",
        "predictions_dir": BASE_DIR / "data" / "laliga" / "predictions",
        "historical_dir": BASE_DIR / "data" / "laliga" / "historical_predictions",
    },
    "euroleague": {
        "raw_file": BASE_DIR / "data" / "euroleague" / "raw" / "euroleague_advanced_history.csv",
        "predictions_dir": BASE_DIR / "data" / "euroleague" / "predictions",
        "historical_dir": BASE_DIR / "data" / "euroleague" / "historical_predictions",
    },
    "ncaa_baseball": {
        "raw_file": BASE_DIR / "data" / "ncaa_baseball" / "raw" / "ncaa_baseball_advanced_history.csv",
        "predictions_dir": BASE_DIR / "data" / "ncaa_baseball" / "predictions",
        "historical_dir": BASE_DIR / "data" / "ncaa_baseball" / "historical_predictions",
    },
}

ODDS_FIELDS = [
    "closing_moneyline_odds", "closing_ml_odds", "closing_odds_ml", "moneyline_odds", "ml_odds", "odds_ml", "odds_moneyline", "opening_moneyline_odds", "opening_ml_odds",
    "closing_spread_odds", "spread_odds", "odds_spread_price", "opening_spread_odds",
    "closing_total_odds", "total_odds", "odds_total_price", "opening_total_odds",
    "closing_q1_odds", "closing_yrfi_odds", "q1_odds", "yrfi_odds", "nrfi_odds", "opening_q1_odds", "opening_yrfi_odds",
    "closing_f5_odds", "f5_odds", "opening_f5_odds",
    "closing_home_over_odds", "home_over_odds", "opening_home_over_odds",
    "closing_corners_odds", "corners_odds", "opening_corners_odds",
    "closing_btts_odds", "btts_odds", "opening_btts_odds",
]


def _read_json_events(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("games"), list):
        return data["games"], True
    if isinstance(data, list):
        return data, False
    return [], False


def _write_json_events(path: Path, events: list, wrap_games: bool):
    payload = {"games": events} if wrap_games else events
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_valid_odds_value(v):
    try:
        x = float(v)
    except Exception:
        return False
    if x == 0:
        return False
    # American odds convention.
    if abs(x) >= 100:
        return True
    # Decimal odds convention.
    if 1.01 <= x <= 20.0:
        return True
    return False


def _extract_odds_payload(row: dict):
    out = {}
    for k in ODDS_FIELDS:
        if k not in row:
            continue
        val = row.get(k)
        if val is None or str(val).strip() in {"", "N/A", "nan"}:
            continue
        if _is_valid_odds_value(val):
            out[k] = float(val)

    # Keep total line if present for total market reconstruction.
    if "odds_over_under" in row:
        try:
            line = float(row.get("odds_over_under"))
            if line > 0:
                out["odds_over_under"] = line
        except Exception:
            pass

    return out


def _load_raw_lookup(raw_file: Path):
    if not raw_file.exists():
        return {}

    df = pd.read_csv(raw_file, dtype={"game_id": str})
    lookup = {}
    for _, row in df.iterrows():
        gid = str(row.get("game_id") or "")
        if not gid:
            continue
        payload = _extract_odds_payload(row.to_dict())
        if payload:
            lookup[gid] = payload
    return lookup


def _patch_events_file(file_path: Path, raw_lookup: dict):
    events, wrapped = _read_json_events(file_path)
    if not events:
        return 0, 0

    touched_events = 0
    inserted_fields = 0

    for event in events:
        gid = str(event.get("game_id") or "")
        if not gid:
            continue
        payload = raw_lookup.get(gid)
        if not payload:
            continue

        event_changed = False
        for k, v in payload.items():
            if event.get(k) in (None, "", "N/A"):
                event[k] = v
                inserted_fields += 1
                event_changed = True

        if event_changed:
            event["odds_data_quality"] = "real"
            touched_events += 1

    if touched_events > 0:
        _write_json_events(file_path, events, wrapped)

    return touched_events, inserted_fields


def _patch_directory(dir_path: Path, raw_lookup: dict):
    if not dir_path.exists():
        return 0, 0, 0

    files = sorted(dir_path.glob("*.json"))
    files_touched = 0
    events_touched = 0
    fields_inserted = 0

    for fp in files:
        e_touched, f_inserted = _patch_events_file(fp, raw_lookup)
        if e_touched > 0:
            files_touched += 1
            events_touched += e_touched
            fields_inserted += f_inserted

    return files_touched, events_touched, fields_inserted


def run_backfill():
    print("[ODDS] Backfilling real market odds into predictions/historical files")
    print("=" * 72)

    grand_files = 0
    grand_events = 0
    grand_fields = 0

    for sport, cfg in SPORTS.items():
        raw_lookup = _load_raw_lookup(cfg["raw_file"])
        raw_games_with_odds = len(raw_lookup)

        p_files, p_events, p_fields = _patch_directory(cfg["predictions_dir"], raw_lookup)
        h_files, h_events, h_fields = _patch_directory(cfg["historical_dir"], raw_lookup)

        total_files = p_files + h_files
        total_events = p_events + h_events
        total_fields = p_fields + h_fields

        grand_files += total_files
        grand_events += total_events
        grand_fields += total_fields

        print(
            f"[{sport}] raw_games_with_real_odds={raw_games_with_odds} | "
            f"files_touched={total_files} | events_touched={total_events} | fields_inserted={total_fields}"
        )

    print("-" * 72)
    print(
        f"[DONE] files_touched={grand_files} | events_touched={grand_events} | fields_inserted={grand_fields}"
    )


if __name__ == "__main__":
    run_backfill()
