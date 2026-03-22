from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ODDS_OVERRIDES_FILE = BASE_DIR / "data" / "odds_provider" / "closing_odds_overrides.csv"

PRICE_KEYS = [
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

LINE_KEYS = [
    "closing_spread_line",
    "closing_total_line",
    "odds_over_under",
]


def _valid_number(value: Any) -> bool:
    try:
        x = float(value)
    except Exception:
        return False
    if math.isnan(x):
        return False
    return x != 0.0


def _existing_value_is_valid(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "n/a", "none"}:
        return False
    return _valid_number(value)


def _valid_line_number(value: Any) -> bool:
    try:
        x = float(value)
    except Exception:
        return False
    return not math.isnan(x)


def _existing_line_is_valid(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "n/a", "none"}:
        return False
    return _valid_line_number(value)


def _load_overrides_map() -> dict[tuple[str, str, str], dict[str, float]]:
    if not ODDS_OVERRIDES_FILE.exists():
        return {}

    try:
        df = pd.read_csv(ODDS_OVERRIDES_FILE, dtype=str)
    except Exception:
        return {}

    if df.empty:
        return {}

    required = {"sport", "date", "game_id"}
    if not required.issubset(set(df.columns)):
        return {}

    # Normalize key columns once to avoid per-row Series construction overhead.
    for key_col in ("sport", "date", "game_id"):
        if key_col not in df.columns:
            return {}

    df["sport"] = df["sport"].fillna("").astype(str).str.strip().str.lower()
    df["date"] = df["date"].fillna("").astype(str).str.strip()
    df["game_id"] = df["game_id"].fillna("").astype(str).str.strip()

    available_price_cols = [c for c in PRICE_KEYS if c in df.columns]
    available_line_cols = [c for c in LINE_KEYS if c in df.columns]
    for c in available_price_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in available_line_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep only rows with valid identity and at least one numeric price.
    candidate_cols = [*available_price_cols, *available_line_cols]
    if candidate_cols:
        df = df[df[candidate_cols].notna().any(axis=1)]
    df = df[(df["sport"] != "") & (df["date"] != "") & (df["game_id"] != "")]

    out: dict[tuple[str, str, str], dict[str, float]] = {}
    if df.empty:
        return out

    cols = ["sport", "date", "game_id", *available_price_cols, *available_line_cols]
    for row in df[cols].to_dict("records"):
        sport = str(row.get("sport") or "")
        date_str = str(row.get("date") or "")
        game_id = str(row.get("game_id") or "")

        payload: dict[str, float] = {}
        for key in available_price_cols:
            val = row.get(key)
            if _valid_number(val):
                payload[key] = float(val)

        for key in available_line_cols:
            val = row.get(key)
            if _valid_line_number(val):
                payload[key] = float(val)

        if payload:
            out[(sport, date_str, game_id)] = payload

    return out


def apply_overrides_to_events(sport: str, date_str: str, events: list[dict]) -> list[dict]:
    if not isinstance(events, list) or not events:
        return events

    overrides = _load_overrides_map()
    if not overrides:
        return events

    out = []
    for event in events:
        if not isinstance(event, dict):
            out.append(event)
            continue

        item = dict(event)
        gid = str(item.get("game_id") or "").strip()
        if not gid:
            out.append(item)
            continue

        payload = overrides.get((str(sport or "").lower(), str(date_str or "").strip(), gid))
        if not payload:
            out.append(item)
            continue

        merged = False
        for key, value in payload.items():
            if key in PRICE_KEYS:
                if not _existing_value_is_valid(item.get(key)):
                    item[key] = value
                    merged = True
                continue

            if key in LINE_KEYS:
                if not _existing_line_is_valid(item.get(key)):
                    item[key] = value
                    merged = True

                if key == "closing_total_line" and not _existing_line_is_valid(item.get("odds_over_under")):
                    item["odds_over_under"] = value
                    merged = True
                continue

        if merged:
            item["odds_data_quality"] = "real"
            item["odds_source_provider"] = "external_override_csv"

        out.append(item)

    return out
