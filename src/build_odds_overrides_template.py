from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OVERRIDES_FILE = BASE_DIR / "data" / "odds_provider" / "closing_odds_overrides.csv"

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


def _read_events_json(path: Path):
    try:
        data = pd.read_json(path)
        if isinstance(data, pd.DataFrame) and "games" in data.columns:
            # Not expected in current files, fallback to python json if needed.
            pass
    except Exception:
        pass

    import json

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(payload, dict) and isinstance(payload.get("games"), list):
        return payload["games"]
    if isinstance(payload, list):
        return payload
    return []


def _safe_date(txt: str):
    try:
        return datetime.strptime(str(txt)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def build_template(days_ahead: int = 14):
    start = date.today()
    end = start + timedelta(days=days_ahead)

    rows = []
    for sport, pred_dir in SPORTS_PREDICTIONS_DIR.items():
        if not pred_dir.exists():
            continue

        for fp in sorted(pred_dir.glob("*.json")):
            day = _safe_date(fp.stem)
            if day is None or day < start or day > end:
                continue

            events = _read_events_json(fp)
            for e in events:
                if not isinstance(e, dict):
                    continue
                gid = str(e.get("game_id") or "").strip()
                d = str(e.get("date") or "").strip()[:10]
                if not gid or not d:
                    continue

                row = {
                    "sport": sport,
                    "date": d,
                    "game_id": gid,
                }
                for key in ODDS_COLUMNS:
                    row[key] = e.get(key)
                rows.append(row)

    template_df = pd.DataFrame(rows)
    if template_df.empty:
        template_df = pd.DataFrame(columns=["sport", "date", "game_id", *ODDS_COLUMNS])

    template_df = template_df.drop_duplicates(subset=["sport", "date", "game_id"], keep="last")

    if OVERRIDES_FILE.exists():
        existing = pd.read_csv(OVERRIDES_FILE, dtype={"sport": str, "date": str, "game_id": str})
        for k in ["sport", "date", "game_id"]:
            if k not in existing.columns:
                existing[k] = ""
            if k == "sport":
                existing[k] = existing[k].fillna("").astype(str).str.strip().str.lower()
            else:
                existing[k] = existing[k].fillna("").astype(str).str.strip()

        for k in ["sport", "date", "game_id"]:
            if k == "sport":
                template_df[k] = template_df[k].fillna("").astype(str).str.strip().str.lower()
            else:
                template_df[k] = template_df[k].fillna("").astype(str).str.strip()

        for c in ODDS_COLUMNS:
            if c not in existing.columns:
                existing[c] = pd.NA
            existing[c] = pd.to_numeric(existing[c], errors="coerce")
            template_df[c] = pd.to_numeric(template_df[c], errors="coerce")

        existing = existing.drop_duplicates(subset=["sport", "date", "game_id"], keep="first")
        template_df = template_df.drop_duplicates(subset=["sport", "date", "game_id"], keep="last")

        merged = existing.merge(
            template_df[["sport", "date", "game_id", *ODDS_COLUMNS]],
            on=["sport", "date", "game_id"],
            how="outer",
            suffixes=("", "__new"),
        )

        for c in ODDS_COLUMNS:
            new_c = f"{c}__new"
            if new_c not in merged.columns:
                continue
            fill_mask = merged[c].isna() & merged[new_c].notna()
            merged.loc[fill_mask, c] = merged.loc[fill_mask, new_c]
            merged.drop(columns=[new_c], inplace=True)
    else:
        merged = template_df

    OVERRIDES_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OVERRIDES_FILE, index=False)

    print(f"[OK] overrides template updated: {OVERRIDES_FILE}")
    print(f"[OK] rows total: {len(merged)} | new_window_rows: {len(template_df)}")


if __name__ == "__main__":
    build_template(days_ahead=21)
