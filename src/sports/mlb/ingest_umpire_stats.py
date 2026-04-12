from pathlib import Path
import requests
import csv
import pandas as pd
import os
import io
import sys

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
CACHE_DIR = BASE_DIR / "data" / "mlb" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
UMP_CACHE = CACHE_DIR / "umpire_stats.csv"


def fetch_umpscorecards(season=None, save_path=UMP_CACHE, timeout=30):
    """Download umpire aggregates from Umpscorecards API and save normalized CSV."""
    target_year = int(season) if season else pd.Timestamp.now("UTC").year
    start_date = f"{target_year}-01-01"
    end_date = f"{target_year}-12-31"

    url = "https://umpscorecards.com/api/umpires"
    params = {
        "startDate": start_date,
        "endDate": end_date,
        "seasonType": "R",
    }
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("Umpire API returned no rows")

    df = pd.DataFrame(rows)
    if "umpire" not in df.columns:
        raise RuntimeError("Umpire API schema missing 'umpire' column")

    # Keep a lightweight normalized file compatible with feature_engineering_mlb_core.py
    norm = pd.DataFrame()
    norm["umpire"] = df["umpire"].astype(str).str.strip()

    # Historical pipeline expects zone_rate; map to 0..1 when possible.
    if "overall_accuracy_wmean" in df.columns:
        zone_rate = pd.to_numeric(df["overall_accuracy_wmean"], errors="coerce")
        zone_rate = zone_rate.where(zone_rate <= 1.0, zone_rate / 100.0)
        norm["zone_rate"] = zone_rate
    elif "consistency_wmean" in df.columns:
        zone_rate = pd.to_numeric(df["consistency_wmean"], errors="coerce")
        zone_rate = zone_rate.where(zone_rate <= 1.0, zone_rate / 100.0)
        norm["zone_rate"] = zone_rate
    else:
        norm["zone_rate"] = pd.NA

    if "n" in df.columns:
        norm["games_sample"] = pd.to_numeric(df["n"], errors="coerce")
    if "consistency_wmean" in df.columns:
        c = pd.to_numeric(df["consistency_wmean"], errors="coerce")
        norm["consistency_rate"] = c.where(c <= 1.0, c / 100.0)
    if "favor_abs_mean" in df.columns:
        norm["favor_abs_mean"] = pd.to_numeric(df["favor_abs_mean"], errors="coerce")
    if "weighted_score" in df.columns:
        norm["weighted_score"] = pd.to_numeric(df["weighted_score"], errors="coerce")

    norm = norm.dropna(subset=["umpire"]).drop_duplicates(subset=["umpire"], keep="first")
    norm.to_csv(save_path, index=False)
    print(f"Umpire stats saved to: {save_path}")
    return save_path


def load_umpire_stats(path=UMP_CACHE):
    if not Path(path).exists():
        raise FileNotFoundError(f"Umpire cache not found: {path}")
    return pd.read_csv(path)


def main():
    # simple CLI: try to download current season
    season = os.environ.get("MLB_SEASON")
    try:
        fetch_umpscorecards(season=season)
    except Exception as e:
        print("Failed to download from umpscorecards:", e)
        print("If you have a local CSV, place it at:", UMP_CACHE)


if __name__ == "__main__":
    main()
