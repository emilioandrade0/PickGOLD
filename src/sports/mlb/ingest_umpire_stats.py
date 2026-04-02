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
    """Best-effort download from umpscorecards.com data endpoint and save CSV.

    If the site or format changes this will fallback to raising an exception.
    """
    url = "https://umpscorecards.com/data/umpires"
    params = {}
    if season:
        params["season"] = season
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    # try to parse as CSV
    txt = resp.text
    try:
        df = pd.read_csv(io.StringIO(txt))
    except Exception:
        # try manual CSV parsing
        rows = list(csv.reader(txt.splitlines()))
        if not rows:
            raise RuntimeError("Downloaded umpire data empty")
        header = rows[0]
        data = rows[1:]
        df = pd.DataFrame(data, columns=header)

    df.to_csv(save_path, index=False)
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
