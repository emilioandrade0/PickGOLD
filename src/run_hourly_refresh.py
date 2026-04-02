from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def run_step(script_name: str):
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    started = time.time()
    print(f"[STEP] {script_name}")
    subprocess.run([sys.executable, str(script_path)], check=True)
    elapsed = time.time() - started
    print(f"[OK] {script_name} ({elapsed:.1f}s)")


# Hourly refresh focuses on fresh data and predictions, not retraining.
STEPS = [
    # NBA
    "sports/nba/data_ingest.py",
    # MLB
    "sports/mlb/data_ingest_mlb.py",
    # KBO
    "sports/kbo/data_ingest_kbo.py",
    # NHL
    "sports/nhl/data_ingest_nhl.py",
    # Liga MX
    "sports/ligamx/data_ingest_liga_mx.py",
    # LaLiga
    "sports/laliga/data_ingest_laliga.py",
    # EuroLeague
    "sports/euroleague/data_ingest_euroleague.py",
    # NCAA Baseball
    "sports/ncaa baseball/data_ingest_ncaa_baseball.py",
    # automation
    "run_odds_automation.py",
    # Predictions
    "sports/nba/predict_today.py",
    "sports/mlb/predict_today_mlb.py",
    "sports/kbo/predict_today_kbo.py",
    "sports/nhl/predict_today_nhl.py",
    "sports/ligamx/predict_today_liga_mx.py",
    "sports/laliga/predict_today_laliga.py",
    "sports/euroleague/predict_today_euroleague.py",
    "sports/ncaa baseball/predict_today_ncaa_baseball.py",
]


def main():
    total_start = time.time()
    for step in STEPS:
        run_step(step)

    elapsed = time.time() - total_start
    print(f"[DONE] Hourly refresh completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
