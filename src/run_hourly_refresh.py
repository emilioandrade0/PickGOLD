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
    "data_ingest.py",
    "data_ingest_mlb.py",
    "data_ingest_kbo.py",
    "data_ingest_nhl.py",
    "data_ingest_liga_mx.py",
    "data_ingest_laliga.py",
    "data_ingest_euroleague.py",
    "data_ingest_ncaa_baseball.py",
    "run_odds_automation.py",
    "predict_today.py",
    "predict_today_mlb.py",
    "predict_today_kbo.py",
    "predict_today_nhl.py",
    "predict_today_liga_mx.py",
    "predict_today_laliga.py",
    "predict_today_euroleague.py",
    "predict_today_ncaa_baseball.py",
]


def main():
    total_start = time.time()
    for step in STEPS:
        run_step(step)

    elapsed = time.time() - total_start
    print(f"[DONE] Hourly refresh completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
