from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


CRITICAL_STEPS = {
    "sports/nba/data_ingest.py",
    "sports/mlb/data_ingest_mlb.py",
    "sports/kbo/data_ingest_kbo.py",
    "sports/nhl/data_ingest_nhl.py",
    "sports/ligamx/data_ingest_liga_mx.py",
    "sports/laliga/data_ingest_laliga.py",
    "sports/bundesliga/data_ingest_bundesliga.py",
    "sports/ligue1/data_ingest_ligue1.py",
    "sports/euroleague/data_ingest_euroleague.py",
    "sports/ncaa baseball/data_ingest_ncaa_baseball.py",
    "sports/triple_a/data_ingest_triple_a.py",
    "run_odds_automation.py",
}


def _step_env() -> dict:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TERM", "xterm-256color")
    return env


def run_step(script_name: str):
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    started = time.time()
    print(f"[STEP] {script_name}")
    try:
        subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=str(BASE_DIR.parent),
            env=_step_env(),
        )
        elapsed = time.time() - started
        print(f"[OK] {script_name} ({elapsed:.1f}s)")
        return {"step": script_name, "ok": True, "elapsed": round(elapsed, 1)}
    except subprocess.CalledProcessError as exc:
        elapsed = time.time() - started
        print(f"[FAIL] {script_name} ({elapsed:.1f}s) exit={exc.returncode}")
        return {
            "step": script_name,
            "ok": False,
            "elapsed": round(elapsed, 1),
            "returncode": int(exc.returncode),
            "critical": script_name in CRITICAL_STEPS,
        }
    except Exception as exc:
        elapsed = time.time() - started
        print(f"[FAIL] {script_name} ({elapsed:.1f}s) error={exc}")
        return {
            "step": script_name,
            "ok": False,
            "elapsed": round(elapsed, 1),
            "returncode": None,
            "critical": script_name in CRITICAL_STEPS,
            "error": str(exc),
        }


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
    # Bundesliga
    "sports/bundesliga/data_ingest_bundesliga.py",
    # Ligue 1
    "sports/ligue1/data_ingest_ligue1.py",
    # EuroLeague
    "sports/euroleague/data_ingest_euroleague.py",
    # NCAA Baseball
    "sports/ncaa baseball/data_ingest_ncaa_baseball.py",
    # Triple-A
    "sports/triple_a/data_ingest_triple_a.py",
    # automation
    "run_odds_automation.py",
    # Predictions
    "sports/nba/predict_today.py",
    "sports/mlb/predict_today_mlb.py",
    "sports/kbo/predict_today_kbo.py",
    "sports/nhl/predict_today_nhl.py",
    "sports/ligamx/predict_today_liga_mx.py",
    "sports/laliga/predict_today_laliga.py",
    "sports/bundesliga/predict_today_bundesliga.py",
    "sports/ligue1/predict_today_ligue1.py",
    "sports/euroleague/predict_today_euroleague.py",
    "sports/ncaa baseball/predict_today_ncaa_baseball.py",
    "sports/triple_a/predict_today_triple_a.py",
]


def main():
    total_start = time.time()
    results = []
    for step in STEPS:
        results.append(run_step(step))

    elapsed = time.time() - total_start
    failed = [r for r in results if not r.get("ok")]
    critical_failed = [r for r in failed if r.get("critical")]

    print()
    print("[SUMMARY] Hourly refresh")
    print(f"[SUMMARY] completed in {elapsed:.1f}s")
    print(f"[SUMMARY] ok={sum(1 for r in results if r.get('ok'))} failed={len(failed)} critical_failed={len(critical_failed)}")
    for item in failed:
        print(f"[SUMMARY][FAIL] {item['step']} | critical={item.get('critical')} | returncode={item.get('returncode')} | elapsed={item.get('elapsed')}")

    if critical_failed and len(critical_failed) == len(results):
        raise SystemExit(1)

    print("[DONE] Hourly refresh finished")


if __name__ == "__main__":
    main()
