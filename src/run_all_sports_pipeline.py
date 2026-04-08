"""Run the full pipeline for every sport under src/sports/."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
SPORTS_DIR = SRC_DIR / "sports"

# Ordered list of (sport_label, pipeline_script_path)
SPORT_PIPELINES: list[tuple[str, Path]] = [
    ("NBA", SPORTS_DIR / "nba" / "run_pipeline_nba.py"),
    ("MLB", SPORTS_DIR / "mlb" / "run_pipeline_mlb.py"),
    ("KBO", SPORTS_DIR / "kbo" / "run_pipeline_kbo.py"),
    ("NHL", SPORTS_DIR / "nhl" / "run_pipeline_nhl.py"),
    ("LaLiga", SPORTS_DIR / "laliga" / "run_pipeline_laliga.py"),
    ("EuroLeague", SPORTS_DIR / "euroleague" / "run_pipeline_euroleague.py"),
    ("Tennis", SPORTS_DIR / "tennis" / "run_pipeline_tennis.py"),
    ("Liga MX", SPORTS_DIR / "ligamx" / "run_pipeline_ligamx.py"),
    ("NCAA Baseball", SPORTS_DIR / "ncaa baseball" / "run_pipeline_ncaa_baseball.py"),
]


def run_sport_pipeline(label: str, script_path: Path) -> dict:
    print()
    print("*" * 72)
    print(f"* SPORT PIPELINE: {label}")
    print("*" * 72)
    started = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(SRC_DIR.parent),
        check=False,
    )
    elapsed = time.time() - started
    ok = result.returncode == 0
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {label} ({elapsed:.1f}s)")
    return {"sport": label, "ok": ok, "elapsed": round(elapsed, 1), "returncode": result.returncode}


def main() -> None:
    total_start = time.time()
    results = []

    for label, script_path in SPORT_PIPELINES:
        if not script_path.exists():
            print(f"[SKIP] {label}: pipeline script not found at {script_path}")
            results.append({"sport": label, "ok": False, "elapsed": 0, "returncode": None, "skipped": True})
            continue
        results.append(run_sport_pipeline(label, script_path))

    elapsed = time.time() - total_start
    failed = [r for r in results if not r.get("ok")]

    print()
    print("=" * 72)
    print("[SUMMARY] All sports pipeline")
    print(f"[SUMMARY] completed in {elapsed:.1f}s")
    print(f"[SUMMARY] ok={sum(1 for r in results if r.get('ok'))} failed={len(failed)}")
    for item in failed:
        skipped = "(skipped)" if item.get("skipped") else f"returncode={item.get('returncode')}"
        print(f"[SUMMARY][FAIL] {item['sport']} | {skipped} | elapsed={item.get('elapsed')}")

    if failed:
        raise SystemExit(1)

    print("[DONE] All sports pipelines completed successfully.")


if __name__ == "__main__":
    main()
