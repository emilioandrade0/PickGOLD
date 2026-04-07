from pathlib import Path
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT


def run_step(script_name: str) -> None:
    script_path = SCRIPT_DIR / script_name
    print("=" * 72)
    print(f"RUNNING: {script_path.name}")
    print("=" * 72)
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(BASE_DIR), check=False)
    if result.returncode != 0:
        raise SystemExit(f"Step failed: {script_name} (exit={result.returncode})")


def main() -> None:
    run_step("data_ingest_ncaa_baseball.py")
    run_step("feature_engineering_ncaa_baseball.py")
    run_step("train_models_ncaa_baseball.py")
    run_step("historical_predictions_ncaa_baseball.py")
    run_step("evaluate_baseline.py")
    run_step("predict_today_ncaa_baseball.py")
    print("\n[OK] NCAA Baseball full pipeline completed.")


if __name__ == "__main__":
    main()
