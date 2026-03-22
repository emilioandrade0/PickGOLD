from pathlib import Path
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parent


def run_step(script_name: str) -> None:
    script_path = BASE_DIR / script_name
    print("=" * 72)
    print(f"RUNNING: {script_path.name}")
    print("=" * 72)
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(BASE_DIR), check=False)
    if result.returncode != 0:
        raise SystemExit(f"Step failed: {script_name} (exit={result.returncode})")


def main() -> None:
    run_step("feature_engineering_liga_mx_v3.py")
    run_step("train_compare_liga_mx_baseline_v3.py")
    run_step("train_models_liga_mx_selective.py")
    print("\n[OK] Liga MX v3 full pipeline completed.")


if __name__ == "__main__":
    main()
