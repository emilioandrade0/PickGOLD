from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "src" / "data" / "mlb" / "walkforward" / "walkforward_summary_mlb.json"

DEFAULT_ROLLBACK_FILES = [
    "src/sports/mlb/feature_engineering_mlb_core.py",
    "src/sports/mlb/train_models_mlb.py",
    "src/sports/mlb/historical_predictions_mlb_walkforward.py",
    "src/data/mlb/walkforward/walkforward_summary_mlb.json",
    "src/data/mlb/walkforward/full_game/walkforward_metrics.json",
    "src/data/mlb/walkforward/full_game/walkforward_predictions_detail.csv",
    "src/data/mlb/walkforward/full_game/walkforward_splits_summary.csv",
]

DEFAULT_EXPERIMENT_COMMAND = (
    "$env:PYTHONIOENCODING='utf-8'; "
    "$env:NBA_MLB_MARKETS='full_game'; "
    "$env:NBA_MLB_DISABLE_PREV_SEASON_BLEND='1'; "
    "$env:NBA_MLB_DISABLE_UMPIRE_EXPANDED='1'; "
    "Remove-Item Env:NBA_MLB_META_GATE* -ErrorAction SilentlyContinue; "
    "Remove-Item Env:NBA_MLB_FULL_GAME_PROB_SHIFT_MIN -ErrorAction SilentlyContinue; "
    "Remove-Item Env:NBA_MLB_FULL_GAME_PROB_SHIFT_MAX -ErrorAction SilentlyContinue; "
    "Remove-Item Env:NBA_MLB_FULL_GAME_PROB_SHIFT_STEP -ErrorAction SilentlyContinue; "
    "& '.venv\\Scripts\\python.exe' 'src\\sports\\mlb\\historical_predictions_mlb_walkforward.py'"
)


def _resolve_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _snapshot_files(paths: list[str]) -> dict[str, bytes | None]:
    snapshot: dict[str, bytes | None] = {}
    for rel_path in paths:
        abs_path = _resolve_path(rel_path)
        if abs_path.exists():
            snapshot[rel_path] = abs_path.read_bytes()
        else:
            snapshot[rel_path] = None
    return snapshot


def _restore_snapshot(snapshot: dict[str, bytes | None]) -> None:
    for rel_path, content in snapshot.items():
        abs_path = _resolve_path(rel_path)
        if content is None:
            if abs_path.exists():
                abs_path.unlink()
            continue
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(content)


def _run_command(command: str) -> int:
    if os.name == "nt":
        cmd = ["powershell", "-NoProfile", "-Command", command]
    else:
        cmd = ["bash", "-lc", command]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return int(proc.returncode)


def _read_full_game_accuracy(summary_path: Path) -> float:
    if not summary_path.exists():
        raise FileNotFoundError(f"No existe summary: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Summary invalido (no es objeto JSON): {summary_path}")
    full_game = payload.get("full_game")
    if not isinstance(full_game, dict):
        raise ValueError(f"Summary sin bloque full_game: {summary_path}")
    accuracy = float(full_game.get("accuracy"))
    return accuracy


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ejecuta experimento MLB full_game con regla de rollback automatico por accuracy."
    )
    parser.add_argument(
        "--baseline-accuracy",
        type=float,
        default=0.5646,
        help="Baseline minimo de accuracy (escala 0-1). Default: 0.5646 (56.46%%).",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.0,
        help="Mejora minima absoluta requerida sobre baseline (escala 0-1).",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=str(DEFAULT_SUMMARY_PATH),
        help="Ruta al summary walk-forward con bloque full_game.accuracy.",
    )
    parser.add_argument(
        "--experiment-command",
        type=str,
        default=DEFAULT_EXPERIMENT_COMMAND,
        help="Comando del experimento (se ejecuta en PowerShell/Bash).",
    )
    parser.add_argument(
        "--rollback-file",
        action="append",
        default=[],
        help="Archivo/ruta a restaurar si no mejora. Repetible.",
    )
    parser.add_argument(
        "--no-rollback-on-failure",
        action="store_true",
        help="Si el experimento falla, no restaura snapshot.",
    )
    args = parser.parse_args()

    rollback_files = list(DEFAULT_ROLLBACK_FILES)
    if args.rollback_file:
        for item in args.rollback_file:
            if item not in rollback_files:
                rollback_files.append(item)

    summary_path = _resolve_path(args.summary_path)
    baseline = float(args.baseline_accuracy)
    target = baseline + float(args.min_improvement)

    print("=" * 70)
    print("MLB FULL_GAME EXPERIMENTO GUARDADO")
    print("=" * 70)
    print(f"Proyecto           : {PROJECT_ROOT}")
    print(f"Summary            : {summary_path}")
    print(f"Baseline           : {baseline:.6f} ({baseline * 100.0:.2f}%)")
    print(f"Target             : {target:.6f} ({target * 100.0:.2f}%)")
    print(f"Rollback files     : {len(rollback_files)}")
    print("-" * 70)

    snapshot = _snapshot_files(rollback_files)

    print("Ejecutando experimento...")
    rc = _run_command(args.experiment_command)
    print(f"Exit code experimento: {rc}")

    if rc != 0:
        if args.no_rollback_on_failure:
            print("Experimento fallo y rollback por fallo esta desactivado.")
            return rc
        print("Experimento fallo. Restaurando snapshot...")
        _restore_snapshot(snapshot)
        return rc

    try:
        current_acc = _read_full_game_accuracy(summary_path)
    except Exception as exc:
        print(f"No se pudo leer accuracy final: {exc}")
        print("Restaurando snapshot por seguridad...")
        _restore_snapshot(snapshot)
        return 2

    delta = current_acc - baseline
    delta_pp = delta * 100.0
    print("-" * 70)
    print(f"Accuracy final      : {current_acc:.6f} ({current_acc * 100.0:.2f}%)")
    print(f"Delta vs baseline   : {delta:+.6f} ({delta_pp:+.2f} pp)")

    if current_acc >= target:
        print("Resultado: KEEP (mejora o empate con objetivo).")
        return 0

    print("Resultado: REVERT (no supera objetivo). Restaurando snapshot...")
    _restore_snapshot(snapshot)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
