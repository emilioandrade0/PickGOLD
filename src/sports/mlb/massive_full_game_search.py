from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WALKFORWARD_SCRIPT = PROJECT_ROOT / "src" / "sports" / "mlb" / "historical_predictions_mlb_walkforward.py"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "src" / "data" / "mlb" / "walkforward" / "walkforward_summary_mlb.json"
DEFAULT_DETAIL_PATH = PROJECT_ROOT / "src" / "data" / "mlb" / "walkforward" / "full_game" / "walkforward_predictions_detail.csv"
DEFAULT_MODEL_READY_PATH = PROJECT_ROOT / "src" / "data" / "mlb" / "processed" / "model_ready_features_mlb.csv"

DEFAULT_RESULTS_JSONL = PROJECT_ROOT / "src" / "data" / "mlb" / "walkforward" / "massive_full_game_results.jsonl"
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "src" / "data" / "mlb" / "walkforward" / "massive_full_game_leaderboard.csv"
DEFAULT_BEST_JSON = PROJECT_ROOT / "src" / "data" / "mlb" / "walkforward" / "massive_full_game_best.json"
DEFAULT_LOG_DIR = PROJECT_ROOT / "src" / "data" / "mlb" / "walkforward" / "massive_logs"
DEFAULT_BEST_SNAPSHOT_DIR = PROJECT_ROOT / "src" / "data" / "mlb" / "walkforward" / "massive_best_snapshot"

DEFAULT_ENV_FIXED = {
    "PYTHONIOENCODING": "utf-8",
    "NBA_MLB_MARKETS": "full_game",
    "NBA_MLB_DISABLE_PREV_SEASON_BLEND": "1",
    "NBA_MLB_DISABLE_UMPIRE_EXPANDED": "1",
}

ABLATON_OPTIONS = ["everything", "baseball_only", "market_basic", "market_full"]
CALIBRATOR_OPTIONS = ["auto", "global_lr", "regime_aware"]
OBJECTIVE_OPTIONS = ["accuracy_cov", "roi"]
XGB_GRID_OPTIONS = [
    "0.00,0.20,0.35,0.50,0.65,0.80,1.00",
    "0.00,0.15,0.30,0.45,0.60,0.75,0.90",
    "0.10,0.25,0.40,0.55,0.70,0.85,1.00",
    "0.00,0.25,0.50,0.75,1.00",
]
THR_MIN_OPTIONS = [0.50, 0.52, 0.54, 0.56, 0.58]
THR_STEP_OPTIONS = [0.01, 0.02]

META_THRESHOLD_MIN_OPTIONS = [0.48, 0.50, 0.52, 0.54, 0.56]
META_THRESHOLD_MAX_OPTIONS = [0.70, 0.74, 0.78, 0.82, 0.86]
META_THRESHOLD_STEP_OPTIONS = [0.01, 0.02]

PROB_SHIFT_MIN_OPTIONS = [-0.06, -0.04, -0.02, -0.01, 0.00]
PROB_SHIFT_MAX_OPTIONS = [0.00, 0.01, 0.02, 0.04, 0.06]
PROB_SHIFT_STEP_OPTIONS = [0.01, 0.02]

DROP_CANDIDATES = [
    "prev_season_data_available",
    "diff_prev_win_pct",
    "diff_prev_run_diff_pg",
    "diff_prev_runs_scored_pg",
    "diff_prev_runs_allowed_pg",
    "umpire_zone_delta",
    "umpire_sample_log",
    "market_micro_missing",
    "market_line_velocity",
]

EXTRA_CANDIDATES = [
    "home_vs_away_matchup_win_pct_pre",
    "home_vs_away_matchup_run_diff_pre",
    "home_vs_away_in_season_record_pre",
    "home_vs_away_matchup_win_pct_L10_pre",
    "home_vs_away_matchup_run_diff_L10_pre",
    "home_vs_away_matchup_games_L10_pre",
    "home_vs_away_matchup_win_pct_L10_shrunk_pre",
    "home_vs_away_matchup_run_diff_L10_shrunk_pre",
    "series_game_number",
    "series_run_diff_so_far",
    "did_home_win_previous_game_in_series",
    "diff_bullpen_ip_yesterday",
    "diff_bullpen_ip_3d",
    "diff_bullpen_outs_yesterday",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_existing_hashes(results_jsonl: Path) -> set[str]:
    if not results_jsonl.exists():
        return set()
    hashes: set[str] = set()
    with results_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if str(payload.get("status", "")).strip().lower() == "dry_run":
                continue
            config_hash = str(payload.get("config_hash", "")).strip()
            if config_hash:
                hashes.add(config_hash)
    return hashes


def _load_feature_columns(model_ready_path: Path) -> set[str]:
    if not model_ready_path.exists():
        return set()
    with model_ready_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return set()
    return {str(col).strip() for col in header if str(col).strip()}


def _pick_sample(rng: random.Random, options: Sequence[float]) -> float:
    return float(rng.choice(list(options)))


@dataclass
class TrialConfig:
    env: Dict[str, str]

    def canonical_items(self) -> List[tuple[str, str]]:
        return sorted((str(k), str(v)) for k, v in self.env.items())

    def config_hash(self) -> str:
        canonical = "|".join(f"{k}={v}" for k, v in self.canonical_items())
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def _weighted_choice(rng: random.Random, weighted_items: Sequence[tuple[str, float]]) -> str:
    items = list(weighted_items)
    total = sum(max(0.0, float(w)) for _, w in items)
    if total <= 0:
        return str(items[0][0])
    roll = rng.random() * total
    acc = 0.0
    for value, weight in items:
        acc += max(0.0, float(weight))
        if roll <= acc:
            return str(value)
    return str(items[-1][0])


def _sample_trial_config(rng: random.Random, available_cols: set[str], profile: str) -> TrialConfig:
    profile = str(profile).strip().lower()
    if profile not in {"stable", "explore"}:
        profile = "stable"

    env: Dict[str, str] = {}

    if profile == "stable":
        env["NBA_MLB_FULL_GAME_ABLATION"] = _weighted_choice(
            rng,
            [
                ("baseball_only", 0.55),
                ("everything", 0.20),
                ("market_full", 0.20),
                ("market_basic", 0.05),
            ],
        )
        env["NBA_MLB_FULL_GAME_BRIER_WEIGHT"] = f"{_pick_sample(rng, [0.06, 0.08, 0.10]):.2f}"
    else:
        env["NBA_MLB_FULL_GAME_ABLATION"] = str(rng.choice(ABLATON_OPTIONS))
        env["NBA_MLB_FULL_GAME_BRIER_WEIGHT"] = f"{_pick_sample(rng, [0.04, 0.06, 0.08, 0.10, 0.12]):.2f}"

    env["NBA_MLB_FULL_GAME_CALIBRATOR_MODE"] = str(rng.choice(CALIBRATOR_OPTIONS))
    env["NBA_MLB_FULL_GAME_THRESHOLD_OBJECTIVE"] = str(rng.choice(OBJECTIVE_OPTIONS))
    env["NBA_MLB_FULL_GAME_XGB_WEIGHT_GRID"] = str(rng.choice(XGB_GRID_OPTIONS))

    if profile == "stable":
        thr_min = _pick_sample(rng, [0.54, 0.55, 0.56, 0.57, 0.58, 0.59])
    else:
        thr_min = _pick_sample(rng, THR_MIN_OPTIONS)
    thr_step = _pick_sample(rng, THR_STEP_OPTIONS)
    thr_max_candidates = [v for v in [0.64, 0.66, 0.68, 0.70, 0.72, 0.74] if v >= thr_min + 0.06]
    thr_max = float(rng.choice(thr_max_candidates)) if thr_max_candidates else float(thr_min + 0.08)
    env["NBA_MLB_FULL_GAME_THR_MIN"] = f"{thr_min:.2f}"
    env["NBA_MLB_FULL_GAME_THR_MAX"] = f"{thr_max:.2f}"
    env["NBA_MLB_FULL_GAME_THR_STEP"] = f"{thr_step:.2f}"

    if profile == "stable":
        meta_gate_enabled = int(rng.random() < 0.90)
    else:
        meta_gate_enabled = int(rng.random() < 0.75)
    env["NBA_MLB_META_GATE_ENABLED"] = str(meta_gate_enabled)
    if meta_gate_enabled:
        if profile == "stable":
            meta_min = _pick_sample(rng, [0.50, 0.52, 0.54, 0.56])
        else:
            meta_min = _pick_sample(rng, META_THRESHOLD_MIN_OPTIONS)
        meta_max_candidates = [v for v in META_THRESHOLD_MAX_OPTIONS if v >= meta_min + 0.08]
        meta_max = float(rng.choice(meta_max_candidates)) if meta_max_candidates else float(meta_min + 0.10)
        if profile == "stable":
            env["NBA_MLB_META_GATE_MODEL_C"] = f"{_pick_sample(rng, [0.8, 1.0, 1.2]):.2f}"
            env["NBA_MLB_META_GATE_MIN_CALIB_ROWS"] = str(int(rng.choice([160, 180, 200])))
            env["NBA_MLB_META_GATE_MIN_BASE_ROWS"] = str(int(rng.choice([20, 25, 30])))
        else:
            env["NBA_MLB_META_GATE_MODEL_C"] = f"{_pick_sample(rng, [0.6, 0.8, 1.0, 1.2, 1.4]):.2f}"
            env["NBA_MLB_META_GATE_MIN_CALIB_ROWS"] = str(int(rng.choice([140, 160, 180, 200, 220])))
            env["NBA_MLB_META_GATE_MIN_BASE_ROWS"] = str(int(rng.choice([20, 25, 30, 35, 40])))
        env["NBA_MLB_META_GATE_THRESHOLD_MIN"] = f"{meta_min:.2f}"
        env["NBA_MLB_META_GATE_THRESHOLD_MAX"] = f"{meta_max:.2f}"
        env["NBA_MLB_META_GATE_THRESHOLD_STEP"] = f"{_pick_sample(rng, META_THRESHOLD_STEP_OPTIONS):.2f}"
        if profile == "stable":
            env["NBA_MLB_META_GATE_MIN_KEEP_ROWS"] = str(int(rng.choice([10, 12, 16])))
            env["NBA_MLB_META_GATE_COVERAGE_BONUS"] = f"{_pick_sample(rng, [0.03, 0.04, 0.05]):.2f}"
            env["NBA_MLB_META_GATE_RETENTION_TARGET"] = f"{_pick_sample(rng, [0.40, 0.45, 0.50]):.2f}"
            env["NBA_MLB_META_GATE_RETENTION_PENALTY"] = f"{_pick_sample(rng, [0.10, 0.12, 0.14, 0.16]):.2f}"
            env["NBA_MLB_META_GATE_MIN_ACC_GAIN"] = f"{_pick_sample(rng, [0.005, 0.01, 0.015]):.3f}"
            env["NBA_MLB_META_GATE_MIN_COVERAGE_RETENTION"] = f"{_pick_sample(rng, [0.35, 0.40, 0.45]):.2f}"
        else:
            env["NBA_MLB_META_GATE_MIN_KEEP_ROWS"] = str(int(rng.choice([8, 10, 12, 16, 20])))
            env["NBA_MLB_META_GATE_COVERAGE_BONUS"] = f"{_pick_sample(rng, [0.02, 0.03, 0.04, 0.05, 0.06]):.2f}"
            env["NBA_MLB_META_GATE_RETENTION_TARGET"] = f"{_pick_sample(rng, [0.35, 0.40, 0.45, 0.50, 0.55]):.2f}"
            env["NBA_MLB_META_GATE_RETENTION_PENALTY"] = f"{_pick_sample(rng, [0.08, 0.10, 0.12, 0.14, 0.16, 0.18]):.2f}"
            env["NBA_MLB_META_GATE_MIN_ACC_GAIN"] = f"{_pick_sample(rng, [0.00, 0.005, 0.01, 0.015, 0.02]):.3f}"
            env["NBA_MLB_META_GATE_MIN_COVERAGE_RETENTION"] = f"{_pick_sample(rng, [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]):.2f}"

    prob_shift_enabled = int(rng.random() < (0.65 if profile == "stable" else 0.80))
    env["NBA_MLB_FULL_GAME_PROB_SHIFT_ENABLED"] = str(prob_shift_enabled)
    if prob_shift_enabled:
        if profile == "stable":
            ps_min = _pick_sample(rng, [-0.03, -0.02, -0.01, 0.00])
            ps_max_candidates = [v for v in [0.00, 0.01, 0.02, 0.03] if v >= ps_min]
        else:
            ps_min = _pick_sample(rng, PROB_SHIFT_MIN_OPTIONS)
            ps_max_candidates = [v for v in PROB_SHIFT_MAX_OPTIONS if v >= ps_min]
        ps_max = float(rng.choice(ps_max_candidates)) if ps_max_candidates else float(ps_min)
        env["NBA_MLB_FULL_GAME_PROB_SHIFT_MIN"] = f"{ps_min:.2f}"
        env["NBA_MLB_FULL_GAME_PROB_SHIFT_MAX"] = f"{ps_max:.2f}"
        env["NBA_MLB_FULL_GAME_PROB_SHIFT_STEP"] = f"{_pick_sample(rng, PROB_SHIFT_STEP_OPTIONS):.2f}"

    vol_enabled = int(rng.random() < (0.25 if profile == "stable" else 0.50))
    env["NBA_MLB_FULL_GAME_VOL_NORM_ENABLED"] = str(vol_enabled)
    if vol_enabled:
        env["NBA_MLB_FULL_GAME_VOL_NORM_ALPHA"] = f"{_pick_sample(rng, [0.12, 0.16, 0.18, 0.22, 0.26]):.2f}"
        env["NBA_MLB_FULL_GAME_VOL_NORM_THRESHOLD_BONUS"] = f"{_pick_sample(rng, [0.00, 0.01, 0.02, 0.03, 0.04]):.2f}"
        env["NBA_MLB_FULL_GAME_VOL_NORM_HIGH_QUANTILE"] = f"{_pick_sample(rng, [0.70, 0.75, 0.80, 0.85]):.2f}"
        env["NBA_MLB_FULL_GAME_VOL_NORM_MIN_ROWS"] = str(int(rng.choice([120, 140, 160, 180, 200])))
        env["NBA_MLB_FULL_GAME_VOL_DECISION_ENABLED"] = str(int(rng.random() < 0.65))
        env["NBA_MLB_FULL_GAME_VOL_DECISION_CENTER"] = f"{_pick_sample(rng, [0.45, 0.50, 0.55]):.2f}"
        env["NBA_MLB_FULL_GAME_VOL_DECISION_MAX_SHIFT"] = f"{_pick_sample(rng, [0.03, 0.05, 0.06, 0.08, 0.10]):.2f}"
        env["NBA_MLB_FULL_GAME_VOL_DECISION_BETA_PENALTY"] = f"{_pick_sample(rng, [0.000, 0.001, 0.002, 0.003, 0.005]):.3f}"

    reliability_enabled = int(rng.random() < (0.25 if profile == "stable" else 0.50))
    env["NBA_MLB_FULL_GAME_RELIABILITY_ENABLED"] = str(reliability_enabled)
    if reliability_enabled:
        env["NBA_MLB_FULL_GAME_RELIABILITY_SHRINK_ALPHA"] = f"{_pick_sample(rng, [0.08, 0.12, 0.14, 0.18, 0.22]):.2f}"
        env["NBA_MLB_FULL_GAME_RELIABILITY_SIDE_SHIFT"] = f"{_pick_sample(rng, [0.02, 0.03, 0.04, 0.05]):.3f}"
        env["NBA_MLB_FULL_GAME_RELIABILITY_CONFLICT_SHIFT"] = f"{_pick_sample(rng, [0.01, 0.015, 0.02, 0.025]):.3f}"

    extra_pool = [c for c in EXTRA_CANDIDATES if c in available_cols]
    if extra_pool and rng.random() < (0.25 if profile == "stable" else 0.60):
        k = rng.randint(1, min(4, len(extra_pool)))
        extra = sorted(rng.sample(extra_pool, k=k))
        env["NBA_MLB_FULL_GAME_EXTRA_FEATURES"] = ",".join(extra)

    drop_pool = [c for c in DROP_CANDIDATES if c in available_cols]
    if drop_pool and rng.random() < (0.10 if profile == "stable" else 0.35):
        k = rng.randint(1, min(3, len(drop_pool)))
        drop = sorted(rng.sample(drop_pool, k=k))
        env["NBA_MLB_FULL_GAME_DROP_FEATURES"] = ",".join(drop)

    return TrialConfig(env=env)


def _load_summary_metrics(summary_path: Path) -> dict:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    full_game = payload.get("full_game") or {}
    rows = int(full_game.get("rows") or 0)
    coverage = float(full_game.get("coverage") or 0.0)
    published_coverage = float(full_game.get("published_coverage") or 0.0)
    published_rows = int(round(rows * published_coverage)) if rows > 0 else 0
    return {
        "rows": rows,
        "accuracy": float(full_game.get("accuracy") or 0.0),
        "brier": float(full_game.get("brier") or 0.0),
        "logloss": float(full_game.get("logloss") or 0.0),
        "roc_auc": float(full_game.get("roc_auc") or 0.0),
        "coverage": coverage,
        "published_accuracy": float(full_game.get("published_accuracy") or 0.0),
        "published_coverage": published_coverage,
        "published_rows": published_rows,
    }


def _run_trial(
    python_exe: str,
    walkforward_script: Path,
    env_trial: Dict[str, str],
    log_path: Path,
    timeout_sec: Optional[int],
) -> tuple[int, float]:
    _ensure_parent(log_path)
    env = dict(os.environ)
    for key in list(env.keys()):
        if key.startswith("NBA_MLB_"):
            env.pop(key, None)
    env.update(DEFAULT_ENV_FIXED)
    env.update(env_trial)

    start = time.time()
    with log_path.open("w", encoding="utf-8", newline="\n") as log_fh:
        log_fh.write(f"[START_UTC] {_now_iso()}\n")
        log_fh.write("[ENV]\n")
        for k in sorted(env_trial.keys()):
            log_fh.write(f"{k}={env_trial[k]}\n")
        log_fh.write("\n")
        log_fh.flush()
        try:
            proc = subprocess.run(
                [python_exe, str(walkforward_script)],
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
                check=False,
            )
            return int(proc.returncode), float(time.time() - start)
        except subprocess.TimeoutExpired:
            log_fh.write("\n[TIMEOUT]\n")
            log_fh.flush()
            return 124, float(time.time() - start)


def _append_jsonl(path: Path, payload: dict) -> None:
    _ensure_parent(path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_leaderboard_csv(path: Path, rows: List[dict]) -> None:
    _ensure_parent(path)
    sorted_rows = sorted(
        rows,
        key=lambda r: (
            float(r.get("accuracy") or -1.0),
            float(r.get("published_accuracy") or -1.0),
            -float(r.get("brier") or 10.0),
        ),
        reverse=True,
    )
    fieldnames = [
        "trial",
        "status",
        "accuracy",
        "coverage",
        "published_accuracy",
        "published_coverage",
        "brier",
        "logloss",
        "roc_auc",
        "rows",
        "published_rows",
        "duration_sec",
        "config_hash",
        "log_file",
        "started_at_utc",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted_rows)


def _save_best_snapshot(best_snapshot_dir: Path, summary_path: Path, detail_path: Path, result: dict) -> None:
    best_snapshot_dir.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        shutil.copy2(summary_path, best_snapshot_dir / "walkforward_summary_mlb.json")
    if detail_path.exists():
        shutil.copy2(detail_path, best_snapshot_dir / "walkforward_predictions_detail.csv")
    (best_snapshot_dir / "best_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _pick_python(python_exe_arg: str) -> str:
    if python_exe_arg:
        return python_exe_arg
    venv_py = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Busqueda masiva reanudable de configuraciones full_game (MLB).",
    )
    parser.add_argument("--trials", type=int, default=2000, help="Numero total de intentos a ejecutar.")
    parser.add_argument("--seed", type=int, default=20260419, help="Semilla RNG para reproducibilidad.")
    parser.add_argument("--python-exe", type=str, default="", help="Ruta al python a usar.")
    parser.add_argument("--timeout-minutes", type=int, default=0, help="Timeout por trial (0 = sin timeout).")
    parser.add_argument("--max-hours", type=float, default=0.0, help="Corte por tiempo total de busqueda (0 = sin corte).")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Pausa entre trials.")
    parser.add_argument(
        "--profile",
        type=str,
        default="stable",
        choices=["stable", "explore"],
        help="stable=busqueda cerca de baseline, explore=busqueda mas agresiva.",
    )
    parser.add_argument("--results-jsonl", type=str, default=str(DEFAULT_RESULTS_JSONL))
    parser.add_argument("--results-csv", type=str, default=str(DEFAULT_RESULTS_CSV))
    parser.add_argument("--best-json", type=str, default=str(DEFAULT_BEST_JSON))
    parser.add_argument("--log-dir", type=str, default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--walkforward-script", type=str, default=str(DEFAULT_WALKFORWARD_SCRIPT))
    parser.add_argument("--summary-path", type=str, default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--detail-path", type=str, default=str(DEFAULT_DETAIL_PATH))
    parser.add_argument("--model-ready-path", type=str, default=str(DEFAULT_MODEL_READY_PATH))
    parser.add_argument("--best-snapshot-dir", type=str, default=str(DEFAULT_BEST_SNAPSHOT_DIR))
    parser.add_argument("--resume", action="store_true", help="Si existe jsonl previo, evita repetir hashes.")
    parser.add_argument("--dry-run", action="store_true", help="No ejecuta walk-forward; solo genera configs.")
    parser.add_argument("--print-every", type=int, default=1, help="Frecuencia de logs de progreso.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rng = random.Random(int(args.seed))

    python_exe = _pick_python(str(args.python_exe).strip())
    walkforward_script = Path(args.walkforward_script)
    summary_path = Path(args.summary_path)
    detail_path = Path(args.detail_path)
    model_ready_path = Path(args.model_ready_path)
    results_jsonl = Path(args.results_jsonl)
    results_csv = Path(args.results_csv)
    best_json = Path(args.best_json)
    best_snapshot_dir = Path(args.best_snapshot_dir)
    log_dir = Path(args.log_dir)
    timeout_sec = int(args.timeout_minutes * 60) if int(args.timeout_minutes) > 0 else None
    max_total_sec = float(args.max_hours) * 3600.0 if float(args.max_hours) > 0 else None
    print_every = max(1, int(args.print_every))

    available_cols = _load_feature_columns(model_ready_path)
    existing_hashes = _read_existing_hashes(results_jsonl) if args.resume else set()
    all_results: List[dict] = []

    if results_jsonl.exists():
        with results_jsonl.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                all_results.append(payload)

    best_record = None
    for rec in all_results:
        if rec.get("status") != "ok":
            continue
        if best_record is None or float(rec.get("accuracy") or -1.0) > float(best_record.get("accuracy") or -1.0):
            best_record = rec

    start_wall = time.time()
    executed = 0
    attempted = 0

    print("=" * 84)
    print("MASSIVE FULL_GAME SEARCH | MLB")
    print("=" * 84)
    print(f"project_root      : {PROJECT_ROOT}")
    print(f"python_exe        : {python_exe}")
    print(f"walkforward       : {walkforward_script}")
    print(f"trials_requested  : {args.trials}")
    print(f"profile           : {args.profile}")
    print(f"resume            : {bool(args.resume)}")
    print(f"known_hashes      : {len(existing_hashes)}")
    print(f"model_ready_cols  : {len(available_cols)}")
    print(f"results_jsonl     : {results_jsonl}")
    print(f"leaderboard_csv   : {results_csv}")
    print(f"best_json         : {best_json}")
    print(f"log_dir           : {log_dir}")
    print("=" * 84)

    while executed < int(args.trials):
        if max_total_sec is not None and (time.time() - start_wall) >= max_total_sec:
            print("STOP: max-hours alcanzado.")
            break

        attempted += 1
        trial_num = len(all_results) + 1
        cfg = _sample_trial_config(rng, available_cols, profile=str(args.profile))
        cfg_hash = cfg.config_hash()
        if cfg_hash in existing_hashes:
            continue
        existing_hashes.add(cfg_hash)

        started_at = _now_iso()
        log_path = log_dir / f"trial_{trial_num:06d}_{cfg_hash[:10]}.log"

        if args.dry_run:
            result = {
                "trial": trial_num,
                "status": "dry_run",
                "config_hash": cfg_hash,
                "duration_sec": 0.0,
                "started_at_utc": started_at,
                "log_file": str(log_path),
                "env": cfg.env,
            }
            _append_jsonl(results_jsonl, result)
            all_results.append(result)
            executed += 1
            if executed % print_every == 0:
                print(f"[{executed}/{args.trials}] dry_run trial={trial_num} hash={cfg_hash[:10]}")
            continue

        rc, duration_sec = _run_trial(
            python_exe=python_exe,
            walkforward_script=walkforward_script,
            env_trial=cfg.env,
            log_path=log_path,
            timeout_sec=timeout_sec,
        )

        result = {
            "trial": trial_num,
            "config_hash": cfg_hash,
            "return_code": rc,
            "duration_sec": round(float(duration_sec), 3),
            "started_at_utc": started_at,
            "log_file": str(log_path),
            "env": cfg.env,
        }

        if rc == 0 and summary_path.exists():
            try:
                metrics = _load_summary_metrics(summary_path)
                result.update(metrics)
                result["status"] = "ok"
            except Exception as exc:
                result["status"] = "summary_error"
                result["error"] = f"{exc.__class__.__name__}: {exc}"
        else:
            result["status"] = "run_error"

        _append_jsonl(results_jsonl, result)
        all_results.append(result)
        executed += 1

        if result.get("status") == "ok":
            if best_record is None or float(result.get("accuracy") or -1.0) > float(best_record.get("accuracy") or -1.0):
                best_record = result
                _ensure_parent(best_json)
                best_json.write_text(json.dumps(best_record, ensure_ascii=False, indent=2), encoding="utf-8")
                _save_best_snapshot(best_snapshot_dir, summary_path, detail_path, best_record)

        if executed % print_every == 0:
            status = str(result.get("status"))
            acc = result.get("accuracy")
            acc_txt = f"{float(acc) * 100.0:.2f}%" if isinstance(acc, (float, int)) else "N/A"
            best_txt = "N/A"
            if best_record is not None and isinstance(best_record.get("accuracy"), (float, int)):
                best_txt = f"{float(best_record['accuracy']) * 100.0:.2f}%"
            print(
                f"[{executed}/{args.trials}] trial={trial_num} status={status} "
                f"acc={acc_txt} best={best_txt} rc={rc} dur={result['duration_sec']}s"
            )

        if float(args.sleep_seconds) > 0:
            time.sleep(float(args.sleep_seconds))

    _write_leaderboard_csv(results_csv, all_results)

    elapsed = time.time() - start_wall
    best_acc = float(best_record["accuracy"]) if best_record and isinstance(best_record.get("accuracy"), (float, int)) else None
    print("=" * 84)
    print("SEARCH FINISHED")
    print("=" * 84)
    print(f"attempted_configs : {attempted}")
    print(f"executed_trials   : {executed}")
    print(f"elapsed_seconds   : {elapsed:.1f}")
    if best_acc is not None:
        print(f"best_accuracy     : {best_acc:.6f} ({best_acc * 100.0:.2f}%)")
        print(f"best_trial        : {best_record.get('trial')}")
        print(f"best_log          : {best_record.get('log_file')}")
    else:
        print("best_accuracy     : N/A")
    print(f"results_jsonl     : {results_jsonl}")
    print(f"leaderboard_csv   : {results_csv}")
    print(f"best_json         : {best_json}")
    print(f"best_snapshot_dir : {best_snapshot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
