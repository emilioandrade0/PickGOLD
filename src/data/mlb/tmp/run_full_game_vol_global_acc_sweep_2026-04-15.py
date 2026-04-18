import csv
import json
import math
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
PYTHON_EXE = str((ROOT / ".venv" / "Scripts" / "python.exe").resolve())
WALKFORWARD_SCRIPT = str((ROOT / "src" / "sports" / "mlb" / "historical_predictions_mlb_walkforward.py").resolve())
SUMMARY_PATH = ROOT / "src" / "data" / "mlb" / "walkforward" / "walkforward_summary_mlb.json"
SPLITS_PATH = ROOT / "src" / "data" / "mlb" / "walkforward" / "full_game" / "walkforward_splits_summary.csv"
TMP_DIR = ROOT / "src" / "data" / "mlb" / "tmp"
OUT_CSV = TMP_DIR / "full_game_vol_global_acc_sweep_2026-04-15.csv"
OUT_MD = TMP_DIR / "full_game_vol_global_acc_sweep_2026-04-15.md"

BASE_ENV = {
    "NBA_MLB_MARKETS": "full_game",
    "NBA_MLB_INPUT_FILE": str((ROOT / "src" / "data" / "mlb" / "processed" / "snapshots" / "model_ready_features_mlb_HEAD.csv").resolve()),
    "NBA_MLB_MAX_TRAIN_DATE": "",
    "NBA_MLB_FULL_GAME_XGB_WEIGHT_GRID": "0.00,0.20,0.35,0.50,0.65,0.80,1.00",
    "NBA_MLB_FULL_GAME_BRIER_WEIGHT": "0.08",
    "NBA_MLB_FULL_GAME_PROB_SHIFT_ENABLED": "1",
    "NBA_MLB_FULL_GAME_PROB_SHIFT_MIN": "-0.02",
    "NBA_MLB_FULL_GAME_PROB_SHIFT_MAX": "0.02",
    "NBA_MLB_FULL_GAME_PROB_SHIFT_STEP": "0.01",
    "NBA_MLB_FULL_GAME_CALIBRATOR_MODE": "global_lr",
    "NBA_MLB_FULL_GAME_THR_MIN": "0.54",
    "NBA_MLB_FULL_GAME_THR_MAX": "0.66",
    "NBA_MLB_FULL_GAME_THR_STEP": "0.01",
    "NBA_MLB_FULL_GAME_EXTRA_FEATURES": "",
    "NBA_MLB_FULL_GAME_DROP_FEATURES": "",
    "NBA_MLB_META_GATE_ENABLED": "0",
    "NBA_MLB_FULL_GAME_THRESHOLD_OBJECTIVE": "roi",
    "NBA_MLB_FULL_GAME_ROI_MIN_EDGE": "0.00",
    "NBA_MLB_FULL_GAME_ROI_MIN_ACCURACY": "0.45",
    "NBA_MLB_FULL_GAME_ROI_MIN_PRICED_ROWS": "8",
    "NBA_MLB_FULL_GAME_ROI_SCORE_ROI_WEIGHT": "1.00",
    "NBA_MLB_FULL_GAME_ROI_SCORE_ACC_WEIGHT": "0.10",
    "NBA_MLB_FULL_GAME_ROI_SCORE_COV_WEIGHT": "0.04",
}

SCENARIOS = [
    (
        "baseline_off",
        {
            "NBA_MLB_FULL_GAME_VOL_NORM_ENABLED": "0",
            "NBA_MLB_FULL_GAME_VOL_DECISION_ENABLED": "0",
        },
    ),
    (
        "vol_norm_only",
        {
            "NBA_MLB_FULL_GAME_VOL_NORM_ENABLED": "1",
            "NBA_MLB_FULL_GAME_VOL_NORM_ALPHA": "0.18",
            "NBA_MLB_FULL_GAME_VOL_NORM_THRESHOLD_BONUS": "0.02",
            "NBA_MLB_FULL_GAME_VOL_NORM_HIGH_QUANTILE": "0.75",
            "NBA_MLB_FULL_GAME_VOL_NORM_MIN_ROWS": "160",
            "NBA_MLB_FULL_GAME_VOL_DECISION_ENABLED": "0",
        },
    ),
    (
        "beta_neg_012",
        {
            "NBA_MLB_FULL_GAME_VOL_NORM_ENABLED": "1",
            "NBA_MLB_FULL_GAME_VOL_NORM_ALPHA": "0.18",
            "NBA_MLB_FULL_GAME_VOL_NORM_THRESHOLD_BONUS": "0.02",
            "NBA_MLB_FULL_GAME_VOL_NORM_HIGH_QUANTILE": "0.75",
            "NBA_MLB_FULL_GAME_VOL_NORM_MIN_ROWS": "160",
            "NBA_MLB_FULL_GAME_VOL_DECISION_ENABLED": "1",
            "NBA_MLB_FULL_GAME_VOL_DECISION_MAX_SHIFT": "0.04",
            "NBA_MLB_FULL_GAME_VOL_DECISION_BETA_GRID": "-0.12",
        },
    ),
    (
        "beta_neg_008",
        {
            "NBA_MLB_FULL_GAME_VOL_NORM_ENABLED": "1",
            "NBA_MLB_FULL_GAME_VOL_NORM_ALPHA": "0.18",
            "NBA_MLB_FULL_GAME_VOL_NORM_THRESHOLD_BONUS": "0.02",
            "NBA_MLB_FULL_GAME_VOL_NORM_HIGH_QUANTILE": "0.75",
            "NBA_MLB_FULL_GAME_VOL_NORM_MIN_ROWS": "160",
            "NBA_MLB_FULL_GAME_VOL_DECISION_ENABLED": "1",
            "NBA_MLB_FULL_GAME_VOL_DECISION_MAX_SHIFT": "0.04",
            "NBA_MLB_FULL_GAME_VOL_DECISION_BETA_GRID": "-0.08",
        },
    ),
    (
        "beta_neg_005",
        {
            "NBA_MLB_FULL_GAME_VOL_NORM_ENABLED": "1",
            "NBA_MLB_FULL_GAME_VOL_NORM_ALPHA": "0.18",
            "NBA_MLB_FULL_GAME_VOL_NORM_THRESHOLD_BONUS": "0.02",
            "NBA_MLB_FULL_GAME_VOL_NORM_HIGH_QUANTILE": "0.75",
            "NBA_MLB_FULL_GAME_VOL_NORM_MIN_ROWS": "160",
            "NBA_MLB_FULL_GAME_VOL_DECISION_ENABLED": "1",
            "NBA_MLB_FULL_GAME_VOL_DECISION_MAX_SHIFT": "0.04",
            "NBA_MLB_FULL_GAME_VOL_DECISION_BETA_GRID": "-0.05",
        },
    ),
    (
        "beta_neg_003",
        {
            "NBA_MLB_FULL_GAME_VOL_NORM_ENABLED": "1",
            "NBA_MLB_FULL_GAME_VOL_NORM_ALPHA": "0.18",
            "NBA_MLB_FULL_GAME_VOL_NORM_THRESHOLD_BONUS": "0.02",
            "NBA_MLB_FULL_GAME_VOL_NORM_HIGH_QUANTILE": "0.75",
            "NBA_MLB_FULL_GAME_VOL_NORM_MIN_ROWS": "160",
            "NBA_MLB_FULL_GAME_VOL_DECISION_ENABLED": "1",
            "NBA_MLB_FULL_GAME_VOL_DECISION_MAX_SHIFT": "0.04",
            "NBA_MLB_FULL_GAME_VOL_DECISION_BETA_GRID": "-0.03",
        },
    ),
    (
        "beta_pos_003",
        {
            "NBA_MLB_FULL_GAME_VOL_NORM_ENABLED": "1",
            "NBA_MLB_FULL_GAME_VOL_NORM_ALPHA": "0.18",
            "NBA_MLB_FULL_GAME_VOL_NORM_THRESHOLD_BONUS": "0.02",
            "NBA_MLB_FULL_GAME_VOL_NORM_HIGH_QUANTILE": "0.75",
            "NBA_MLB_FULL_GAME_VOL_NORM_MIN_ROWS": "160",
            "NBA_MLB_FULL_GAME_VOL_DECISION_ENABLED": "1",
            "NBA_MLB_FULL_GAME_VOL_DECISION_MAX_SHIFT": "0.04",
            "NBA_MLB_FULL_GAME_VOL_DECISION_BETA_GRID": "0.03",
        },
    ),
    (
        "beta_pos_005",
        {
            "NBA_MLB_FULL_GAME_VOL_NORM_ENABLED": "1",
            "NBA_MLB_FULL_GAME_VOL_NORM_ALPHA": "0.18",
            "NBA_MLB_FULL_GAME_VOL_NORM_THRESHOLD_BONUS": "0.02",
            "NBA_MLB_FULL_GAME_VOL_NORM_HIGH_QUANTILE": "0.75",
            "NBA_MLB_FULL_GAME_VOL_NORM_MIN_ROWS": "160",
            "NBA_MLB_FULL_GAME_VOL_DECISION_ENABLED": "1",
            "NBA_MLB_FULL_GAME_VOL_DECISION_MAX_SHIFT": "0.04",
            "NBA_MLB_FULL_GAME_VOL_DECISION_BETA_GRID": "0.05",
        },
    ),
    (
        "beta_pos_008",
        {
            "NBA_MLB_FULL_GAME_VOL_NORM_ENABLED": "1",
            "NBA_MLB_FULL_GAME_VOL_NORM_ALPHA": "0.18",
            "NBA_MLB_FULL_GAME_VOL_NORM_THRESHOLD_BONUS": "0.02",
            "NBA_MLB_FULL_GAME_VOL_NORM_HIGH_QUANTILE": "0.75",
            "NBA_MLB_FULL_GAME_VOL_NORM_MIN_ROWS": "160",
            "NBA_MLB_FULL_GAME_VOL_DECISION_ENABLED": "1",
            "NBA_MLB_FULL_GAME_VOL_DECISION_MAX_SHIFT": "0.04",
            "NBA_MLB_FULL_GAME_VOL_DECISION_BETA_GRID": "0.08",
        },
    ),
    (
        "beta_pos_012",
        {
            "NBA_MLB_FULL_GAME_VOL_NORM_ENABLED": "1",
            "NBA_MLB_FULL_GAME_VOL_NORM_ALPHA": "0.18",
            "NBA_MLB_FULL_GAME_VOL_NORM_THRESHOLD_BONUS": "0.02",
            "NBA_MLB_FULL_GAME_VOL_NORM_HIGH_QUANTILE": "0.75",
            "NBA_MLB_FULL_GAME_VOL_NORM_MIN_ROWS": "160",
            "NBA_MLB_FULL_GAME_VOL_DECISION_ENABLED": "1",
            "NBA_MLB_FULL_GAME_VOL_DECISION_MAX_SHIFT": "0.04",
            "NBA_MLB_FULL_GAME_VOL_DECISION_BETA_GRID": "0.12",
        },
    ),
]

CLEAR_KEYS = set(BASE_ENV) | {
    "NBA_MLB_FULL_GAME_VOL_NORM_ENABLED",
    "NBA_MLB_FULL_GAME_VOL_NORM_ALPHA",
    "NBA_MLB_FULL_GAME_VOL_NORM_THRESHOLD_BONUS",
    "NBA_MLB_FULL_GAME_VOL_NORM_HIGH_QUANTILE",
    "NBA_MLB_FULL_GAME_VOL_NORM_MIN_ROWS",
    "NBA_MLB_FULL_GAME_VOL_DECISION_ENABLED",
    "NBA_MLB_FULL_GAME_VOL_DECISION_BETA",
    "NBA_MLB_FULL_GAME_VOL_DECISION_BETA_GRID",
    "NBA_MLB_FULL_GAME_VOL_DECISION_MAX_SHIFT",
    "NBA_MLB_FULL_GAME_VOL_DECISION_CENTER",
    "NBA_MLB_FULL_GAME_VOL_DECISION_BETA_PENALTY",
}


def _safe_float(value, default=float("nan")) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value, default=0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _format_float(value: float, digits: int = 6) -> str:
    if value is None or not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _read_split_metrics() -> dict:
    if not SPLITS_PATH.exists():
        return {
            "split_count": 0,
            "avg_vol_norm_mean_test": float("nan"),
            "avg_vol_norm_high_rate_test": float("nan"),
            "avg_vol_decision_beta": float("nan"),
            "vol_decision_enabled_rate": float("nan"),
        }

    rows = []
    with open(SPLITS_PATH, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    if not rows:
        return {
            "split_count": 0,
            "avg_vol_norm_mean_test": float("nan"),
            "avg_vol_norm_high_rate_test": float("nan"),
            "avg_vol_decision_beta": float("nan"),
            "vol_decision_enabled_rate": float("nan"),
        }

    def _avg(key: str) -> float:
        vals = []
        for row in rows:
            val = _safe_float(row.get(key), default=float("nan"))
            if math.isfinite(val):
                vals.append(val)
        if not vals:
            return float("nan")
        return float(sum(vals) / len(vals))

    return {
        "split_count": int(len(rows)),
        "avg_vol_norm_mean_test": _avg("vol_norm_mean_test"),
        "avg_vol_norm_high_rate_test": _avg("vol_norm_high_rate_test"),
        "avg_vol_decision_beta": _avg("vol_decision_beta"),
        "vol_decision_enabled_rate": _avg("vol_decision_enabled"),
    }


def run_scenarios() -> list[dict]:
    rows = []

    for idx, (name, overrides) in enumerate(SCENARIOS, start=1):
        env = os.environ.copy()
        for key in CLEAR_KEYS:
            env.pop(key, None)
        env.update(BASE_ENV)
        env.update(overrides)

        log_file = TMP_DIR / f"full_game_vol_global_acc_{name}.log"
        print(f"[{idx}/{len(SCENARIOS)}] START {name}")

        with open(log_file, "w", encoding="utf-8") as handle:
            proc = subprocess.run(
                [PYTHON_EXE, WALKFORWARD_SCRIPT],
                cwd=str(ROOT),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )

        row = {
            "scenario": name,
            "vol_norm_enabled": _safe_int(overrides.get("NBA_MLB_FULL_GAME_VOL_NORM_ENABLED"), default=0),
            "vol_decision_enabled": _safe_int(overrides.get("NBA_MLB_FULL_GAME_VOL_DECISION_ENABLED"), default=0),
            "vol_decision_beta_grid": str(overrides.get("NBA_MLB_FULL_GAME_VOL_DECISION_BETA_GRID", "")),
            "accuracy": float("nan"),
            "published_accuracy": float("nan"),
            "published_coverage": float("nan"),
            "published_roi_per_bet": float("nan"),
            "published_priced_picks": -1,
            "split_count": 0,
            "avg_vol_norm_mean_test": float("nan"),
            "avg_vol_norm_high_rate_test": float("nan"),
            "avg_vol_decision_beta": float("nan"),
            "vol_decision_enabled_rate": float("nan"),
            "exit_code": int(proc.returncode),
            "log_file": str(log_file),
        }

        if proc.returncode == 0 and SUMMARY_PATH.exists():
            summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
            fg = summary.get("full_game", {})
            row.update(
                {
                    "accuracy": _safe_float(fg.get("accuracy")),
                    "published_accuracy": _safe_float(fg.get("published_accuracy")),
                    "published_coverage": _safe_float(fg.get("published_coverage")),
                    "published_roi_per_bet": _safe_float(fg.get("published_roi_per_bet"), default=0.0),
                    "published_priced_picks": _safe_int(fg.get("published_priced_picks"), default=0),
                }
            )
            row.update(_read_split_metrics())
            print(
                f"[{idx}/{len(SCENARIOS)}] DONE {name} "
                f"acc={_format_float(row['accuracy'], 12)} "
                f"pub_acc={_format_float(row['published_accuracy'], 6)} "
                f"beta_mean={_format_float(row['avg_vol_decision_beta'], 4)}"
            )
        else:
            print(f"[{idx}/{len(SCENARIOS)}] FAIL {name} exit={proc.returncode}")

        rows.append(row)

    return rows


def apply_decision_flags(rows: list[dict]) -> list[dict]:
    baseline = next((r for r in rows if r.get("scenario") == "baseline_off" and r.get("exit_code") == 0), None)
    if baseline is None:
        for row in rows:
            row["global_accuracy_improved"] = 0
            row["decision"] = "reject"
            row["decision_reason"] = "no_valid_baseline"
        return rows

    base_acc = _safe_float(baseline.get("accuracy"), default=float("nan"))

    for row in rows:
        if row.get("exit_code") != 0:
            row["global_accuracy_improved"] = 0
            row["decision"] = "reject"
            row["decision_reason"] = "run_failed"
            continue

        acc = _safe_float(row.get("accuracy"), default=float("nan"))
        improved = int(math.isfinite(base_acc) and math.isfinite(acc) and (acc > (base_acc + 1e-9)))
        row["global_accuracy_improved"] = improved

        if row.get("scenario") == "baseline_off":
            row["decision"] = "baseline"
            row["decision_reason"] = "reference"
        elif improved == 1:
            row["decision"] = "promote_global_accuracy"
            row["decision_reason"] = "global_accuracy_up"
        else:
            row["decision"] = "reject"
            row["decision_reason"] = "global_accuracy_not_up"

    return rows


def write_outputs(rows: list[dict]) -> None:
    fieldnames = [
        "scenario",
        "vol_norm_enabled",
        "vol_decision_enabled",
        "vol_decision_beta_grid",
        "accuracy",
        "published_accuracy",
        "published_coverage",
        "published_roi_per_bet",
        "published_priced_picks",
        "split_count",
        "avg_vol_norm_mean_test",
        "avg_vol_norm_high_rate_test",
        "avg_vol_decision_beta",
        "vol_decision_enabled_rate",
        "global_accuracy_improved",
        "decision",
        "decision_reason",
        "exit_code",
        "log_file",
    ]

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    lines = []
    lines.append("# Full Game Volatility Global Accuracy Sweep (2026-04-15)")
    lines.append("")
    lines.append("| scenario | acc | pub_acc | pub_cov | roi/bet | priced | beta_mean | decision | reason |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|")

    for row in rows:
        lines.append(
            "| {scenario} | {acc} | {pub_acc} | {pub_cov} | {roi} | {priced} | {beta_mean} | {decision} | {reason} |".format(
                scenario=row.get("scenario", ""),
                acc=_format_float(_safe_float(row.get("accuracy"), default=float("nan")), 6),
                pub_acc=_format_float(_safe_float(row.get("published_accuracy"), default=float("nan")), 6),
                pub_cov=_format_float(_safe_float(row.get("published_coverage"), default=float("nan")), 6),
                roi=_format_float(_safe_float(row.get("published_roi_per_bet"), default=float("nan")), 6),
                priced=_safe_int(row.get("published_priced_picks"), default=0),
                beta_mean=_format_float(_safe_float(row.get("avg_vol_decision_beta"), default=float("nan")), 4),
                decision=str(row.get("decision", "")),
                reason=str(row.get("decision_reason", "")),
            )
        )

    lines.append("")
    lines.append("## Objective")
    lines.append("- maximize full_game global accuracy (not published metrics)")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = run_scenarios()
    rows = apply_decision_flags(rows)
    rows = sorted(rows, key=lambda r: (_safe_float(r.get("accuracy"), default=-999.0), -_safe_int(r.get("exit_code"), default=1)), reverse=True)
    write_outputs(rows)
    print(f"\nSaved CSV: {OUT_CSV}")
    print(f"Saved MD : {OUT_MD}")


if __name__ == "__main__":
    main()
