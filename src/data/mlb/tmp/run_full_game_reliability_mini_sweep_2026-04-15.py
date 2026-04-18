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
OUT_CSV = TMP_DIR / "full_game_reliability_mini_sweep_2026-04-15.csv"
OUT_MD = TMP_DIR / "full_game_reliability_mini_sweep_2026-04-15.md"

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
    "NBA_MLB_FULL_GAME_VOL_NORM_ENABLED": "1",
    "NBA_MLB_FULL_GAME_VOL_NORM_ALPHA": "0.18",
    "NBA_MLB_FULL_GAME_VOL_NORM_THRESHOLD_BONUS": "0.02",
    "NBA_MLB_FULL_GAME_VOL_NORM_HIGH_QUANTILE": "0.75",
    "NBA_MLB_FULL_GAME_VOL_NORM_MIN_ROWS": "160",
    "NBA_MLB_FULL_GAME_VOL_DECISION_ENABLED": "1",
    "NBA_MLB_FULL_GAME_VOL_DECISION_MAX_SHIFT": "0.04",
    "NBA_MLB_FULL_GAME_VOL_DECISION_BETA_GRID": "-0.08",
}

SCENARIOS = [
    ("rel_off", 0, 0.00, 0.0000, 0.0000),
    ("rel_a002_s000_c000", 1, 0.02, 0.0000, 0.0000),
    ("rel_a002_s001_c0005", 1, 0.02, 0.0010, 0.0005),
    ("rel_a004_s000_c000", 1, 0.04, 0.0000, 0.0000),
    ("rel_a004_s002_c001", 1, 0.04, 0.0020, 0.0010),
    ("rel_a006_s002_c001", 1, 0.06, 0.0020, 0.0010),
    ("rel_a008_s003_c0015", 1, 0.08, 0.0030, 0.0015),
]

CLEAR_KEYS = set(BASE_ENV) | {
    "NBA_MLB_FULL_GAME_RELIABILITY_ENABLED",
    "NBA_MLB_FULL_GAME_RELIABILITY_SHRINK_ALPHA",
    "NBA_MLB_FULL_GAME_RELIABILITY_SIDE_SHIFT",
    "NBA_MLB_FULL_GAME_RELIABILITY_CONFLICT_SHIFT",
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
    default = {
        "split_count": 0,
        "avg_reliability_mean_test": float("nan"),
        "avg_reliability_abs_shift_mean_test": float("nan"),
        "avg_reliability_conflict_rate_test": float("nan"),
    }
    if not SPLITS_PATH.exists():
        return default

    rows = []
    with open(SPLITS_PATH, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        return default

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
        "avg_reliability_mean_test": _avg("reliability_mean_test"),
        "avg_reliability_abs_shift_mean_test": _avg("reliability_abs_shift_mean_test"),
        "avg_reliability_conflict_rate_test": _avg("reliability_conflict_rate_test"),
    }


def run_scenarios() -> list[dict]:
    rows = []
    total = len(SCENARIOS)

    for idx, (name, enabled, alpha, side_shift, conflict_shift) in enumerate(SCENARIOS, start=1):
        env = os.environ.copy()
        for key in CLEAR_KEYS:
            env.pop(key, None)
        env.update(BASE_ENV)
        env.update(
            {
                "NBA_MLB_FULL_GAME_RELIABILITY_ENABLED": str(enabled),
                "NBA_MLB_FULL_GAME_RELIABILITY_SHRINK_ALPHA": f"{alpha:.4f}",
                "NBA_MLB_FULL_GAME_RELIABILITY_SIDE_SHIFT": f"{side_shift:.4f}",
                "NBA_MLB_FULL_GAME_RELIABILITY_CONFLICT_SHIFT": f"{conflict_shift:.4f}",
            }
        )

        log_file = TMP_DIR / f"full_game_reliability_mini_sweep_{name}.log"
        print(f"[{idx}/{total}] START {name}")

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
            "reliability_enabled": enabled,
            "reliability_shrink_alpha": alpha,
            "reliability_side_shift": side_shift,
            "reliability_conflict_shift": conflict_shift,
            "accuracy": float("nan"),
            "published_accuracy": float("nan"),
            "published_coverage": float("nan"),
            "published_roi_per_bet": float("nan"),
            "published_priced_picks": -1,
            "reliability_enabled_split_rate": float("nan"),
            "avg_reliability_mean_test_summary": float("nan"),
            "avg_reliability_abs_shift_mean_test_summary": float("nan"),
            "avg_reliability_conflict_rate_test_summary": float("nan"),
            "split_count": 0,
            "avg_reliability_mean_test": float("nan"),
            "avg_reliability_abs_shift_mean_test": float("nan"),
            "avg_reliability_conflict_rate_test": float("nan"),
            "threshold_objective_mode": "run_failed",
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
                    "threshold_objective_mode": str(fg.get("threshold_objective_mode", "accuracy_cov")),
                    "reliability_enabled_split_rate": _safe_float(fg.get("reliability_enabled_split_rate"), default=0.0),
                    "avg_reliability_mean_test_summary": _safe_float(fg.get("avg_reliability_mean_test"), default=float("nan")),
                    "avg_reliability_abs_shift_mean_test_summary": _safe_float(
                        fg.get("avg_reliability_abs_shift_mean_test"), default=float("nan")
                    ),
                    "avg_reliability_conflict_rate_test_summary": _safe_float(
                        fg.get("avg_reliability_conflict_rate_test"), default=float("nan")
                    ),
                }
            )
            row.update(_read_split_metrics())
            print(
                f"[{idx}/{total}] DONE {name} "
                f"acc={_format_float(row['accuracy'], 12)} "
                f"pub_acc={_format_float(row['published_accuracy'], 6)} "
                f"roi={_format_float(row['published_roi_per_bet'], 5)} "
                f"priced={row['published_priced_picks']}"
            )
        else:
            print(f"[{idx}/{total}] FAIL {name} exit={proc.returncode}")

        rows.append(row)

    return rows


def apply_decision(rows: list[dict]) -> list[dict]:
    baseline = next((r for r in rows if r.get("scenario") == "rel_off" and r.get("exit_code") == 0), None)
    if baseline is None:
        for row in rows:
            row["delta_accuracy"] = float("nan")
            row["delta_published_accuracy"] = float("nan")
            row["delta_published_roi_per_bet"] = float("nan")
            row["delta_published_priced_picks"] = 0
            row["no_accuracy_regression"] = 0
            row["global_accuracy_improved"] = 0
            row["decision"] = "reject"
            row["decision_reason"] = "no_valid_baseline"
        return rows

    base_acc = _safe_float(baseline.get("accuracy"), default=float("nan"))
    base_pub_acc = _safe_float(baseline.get("published_accuracy"), default=float("nan"))
    base_roi = _safe_float(baseline.get("published_roi_per_bet"), default=float("nan"))
    base_priced = _safe_int(baseline.get("published_priced_picks"), default=0)

    for row in rows:
        acc = _safe_float(row.get("accuracy"), default=float("nan"))
        pub_acc = _safe_float(row.get("published_accuracy"), default=float("nan"))
        roi = _safe_float(row.get("published_roi_per_bet"), default=float("nan"))
        priced = _safe_int(row.get("published_priced_picks"), default=0)

        if row.get("exit_code") != 0:
            row["delta_accuracy"] = float("nan")
            row["delta_published_accuracy"] = float("nan")
            row["delta_published_roi_per_bet"] = float("nan")
            row["delta_published_priced_picks"] = 0
            row["no_accuracy_regression"] = 0
            row["global_accuracy_improved"] = 0
            row["decision"] = "reject"
            row["decision_reason"] = "run_failed"
            continue

        row["delta_accuracy"] = acc - base_acc if math.isfinite(acc) and math.isfinite(base_acc) else float("nan")
        row["delta_published_accuracy"] = (
            pub_acc - base_pub_acc if math.isfinite(pub_acc) and math.isfinite(base_pub_acc) else float("nan")
        )
        row["delta_published_roi_per_bet"] = roi - base_roi if math.isfinite(roi) and math.isfinite(base_roi) else float("nan")
        row["delta_published_priced_picks"] = priced - base_priced

        no_acc_reg = int(math.isfinite(base_acc) and math.isfinite(acc) and (acc >= (base_acc - 1e-9)))
        global_acc_improved = int(math.isfinite(base_acc) and math.isfinite(acc) and (acc > (base_acc + 1e-9)))
        row["no_accuracy_regression"] = no_acc_reg
        row["global_accuracy_improved"] = global_acc_improved

        if row.get("scenario") == "rel_off":
            row["decision"] = "baseline"
            row["decision_reason"] = "reference"
        elif global_acc_improved == 1:
            row["decision"] = "promote_global_accuracy"
            row["decision_reason"] = "global_accuracy_up"
        elif no_acc_reg == 1:
            row["decision"] = "candidate"
            row["decision_reason"] = "no_accuracy_drop"
        else:
            row["decision"] = "reject"
            row["decision_reason"] = "accuracy_drop"

    return rows


def write_outputs(rows: list[dict]) -> None:
    valid_rows = [r for r in rows if r.get("exit_code") == 0 and math.isfinite(_safe_float(r.get("accuracy"), float("nan")))]
    best = None
    if valid_rows:
        best = max(valid_rows, key=lambda r: _safe_float(r.get("accuracy"), float("-inf")))

    fieldnames = [
        "scenario",
        "reliability_enabled",
        "reliability_shrink_alpha",
        "reliability_side_shift",
        "reliability_conflict_shift",
        "accuracy",
        "delta_accuracy",
        "published_accuracy",
        "delta_published_accuracy",
        "published_coverage",
        "published_roi_per_bet",
        "delta_published_roi_per_bet",
        "published_priced_picks",
        "delta_published_priced_picks",
        "reliability_enabled_split_rate",
        "avg_reliability_mean_test_summary",
        "avg_reliability_abs_shift_mean_test_summary",
        "avg_reliability_conflict_rate_test_summary",
        "split_count",
        "avg_reliability_mean_test",
        "avg_reliability_abs_shift_mean_test",
        "avg_reliability_conflict_rate_test",
        "threshold_objective_mode",
        "no_accuracy_regression",
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
    lines.append("# Full Game Reliability Mini Sweep (2026-04-15)")
    lines.append("")
    lines.append("| scenario | alpha | side | conflict | acc | d_acc | pub_acc | d_pub_acc | roi/bet | d_roi | priced | d_priced | decision |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for row in rows:
        lines.append(
            "| {scenario} | {alpha} | {side} | {conflict} | {acc} | {d_acc} | {pub_acc} | {d_pub_acc} | {roi} | {d_roi} | {priced} | {d_priced} | {decision} |".format(
                scenario=row.get("scenario", ""),
                alpha=_format_float(_safe_float(row.get("reliability_shrink_alpha"), float("nan")), 4),
                side=_format_float(_safe_float(row.get("reliability_side_shift"), float("nan")), 4),
                conflict=_format_float(_safe_float(row.get("reliability_conflict_shift"), float("nan")), 4),
                acc=_format_float(_safe_float(row.get("accuracy"), float("nan")), 6),
                d_acc=_format_float(_safe_float(row.get("delta_accuracy"), float("nan")), 6),
                pub_acc=_format_float(_safe_float(row.get("published_accuracy"), float("nan")), 6),
                d_pub_acc=_format_float(_safe_float(row.get("delta_published_accuracy"), float("nan")), 6),
                roi=_format_float(_safe_float(row.get("published_roi_per_bet"), float("nan")), 6),
                d_roi=_format_float(_safe_float(row.get("delta_published_roi_per_bet"), float("nan")), 6),
                priced=_safe_int(row.get("published_priced_picks"), default=0),
                d_priced=_safe_int(row.get("delta_published_priced_picks"), default=0),
                decision=str(row.get("decision", "")),
            )
        )

    lines.append("")
    if best is not None:
        lines.append(
            "Best by global accuracy: {scenario} with acc={acc} (delta={delta}).".format(
                scenario=best.get("scenario", ""),
                acc=_format_float(_safe_float(best.get("accuracy"), float("nan")), 12),
                delta=_format_float(_safe_float(best.get("delta_accuracy"), float("nan")), 12),
            )
        )

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")



def main() -> None:
    rows = run_scenarios()
    rows = apply_decision(rows)
    write_outputs(rows)
    print(f"\nSaved CSV: {OUT_CSV}")
    print(f"Saved MD : {OUT_MD}")


if __name__ == "__main__":
    main()
