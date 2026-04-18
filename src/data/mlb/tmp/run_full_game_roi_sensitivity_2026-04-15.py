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
TMP_DIR = ROOT / "src" / "data" / "mlb" / "tmp"
OUT_CSV = TMP_DIR / "full_game_roi_sensitivity_2026-04-15.csv"
OUT_MD = TMP_DIR / "full_game_roi_sensitivity_2026-04-15.md"

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
}

SCENARIOS = [
    (
        "accuracy_cov_ref",
        {
            "NBA_MLB_FULL_GAME_THRESHOLD_OBJECTIVE": "accuracy_cov",
        },
    ),
    (
        "roi_base",
        {
            "NBA_MLB_FULL_GAME_THRESHOLD_OBJECTIVE": "roi",
            "NBA_MLB_FULL_GAME_ROI_MIN_EDGE": "0.00",
            "NBA_MLB_FULL_GAME_ROI_MIN_ACCURACY": "0.45",
            "NBA_MLB_FULL_GAME_ROI_MIN_PRICED_ROWS": "8",
            "NBA_MLB_FULL_GAME_ROI_SCORE_ROI_WEIGHT": "1.00",
            "NBA_MLB_FULL_GAME_ROI_SCORE_ACC_WEIGHT": "0.10",
            "NBA_MLB_FULL_GAME_ROI_SCORE_COV_WEIGHT": "0.04",
        },
    ),
    (
        "roi_edge_0005_basew",
        {
            "NBA_MLB_FULL_GAME_THRESHOLD_OBJECTIVE": "roi",
            "NBA_MLB_FULL_GAME_ROI_MIN_EDGE": "0.005",
            "NBA_MLB_FULL_GAME_ROI_MIN_ACCURACY": "0.45",
            "NBA_MLB_FULL_GAME_ROI_MIN_PRICED_ROWS": "8",
            "NBA_MLB_FULL_GAME_ROI_SCORE_ROI_WEIGHT": "1.00",
            "NBA_MLB_FULL_GAME_ROI_SCORE_ACC_WEIGHT": "0.10",
            "NBA_MLB_FULL_GAME_ROI_SCORE_COV_WEIGHT": "0.04",
        },
    ),
    (
        "roi_edge_0010_basew",
        {
            "NBA_MLB_FULL_GAME_THRESHOLD_OBJECTIVE": "roi",
            "NBA_MLB_FULL_GAME_ROI_MIN_EDGE": "0.010",
            "NBA_MLB_FULL_GAME_ROI_MIN_ACCURACY": "0.45",
            "NBA_MLB_FULL_GAME_ROI_MIN_PRICED_ROWS": "8",
            "NBA_MLB_FULL_GAME_ROI_SCORE_ROI_WEIGHT": "1.00",
            "NBA_MLB_FULL_GAME_ROI_SCORE_ACC_WEIGHT": "0.10",
            "NBA_MLB_FULL_GAME_ROI_SCORE_COV_WEIGHT": "0.04",
        },
    ),
    (
        "roi_edge_0015_basew",
        {
            "NBA_MLB_FULL_GAME_THRESHOLD_OBJECTIVE": "roi",
            "NBA_MLB_FULL_GAME_ROI_MIN_EDGE": "0.015",
            "NBA_MLB_FULL_GAME_ROI_MIN_ACCURACY": "0.45",
            "NBA_MLB_FULL_GAME_ROI_MIN_PRICED_ROWS": "8",
            "NBA_MLB_FULL_GAME_ROI_SCORE_ROI_WEIGHT": "1.00",
            "NBA_MLB_FULL_GAME_ROI_SCORE_ACC_WEIGHT": "0.10",
            "NBA_MLB_FULL_GAME_ROI_SCORE_COV_WEIGHT": "0.04",
        },
    ),
    (
        "roi_edge_0000_accup",
        {
            "NBA_MLB_FULL_GAME_THRESHOLD_OBJECTIVE": "roi",
            "NBA_MLB_FULL_GAME_ROI_MIN_EDGE": "0.00",
            "NBA_MLB_FULL_GAME_ROI_MIN_ACCURACY": "0.45",
            "NBA_MLB_FULL_GAME_ROI_MIN_PRICED_ROWS": "8",
            "NBA_MLB_FULL_GAME_ROI_SCORE_ROI_WEIGHT": "1.00",
            "NBA_MLB_FULL_GAME_ROI_SCORE_ACC_WEIGHT": "0.15",
            "NBA_MLB_FULL_GAME_ROI_SCORE_COV_WEIGHT": "0.05",
        },
    ),
    (
        "roi_edge_0000_roiup",
        {
            "NBA_MLB_FULL_GAME_THRESHOLD_OBJECTIVE": "roi",
            "NBA_MLB_FULL_GAME_ROI_MIN_EDGE": "0.00",
            "NBA_MLB_FULL_GAME_ROI_MIN_ACCURACY": "0.45",
            "NBA_MLB_FULL_GAME_ROI_MIN_PRICED_ROWS": "8",
            "NBA_MLB_FULL_GAME_ROI_SCORE_ROI_WEIGHT": "1.25",
            "NBA_MLB_FULL_GAME_ROI_SCORE_ACC_WEIGHT": "0.08",
            "NBA_MLB_FULL_GAME_ROI_SCORE_COV_WEIGHT": "0.03",
        },
    ),
    (
        "roi_edge_0005_roiup",
        {
            "NBA_MLB_FULL_GAME_THRESHOLD_OBJECTIVE": "roi",
            "NBA_MLB_FULL_GAME_ROI_MIN_EDGE": "0.005",
            "NBA_MLB_FULL_GAME_ROI_MIN_ACCURACY": "0.45",
            "NBA_MLB_FULL_GAME_ROI_MIN_PRICED_ROWS": "8",
            "NBA_MLB_FULL_GAME_ROI_SCORE_ROI_WEIGHT": "1.25",
            "NBA_MLB_FULL_GAME_ROI_SCORE_ACC_WEIGHT": "0.08",
            "NBA_MLB_FULL_GAME_ROI_SCORE_COV_WEIGHT": "0.03",
        },
    ),
]

CLEAR_KEYS = set(BASE_ENV) | {
    "NBA_MLB_FULL_GAME_THRESHOLD_OBJECTIVE",
    "NBA_MLB_FULL_GAME_ROI_MIN_EDGE",
    "NBA_MLB_FULL_GAME_ROI_MIN_ACCURACY",
    "NBA_MLB_FULL_GAME_ROI_MIN_PRICED_ROWS",
    "NBA_MLB_FULL_GAME_ROI_SCORE_ROI_WEIGHT",
    "NBA_MLB_FULL_GAME_ROI_SCORE_ACC_WEIGHT",
    "NBA_MLB_FULL_GAME_ROI_SCORE_COV_WEIGHT",
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


def run_scenarios() -> list[dict]:
    rows: list[dict] = []

    for idx, (name, overrides) in enumerate(SCENARIOS, start=1):
        env = os.environ.copy()
        for key in CLEAR_KEYS:
            env.pop(key, None)
        env.update(BASE_ENV)
        env.update(overrides)

        log_file = TMP_DIR / f"full_game_roi_sensitivity_{name}.log"
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
            "objective": str(overrides.get("NBA_MLB_FULL_GAME_THRESHOLD_OBJECTIVE", "accuracy_cov")),
            "roi_min_edge": _safe_float(overrides.get("NBA_MLB_FULL_GAME_ROI_MIN_EDGE"), default=0.0),
            "roi_score_roi_weight": _safe_float(overrides.get("NBA_MLB_FULL_GAME_ROI_SCORE_ROI_WEIGHT"), default=0.0),
            "roi_score_acc_weight": _safe_float(overrides.get("NBA_MLB_FULL_GAME_ROI_SCORE_ACC_WEIGHT"), default=0.0),
            "roi_score_cov_weight": _safe_float(overrides.get("NBA_MLB_FULL_GAME_ROI_SCORE_COV_WEIGHT"), default=0.0),
            "accuracy": float("nan"),
            "published_accuracy": float("nan"),
            "published_coverage": float("nan"),
            "published_roi_per_bet": float("nan"),
            "published_total_return_units": float("nan"),
            "published_priced_picks": -1,
            "roi_threshold_split_rate": float("nan"),
            "roi_threshold_splits": -1,
            "threshold_objective_mode": "run_failed",
            "log_file": str(log_file),
            "exit_code": int(proc.returncode),
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
                    "published_total_return_units": _safe_float(fg.get("published_total_return_units"), default=0.0),
                    "published_priced_picks": _safe_int(fg.get("published_priced_picks"), default=0),
                    "roi_threshold_split_rate": _safe_float(fg.get("roi_threshold_split_rate"), default=0.0),
                    "roi_threshold_splits": _safe_int(fg.get("roi_threshold_splits"), default=0),
                    "threshold_objective_mode": str(fg.get("threshold_objective_mode", "accuracy_cov")),
                }
            )
            print(
                f"[{idx}/{len(SCENARIOS)}] DONE {name} "
                f"acc={_format_float(row['accuracy'], 12)} "
                f"roi={_format_float(row['published_roi_per_bet'], 5)} "
                f"priced={row['published_priced_picks']} "
                f"roi_split_rate={_format_float(row['roi_threshold_split_rate'], 4)}"
            )
        else:
            print(f"[{idx}/{len(SCENARIOS)}] FAIL {name} exit={proc.returncode}")

        rows.append(row)

    return rows


def apply_decision_flags(rows: list[dict]) -> list[dict]:
    baseline = next((r for r in rows if r["scenario"] == "roi_base" and r["exit_code"] == 0), None)
    if baseline is None:
        baseline = next((r for r in rows if r["objective"] == "roi" and r["exit_code"] == 0), None)

    if baseline is None:
        for row in rows:
            row["no_regression"] = 0
            row["roi_improved"] = 0
            row["priced_non_drop"] = 0
            row["roi_split_non_drop"] = 0
            row["decision"] = "reject"
            row["decision_reason"] = "no_valid_baseline"
        return rows

    base_acc = _safe_float(baseline.get("accuracy"), default=float("nan"))
    base_roi = _safe_float(baseline.get("published_roi_per_bet"), default=float("nan"))
    base_priced = _safe_int(baseline.get("published_priced_picks"), default=0)
    base_split_rate = _safe_float(baseline.get("roi_threshold_split_rate"), default=float("nan"))

    for row in rows:
        if row["exit_code"] != 0 or row["objective"] != "roi":
            row["no_regression"] = 0
            row["roi_improved"] = 0
            row["priced_non_drop"] = 0
            row["roi_split_non_drop"] = 0
            row["decision"] = "reject"
            row["decision_reason"] = "not_roi_or_failed"
            continue

        no_regression = int(_safe_float(row.get("accuracy"), float("nan")) >= (base_acc - 1e-12))
        roi_improved = int(_safe_float(row.get("published_roi_per_bet"), float("nan")) >= (base_roi - 1e-12))
        priced_non_drop = int(_safe_int(row.get("published_priced_picks"), 0) >= base_priced)
        roi_split_non_drop = int(_safe_float(row.get("roi_threshold_split_rate"), float("nan")) >= (base_split_rate - 1e-12))

        row["no_regression"] = no_regression
        row["roi_improved"] = roi_improved
        row["priced_non_drop"] = priced_non_drop
        row["roi_split_non_drop"] = roi_split_non_drop

        if no_regression and roi_improved and priced_non_drop and roi_split_non_drop:
            row["decision"] = "promote"
            row["decision_reason"] = "all_guardrails_pass"
        else:
            reasons = []
            if not no_regression:
                reasons.append("accuracy_regression")
            if not roi_improved:
                reasons.append("roi_below_baseline")
            if not priced_non_drop:
                reasons.append("priced_picks_below_baseline")
            if not roi_split_non_drop:
                reasons.append("roi_split_rate_below_baseline")
            row["decision"] = "reject"
            row["decision_reason"] = ",".join(reasons)

    return rows


def write_outputs(rows: list[dict]) -> None:
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            -1e9 if not math.isfinite(_safe_float(r.get("published_roi_per_bet"), float("nan"))) else -_safe_float(r.get("published_roi_per_bet"), float("nan")),
            -1e9 if not math.isfinite(_safe_float(r.get("accuracy"), float("nan"))) else -_safe_float(r.get("accuracy"), float("nan")),
            -1e9 if not math.isfinite(_safe_float(r.get("roi_threshold_split_rate"), float("nan"))) else -_safe_float(r.get("roi_threshold_split_rate"), float("nan")),
        ),
    )

    fields = [
        "scenario",
        "objective",
        "roi_min_edge",
        "roi_score_roi_weight",
        "roi_score_acc_weight",
        "roi_score_cov_weight",
        "accuracy",
        "published_accuracy",
        "published_coverage",
        "published_roi_per_bet",
        "published_total_return_units",
        "published_priced_picks",
        "roi_threshold_splits",
        "roi_threshold_split_rate",
        "threshold_objective_mode",
        "no_regression",
        "roi_improved",
        "priced_non_drop",
        "roi_split_non_drop",
        "decision",
        "decision_reason",
        "log_file",
        "exit_code",
    ]

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_sorted)

    lines = []
    lines.append("# Full Game ROI Sensitivity Sweep (2026-04-15)")
    lines.append("")
    lines.append("| scenario | objective | edge | roi_w | acc_w | cov_w | accuracy | pub_roi | priced | roi_split_rate | decision |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows_sorted:
        lines.append(
            "| {scenario} | {objective} | {edge} | {roi_w} | {acc_w} | {cov_w} | {acc} | {roi} | {priced} | {split_rate} | {decision} |".format(
                scenario=row["scenario"],
                objective=row["objective"],
                edge=_format_float(row["roi_min_edge"], 4),
                roi_w=_format_float(row["roi_score_roi_weight"], 2),
                acc_w=_format_float(row["roi_score_acc_weight"], 2),
                cov_w=_format_float(row["roi_score_cov_weight"], 2),
                acc=_format_float(row["accuracy"], 6),
                roi=_format_float(row["published_roi_per_bet"], 5),
                priced=_safe_int(row["published_priced_picks"], -1),
                split_rate=_format_float(row["roi_threshold_split_rate"], 4),
                decision=row["decision"],
            )
        )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("SAVED", OUT_CSV)
    print("SAVED", OUT_MD)
    for row in rows_sorted:
        print(
            f"{row['scenario']}: acc={_format_float(row['accuracy'], 12)} "
            f"pub_roi={_format_float(row['published_roi_per_bet'], 6)} "
            f"priced={_safe_int(row['published_priced_picks'], -1)} "
            f"roi_split_rate={_format_float(row['roi_threshold_split_rate'], 6)} "
            f"decision={row['decision']}"
        )


def main() -> None:
    rows = run_scenarios()
    rows = apply_decision_flags(rows)
    write_outputs(rows)


if __name__ == "__main__":
    main()
