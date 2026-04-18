from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


DATE_FILE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.json$")


@dataclass
class DayScore:
    date: str
    games: int
    hits: int

    @property
    def accuracy(self) -> float:
        if self.games <= 0:
            return 0.0
        return float(self.hits) / float(self.games)


def _load_target_accuracy(default_value: float, baseline_file: Path) -> float:
    if not baseline_file.exists():
        return default_value

    try:
        payload = json.loads(baseline_file.read_text(encoding="utf-8"))
        value = float(((payload.get("metrics") or {}).get("accuracy")) or default_value)
        if np.isnan(value) or value <= 0 or value > 1:
            return default_value
        return value
    except Exception:
        return default_value


def _extract_actual_winner(raw_day: pd.DataFrame) -> pd.Series:
    return np.where(
        raw_day["home_runs_total"] > raw_day["away_runs_total"],
        raw_day["home_team"],
        np.where(
            raw_day["away_runs_total"] > raw_day["home_runs_total"],
            raw_day["away_team"],
            np.nan,
        ),
    )


def _score_one_day(pred_file: Path, raw_df: pd.DataFrame) -> Optional[DayScore]:
    match = DATE_FILE_RE.match(pred_file.name)
    if not match:
        return None
    date_str = match.group(1)

    try:
        pred = pd.read_json(pred_file)
    except Exception:
        return None

    if pred.empty or "full_game_pick" not in pred.columns:
        return None

    pred = pred.copy()
    if "game_id" not in pred.columns or "home_team" not in pred.columns or "away_team" not in pred.columns:
        return None

    pred["game_id"] = pred["game_id"].astype(str)

    raw_day = raw_df.loc[raw_df["date"] == date_str].copy()
    if raw_day.empty:
        return None

    for col in ["home_runs_total", "away_runs_total"]:
        raw_day[col] = pd.to_numeric(raw_day.get(col), errors="coerce")

    raw_day = raw_day.dropna(subset=["home_runs_total", "away_runs_total"]).copy()
    if raw_day.empty:
        return None

    raw_day["actual_winner"] = _extract_actual_winner(raw_day)

    merge_cols = ["game_id", "home_team", "away_team", "actual_winner"]
    merged = pred.merge(raw_day[merge_cols], on=["game_id", "home_team", "away_team"], how="left")

    valid_mask = merged["full_game_pick"].notna() & merged["actual_winner"].notna()
    games = int(valid_mask.sum())
    if games <= 0:
        return None

    hits = int((merged.loc[valid_mask, "full_game_pick"] == merged.loc[valid_mask, "actual_winner"]).sum())
    return DayScore(date=date_str, games=games, hits=hits)


def _decide_status(
    rolling_accuracy: float,
    rolling_games: int,
    yesterday_accuracy: Optional[float],
    target_accuracy: float,
    min_games: int,
) -> tuple[str, str, str, dict]:
    green_min = max(target_accuracy - 0.005, 0.57)
    yellow_min = 0.53
    yesterday_floor_for_green = 0.50

    thresholds = {
        "green_min_rolling_accuracy": green_min,
        "yellow_min_rolling_accuracy": yellow_min,
        "min_games": min_games,
        "yesterday_floor_for_green": yesterday_floor_for_green,
    }

    if rolling_games < min_games:
        return "YELLOW", "baseline578", "low_sample", thresholds

    if rolling_accuracy >= green_min and (
        yesterday_accuracy is None or yesterday_accuracy >= yesterday_floor_for_green
    ):
        return "GREEN", "live", "rolling_strong", thresholds

    if rolling_accuracy >= yellow_min:
        return "YELLOW", "baseline578", "rolling_mixed", thresholds

    return "RED", "baseline578", "rolling_weak", thresholds


def build_report(
    base_dir: Path,
    window_days: int,
    default_target: float,
    min_games: int,
) -> dict:
    predictions_dir = base_dir / "src" / "data" / "mlb" / "predictions"
    raw_file = base_dir / "src" / "data" / "mlb" / "raw" / "mlb_advanced_history.csv"
    baseline_file = base_dir / "src" / "sports" / "mlb" / "documentacion_mejoras" / "baseline_oficial_full_game.json"

    target_accuracy = _load_target_accuracy(default_target, baseline_file)

    report: dict = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_accuracy": target_accuracy,
        "window_days": int(window_days),
        "status": "YELLOW",
        "recommended_mode": "baseline578",
        "reason": "no_data",
        "completed_days": [],
        "rolling": {"games": 0, "hits": 0, "accuracy": None},
        "yesterday": None,
        "paths": {
            "predictions_dir": str(predictions_dir),
            "raw_history": str(raw_file),
            "baseline_file": str(baseline_file),
        },
    }

    if not predictions_dir.exists() or not raw_file.exists():
        report["reason"] = "missing_inputs"
        return report

    raw_df = pd.read_csv(raw_file, low_memory=False, dtype={"game_id": str})
    raw_df["date"] = raw_df["date"].astype(str)

    day_scores: list[DayScore] = []
    for file_path in sorted(predictions_dir.glob("*.json")):
        score = _score_one_day(file_path, raw_df)
        if score is not None:
            day_scores.append(score)

    if not day_scores:
        report["reason"] = "no_completed_prediction_days"
        return report

    day_scores = sorted(day_scores, key=lambda x: x.date, reverse=True)
    selected = day_scores[: max(int(window_days), 1)]

    rolling_games = int(sum(d.games for d in selected))
    rolling_hits = int(sum(d.hits for d in selected))
    rolling_accuracy = (float(rolling_hits) / float(rolling_games)) if rolling_games > 0 else 0.0

    yesterday_accuracy: Optional[float] = selected[0].accuracy if selected else None

    status, recommended_mode, reason, thresholds = _decide_status(
        rolling_accuracy=rolling_accuracy,
        rolling_games=rolling_games,
        yesterday_accuracy=yesterday_accuracy,
        target_accuracy=target_accuracy,
        min_games=min_games,
    )

    report.update(
        {
            "status": status,
            "recommended_mode": recommended_mode,
            "reason": reason,
            "thresholds": thresholds,
            "completed_days": [
                {
                    "date": d.date,
                    "games": int(d.games),
                    "hits": int(d.hits),
                    "accuracy": d.accuracy,
                }
                for d in selected
            ],
            "rolling": {
                "games": rolling_games,
                "hits": rolling_hits,
                "accuracy": rolling_accuracy if rolling_games > 0 else None,
            },
            "yesterday": {
                "date": selected[0].date,
                "games": int(selected[0].games),
                "hits": int(selected[0].hits),
                "accuracy": selected[0].accuracy,
            }
            if selected
            else None,
        }
    )

    return report


def _save_report(report: dict, report_file: Path) -> None:
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_mode_file(mode_file: Path, mode_value: str) -> None:
    mode_file.parent.mkdir(parents=True, exist_ok=True)
    mode_file.write_text(f"{mode_value}\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="MLB daily traffic-light decision helper")
    parser.add_argument("--window-days", type=int, default=3, help="Number of completed days to evaluate")
    parser.add_argument("--target-accuracy", type=float, default=0.5782312925, help="Fallback target accuracy")
    parser.add_argument("--min-games", type=int, default=25, help="Minimum combined games in window")
    parser.add_argument("--apply-profile", action="store_true", help="Write recommended mode into tools/mlb_profile_mode.txt")
    parser.add_argument("--quiet", action="store_true", help="Print only critical output")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    report_file = base_dir / "src" / "data" / "mlb" / "reports" / "mlb_daily_semaforo_latest.json"
    mode_file = base_dir / "tools" / "mlb_profile_mode.txt"

    report = build_report(
        base_dir=base_dir,
        window_days=args.window_days,
        default_target=float(args.target_accuracy),
        min_games=int(args.min_games),
    )
    _save_report(report, report_file)

    if args.apply_profile:
        _write_mode_file(mode_file, report["recommended_mode"])

    if not args.quiet:
        rolling = report.get("rolling") or {}
        yesterday = report.get("yesterday") or {}
        rolling_acc = rolling.get("accuracy")
        rolling_acc_txt = "N/A" if rolling_acc is None else f"{rolling_acc:.4f}"
        yesterday_acc = yesterday.get("accuracy")
        yesterday_acc_txt = "N/A" if yesterday_acc is None else f"{yesterday_acc:.4f}"

        print(
            "MLB SEMAFORO "
            + f"status={report.get('status')} "
            + f"recommended={report.get('recommended_mode')} "
            + f"rolling_acc={rolling_acc_txt} "
            + f"yesterday_acc={yesterday_acc_txt}"
        )
        print(f"Report: {report_file}")
        if args.apply_profile:
            print(f"Mode file updated: {mode_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
