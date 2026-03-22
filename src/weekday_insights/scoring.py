from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

WEEKDAY_NAMES = {
    0: "Lunes",
    1: "Martes",
    2: "Miercoles",
    3: "Jueves",
    4: "Viernes",
    5: "Sabado",
    6: "Domingo",
}

WEEKDAY_ORDER = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]


@dataclass
class SportScoringConfig:
    key: str
    label: str
    raw_file: Path
    home_col: str
    away_col: str
    metric_label: str
    date_shift_days: int = 0


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _compute_weekday_summary_for_sport(cfg: SportScoringConfig):
    if not cfg.raw_file.exists():
        return {
            "sport": cfg.key,
            "label": cfg.label,
            "metric_label": cfg.metric_label,
            "total_games": 0,
            "overall_avg_total": 0.0,
            "overall_median_total": 0.0,
            "highest_day": None,
            "lowest_day": None,
            "weekday_breakdown": [],
        }

    try:
        df = pd.read_csv(cfg.raw_file, usecols=["date", cfg.home_col, cfg.away_col])
    except Exception:
        return {
            "sport": cfg.key,
            "label": cfg.label,
            "metric_label": cfg.metric_label,
            "total_games": 0,
            "overall_avg_total": 0.0,
            "overall_median_total": 0.0,
            "highest_day": None,
            "lowest_day": None,
            "weekday_breakdown": [],
        }

    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    if cfg.date_shift_days != 0:
        df["date_dt"] = df["date_dt"] + timedelta(days=cfg.date_shift_days)

    df[cfg.home_col] = pd.to_numeric(df[cfg.home_col], errors="coerce")
    df[cfg.away_col] = pd.to_numeric(df[cfg.away_col], errors="coerce")

    df = df.dropna(subset=["date_dt", cfg.home_col, cfg.away_col]).copy()
    if df.empty:
        return {
            "sport": cfg.key,
            "label": cfg.label,
            "metric_label": cfg.metric_label,
            "total_games": 0,
            "overall_avg_total": 0.0,
            "overall_median_total": 0.0,
            "highest_day": None,
            "lowest_day": None,
            "weekday_breakdown": [],
        }

    df["total_metric"] = df[cfg.home_col] + df[cfg.away_col]
    overall_median = float(df["total_metric"].median())

    df["is_high"] = (df["total_metric"] >= overall_median).astype(int)
    df["is_low"] = (df["total_metric"] < overall_median).astype(int)
    df["weekday_idx"] = df["date_dt"].dt.weekday
    df["weekday"] = df["weekday_idx"].map(WEEKDAY_NAMES)

    grouped = (
        df.groupby(["weekday_idx", "weekday"], as_index=False)
        .agg(
            games=("total_metric", "size"),
            avg_total=("total_metric", "mean"),
            median_total=("total_metric", "median"),
            high_games=("is_high", "sum"),
            low_games=("is_low", "sum"),
        )
    )

    grouped["high_rate"] = grouped["high_games"] / grouped["games"]
    grouped["low_rate"] = grouped["low_games"] / grouped["games"]
    grouped = grouped.sort_values("weekday_idx")

    rows = []
    for _, row in grouped.iterrows():
        rows.append(
            {
                "weekday": str(row["weekday"]),
                "games": int(row["games"]),
                "avg_total": round(float(row["avg_total"]), 3),
                "median_total": round(float(row["median_total"]), 3),
                "high_games": int(row["high_games"]),
                "low_games": int(row["low_games"]),
                "high_rate": round(float(row["high_rate"]), 4),
                "low_rate": round(float(row["low_rate"]), 4),
            }
        )

    highest = None
    lowest = None
    if rows:
        highest = max(rows, key=lambda x: (x["avg_total"], x["games"]))
        lowest = min(rows, key=lambda x: (x["avg_total"], -x["games"]))

    return {
        "sport": cfg.key,
        "label": cfg.label,
        "metric_label": cfg.metric_label,
        "total_games": int(len(df)),
        "overall_avg_total": round(float(df["total_metric"].mean()), 3),
        "overall_median_total": round(overall_median, 3),
        "highest_day": {
            "weekday": highest["weekday"],
            "avg_total": highest["avg_total"],
            "games": highest["games"],
        } if highest else None,
        "lowest_day": {
            "weekday": lowest["weekday"],
            "avg_total": lowest["avg_total"],
            "games": lowest["games"],
        } if lowest else None,
        "weekday_breakdown": rows,
    }


def build_weekday_scoring_summary(configs: list[SportScoringConfig]):
    return {
        "generated_at": datetime.now().isoformat(),
        "weekday_order": WEEKDAY_ORDER,
        "sports": [_compute_weekday_summary_for_sport(cfg) for cfg in configs],
    }
