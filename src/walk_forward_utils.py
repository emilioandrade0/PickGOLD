from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


EPSILON = 1e-6


@dataclass
class DateSplit:
    split_id: int
    train_dates: List[str]
    calib_dates: List[str]
    test_dates: List[str]


class IdentityCalibrator:
    def predict(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        return np.clip(probs, EPSILON, 1.0 - EPSILON)


class SafeIsotonicCalibrator:
    def __init__(self) -> None:
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=EPSILON, y_max=1.0 - EPSILON)
        self.is_fitted = False

    def fit(self, probs: Sequence[float], y: Sequence[int]) -> "SafeIsotonicCalibrator":
        x = np.asarray(probs, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        if len(x) < 20 or len(np.unique(y_arr)) < 2:
            self.is_fitted = False
            return self
        self.model.fit(x, y_arr)
        self.is_fitted = True
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        if not self.is_fitted:
            return np.clip(probs, EPSILON, 1.0 - EPSILON)
        return np.clip(self.model.predict(probs), EPSILON, 1.0 - EPSILON)


def clamp_probs(probs: Sequence[float]) -> np.ndarray:
    arr = np.asarray(probs, dtype=float)
    return np.clip(arr, EPSILON, 1.0 - EPSILON)


def sanitize_feature_frame(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas de features: {missing[:10]}")
    X = df.loc[:, list(feature_cols)].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0.0)
    return X


def generate_date_walk_forward_splits(
    df: pd.DataFrame,
    date_col: str,
    *,
    min_train_dates: int,
    calibration_dates: int,
    test_dates: int = 1,
    step_dates: int = 1,
) -> List[DateSplit]:
    if date_col not in df.columns:
        raise KeyError(f"No existe la columna de fecha: {date_col}")

    unique_dates = [str(x) for x in pd.Series(df[date_col]).dropna().astype(str).sort_values().unique().tolist()]
    needed = min_train_dates + calibration_dates + test_dates
    if len(unique_dates) < needed:
        raise ValueError(
            f"No hay suficientes fechas para walk-forward. Fechas={len(unique_dates)} requeridas={needed}"
        )

    splits: List[DateSplit] = []
    split_id = 0
    last_start = len(unique_dates) - calibration_dates - test_dates
    for start_idx in range(min_train_dates, last_start + 1, step_dates):
        train_dates = unique_dates[:start_idx]
        calib_dates = unique_dates[start_idx : start_idx + calibration_dates]
        test_block = unique_dates[start_idx + calibration_dates : start_idx + calibration_dates + test_dates]
        if not train_dates or not calib_dates or not test_block:
            continue
        splits.append(
            DateSplit(
                split_id=split_id,
                train_dates=train_dates,
                calib_dates=calib_dates,
                test_dates=test_block,
            )
        )
        split_id += 1
    return splits


def safe_binary_metrics(y_true: Sequence[int], probs: Sequence[float], threshold: float) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = clamp_probs(probs)
    pred = (p >= threshold).astype(int)
    out = {
        "rows": int(len(y)),
        "threshold": float(threshold),
        "positive_rate": float(pred.mean()) if len(pred) else 0.0,
        "accuracy": float(accuracy_score(y, pred)) if len(y) else 0.0,
        "brier": float(brier_score_loss(y, p)) if len(np.unique(y)) > 1 else 0.0,
        "logloss": float(log_loss(y, p, labels=[0, 1])) if len(y) else 0.0,
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5
    except Exception:
        out["roc_auc"] = 0.5
    return out


def summarize_predictions(detail_df: pd.DataFrame) -> Dict[str, float]:
    if detail_df.empty:
        return {
            "rows": 0,
            "accuracy": 0.0,
            "brier": 0.0,
            "logloss": 0.0,
            "roc_auc": 0.5,
            "coverage": 0.0,
            "published_accuracy": 0.0,
        }

    y = detail_df["y_true"].astype(int).to_numpy()
    p = clamp_probs(detail_df["ensemble_prob_calibrated"].astype(float).to_numpy())
    pred = detail_df["pred_label"].astype(int).to_numpy()
    publish = detail_df["publish_pick"].astype(int).to_numpy()

    summary = {
        "rows": int(len(detail_df)),
        "accuracy": float(accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, p)) if len(np.unique(y)) > 1 else 0.0,
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "coverage": float(publish.mean()),
    }
    try:
        summary["roc_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5
    except Exception:
        summary["roc_auc"] = 0.5

    if publish.sum() > 0:
        summary["published_accuracy"] = float((detail_df.loc[publish == 1, "y_true"] == detail_df.loc[publish == 1, "pred_label"]).mean())
    else:
        summary["published_accuracy"] = 0.0
    return summary


def choose_threshold_from_calibration(
    y_true: Sequence[int],
    probs: Sequence[float],
    *,
    min_threshold: float = 0.50,
    max_threshold: float = 0.70,
    step: float = 0.01,
    min_coverage: float = 0.20,
    target_coverage: Optional[float] = None,
    coverage_tolerance: float = 0.02,
    min_published_rows: int = 25,
    accuracy_weight: float = 1.0,
    coverage_weight: float = 0.03,
    balance_weight: float = 0.03,
) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = clamp_probs(probs)
    candidates = np.arange(min_threshold, max_threshold + 1e-9, step)

    best = {
        "threshold": 0.55,
        "score": -1e9,
        "accuracy": 0.0,
        "coverage": 0.0,
        "positive_rate": 0.0,
        "published_rows": 0,
        "target_penalty": 0.0,
    }

    conf = np.maximum(p, 1.0 - p)
    pred_side = (p >= 0.5).astype(int)

    for thr in candidates:
        take = conf >= thr
        published_rows = int(take.sum())
        coverage = float(take.mean()) if len(take) else 0.0
        if coverage < min_coverage or published_rows < int(max(1, min_published_rows)):
            continue

        acc = float((pred_side[take] == y[take]).mean())
        positive_rate = float(pred_side[take].mean()) if published_rows else 0.0

        target_penalty = 0.0
        if target_coverage is not None and float(target_coverage) > 0.0:
            target_cov = float(target_coverage)
            cov_shortfall = max(0.0, target_cov - coverage)
            cov_excess = max(0.0, coverage - (target_cov + float(max(0.0, coverage_tolerance))))
            target_penalty = (cov_shortfall * 0.25) + (cov_excess * 0.20)

        score = (
            (float(accuracy_weight) * acc)
            + (coverage * float(coverage_weight))
            - (abs(positive_rate - 0.5) * float(balance_weight))
            - target_penalty
        )

        if score > best["score"]:
            best = {
                "threshold": float(round(thr, 2)),
                "score": float(score),
                "accuracy": acc,
                "coverage": coverage,
                "positive_rate": positive_rate,
                "published_rows": int(published_rows),
                "target_penalty": float(target_penalty),
            }

    if best["score"] <= -1e8:
        relaxed_min_coverage = max(0.0, float(min_coverage) * 0.5)
        for thr in candidates:
            take = conf >= thr
            published_rows = int(take.sum())
            coverage = float(take.mean()) if len(take) else 0.0
            if coverage < relaxed_min_coverage or published_rows == 0:
                continue

            acc = float((pred_side[take] == y[take]).mean())
            positive_rate = float(pred_side[take].mean()) if published_rows else 0.0
            score = acc + (coverage * 0.02) - (abs(positive_rate - 0.5) * float(balance_weight))

            if score > best["score"]:
                best = {
                    "threshold": float(round(thr, 2)),
                    "score": float(score),
                    "accuracy": acc,
                    "coverage": coverage,
                    "positive_rate": positive_rate,
                    "published_rows": int(published_rows),
                    "target_penalty": 0.0,
                }

    return best
