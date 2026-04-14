from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ERROR_RISK_FEATURES = [
    "prob_home",
    "confidence",
    "uncertainty",
    "spread_abs",
    "market_missing",
    "has_market_side",
    "model_vs_market",
    "pick_is_home",
    "market_pick_is_home",
    "threshold_used",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        if np.isnan(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def market_pick_from_row(row: pd.Series):
    hs = _safe_float(row.get("home_spread"), default=0.0)
    if hs < 0:
        return row.get("home_team")
    if hs > 0:
        return row.get("away_team")
    return None


def build_error_risk_feature_dict(
    row: pd.Series,
    calibrated_prob_home: float,
    threshold_used: float,
    current_pick: str,
):
    prob_home = _safe_float(calibrated_prob_home, default=0.5)
    confidence = max(prob_home, 1.0 - prob_home)
    uncertainty = abs(prob_home - _safe_float(threshold_used, default=0.5))
    hs = _safe_float(row.get("home_spread"), default=0.0)
    spread_abs = abs(hs)
    market_pick = market_pick_from_row(row)

    has_market_side = 1.0 if market_pick is not None else 0.0
    market_missing = 0.0 if has_market_side == 1.0 else 1.0
    model_vs_market = 1.0 if (market_pick is not None and str(current_pick) != str(market_pick)) else 0.0
    pick_is_home = 1.0 if str(current_pick) == str(row.get("home_team")) else 0.0
    market_pick_is_home = 1.0 if (market_pick is not None and str(market_pick) == str(row.get("home_team"))) else 0.0

    return {
        "prob_home": prob_home,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "spread_abs": spread_abs,
        "market_missing": market_missing,
        "has_market_side": has_market_side,
        "model_vs_market": model_vs_market,
        "pick_is_home": pick_is_home,
        "market_pick_is_home": market_pick_is_home,
        "threshold_used": _safe_float(threshold_used, default=0.5),
    }


def feature_vector_from_dict(feat: dict, feature_names: list[str] | None = None) -> np.ndarray:
    names = list(feature_names or ERROR_RISK_FEATURES)
    return np.array([[float(feat.get(name, 0.0)) for name in names]], dtype=float)


def load_error_risk_bundle(path: Path):
    if not path.exists():
        return None
    try:
        obj = joblib.load(path)
        if not isinstance(obj, dict):
            return None
        if "model" not in obj:
            return None
        if "feature_names" not in obj or not isinstance(obj["feature_names"], list):
            obj["feature_names"] = list(ERROR_RISK_FEATURES)
        if "policy" not in obj or not isinstance(obj["policy"], dict):
            obj["policy"] = {}
        return obj
    except Exception:
        return None


def predict_error_risk_prob(
    bundle: dict | None,
    row: pd.Series,
    calibrated_prob_home: float,
    threshold_used: float,
    current_pick: str,
):
    if not isinstance(bundle, dict):
        return None
    model = bundle.get("model")
    if model is None:
        return None
    feat = build_error_risk_feature_dict(
        row=row,
        calibrated_prob_home=calibrated_prob_home,
        threshold_used=threshold_used,
        current_pick=current_pick,
    )
    x = feature_vector_from_dict(feat, bundle.get("feature_names"))
    try:
        p = float(model.predict_proba(x)[0, 1])
        if np.isnan(p):
            return None
        return max(0.0, min(1.0, p))
    except Exception:
        return None


def apply_error_risk_override(
    bundle: dict | None,
    row: pd.Series,
    calibrated_prob_home: float,
    threshold_used: float,
    current_pick: str,
    current_rule: str,
):
    risk_prob = predict_error_risk_prob(
        bundle=bundle,
        row=row,
        calibrated_prob_home=calibrated_prob_home,
        threshold_used=threshold_used,
        current_pick=current_pick,
    )
    if risk_prob is None or not isinstance(bundle, dict):
        return current_pick, current_rule, None

    policy = bundle.get("policy", {}) if isinstance(bundle.get("policy", {}), dict) else {}
    enabled = bool(policy.get("enabled", True))
    if not enabled:
        return current_pick, current_rule, risk_prob

    risk_threshold = _safe_float(policy.get("risk_threshold"), default=0.62)
    min_spread_abs = _safe_float(policy.get("min_spread_abs"), default=0.0)
    only_when_disagree = bool(policy.get("only_when_disagree", True))

    market_pick = market_pick_from_row(row)
    if market_pick is None:
        return current_pick, current_rule, risk_prob

    spread_abs = abs(_safe_float(row.get("home_spread"), default=0.0))
    if spread_abs < min_spread_abs:
        return current_pick, current_rule, risk_prob

    if only_when_disagree and str(current_pick) == str(market_pick):
        return current_pick, current_rule, risk_prob

    if risk_prob >= risk_threshold:
        if str(current_pick) != str(market_pick):
            return market_pick, "error_risk_override", risk_prob

    return current_pick, current_rule, risk_prob
