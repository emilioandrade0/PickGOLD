from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List


DEFAULT_BUCKETS = [
    {"min": 0.5, "max": 1.0, "scale": 1.0, "offset": 0.0},
]


def clamp_prob(prob: float) -> float:
    return float(min(max(prob, 1e-6), 1 - 1e-6))


def apply_bucket_calibration(prob: float, buckets: List[Dict]) -> float:
    p = clamp_prob(prob)
    for b in buckets:
        bmin = float(b.get("min", 0.0))
        bmax = float(b.get("max", 1.0))
        if bmin <= p <= bmax:
            scale = float(b.get("scale", 1.0))
            offset = float(b.get("offset", 0.0))
            return clamp_prob((p * scale) + offset)
    return p


def apply_confidence_bucket_calibration(confidence: float, buckets: List[Dict]) -> float:
    c = float(min(max(confidence, 0.5), 1.0))
    for b in buckets:
        bmin = float(b.get("min", 0.5))
        bmax = float(b.get("max", 1.0))
        if bmin <= c <= bmax:
            scale = float(b.get("scale", 1.0))
            offset = float(b.get("offset", 0.0))
            return float(min(max((c * scale) + offset, 0.5), 1.0))
    return c


def _safe_list(values):
    if not isinstance(values, list):
        return []
    out = []
    for v in values:
        try:
            fv = float(v)
        except Exception:
            continue
        if fv == fv:
            out.append(fv)
    return out


def apply_isotonic_confidence_calibration(confidence: float, isotonic_cfg: Dict) -> float:
    c = float(min(max(confidence, 0.5), 1.0))
    x = _safe_list(isotonic_cfg.get("x"))
    y = _safe_list(isotonic_cfg.get("y"))
    if len(x) < 2 or len(x) != len(y):
        return c

    pairs = sorted(zip(x, y), key=lambda t: t[0])
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    if c <= xs[0]:
        return float(min(max(ys[0], 0.5), 1.0))
    if c >= xs[-1]:
        return float(min(max(ys[-1], 0.5), 1.0))

    for idx in range(1, len(xs)):
        if c <= xs[idx]:
            x0, x1 = xs[idx - 1], xs[idx]
            y0, y1 = ys[idx - 1], ys[idx]
            if x1 <= x0:
                return float(min(max(y1, 0.5), 1.0))
            t = (c - x0) / (x1 - x0)
            yhat = y0 + t * (y1 - y0)
            return float(min(max(yhat, 0.5), 1.0))

    return c


def apply_platt_confidence_calibration(confidence: float, platt_cfg: Dict) -> float:
    c = float(min(max(confidence, 0.5), 1.0))
    try:
        a = float(platt_cfg.get("a", 1.0))
        b = float(platt_cfg.get("b", 0.0))
    except Exception:
        return c

    z = (a * c) + b
    if z >= 0:
        exp_neg = math.exp(-z)
        calibrated = 1.0 / (1.0 + exp_neg)
    else:
        exp_pos = math.exp(z)
        calibrated = exp_pos / (1.0 + exp_pos)

    return float(min(max(calibrated, 0.5), 1.0))


def load_calibration_config(config_path: Path) -> Dict:
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def calibrate_probability(
    model_prob: float,
    sport: str,
    market: str,
    calibration_config: Dict,
) -> float:
    sport_cfg = calibration_config.get(sport, {}) if isinstance(calibration_config, dict) else {}
    market_cfg = sport_cfg.get(market, {}) if isinstance(sport_cfg, dict) else {}
    buckets = market_cfg.get("buckets", DEFAULT_BUCKETS)
    method = str(market_cfg.get("method", "bucket") or "bucket").strip().lower()

    p = clamp_prob(model_prob)
    side_positive = p >= 0.5
    raw_conf = p if side_positive else (1.0 - p)

    if method == "isotonic":
        calibrated_conf = apply_isotonic_confidence_calibration(raw_conf, market_cfg.get("isotonic", {}))
    elif method == "platt":
        calibrated_conf = apply_platt_confidence_calibration(raw_conf, market_cfg.get("platt", {}))
    else:
        calibrated_conf = apply_confidence_bucket_calibration(raw_conf, buckets)

    calibrated = calibrated_conf if side_positive else (1.0 - calibrated_conf)
    return clamp_prob(calibrated)


def confidence_pct_from_prob(prob: float) -> float:
    p = clamp_prob(prob)
    return max(p, 1.0 - p) * 100.0
