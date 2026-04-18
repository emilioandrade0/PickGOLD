import json
import importlib
import math
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

TARGETS = {
    "nba": {
        "output": BASE_DIR / "models" / "calibration_params.json",
        "historical_dir": BASE_DIR / "data" / "historical_predictions",
        "markets": [{"source": "full_game", "output": "full_game"}, {"source": "q1_yrfi", "output": "q1"}],
    },
    "mlb": {
        "output": BASE_DIR / "data" / "mlb" / "models" / "calibration_params.json",
        "historical_dir": BASE_DIR / "data" / "mlb" / "historical_predictions",
        "markets": [
            {"source": "full_game", "output": "full_game"},
            {"source": "q1_yrfi", "output": "yrfi"},
            {"source": "f5", "output": "f5"},
        ],
    },
    "lmb": {
        "output": BASE_DIR / "data" / "lmb" / "models" / "calibration_params.json",
        "historical_dir": BASE_DIR / "data" / "lmb" / "historical_predictions",
        "markets": [
            {"source": "full_game", "output": "full_game"},
            {"source": "q1_yrfi", "output": "yrfi"},
            {"source": "f5", "output": "f5"},
        ],
    },
    "kbo": {
        "output": BASE_DIR / "data" / "kbo" / "models" / "calibration_params.json",
        "historical_dir": BASE_DIR / "data" / "kbo" / "historical_predictions",
        "markets": [
            {"source": "full_game", "output": "full_game"},
            {"source": "q1_yrfi", "output": "yrfi"},
            {"source": "f5", "output": "f5"},
        ],
    },
    "nhl": {
        "output": BASE_DIR / "data" / "nhl" / "models" / "calibration_params.json",
        "historical_dir": BASE_DIR / "data" / "nhl" / "historical_predictions",
        "markets": [
            {"source": "full_game", "output": "full_game"},
            {"source": "spread", "output": "spread"},
            {"source": "home_over", "output": "home_over_25"},
        ],
    },
    "liga_mx": {
        "output": BASE_DIR / "data" / "liga_mx" / "models" / "calibration_params.json",
        "historical_dir": BASE_DIR / "data" / "liga_mx" / "historical_predictions",
        "markets": [
            {"source": "full_game", "output": "full_game"},
            {"source": "total", "output": "total"},
            {"source": "btts", "output": "btts"},
        ],
    },
    "laliga": {
        "output": BASE_DIR / "data" / "laliga" / "models" / "calibration_params.json",
        "historical_dir": BASE_DIR / "data" / "laliga" / "historical_predictions",
        "markets": [
            {"source": "full_game", "output": "full_game"},
            {"source": "total", "output": "total"},
            {"source": "btts", "output": "btts"},
        ],
    },
    "euroleague": {
        "output": BASE_DIR / "data" / "euroleague" / "models" / "calibration_params.json",
        "historical_dir": BASE_DIR / "data" / "euroleague" / "historical_predictions",
        "markets": [{"source": "full_game", "output": "full_game"}, {"source": "q1_yrfi", "output": "q1"}],
    },
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _to_prob(value):
    try:
        v = float(value)
    except Exception:
        return None
    if v != v:
        return None
    if 0.0 <= v <= 1.0:
        return v
    if 1.0 < v <= 100.0:
        return v / 100.0
    return None


def _to_hit(value):
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(int(value))
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "1", "yes", "si", "acierto"}:
            return True
        if s in {"false", "0", "no", "fallo"}:
            return False
    return None


def _event_prob_for_market(event: dict, market: str):
    keys_by_market = {
        "full_game": [
            "full_game_calibrated_prob_home",
            "full_game_model_prob_home",
            "full_game_calibrated_prob_pick",
            "full_game_model_prob_pick",
            "full_game_confidence",
            "recommended_confidence",
        ],
        "q1_yrfi": ["q1_calibrated_prob_yes", "q1_model_prob_yes", "q1_calibrated_prob_home", "q1_confidence"],
        "spread": ["spread_calibrated_prob_pick", "spread_model_prob_pick", "spread_confidence"],
        "total": ["total_adjusted_probability", "total_base_probability", "total_confidence"],
        "btts": ["btts_adjusted_probability", "btts_base_probability", "btts_confidence"],
        "f5": ["extra_f5_calibrated_prob_home", "extra_f5_model_prob_home", "extra_f5_confidence"],
        "home_over": ["home_over_calibrated_prob_pick", "home_over_model_prob_pick", "home_over_confidence"],
    }
    for key in keys_by_market.get(market, []):
        if key in event:
            p = _to_prob(event.get(key))
            if p is not None:
                return p
    return None


def _event_hit_for_market(event: dict, market: str):
    keys_by_market = {
        "full_game": ["correct_full_game_adjusted", "correct_full_game", "full_game_hit", "correct_full_game_base"],
        "q1_yrfi": ["q1_hit"],
        "spread": ["correct_spread"],
        "total": ["correct_total_adjusted", "correct_total"],
        "btts": ["correct_btts_adjusted", "correct_btts"],
        "f5": ["correct_home_win_f5", "correct_f5"],
        "home_over": ["correct_home_over", "correct_home_total"],
    }
    for key in keys_by_market.get(market, []):
        if key in event:
            h = _to_hit(event.get(key))
            if h is not None:
                return h
    return None


def _brier_score(probs, outcomes):
    if not probs:
        return None
    return float(sum((p - y) ** 2 for p, y in zip(probs, outcomes)) / len(probs))


def _binary_log_loss(probs, outcomes, eps=1e-9):
    if not probs:
        return None
    total = 0.0
    n = len(probs)
    for p, y in zip(probs, outcomes):
        p = _clamp(float(p), eps, 1.0 - eps)
        y = int(bool(y))
        total += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
    return float(total / n)


def _ece(probs, outcomes, bins=10):
    n = len(probs)
    if n <= 0:
        return None
    total = 0.0
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        bucket = [(p, y) for p, y in zip(probs, outcomes) if (lo <= p < hi) or (i == bins - 1 and p == 1.0)]
        if not bucket:
            continue
        mean_pred = sum(p for p, _ in bucket) / len(bucket)
        mean_hit = sum(y for _, y in bucket) / len(bucket)
        total += (len(bucket) / n) * abs(mean_pred - mean_hit)
    return float(total)


def _reliability_curve(probs, outcomes, bins=10):
    out = []
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        bucket = [(p, y) for p, y in zip(probs, outcomes) if (lo <= p < hi) or (i == bins - 1 and p == 1.0)]
        if not bucket:
            continue
        mean_pred = sum(p for p, _ in bucket) / len(bucket)
        mean_hit = sum(y for _, y in bucket) / len(bucket)
        out.append(
            {
                "min": round(lo, 4),
                "max": round(hi, 4),
                "count": len(bucket),
                "mean_pred": round(mean_pred, 4),
                "mean_hit": round(mean_hit, 4),
            }
        )
    return out


def _split_train_valid(values, ratio=0.70):
    n = len(values)
    if n < 30:
        return values, values
    cut = int(n * ratio)
    cut = max(20, min(n - 10, cut))
    return values[:cut], values[cut:]


def _fit_isotonic(train_conf, train_hit):
    try:
        isotonic_module = importlib.import_module("sklearn.isotonic")
        IsotonicRegression = getattr(isotonic_module, "IsotonicRegression")
    except Exception:
        return None
    if len(train_conf) < 20:
        return None
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(train_conf, train_hit)

    x = [float(v) for v in getattr(iso, "X_thresholds_", [])]
    y = [float(v) for v in getattr(iso, "y_thresholds_", [])]
    if len(x) < 2 or len(x) != len(y):
        return None

    y = [float(_clamp(v, 0.5, 1.0)) for v in y]
    return {"x": [round(float(v), 6) for v in x], "y": [round(float(v), 6) for v in y]}


def _predict_isotonic(conf_values, model):
    xs = model.get("x", [])
    ys = model.get("y", [])
    if len(xs) < 2 or len(xs) != len(ys):
        return list(conf_values)

    pred = []
    for c in conf_values:
        c = float(_clamp(c, 0.5, 1.0))
        if c <= xs[0]:
            pred.append(float(_clamp(ys[0], 0.5, 1.0)))
            continue
        if c >= xs[-1]:
            pred.append(float(_clamp(ys[-1], 0.5, 1.0)))
            continue
        yhat = c
        for idx in range(1, len(xs)):
            if c <= xs[idx]:
                x0, x1 = xs[idx - 1], xs[idx]
                y0, y1 = ys[idx - 1], ys[idx]
                if x1 <= x0:
                    yhat = y1
                else:
                    t = (c - x0) / (x1 - x0)
                    yhat = y0 + t * (y1 - y0)
                break
        pred.append(float(_clamp(yhat, 0.5, 1.0)))
    return pred


def _fit_platt(train_conf, train_hit):
    try:
        linear_module = importlib.import_module("sklearn.linear_model")
        LogisticRegression = getattr(linear_module, "LogisticRegression")
    except Exception:
        return None
    if len(train_conf) < 30:
        return None
    if len(set(int(v) for v in train_hit)) < 2:
        return None
    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    X = [[float(v)] for v in train_conf]
    y = [int(v) for v in train_hit]
    lr.fit(X, y)
    a = float(lr.coef_[0][0])
    b = float(lr.intercept_[0])
    return {"a": round(a, 8), "b": round(b, 8)}


def _predict_platt(conf_values, model):
    try:
        a = float(model.get("a", 1.0))
        b = float(model.get("b", 0.0))
    except Exception:
        return list(conf_values)

    out = []
    for c in conf_values:
        c = float(_clamp(c, 0.5, 1.0))
        z = (a * c) + b
        if z >= 0:
            p = 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            p = ez / (1.0 + ez)
        out.append(float(_clamp(p, 0.5, 1.0)))
    return out


def _default_bucket_config():
    return {"method": "bucket", "buckets": [{"min": 0.5, "max": 1.0, "scale": 1.0, "offset": 0.0}]}


def _build_market_calibrator(rows):
    if not rows:
        payload = _default_bucket_config()
        payload["metrics"] = {"sample_size": 0}
        return payload

    # Use confidence symmetry so calibrator works for both home and away picks.
    values = []
    for p, hit in rows:
        p = float(_clamp(p, 1e-6, 1 - 1e-6))
        conf = max(p, 1.0 - p)
        values.append((conf, int(bool(hit))))

    train_rows, valid_rows = _split_train_valid(values)
    train_conf = [r[0] for r in train_rows]
    train_hit = [r[1] for r in train_rows]
    valid_conf = [r[0] for r in valid_rows]
    valid_hit = [r[1] for r in valid_rows]

    raw_pred = list(valid_conf)
    raw_brier = _brier_score(raw_pred, valid_hit)
    raw_logloss = _binary_log_loss(raw_pred, valid_hit)
    raw_ece = _ece(raw_pred, valid_hit)
    raw_curve = _reliability_curve(raw_pred, valid_hit)

    isotonic_model = _fit_isotonic(train_conf, train_hit)
    isotonic_pred = None
    isotonic_brier = None
    isotonic_logloss = None
    isotonic_ece = None
    isotonic_curve = None
    if isotonic_model is not None:
        isotonic_pred = _predict_isotonic(valid_conf, isotonic_model)
        isotonic_brier = _brier_score(isotonic_pred, valid_hit)
        isotonic_logloss = _binary_log_loss(isotonic_pred, valid_hit)
        isotonic_ece = _ece(isotonic_pred, valid_hit)
        isotonic_curve = _reliability_curve(isotonic_pred, valid_hit)

    platt_model = _fit_platt(train_conf, train_hit)
    platt_pred = None
    platt_brier = None
    platt_logloss = None
    platt_ece = None
    platt_curve = None
    if platt_model is not None:
        platt_pred = _predict_platt(valid_conf, platt_model)
        platt_brier = _brier_score(platt_pred, valid_hit)
        platt_logloss = _binary_log_loss(platt_pred, valid_hit)
        platt_ece = _ece(platt_pred, valid_hit)
        platt_curve = _reliability_curve(platt_pred, valid_hit)

    method = "bucket"
    best_brier = raw_brier
    if isotonic_brier is not None and (best_brier is None or isotonic_brier <= (best_brier - 1e-4)):
        method = "isotonic"
        best_brier = isotonic_brier
    if platt_brier is not None and (best_brier is None or platt_brier <= (best_brier - 1e-4)):
        method = "platt"
        best_brier = platt_brier

    selected_pred = raw_pred
    selected_brier = raw_brier
    selected_logloss = raw_logloss
    selected_ece = raw_ece
    selected_curve = raw_curve
    if method == "isotonic" and isotonic_pred is not None:
        selected_pred = isotonic_pred
        selected_brier = isotonic_brier
        selected_logloss = isotonic_logloss
        selected_ece = isotonic_ece
        selected_curve = isotonic_curve
    elif method == "platt" and platt_pred is not None:
        selected_pred = platt_pred
        selected_brier = platt_brier
        selected_logloss = platt_logloss
        selected_ece = platt_ece
        selected_curve = platt_curve

    payload = _default_bucket_config()
    payload["method"] = method
    if isotonic_model is not None:
        payload["isotonic"] = isotonic_model
    if platt_model is not None:
        payload["platt"] = platt_model

    payload["metrics"] = {
        "sample_size": len(values),
        "train_size": len(train_rows),
        "valid_size": len(valid_rows),
        "brier_raw": round(raw_brier, 6) if raw_brier is not None else None,
        "brier_isotonic": round(isotonic_brier, 6) if isotonic_brier is not None else None,
        "brier_platt": round(platt_brier, 6) if platt_brier is not None else None,
        "brier_selected": round(selected_brier, 6) if selected_brier is not None else None,
        "logloss_raw": round(raw_logloss, 6) if raw_logloss is not None else None,
        "logloss_isotonic": round(isotonic_logloss, 6) if isotonic_logloss is not None else None,
        "logloss_platt": round(platt_logloss, 6) if platt_logloss is not None else None,
        "logloss_selected": round(selected_logloss, 6) if selected_logloss is not None else None,
        "ece_raw": round(raw_ece, 6) if raw_ece is not None else None,
        "ece_isotonic": round(isotonic_ece, 6) if isotonic_ece is not None else None,
        "ece_platt": round(platt_ece, 6) if platt_ece is not None else None,
        "ece_selected": round(selected_ece, 6) if selected_ece is not None else None,
        "reliability_curve_raw": raw_curve,
        "reliability_curve_isotonic": isotonic_curve,
        "reliability_curve_platt": platt_curve,
        "reliability_curve_selected": selected_curve,
        "selected_method": method,
    }
    return payload


def _read_events(file_path: Path):
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, list):
        return [e for e in payload if isinstance(e, dict)]
    if isinstance(payload, dict):
        for key in ("events", "predictions", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [e for e in value if isinstance(e, dict)]
    return []


def main():
    for sport, cfg in TARGETS.items():
        hist_dir = cfg["historical_dir"]
        files = sorted(hist_dir.glob("*.json")) if hist_dir.exists() else []
        grouped_rows = {}

        for fp in files:
            for event in _read_events(fp):
                for market_cfg in cfg["markets"]:
                    source_market = market_cfg["source"]
                    output_market = market_cfg["output"]
                    p = _event_prob_for_market(event, source_market)
                    h = _event_hit_for_market(event, source_market)
                    if p is None or h is None:
                        continue
                    grouped_rows.setdefault(output_market, []).append((float(p), int(bool(h))))

        payload = {sport: {}}

        for market_cfg in cfg["markets"]:
            output_market = market_cfg["output"]
            rows = grouped_rows.get(output_market, [])
            payload[sport][output_market] = _build_market_calibrator(rows)

        out_path = cfg["output"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] Wrote calibration params: {out_path}")
        for market_cfg in cfg["markets"]:
            output_market = market_cfg["output"]
            metrics = payload[sport][output_market].get("metrics", {})
            print(
                "   "
                + f"{sport}:{output_market} sample={metrics.get('sample_size', 0)} "
                + f"method={payload[sport][output_market].get('method')} "
                + f"brier_raw={metrics.get('brier_raw')} "
                + f"brier_iso={metrics.get('brier_isotonic')} "
                + f"brier_platt={metrics.get('brier_platt')}"
            )


if __name__ == "__main__":
    main()
