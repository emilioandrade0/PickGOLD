from __future__ import annotations

from datetime import datetime
import math
import json
from pathlib import Path

import joblib

SPORT_LABELS = {
    "nba": "NBA",
    "mlb": "MLB",
    "kbo": "KBO",
    "nhl": "NHL",
    "liga_mx": "Liga MX",
    "laliga": "LaLiga EA Sports",
    "euroleague": "EuroLeague",
    "ncaa_baseball": "NCAA Baseball",
}

# Temporary toggle: NCAA Baseball excluded from Best Picks output.
EXCLUDED_BEST_PICKS_SPORTS = {"ncaa_baseball"}

# Temporary freeze for sport/market combinations pending homogeneous re-audit.
FROZEN_SPORT_MARKETS = {
    "nhl": {"spread"},
    "liga_mx": {"full_game", "total", "btts"},
    "mlb": {"q1_yrfi"},
}

MARKET_DEFINITIONS = [
    {
        "market": "full_game",
        "display": "Full Game",
        "pick_keys": ["full_game_pick", "recommended_pick"],
        "tier_keys": ["full_game_tier"],
        "score_keys": [
            "full_game_recommended_score",
            "recommended_score",
            "full_game_confidence",
            "recommended_confidence",
        ],
    },
    {
        "market": "q1_yrfi",
        "display": "Q1 / YRFI",
        "pick_keys": ["q1_pick"],
        "score_keys": ["q1_recommended_score", "q1_confidence"],
        "action_keys": ["q1_action"],
    },
    {
        "market": "spread",
        "display": "Spread",
        "pick_keys": ["spread_pick"],
        "score_keys": ["spread_recommended_score", "spread_confidence"],
    },
    {
        "market": "total",
        "display": "Total",
        "pick_keys": ["total_recommended_pick", "total_pick"],
        "tier_keys": ["total_tier"],
        "score_keys": ["total_recommended_score", "total_confidence"],
    },
    {
        "market": "btts",
        "display": "BTTS",
        "pick_keys": ["btts_recommended_pick", "btts_pick"],
        "tier_keys": ["btts_tier"],
        "score_keys": ["btts_recommended_score", "btts_confidence"],
    },
    {
        "market": "f5",
        "display": "F5",
        "sports": ["mlb", "kbo", "ncaa_baseball"],
        "pick_keys": ["f5_pick", "assists_pick"],
        "tier_keys": ["extra_f5_tier"],
        "score_keys": ["extra_f5_recommended_score", "extra_f5_confidence"],
    },
    {
        "market": "home_over",
        "display": "Home Team Total",
        "pick_keys": ["home_over_pick"],
        "score_keys": ["home_over_recommended_score", "home_over_confidence"],
    },
    {
        "market": "corners",
        "display": "Corners O/U",
        "pick_keys": ["corners_recommended_pick", "corners_pick"],
        "score_keys": ["corners_recommended_score", "corners_confidence"],
    },
]

# Stability factors: lower means higher variance/risk profile for that market type.
MARKET_STABILITY_FACTORS = {
    "full_game": 0.97,
    "spread": 1.00,
    "total": 0.95,
    "home_over": 0.85,
    "corners": 0.84,
    "btts": 0.80,
    "f5": 0.82,
    "q1_yrfi": 0.80,
}

# Fallback odds when no bookmaker price is available in the event payload.
DEFAULT_MARKET_ODDS_AMERICAN = {
    "full_game": -115.0,
    "spread": -110.0,
    "total": -110.0,
    "home_over": -112.0,
    "corners": -110.0,
    "btts": -110.0,
    "f5": -110.0,
    "q1_yrfi": -110.0,
}

BASE_DIR = Path(__file__).resolve().parent.parent
META_MODEL_FILE = BASE_DIR / "data" / "insights" / "meta_model" / "best_picks_meta.pkl"
UNIFIED_AUDIT_FILE = BASE_DIR / "reports" / "unified_backtest_audit.json"

MIN_AUDIT_SAMPLE_SIZE = 200
MIN_CALIBRATION_BIN_COUNT = 40
MIN_CALIBRATION_BIN_COUNT_VALUE = 5
MIN_RELIABILITY_FACTOR = 0.74
MIN_CALIBRATION_BIN_COUNT_HIT_RATE = 1
MIN_RELIABILITY_FACTOR_HIT_RATE = 0.68
MIN_EDGE_PROXY = 0.010
MIN_EDGE_PROXY_HIT_RATE = -0.050
MIN_EV_PER_UNIT = 0.005
MIN_CORRELATION_PENALTY_STRICT = 0.84
MIN_CORRELATION_PENALTY_HIT_RATE = 0.80

_AUDIT_STATUS_CACHE = None


def _gate_increment(gate_stats: dict | None, key: str):
    if not isinstance(gate_stats, dict):
        return
    gate_stats[key] = int(gate_stats.get(key, 0) or 0) + 1


def _market_audit_key(sport: str, market: str):
    m = str(market or "")
    if m == "q1_yrfi":
        return "q1"
    if str(sport or "") == "nhl" and m == "spread":
        return "total_goals_55"
    return m


def _load_unified_audit_market_status():
    global _AUDIT_STATUS_CACHE
    if isinstance(_AUDIT_STATUS_CACHE, dict):
        return _AUDIT_STATUS_CACHE

    out = {}
    if not UNIFIED_AUDIT_FILE.exists():
        _AUDIT_STATUS_CACHE = out
        return out

    try:
        payload = json.loads(UNIFIED_AUDIT_FILE.read_text(encoding="utf-8"))
    except Exception:
        _AUDIT_STATUS_CACHE = out
        return out

    sports = payload.get("sports") if isinstance(payload, dict) else None
    if not isinstance(sports, dict):
        _AUDIT_STATUS_CACHE = out
        return out

    for sport, sport_payload in sports.items():
        markets = sport_payload.get("markets") if isinstance(sport_payload, dict) else None
        if not isinstance(markets, dict):
            continue
        for market, market_payload in markets.items():
            if not isinstance(market_payload, dict):
                continue
            try:
                picks = int(market_payload.get("picks", 0) or 0)
            except Exception:
                picks = 0
            suspicious = bool(market_payload.get("suspicious_high_accuracy", False))
            enabled = (picks >= MIN_AUDIT_SAMPLE_SIZE) and (not suspicious)
            out[(str(sport), str(market))] = {
                "enabled": enabled,
                "sample_size": picks,
                "suspicious": suspicious,
            }

    _AUDIT_STATUS_CACHE = out
    return out


def _audit_market_status(sport: str, market: str):
    status = _load_unified_audit_market_status().get((str(sport), _market_audit_key(sport, market)))
    if isinstance(status, dict):
        return status
    return {
        "enabled": False,
        "sample_size": 0,
        "suspicious": True,
    }


def _load_meta_model_payload():
    if not META_MODEL_FILE.exists():
        return None
    try:
        payload = joblib.load(META_MODEL_FILE)
        if isinstance(payload, dict) and payload.get("model") is not None:
            return payload
    except Exception:
        return None
    return None


def _meta_feature_row(pick: dict):
    return {
        "sport": str(pick.get("sport") or ""),
        "market": str(pick.get("market") or ""),
        "tier": str(pick.get("tier") or ""),
        "score": float(pick.get("score") or 0.0),
        "final_rank_score": float(pick.get("final_rank_score") or 0.0),
        "model_probability": float(pick.get("model_probability") or 0.0),
        "implied_probability_estimate": float(pick.get("implied_probability_estimate") or 0.0),
        "edge_proxy": float(pick.get("edge_proxy") or 0.0),
        "stability_factor": float(pick.get("stability_factor") or 0.0),
        "reliability_factor": float(pick.get("reliability_factor") or 0.0),
        "correlation_penalty": float(pick.get("correlation_penalty") or 1.0),
        "decimal_odds_used": float(pick.get("decimal_odds_used") or 0.0),
        "odds_is_fallback": int(bool(pick.get("odds_is_fallback"))),
    }


def _apply_meta_ranking(picks: list[dict]):
    payload = _load_meta_model_payload()
    if not payload:
        return False

    model = payload.get("model")
    vectorizer = payload.get("vectorizer")
    if model is None or vectorizer is None:
        return False

    try:
        feature_rows = [_meta_feature_row(p) for p in picks]
        X = vectorizer.transform(feature_rows)
        probs = model.predict_proba(X)[:, 1]
    except Exception:
        return False

    for idx, pick in enumerate(picks):
        p = float(probs[idx])
        meta_score = (
            p
            * float(pick.get("stability_factor", 0.9) or 0.9)
            * float(pick.get("reliability_factor", 1.0) or 1.0)
            * 100.0
        )
        pick["meta_profit_probability"] = round(p, 4)
        pick["risk_adjusted_rank_score"] = round(meta_score, 3)

    return True


def _clamp(value: float, low: float, high: float):
    return max(low, min(high, value))


def _first_non_empty(event: dict, keys: list[str]):
    for key in keys:
        if key not in event:
            continue
        val = event.get(key)
        if val is None:
            continue
        text = str(val).strip()
        if text and text.lower() != "nan":
            return text
    return None


def _first_float(event: dict, keys: list[str]):
    for key in keys:
        if key not in event:
            continue
        try:
            value = float(event.get(key))
            if value != value:
                continue
            return value
        except Exception:
            continue
    return None


def _event_status_allows_pick(event: dict) -> bool:
    # Some historical records may carry resolved result flags even when status fields are incomplete.
    result_keys = [
        "correct_full_game",
        "correct_full_game_adjusted",
        "correct_total",
        "correct_total_adjusted",
        "correct_btts",
        "correct_btts_adjusted",
        "correct_corners",
        "correct_corners_adjusted",
        "correct_corners_base",
        "correct_spread",
        "correct_q1",
        "correct_yrfi",
        "correct_f5",
        "correct_home_win_f5",
        "correct_home_over",
    ]
    for key in result_keys:
        if key not in event:
            continue
        raw = str(event.get(key, "")).strip().lower()
        if raw in {"true", "false", "1", "0", "acierto", "fallo", "win", "won", "lose", "lost"}:
            return False

    completed = event.get("status_completed")
    if completed is not None:
        try:
            if int(completed) == 1:
                return False
        except Exception:
            pass

    state = str(event.get("status_state", "") or "").strip().lower()
    if state in {"post", "final", "completed"}:
        return False
    return True


def _market_action_allows_pick(event: dict, action_keys: list[str] | None):
    if not action_keys:
        return True
    for key in action_keys:
        action = str(event.get(key, "") or "").strip().upper()
        if not action:
            continue
        if "PASS" in action or "PASAR" in action:
            return False
    return True


def _normalize_tier(raw_tier: str | None):
    if not raw_tier:
        return None
    tier = str(raw_tier).strip().upper()
    if not tier:
        return None
    if tier in {"ELITE", "PREMIUM", "STRONG", "NORMAL"}:
        return tier
    if tier == "PASS":
        return "NORMAL"
    return tier


def _tier_from_score(score: float):
    if score >= 80:
        return "ELITE"
    if score >= 65:
        return "PREMIUM"
    if score >= 55:
        return "STRONG"
    return "NORMAL"


def _tier_priority(tier: str | None):
    normalized = _normalize_tier(tier)
    if normalized == "ELITE":
        return 3
    if normalized == "PREMIUM":
        return 2
    if normalized == "STRONG":
        return 1
    return 0


def _tier_rank_multiplier(tier: str | None):
    priority = _tier_priority(tier)
    # Keep tier impact meaningful but bounded to avoid overpowering model quality.
    return 1.0 + (0.02 * float(priority))


def _resolve_tier(event: dict, market_def: dict, score: float):
    tier_keys = market_def.get("tier_keys") or []
    tier = _normalize_tier(_first_non_empty(event, tier_keys))
    if tier:
        return tier

    fallback = _normalize_tier(_first_non_empty(event, ["full_game_tier"]))
    if fallback:
        return fallback

    return _tier_from_score(score)


def _probability_from_event(event: dict, market: str, score: float):
    def _to_prob(raw):
        try:
            val = float(raw)
            if val != val:
                return None
            if 0.0 <= val <= 1.0:
                return val
            if 1.0 < val <= 100.0:
                return val / 100.0
        except Exception:
            return None
        return None

    def _pick_side_from_text(pick_text: str):
        p = str(pick_text or "").strip().upper()
        if not p:
            return None

        home_team = str(event.get("home_team") or "").strip().upper()
        away_team = str(event.get("away_team") or "").strip().upper()

        if p in {"HOME WIN", "HOME", "LOCAL", "1"}:
            return "home"
        if p in {"AWAY WIN", "AWAY", "VISITOR", "VISITANTE", "2"}:
            return "away"
        if home_team and home_team in p:
            return "home"
        if away_team and away_team in p:
            return "away"
        return None

    pick_text = ""
    if market == "full_game":
        pick_text = str(event.get("full_game_pick") or event.get("recommended_pick") or "")
    elif market == "f5":
        pick_text = str(event.get("f5_pick") or event.get("assists_pick") or "")
    elif market == "q1_yrfi":
        pick_text = str(event.get("q1_pick") or "")
    elif market == "total":
        pick_text = str(event.get("total_recommended_pick") or event.get("total_pick") or "")
    elif market == "btts":
        pick_text = str(event.get("btts_recommended_pick") or event.get("btts_pick") or "")
    elif market == "corners":
        pick_text = str(event.get("corners_recommended_pick") or event.get("corners_pick") or "")
    elif market == "home_over":
        pick_text = str(event.get("home_over_pick") or "")

    prob_keys = {
        "full_game": [
            "full_game_calibrated_prob_pick",
            "full_game_model_prob_pick",
            "full_game_calibrated_prob_home",
            "full_game_model_prob_home",
        ],
        "spread": ["spread_calibrated_prob_pick", "spread_model_prob_pick"],
        "total": [
            "total_adjusted_probability",
            "total_base_probability",
            "total_confidence",
            "total_recommended_score",
        ],
        "btts": [
            "btts_adjusted_probability",
            "btts_base_probability",
            "btts_confidence",
            "btts_recommended_score",
        ],
        "f5": ["extra_f5_calibrated_prob_home", "extra_f5_model_prob_home"],
        "home_over": [
            "home_over_calibrated_prob_pick",
            "home_over_model_prob_pick",
            "home_over_confidence",
            "home_over_recommended_score",
        ],
        "corners": [
            "corners_model_prob_over",
            "corners_confidence",
            "corners_recommended_score",
        ],
        "q1_yrfi": ["q1_calibrated_prob_yes", "q1_model_prob_yes", "q1_calibrated_prob_home", "q1_model_prob_home"],
    }

    for key in prob_keys.get(market, []):
        if key not in event:
            continue
        val = _to_prob(event.get(key))
        if val is None:
            continue

        key_lower = key.lower()

        if "_prob_pick" in key_lower:
            return {
                "probability": val,
                "pick_side_available": True,
                "probability_source": key,
            }

        if "_prob_home" in key_lower:
            side = _pick_side_from_text(pick_text)
            if side == "away":
                return {
                    "probability": _clamp(1.0 - val, 0.0, 1.0),
                    "pick_side_available": True,
                    "probability_source": key,
                }
            if side == "home":
                return {
                    "probability": val,
                    "pick_side_available": True,
                    "probability_source": key,
                }
            return {
                "probability": val,
                "pick_side_available": False,
                "probability_source": key,
            }

        if market == "q1_yrfi" and "_prob_yes" in key_lower:
            p = str(pick_text or "").strip().upper()
            if "NRFI" in p:
                return {
                    "probability": _clamp(1.0 - val, 0.0, 1.0),
                    "pick_side_available": True,
                    "probability_source": key,
                }
            return {
                "probability": val,
                "pick_side_available": True,
                "probability_source": key,
            }

        if market in {"total", "corners", "home_over"}:
            p = str(pick_text or "").strip().upper()
            if "UNDER" in p:
                return {
                    "probability": _clamp(1.0 - val, 0.0, 1.0),
                    "pick_side_available": True,
                    "probability_source": key,
                }
            return {
                "probability": val,
                "pick_side_available": True,
                "probability_source": key,
            }

        if market == "btts":
            p = str(pick_text or "").strip().upper()
            if "NO" in p:
                return {
                    "probability": _clamp(1.0 - val, 0.0, 1.0),
                    "pick_side_available": True,
                    "probability_source": key,
                }
            return {
                "probability": val,
                "pick_side_available": True,
                "probability_source": key,
            }

        return {
            "probability": val,
            "pick_side_available": True,
            "probability_source": key,
        }

    # Conservative fallback mapping from score to probability proxy.
    # 50 -> 0.50, 90 -> 0.70, 100 -> 0.75
    return {
        "probability": _clamp(0.5 + ((float(score) - 50.0) / 200.0), 0.40, 0.80),
        "pick_side_available": False,
        "probability_source": "score_proxy_fallback",
    }


def _american_odds_to_implied(value):
    try:
        odds = float(value)
    except Exception:
        return None

    if odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def _american_odds_to_decimal(value):
    try:
        odds = float(value)
    except Exception:
        return None

    if odds == 0:
        return None
    if odds > 0:
        return 1.0 + (odds / 100.0)
    return 1.0 + (100.0 / abs(odds))


def _decimal_odds_to_implied(value):
    try:
        odds = float(value)
    except Exception:
        return None
    if odds <= 1.0:
        return None
    return 1.0 / odds


def _decimal_odds_normalized(value):
    try:
        odds = float(value)
    except Exception:
        return None
    if odds <= 1.0:
        return None
    return odds


def _to_implied_probability(value):
    implied = _american_odds_to_implied(value)
    if implied is not None:
        return implied
    return _decimal_odds_to_implied(value)


def _to_decimal_odds(value):
    dec = _american_odds_to_decimal(value)
    if dec is not None:
        return dec
    return _decimal_odds_normalized(value)


def _extract_market_odds(event: dict, market: str):
    odds_keys = {
        "full_game": [
            "closing_moneyline_odds",
            "closing_ml_odds",
            "closing_odds_ml",
            "moneyline_odds",
            "ml_odds",
            "odds_ml",
            "odds_moneyline",
            "opening_moneyline_odds",
            "opening_ml_odds",
        ],
        "spread": [
            "closing_spread_odds",
            "spread_odds",
            "odds_spread_price",
            "opening_spread_odds",
        ],
        "total": [
            "closing_total_odds",
            "total_odds",
            "odds_total_price",
            "opening_total_odds",
        ],
        "q1_yrfi": [
            "closing_q1_odds",
            "closing_yrfi_odds",
            "q1_odds",
            "yrfi_odds",
            "nrfi_odds",
            "opening_q1_odds",
            "opening_yrfi_odds",
        ],
        "f5": [
            "closing_f5_odds",
            "f5_odds",
            "opening_f5_odds",
        ],
        "home_over": [
            "closing_home_over_odds",
            "home_over_odds",
            "opening_home_over_odds",
        ],
        "corners": [
            "closing_corners_odds",
            "corners_odds",
            "opening_corners_odds",
        ],
        "btts": [
            "closing_btts_odds",
            "btts_odds",
            "opening_btts_odds",
        ],
    }

    for key in odds_keys.get(market, []):
        if key not in event:
            continue
        implied = _to_implied_probability(event.get(key))
        decimal = _to_decimal_odds(event.get(key))
        if implied is not None and decimal is not None:
            return {
                "odds_value": event.get(key),
                "implied_probability": _clamp(float(implied), 0.02, 0.98),
                "decimal_odds": float(decimal),
                "odds_source": key,
            }

    default_american = DEFAULT_MARKET_ODDS_AMERICAN.get(market, -110.0)
    default_implied = _to_implied_probability(default_american)
    default_decimal = _to_decimal_odds(default_american)
    return {
        "odds_value": default_american,
        "implied_probability": _clamp(float(default_implied or 0.5238), 0.02, 0.98),
        "decimal_odds": float(default_decimal or 1.9091),
        "odds_source": "default_market_baseline",
    }


def _implied_probability_estimate(event: dict, market: str):
    odds_info = _extract_market_odds(event, market)
    return float(odds_info["implied_probability"])


def _teams_overlap(a: dict, b: dict):
    teams_a = {str(a.get("home_team", "")).upper(), str(a.get("away_team", "")).upper()}
    teams_b = {str(b.get("home_team", "")).upper(), str(b.get("away_team", "")).upper()}
    teams_a.discard("")
    teams_b.discard("")
    return bool(teams_a.intersection(teams_b))


def _correlation_penalty_dynamic(candidate: dict, selected_picks: list[dict]):
    penalty = 1.0
    for pick in selected_picks:
        if pick.get("sport") == candidate.get("sport"):
            penalty *= 0.97
        if pick.get("market") == candidate.get("market"):
            penalty *= 0.98
        if _teams_overlap(pick, candidate):
            penalty *= 0.94

        pick_token_a = str(pick.get("pick", "")).strip().upper()
        pick_token_b = str(candidate.get("pick", "")).strip().upper()
        if pick_token_a and pick_token_a == pick_token_b:
            penalty *= 0.95

    return _clamp(penalty, 0.75, 1.0)


def _rank_metrics(event: dict, market: str, raw_score: float):
    score = float(raw_score)
    confidence = _clamp(score / 100.0, 0.0, 1.0)
    prob_info = _probability_from_event(event, market, score)
    raw_model_prob = float(prob_info.get("probability", 0.5))
    implied_prob = _implied_probability_estimate(event, market)
    odds_info = _extract_market_odds(event, market)
    decimal_odds = float(odds_info["decimal_odds"])
    using_fallback_odds = str(odds_info.get("odds_source") or "") == "default_market_baseline"
    odds_quality = str(event.get("odds_data_quality") or "").strip().lower()
    has_real_odds = (not using_fallback_odds) and (odds_quality == "real")

    # Moderate shrinkage keeps edge realistic but with useful variance.
    model_prob = implied_prob + (raw_model_prob - implied_prob) * 0.70
    # Compress extremes so very high model probabilities keep separation but avoid saturation.
    model_prob = 0.5 + (model_prob - 0.5) * 0.60
    model_prob = _clamp(model_prob, 0.40, 0.80)
    edge_raw = model_prob - implied_prob
    # Soft cap preserves ranking variance better than hard clipping.
    edge = 0.18 * math.tanh(edge_raw / 0.18)
    edge = _clamp(edge, -0.08, 0.18)

    # Positive floor so edge contributes without collapsing rank diversity.
    edge_for_rank = max(0.02, edge)

    # EV per 1 unit stake using decimal odds.
    ev_per_unit = (model_prob * decimal_odds) - 1.0
    ev_for_rank = 0.0 if using_fallback_odds else _clamp(ev_per_unit, -0.08, 0.20)

    # Non-linear edge factor: prevents tiny edge from killing solid picks.
    base_rank = confidence * (1.0 + edge_for_rank + (max(0.0, ev_for_rank) * 0.8))
    stability = MARKET_STABILITY_FACTORS.get(market, 0.9)
    adjusted_rank = base_rank * stability

    return {
        "stability_factor": round(stability, 4),
        "model_probability": round(model_prob, 4),
        "probability_source": str(prob_info.get("probability_source") or "score_proxy_fallback"),
        "pick_side_probability_available": bool(prob_info.get("pick_side_available") is True),
        "implied_probability_estimate": round(implied_prob, 4),
        "edge_proxy": round(edge, 4),
        "odds_used": odds_info["odds_value"],
        "odds_source": odds_info["odds_source"],
        "odds_data_quality": odds_quality or "fallback",
        "has_real_odds": has_real_odds,
        "odds_is_fallback": using_fallback_odds,
        "decimal_odds_used": round(decimal_odds, 4),
        "expected_value_per_unit": round(ev_per_unit, 4),
        "base_rank_score": round(base_rank * 100.0, 3),
        "risk_adjusted_rank_score": round(adjusted_rank * 100.0, 3),
    }


def _reliability_from_profiles(
    sport: str,
    market: str,
    model_probability: float,
    calibration_profiles: dict | None,
):
    if not isinstance(calibration_profiles, dict):
        return {"reliability": 1.0, "bin_count": 0, "calibration_gap": 0.0}

    sport_profile = calibration_profiles.get(str(sport), {})
    if not isinstance(sport_profile, dict):
        return {"reliability": 1.0, "bin_count": 0, "calibration_gap": 0.0}

    market_profile = sport_profile.get(str(market), {})
    if not isinstance(market_profile, dict):
        return {"reliability": 1.0, "bin_count": 0, "calibration_gap": 0.0}

    bins = market_profile.get("bins", [])
    if not isinstance(bins, list) or not bins:
        return {"reliability": 1.0, "bin_count": 0, "calibration_gap": 0.0}

    p = _clamp(float(model_probability), 0.001, 0.999)
    chosen = None
    for b in bins:
        try:
            lo = float(b.get("min", 0.0))
            hi = float(b.get("max", 1.0))
            if (lo <= p < hi) or (hi >= 1.0 and p <= hi):
                chosen = b
                break
        except Exception:
            continue

    if not isinstance(chosen, dict):
        return {"reliability": 1.0, "bin_count": 0, "calibration_gap": 0.0}

    try:
        mean_pred = float(chosen.get("mean_pred", p))
        mean_hit = float(chosen.get("mean_hit", p))
        count = int(chosen.get("count", 0))
    except Exception:
        return {"reliability": 1.0, "bin_count": 0, "calibration_gap": 0.0}

    calibration_gap = abs(mean_pred - mean_hit)
    sample_factor = _clamp(count / 80.0, 0.0, 1.0)
    # Penalize poorly calibrated bins but avoid overreacting on small samples.
    reliability = 1.0 - (calibration_gap * (0.5 + 0.5 * sample_factor))
    return {
        "reliability": _clamp(reliability, 0.65, 1.0),
        "bin_count": count,
        "calibration_gap": round(calibration_gap, 4),
    }


def _build_pick_item(
    sport: str,
    event: dict,
    market_def: dict,
    calibration_profiles: dict | None = None,
    gate_stats: dict | None = None,
    ranking_mode: str = "balanced",
):
    _gate_increment(gate_stats, "candidates_considered")
    mode = str(ranking_mode or "balanced").strip().lower()
    require_market_audit = mode in {"balanced", "best_hit_rate", "meta"}

    market_name = str(market_def.get("market") or "")
    if market_name in FROZEN_SPORT_MARKETS.get(sport, set()):
        _gate_increment(gate_stats, "rejected_market_frozen")
        return None

    audit_status = _audit_market_status(sport, market_name)
    if require_market_audit and not bool(audit_status.get("enabled") is True):
        _gate_increment(gate_stats, "rejected_market_audit_disabled")
        return None
    if (not require_market_audit) and (not bool(audit_status.get("enabled") is True)):
        _gate_increment(gate_stats, "audit_relaxed_for_value_mode")

    pick_text = _first_non_empty(event, market_def["pick_keys"])
    if not pick_text:
        _gate_increment(gate_stats, "rejected_missing_pick")
        return None

    score = _first_float(event, market_def["score_keys"])
    if score is None:
        _gate_increment(gate_stats, "rejected_missing_score")
        return None

    if not _market_action_allows_pick(event, market_def.get("action_keys")):
        _gate_increment(gate_stats, "rejected_action_pass")
        return None

    allowed_sports = market_def.get("sports")
    if isinstance(allowed_sports, list) and allowed_sports and sport not in allowed_sports:
        _gate_increment(gate_stats, "rejected_market_not_allowed_for_sport")
        return None

    if market_def.get("market") == "f5" and "F5" not in str(pick_text).upper():
        _gate_increment(gate_stats, "rejected_f5_pick_format")
        return None

    tier = _resolve_tier(event, market_def, float(score))
    rank = _rank_metrics(event, market_name, float(score))
    reliability_info = _reliability_from_profiles(
        sport=sport,
        market=market_name,
        model_probability=rank["model_probability"],
        calibration_profiles=calibration_profiles,
    )
    reliability = float(reliability_info.get("reliability", 1.0))
    require_real_odds = mode in {"balanced", "best_ev_real_only", "meta"}
    require_min_ev = mode in {"balanced", "best_ev_real_only", "meta"}
    min_edge_required = MIN_EDGE_PROXY_HIT_RATE if mode == "best_hit_rate" else MIN_EDGE_PROXY
    min_bin_count_required = MIN_CALIBRATION_BIN_COUNT
    if mode == "best_hit_rate":
        min_bin_count_required = 0
    elif mode == "best_ev_real_only":
        min_bin_count_required = MIN_CALIBRATION_BIN_COUNT_VALUE
    min_reliability_required = (
        MIN_RELIABILITY_FACTOR_HIT_RATE
        if mode == "best_hit_rate"
        else MIN_RELIABILITY_FACTOR
    )

    if not bool(rank.get("pick_side_probability_available") is True):
        _gate_increment(gate_stats, "rejected_missing_pick_side_probability")
        return None
    if require_real_odds and not bool(rank.get("has_real_odds") is True):
        _gate_increment(gate_stats, "rejected_missing_real_odds")
        return None
    if float(rank.get("edge_proxy", 0.0) or 0.0) < float(min_edge_required):
        _gate_increment(gate_stats, "rejected_edge_below_min")
        return None
    if require_min_ev and float(rank.get("expected_value_per_unit", 0.0) or 0.0) < MIN_EV_PER_UNIT:
        _gate_increment(gate_stats, "rejected_ev_below_min")
        return None
    if float(reliability) < float(min_reliability_required):
        _gate_increment(gate_stats, "rejected_reliability_below_min")
        return None
    if int(min_bin_count_required) > 0 and int(reliability_info.get("bin_count", 0) or 0) < int(min_bin_count_required):
        _gate_increment(gate_stats, "rejected_calibration_bin_count_below_min")
        return None

    _gate_increment(gate_stats, "passed_all_entry_gates")

    tier_priority = _tier_priority(tier)
    tier_multiplier = _tier_rank_multiplier(tier)
    risk_adjusted_rank_score = round(
        float(rank["risk_adjusted_rank_score"]) * float(reliability) * float(tier_multiplier),
        3,
    )

    market_label = str(market_def["display"])
    if market_name == "spread":
        spread_market_text = str(event.get("spread_market") or "").strip().lower()
        spread_pick_text = str(event.get("spread_pick") or "").strip().lower()
        if (
            "total" in spread_market_text
            or "over" in spread_pick_text
            or "under" in spread_pick_text
        ):
            market_label = "Total Goals O/U"

    return {
        "sport": sport,
        "sport_label": SPORT_LABELS.get(sport, sport.upper()),
        "market": market_name,
        "market_label": market_label,
        "score": round(float(score), 2),
        "tier": tier,
        "tier_priority": tier_priority,
        "tier_rank_multiplier": round(float(tier_multiplier), 3),
        "stability_factor": rank["stability_factor"],
        "model_probability": rank["model_probability"],
        "implied_probability_estimate": rank["implied_probability_estimate"],
        "probability_source": rank["probability_source"],
        "pick_side_probability_available": rank["pick_side_probability_available"],
        "edge_proxy": rank["edge_proxy"],
        "odds_used": rank["odds_used"],
        "odds_source": rank["odds_source"],
        "odds_data_quality": rank["odds_data_quality"],
        "has_real_odds": rank["has_real_odds"],
        "decimal_odds_used": rank["decimal_odds_used"],
        "expected_value_per_unit": rank["expected_value_per_unit"],
        "base_rank_score": rank["base_rank_score"],
        "audit_sample_size": int(audit_status.get("sample_size", 0) or 0),
        "audit_suspicious": bool(audit_status.get("suspicious") is True),
        "calibration_bin_count": int(reliability_info.get("bin_count", 0) or 0),
        "calibration_gap": float(reliability_info.get("calibration_gap", 0.0) or 0.0),
        "reliability_factor": round(float(reliability), 4),
        "risk_adjusted_rank_score": risk_adjusted_rank_score,
        "correlation_penalty": 1.0,
        "final_rank_score": risk_adjusted_rank_score,
        "pick": pick_text,
        "game_id": str(event.get("game_id", "")),
        "date": str(event.get("date", "")),
        "time": str(event.get("time", "")),
        "game_name": str(event.get("game_name") or "").strip() or None,
        "home_team": str(event.get("home_team") or "").strip(),
        "away_team": str(event.get("away_team") or "").strip(),
        "status_state": str(event.get("status_state") or "").strip().lower() or None,
        "status_description": str(event.get("status_description") or "").strip() or None,
    }


def _portfolio_limits(top_n: int, ranking_mode: str, candidate_count: int):
    # Keep exposure tight to avoid league saturation.
    mode = str(ranking_mode or "balanced").strip().lower()

    if mode == "best_hit_rate" and int(candidate_count) <= 30:
        # When valid inventory is already low, relax concentration limits slightly
        # to avoid returning too few picks despite good candidates.
        return {"max_per_sport": min(5, max(1, int(top_n))), "max_same_market": 5}

    max_per_sport = min(3, max(1, int(top_n)))
    if top_n <= 12:
        return {"max_per_sport": max_per_sport, "max_same_market": 2}
    if top_n <= 20:
        return {"max_per_sport": max_per_sport, "max_same_market": 3}
    return {"max_per_sport": max_per_sport, "max_same_market": 4}


def _build_diversified_portfolio(
    candidates: list[dict],
    top_n: int,
    ranking_mode: str,
    min_correlation_penalty: float,
):
    limits = _portfolio_limits(top_n, ranking_mode, len(candidates))
    max_per_sport = limits["max_per_sport"]
    max_same_market = limits["max_same_market"]

    selected = []
    remaining = [dict(item) for item in candidates]

    while remaining and len(selected) < top_n:
        sport_counts = {}
        market_counts = {}
        for pick in selected:
            sport_counts[pick["sport"]] = sport_counts.get(pick["sport"], 0) + 1
            market_counts[pick["market"]] = market_counts.get(pick["market"], 0) + 1

        best_index = None
        best_score = None
        best_penalty = None
        best_priority = None

        for idx, cand in enumerate(remaining):
            if sport_counts.get(cand["sport"], 0) >= max_per_sport:
                continue
            if market_counts.get(cand["market"], 0) >= max_same_market:
                continue

            penalty = _correlation_penalty_dynamic(cand, selected)
            if penalty < float(min_correlation_penalty):
                continue
            final_score = float(cand["risk_adjusted_rank_score"]) * penalty
            tier_priority = int(cand.get("tier_priority", 0) or 0)

            if (
                best_score is None
                or tier_priority > int(best_priority or 0)
                or (tier_priority == int(best_priority or 0) and final_score > float(best_score or 0.0))
            ):
                best_index = idx
                best_score = final_score
                best_penalty = penalty
                best_priority = tier_priority

        if best_index is None:
            break

        chosen = remaining.pop(best_index)
        chosen["correlation_penalty"] = round(float(best_penalty), 4)
        chosen["final_rank_score"] = round(float(best_score), 3)
        selected.append(chosen)

    return selected, limits


def build_daily_best_picks(
    events_by_sport: dict[str, list],
    top_n: int = 25,
    calibration_profiles: dict | None = None,
    include_completed: bool = False,
    ranking_mode: str = "balanced",
):
    requested_mode = str(ranking_mode or "balanced").strip().lower()
    if requested_mode not in {"balanced", "best_hit_rate", "best_ev_real_only", "meta"}:
        requested_mode = "balanced"

    picks = []
    gate_stats = {
        "events_seen": 0,
        "events_status_filtered": 0,
        "candidates_considered": 0,
    }

    for sport, events in events_by_sport.items():
        if sport in EXCLUDED_BEST_PICKS_SPORTS:
            continue
        for event in events or []:
            _gate_increment(gate_stats, "events_seen")
            if not isinstance(event, dict):
                _gate_increment(gate_stats, "events_non_dict_skipped")
                continue
            if (not include_completed) and (not _event_status_allows_pick(event)):
                _gate_increment(gate_stats, "events_status_filtered")
                continue

            for market_def in MARKET_DEFINITIONS:
                pick_item = _build_pick_item(
                    sport,
                    event,
                    market_def,
                    calibration_profiles=calibration_profiles,
                    gate_stats=gate_stats,
                    ranking_mode=requested_mode,
                )
                if pick_item is not None:
                    picks.append(pick_item)

    if requested_mode == "best_ev_real_only":
        before = len(picks)
        picks = [p for p in picks if not bool(p.get("odds_is_fallback"))]
        gate_stats["post_mode_removed_fallback_odds"] = int(before - len(picks))

    if requested_mode == "best_hit_rate":
        for pick in picks:
            hit_rate_score = (
                float(pick.get("model_probability", 0.5))
                * float(pick.get("reliability_factor", 1.0))
                * float(pick.get("stability_factor", 0.9))
                * float(pick.get("tier_rank_multiplier", 1.0))
                * 100.0
            )
            pick["risk_adjusted_rank_score"] = round(hit_rate_score, 3)

    if requested_mode == "meta":
        loaded = _apply_meta_ranking(picks)
        if not loaded:
            for pick in picks:
                fallback_score = (
                    float(pick.get("model_probability", 0.5))
                    * float(pick.get("reliability_factor", 1.0))
                    * float(pick.get("stability_factor", 0.9))
                    * float(pick.get("tier_rank_multiplier", 1.0))
                    * 100.0
                )
                pick["risk_adjusted_rank_score"] = round(fallback_score, 3)

    picks.sort(
        key=lambda x: (
            int(x.get("tier_priority", 0) or 0),
            float(x.get("risk_adjusted_rank_score", 0.0) or 0.0),
            float(x.get("score", 0.0) or 0.0),
        ),
        reverse=True,
    )

    min_correlation_penalty = (
        MIN_CORRELATION_PENALTY_HIT_RATE
        if requested_mode == "best_hit_rate"
        else MIN_CORRELATION_PENALTY_STRICT
    )

    trimmed, limits = _build_diversified_portfolio(
        picks,
        max(1, int(top_n)),
        ranking_mode=requested_mode,
        min_correlation_penalty=min_correlation_penalty,
    )

    by_sport = {}
    for p in trimmed:
        stats = by_sport.setdefault(
            p["sport"],
            {
                "sport": p["sport"],
                "label": p["sport_label"],
                "count": 0,
                "avg_score": 0.0,
                "avg_final_rank_score": 0.0,
                "avg_expected_value_per_unit": 0.0,
                "max_score": 0.0,
                "max_final_rank_score": 0.0,
                "max_expected_value_per_unit": -999.0,
            },
        )
        stats["count"] += 1
        stats["avg_score"] += p["score"]
        stats["avg_final_rank_score"] += p["final_rank_score"]
        stats["avg_expected_value_per_unit"] += float(p.get("expected_value_per_unit", 0.0) or 0.0)
        stats["max_score"] = max(stats["max_score"], p["score"])
        stats["max_final_rank_score"] = max(stats["max_final_rank_score"], p["final_rank_score"])
        stats["max_expected_value_per_unit"] = max(
            stats["max_expected_value_per_unit"],
            float(p.get("expected_value_per_unit", 0.0) or 0.0),
        )

    for stats in by_sport.values():
        if stats["count"]:
            stats["avg_score"] = round(stats["avg_score"] / stats["count"], 2)
            stats["avg_final_rank_score"] = round(stats["avg_final_rank_score"] / stats["count"], 3)
            stats["avg_expected_value_per_unit"] = round(
                stats["avg_expected_value_per_unit"] / stats["count"],
                4,
            )
            stats["max_expected_value_per_unit"] = round(stats["max_expected_value_per_unit"], 4)

    sports_summary = sorted(
        by_sport.values(),
        key=lambda x: (x["max_final_rank_score"], x["avg_final_rank_score"], x["max_score"]),
        reverse=True,
    )

    return {
        "generated_at": datetime.now().isoformat(),
        "top_n": int(top_n),
        "total_candidates": len(picks),
        "ranking_model": f"v6_{requested_mode}_pick_side_prob_safe_ev",
        "ranking_mode": requested_mode,
        "entry_gates": {
            "market_audit_enabled_required": requested_mode in {"balanced", "best_hit_rate", "meta"},
            "pick_side_probability_required": True,
            "real_odds_required": requested_mode in {"balanced", "best_ev_real_only", "meta"},
            "min_edge_proxy": MIN_EDGE_PROXY_HIT_RATE if requested_mode == "best_hit_rate" else MIN_EDGE_PROXY,
            "min_ev_per_unit": MIN_EV_PER_UNIT if requested_mode in {"balanced", "best_ev_real_only", "meta"} else None,
            "min_calibration_bin_count": (
                None
                if requested_mode == "best_hit_rate"
                else (
                    MIN_CALIBRATION_BIN_COUNT_VALUE
                    if requested_mode == "best_ev_real_only"
                    else MIN_CALIBRATION_BIN_COUNT
                )
            ),
            "min_reliability_factor": (
                MIN_RELIABILITY_FACTOR_HIT_RATE
                if requested_mode == "best_hit_rate"
                else MIN_RELIABILITY_FACTOR
            ),
            "min_correlation_penalty": min_correlation_penalty,
        },
        "gate_diagnostics": gate_stats,
        "portfolio_limits": limits,
        "sports_summary": sports_summary,
        "picks": trimmed,
    }
