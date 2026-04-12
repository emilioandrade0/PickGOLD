"""
Predict today's upcoming NHL games using 4-model ensemble.
Loads real data from nhl_upcoming_schedule.csv and builds features from historical data.
Mirrors the structure of MLB's predict_today_mlb.py
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import sys
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import joblib
import numpy as np
import pandas as pd
from calibration import calibrate_probability, load_calibration_config
from pattern_engine import aggregate_pattern_edge
from pick_selector import fuse_with_pattern_score, recommendation_score

try:
    from .pattern_engine_nhl import generate_nhl_patterns
except ImportError:
    from sports.nhl.pattern_engine_nhl import generate_nhl_patterns

BASE_DIR = SRC_ROOT
HISTORY_FILE = BASE_DIR / "data" / "nhl" / "processed" / "model_ready_features_nhl.csv"
RAW_HISTORY_FILE = BASE_DIR / "data" / "nhl" / "raw" / "nhl_advanced_history.csv"
UPCOMING_FILE = BASE_DIR / "data" / "nhl" / "raw" / "nhl_upcoming_schedule.csv"
MODELS_DIR = BASE_DIR / "data" / "nhl" / "models"
PREDICTIONS_DIR = BASE_DIR / "data" / "nhl" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_FILE = MODELS_DIR / "calibration_params.json"

# Threshold behavior for live binary markets:
# - trained: use threshold from each market metadata.json
# - fixed: use values in LIVE_FIXED_THRESHOLDS
NHL_LIVE_THRESHOLD_MODE = os.getenv("NHL_LIVE_THRESHOLD_MODE", "trained").strip().lower()
LIVE_FIXED_THRESHOLDS = {
    "spread_2_5": 0.5,
    "home_over_2_5": 0.5,
}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, str(default))).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


# Conservative sweep knobs for NHL full-game meta scoring/publication.
FULL_GAME_META_CFG = {
    "spread_prob_high": _env_float("NHL_FG_SPREAD_PROB_HIGH", 0.58),
    "spread_prob_low": _env_float("NHL_FG_SPREAD_PROB_LOW", 0.42),
    "spread_aligned_bonus": _env_float("NHL_FG_SPREAD_ALIGNED_BONUS", 0.016),
    "spread_conflicted_penalty": _env_float("NHL_FG_SPREAD_CONFLICTED_PENALTY", -0.018),
    "market_align_home_threshold": _env_float("NHL_FG_MARKET_ALIGN_HOME_THRESHOLD", 0.53),
    "market_align_away_threshold": _env_float("NHL_FG_MARKET_ALIGN_AWAY_THRESHOLD", 0.47),
    "market_min_gap_for_signal": _env_float("NHL_FG_MARKET_MIN_GAP_FOR_SIGNAL", 0.035),
    "market_aligned_bonus": _env_float("NHL_FG_MARKET_ALIGNED_BONUS", 0.004),
    "market_conflicted_penalty": _env_float("NHL_FG_MARKET_CONFLICTED_PENALTY", -0.028),
    "market_edge_bonus_scale": _env_float("NHL_FG_MARKET_EDGE_BONUS_SCALE", 0.040),
    "meta_score_floor": _env_float("NHL_FG_META_SCORE_FLOOR", 0.50),
    "meta_score_ceiling": _env_float("NHL_FG_META_SCORE_CEILING", 0.78),
}

FULL_GAME_BUCKET_THRESHOLDS = {
    "elite": _env_float("NHL_FG_BUCKET_ELITE_MIN", 0.67),
    "strong": _env_float("NHL_FG_BUCKET_STRONG_MIN", 0.60),
    "normal": _env_float("NHL_FG_BUCKET_NORMAL_MIN", 0.55),
}

NHL_FULL_GAME_PUBLISH_RULE = os.getenv("NHL_FULL_GAME_PUBLISH_RULE", "elite_or_strong").strip().lower()
NHL_FULL_GAME_PUBLISH_MIN_META_CONF = _env_int("NHL_FULL_GAME_PUBLISH_MIN_META_CONF", 62)
NHL_FULL_GAME_PUBLISH_EXCLUDE_CONFLICTED = _env_bool("NHL_FULL_GAME_PUBLISH_EXCLUDE_CONFLICTED", True)

NHL_FULL_GAME_ACTIVE_VARIANT = os.getenv("NHL_FULL_GAME_ACTIVE_VARIANT", "base").strip().lower()
FULL_GAME_V2_MODEL_FILE = MODELS_DIR / "full_game" / "meta_model_full_game_v2.pkl"
FULL_GAME_V2_METADATA_FILE = MODELS_DIR / "full_game" / "meta_model_full_game_v2_metadata.json"

NHL_FULL_GAME_HYBRID_V2_BUCKETS = {
    x.strip().upper()
    for x in os.getenv("NHL_FULL_GAME_HYBRID_V2_BUCKETS", "ELITE,STRONG,NORMAL").split(",")
    if x.strip()
}
NHL_FULL_GAME_HYBRID_BLOCK_CONFLICTED = _env_bool("NHL_FULL_GAME_HYBRID_BLOCK_CONFLICTED", False)

NON_FEATURE_COLUMNS = {
    "game_id", "date", "date_dt", "time", "season", "home_team", "away_team",
    "home_score", "away_score", "total_goals", "is_draw", "completed",
    "home_p1_goals", "away_p1_goals", "total_p1_goals",
    "venue_name", "odds_details", "odds_over_under",
    "TARGET_full_game", "TARGET_over_55", "TARGET_home_over_25",
    "TARGET_spread_1_5", "TARGET_p1_over_15",
}


def load_history() -> pd.DataFrame:
    """Load historical features for building snapshots."""
    if not HISTORY_FILE.exists():
        raise FileNotFoundError(f"History file not found: {HISTORY_FILE}")

    df = pd.read_csv(HISTORY_FILE, dtype={"game_id": str})
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df["date"] = df["date_dt"].astype(str)

    # Feature engineering NHL currently leaves team labels corrupted in the processed file.
    # Recover them from raw history so live pregame snapshots can match upcoming abbreviations.
    if RAW_HISTORY_FILE.exists():
        raw_df = pd.read_csv(RAW_HISTORY_FILE, dtype={"game_id": str})
        if {"game_id", "home_team", "away_team"}.issubset(raw_df.columns):
            raw_map = raw_df[["game_id", "home_team", "away_team"]].drop_duplicates("game_id", keep="last")
            df = df.drop(columns=["home_team", "away_team"], errors="ignore")
            df = df.merge(raw_map, on="game_id", how="left")

    return df.sort_values(["date", "game_id"]).reset_index(drop=True)


def get_team_snapshot(history_df: pd.DataFrame, team: str, cutoff_date: str) -> Optional[Dict]:
    """Get the last game features snapshot for a team before cutoff_date."""
    prior = history_df[history_df["date"] < cutoff_date].copy()
    if prior.empty:
        return None

    home_rows = prior[prior["home_team"] == team]
    away_rows = prior[prior["away_team"] == team]

    last_home = home_rows.iloc[-1] if not home_rows.empty else None
    last_away = away_rows.iloc[-1] if not away_rows.empty else None

    last_game = None
    if last_home is not None and last_away is not None:
        last_game = last_home if last_home["date"] >= last_away["date"] else last_away
    elif last_home is not None:
        last_game = last_home
    elif last_away is not None:
        last_game = last_away
    else:
        return None

    prefix = "home_" if last_game["home_team"] == team else "away_"

    snapshot = {}
    for col in history_df.columns:
        if col.startswith(prefix) and col not in NON_FEATURE_COLUMNS:
            feature_name = col.replace(prefix, "")
            snapshot[feature_name] = last_game[col]

    return snapshot


def build_pregame_features(history_df: pd.DataFrame, upcoming_game: Dict) -> Optional[Dict]:
    """Build feature vector for an upcoming game."""
    cutoff_date = upcoming_game.get("date")
    home_team = upcoming_game.get("home_team")
    away_team = upcoming_game.get("away_team")

    if not cutoff_date or not home_team or not away_team:
        return None

    home_snap = get_team_snapshot(history_df, home_team, cutoff_date)
    away_snap = get_team_snapshot(history_df, away_team, cutoff_date)

    if not home_snap or not away_snap:
        return None

    features = {}
    for key, val in home_snap.items():
        features[f"home_{key}"] = val
    for key, val in away_snap.items():
        features[f"away_{key}"] = val

    return features


def load_model_assets(market: str) -> Dict:
    """Load trained models and metadata for a market."""
    market_dir = MODELS_DIR / market

    if not market_dir.exists():
        raise FileNotFoundError(f"Models not found for {market}")

    try:
        assets = {
            "xgb": joblib.load(market_dir / "xgboost_model.pkl"),
            "lgbm": joblib.load(market_dir / "lgbm_model.pkl"),
            "lgbm_secondary": joblib.load(market_dir / "lgbm_secondary_model.pkl"),
            "catboost": joblib.load(market_dir / "catboost_model.pkl"),
        }

        with open(market_dir / "metadata.json", "r") as f:
            assets["metadata"] = json.load(f)

        return assets
    except Exception as e:
        raise FileNotFoundError(f"Could not load models for {market}: {e}")


def predict_market(assets: Dict, features_dict: Dict, market: str = None) -> Tuple[int, float, np.ndarray]:
    """Generate 4-model ensemble predictions for a single game."""
    metadata = assets["metadata"]
    feature_names = metadata["feature_columns"]

    X = pd.DataFrame(
        [{f: features_dict.get(f, 0.0) for f in feature_names}],
        columns=feature_names,
    )

    xgb_probs = assets["xgb"].predict_proba(X)
    lgbm_probs = assets["lgbm"].predict_proba(X)
    lgbm_sec_probs = assets["lgbm_secondary"].predict_proba(X)
    catboost_probs = assets["catboost"].predict_proba(X)

    ensemble_probs = (xgb_probs + lgbm_probs + lgbm_sec_probs + catboost_probs) / 4.0

    problem_type = metadata["problem_type"]
    threshold = metadata.get("threshold", 0.5)

    if problem_type != "multiclass" and market:
        if NHL_LIVE_THRESHOLD_MODE == "fixed":
            threshold = float(LIVE_FIXED_THRESHOLDS.get(market, threshold))
        else:
            threshold = float(threshold)

    if problem_type == "multiclass":
        preds = np.argmax(ensemble_probs, axis=1)
        confidences = np.max(ensemble_probs, axis=1)
    else:
        if threshold:
            preds = (ensemble_probs[:, 1] >= threshold).astype(int)
        else:
            preds = np.argmax(ensemble_probs, axis=1)
        confidences = np.abs(ensemble_probs[:, 1] - 0.5) + 0.5

    if market == "full_game":
        preds = preds + 1

    return int(preds[0]), float(confidences[0]), ensemble_probs[0]


def derive_nhl_first_period_pick(prob_over_55: float) -> Dict:
    """Derive 1P O/U 1.5 prediction from full-game totals probability.

    This is a calibrated proxy while first-period training labels are unavailable.
    """
    p = float(np.clip(prob_over_55, 0.01, 0.99))

    expected_total_goals = 5.5 + 1.4 * (p - 0.5)
    expected_total_goals = float(np.clip(expected_total_goals, 4.6, 6.4))
    lambda_p1 = expected_total_goals * 0.30

    p_over_15 = 1.0 - float(np.exp(-lambda_p1) * (1.0 + lambda_p1))
    p_over_15 = float(np.clip(p_over_15, 0.01, 0.99))

    pick_over = p_over_15 >= 0.53
    pick = "Over 1.5" if pick_over else "Under 1.5"
    confidence = int((0.5 + abs(p_over_15 - 0.5)) * 100)
    action = "JUGAR" if confidence >= 56 else "PASS"

    return {
        "q1_pick": pick,
        "q1_market": "1P Goals O/U 1.5",
        "q1_line": 1.5,
        "q1_confidence": confidence,
        "q1_action": action,
        "q1_model_prob_yes": round(p_over_15, 4),
        "q1_calibrated_prob_yes": round(p_over_15, 4),
    }


def _resolve_nhl_total_line(raw_value) -> float:
    try:
        line = float(raw_value)
        if line > 0:
            return float(round(line * 2) / 2.0)
    except Exception:
        pass
    return 5.5


def _goal_line_label(line: float) -> str:
    return str(int(line)) if float(line).is_integer() else f"{line:.1f}"


def _pick_side_from_text(pick_text: str, home_team: str, away_team: str) -> Optional[str]:
    text = str(pick_text or "").strip().lower()
    if not text:
        return None

    home = str(home_team or "").strip().lower()
    away = str(away_team or "").strip().lower()

    if "home win" in text:
        return "home"
    if "away win" in text:
        return "away"
    if text.startswith(home):
        return "home"
    if text.startswith(away):
        return "away"
    if home and home in text:
        return "home"
    if away and away in text:
        return "away"
    return None


def _safe_prob(value) -> Optional[float]:
    """Convert confidence-like values into [0, 1] probability space."""
    try:
        prob = float(value)
        if np.isnan(prob):
            return None
        if prob > 1.0:
            prob = prob / 100.0
        return float(np.clip(prob, 0.0, 1.0))
    except Exception:
        return None


def _resolve_pick_probability(game_dict: Dict, prob_key: str, conf_key: str) -> Optional[float]:
    """Resolve side probability from explicit probability field, then fallback to confidence."""
    prob = _safe_prob(game_dict.get(prob_key))
    if prob is not None:
        return prob
    return _safe_prob(game_dict.get(conf_key))


def _resolve_pick_probability_with_source(game_dict: Dict, prob_key: str, conf_key: str) -> Tuple[Optional[float], str]:
    """Resolve side probability and annotate whether fallback was used."""
    prob = _safe_prob(game_dict.get(prob_key))
    if prob is not None:
        return prob, prob_key

    conf_prob = _safe_prob(game_dict.get(conf_key))
    if conf_prob is not None:
        return conf_prob, f"{conf_key}_fallback"

    return None, "missing"


def _american_to_implied_prob(odds_value) -> Optional[float]:
    """Convert American odds to implied probability in [0, 1]."""
    try:
        odds = float(odds_value)
    except Exception:
        return None

    if np.isnan(odds) or odds == 0:
        return None

    if odds > 0:
        implied = 100.0 / (odds + 100.0)
    else:
        implied = abs(odds) / (abs(odds) + 100.0)
    return float(np.clip(implied, 0.0, 1.0))


def _resolve_market_home_prob_no_vig(game_dict: Dict) -> Tuple[Optional[float], str]:
    """Resolve market home win probability (prefer no-vig), with defensive fallbacks."""
    for key in [
        "ml_implied_home_prob_no_vig",
        "moneyline_prob_home",
        "ml_implied_home_prob",
    ]:
        value = _safe_prob(game_dict.get(key))
        if value is not None:
            return value, key

    home_ml = _american_to_implied_prob(game_dict.get("home_moneyline_odds"))
    away_ml = _american_to_implied_prob(game_dict.get("away_moneyline_odds"))
    if home_ml is not None and away_ml is not None and (home_ml + away_ml) > 0:
        no_vig_home = home_ml / (home_ml + away_ml)
        return float(np.clip(no_vig_home, 0.0, 1.0)), "home_away_moneyline_odds_no_vig"

    closing_ml = game_dict.get("closing_moneyline_odds")
    if isinstance(closing_ml, dict):
        home_val = closing_ml.get("home")
        away_val = closing_ml.get("away")
        home_close = _american_to_implied_prob(home_val)
        away_close = _american_to_implied_prob(away_val)
        if home_close is not None and away_close is not None and (home_close + away_close) > 0:
            no_vig_home = home_close / (home_close + away_close)
            return float(np.clip(no_vig_home, 0.0, 1.0)), "closing_moneyline_odds_dict_no_vig"

    return None, "missing"


def _resolve_spread_prob_home(game_dict: Dict, home_team: str, away_team: str) -> Tuple[Optional[float], str]:
    """Resolve implied probability that home covers -1.5 from spread fields."""
    spread_pick_prob, spread_prob_source = _resolve_pick_probability_with_source(
        game_dict,
        prob_key="spread_calibrated_prob_pick",
        conf_key="spread_confidence",
    )
    if spread_pick_prob is None:
        return None, "missing"

    spread_pick = game_dict.get("spread_pick")
    spread_side = _pick_side_from_text(spread_pick, home_team=home_team, away_team=away_team)
    if spread_side == "home":
        return float(np.clip(spread_pick_prob, 0.0, 1.0)), f"{spread_prob_source}:home_pick"
    if spread_side == "away":
        return float(np.clip(1.0 - spread_pick_prob, 0.0, 1.0)), f"{spread_prob_source}:away_pick_inverted"

    return None, f"{spread_prob_source}:side_unresolved"


def _meta_bucket_from_score(meta_score: float) -> str:
    score = float(np.clip(meta_score, 0.0, 1.0))
    if score >= FULL_GAME_BUCKET_THRESHOLDS["elite"]:
        return "ELITE"
    if score >= FULL_GAME_BUCKET_THRESHOLDS["strong"]:
        return "STRONG"
    if score >= FULL_GAME_BUCKET_THRESHOLDS["normal"]:
        return "NORMAL"
    return "PASS"


def _normalize_meta_bucket(bucket: Optional[str]) -> str:
    normalized = str(bucket or "PASS").strip().upper()
    if normalized == "LOW":
        return "PASS"
    if normalized in {"ELITE", "STRONG", "NORMAL", "PASS"}:
        return normalized
    return "PASS"


def _shift_meta_bucket(bucket: str, steps: int) -> str:
    order = ["PASS", "NORMAL", "STRONG", "ELITE"]
    normalized = _normalize_meta_bucket(bucket)
    idx = order.index(normalized)
    new_idx = int(np.clip(idx + steps, 0, len(order) - 1))
    return order[new_idx]


def _safe_boolish(value) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    txt = str(value).strip().lower()
    if txt in {"1", "true", "yes", "si", "y"}:
        return True
    if txt in {"0", "false", "no", "n"}:
        return False
    return None


def _resolve_market_gap(game_dict: Dict, market_home_prob: Optional[float]) -> Tuple[Optional[float], str]:
    gap = _safe_prob(game_dict.get("ml_prob_gap_no_vig"))
    if gap is not None:
        return float(np.clip(gap, 0.0, 1.0)), "ml_prob_gap_no_vig"

    away_prob = _safe_prob(game_dict.get("ml_implied_away_prob_no_vig"))
    if market_home_prob is not None and away_prob is not None:
        return float(np.clip(abs(market_home_prob - away_prob), 0.0, 1.0)), "home_away_no_vig_gap"

    if market_home_prob is not None:
        return float(np.clip(abs(market_home_prob - 0.5) * 2.0, 0.0, 1.0)), "home_prob_distance"

    return None, "missing"


def _resolve_market_ml_alignment(
    game_dict: Dict,
    full_side: Optional[str],
    market_home_prob: Optional[float],
    market_gap: Optional[float],
) -> Tuple[str, str]:
    if full_side not in {"home", "away"}:
        return "neutral", "full_side_unresolved"

    min_gap = FULL_GAME_META_CFG["market_min_gap_for_signal"]
    if market_gap is not None and market_gap < min_gap:
        return "neutral", f"market_gap_ambiguous<{min_gap:.3f}"

    home_align = FULL_GAME_META_CFG["market_align_home_threshold"]
    away_align = FULL_GAME_META_CFG["market_align_away_threshold"]
    if market_home_prob is not None:
        if full_side == "home":
            if market_home_prob >= home_align:
                return "aligned", f"home_prob={market_home_prob:.4f}>={home_align:.2f}"
            if market_home_prob <= away_align:
                return "conflicted", f"home_prob={market_home_prob:.4f}<={away_align:.2f}"
        if full_side == "away":
            if market_home_prob <= away_align:
                return "aligned", f"home_prob={market_home_prob:.4f}<={away_align:.2f}"
            if market_home_prob >= home_align:
                return "conflicted", f"home_prob={market_home_prob:.4f}>={home_align:.2f}"

    market_fav_home = _safe_boolish(game_dict.get("ml_home_is_favorite_market"))
    if market_fav_home is not None:
        if (full_side == "home" and market_fav_home) or (full_side == "away" and not market_fav_home):
            return "aligned", "favorite_flag_support"
        return "conflicted", "favorite_flag_conflict"

    return "neutral", "market_signal_missing"


def _should_publish_full_game(meta_bucket: str, market_ml_alignment: str, meta_confidence: int) -> Tuple[bool, str]:
    bucket = _normalize_meta_bucket(meta_bucket)
    alignment = str(market_ml_alignment or "neutral").strip().lower()

    if alignment == "conflicted" and NHL_FULL_GAME_PUBLISH_EXCLUDE_CONFLICTED:
        return False, "blocked_market_conflicted"

    if NHL_FULL_GAME_PUBLISH_RULE == "elite_only":
        publish = bucket == "ELITE"
        return publish, f"rule=elite_only bucket={bucket}"

    if NHL_FULL_GAME_PUBLISH_RULE == "elite_or_strong_min_conf":
        bucket_ok = bucket in {"ELITE", "STRONG"}
        conf_ok = int(meta_confidence) >= int(NHL_FULL_GAME_PUBLISH_MIN_META_CONF)
        publish = bucket_ok and conf_ok
        return publish, (
            f"rule=elite_or_strong_min_conf bucket={bucket} conf={int(meta_confidence)} "
            f"min_conf={int(NHL_FULL_GAME_PUBLISH_MIN_META_CONF)}"
        )

    publish = bucket in {"ELITE", "STRONG"}
    return publish, f"rule=elite_or_strong bucket={bucket}"


def _load_full_game_v2_meta_assets() -> Optional[Dict]:
    if not FULL_GAME_V2_MODEL_FILE.exists() or not FULL_GAME_V2_METADATA_FILE.exists():
        return None
    try:
        model = joblib.load(FULL_GAME_V2_MODEL_FILE)
        with open(FULL_GAME_V2_METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        threshold = float(metadata.get("threshold", 0.5))
        return {"model": model, "threshold": threshold, "metadata": metadata}
    except Exception:
        return None


def _build_full_game_v2_meta_matrix(
    full_home_prob: float,
    spread_home_cover_prob: Optional[float],
    market_home_prob: Optional[float],
    market_gap: Optional[float],
) -> np.ndarray:
    base_p = float(np.clip(full_home_prob, 1e-6, 1 - 1e-6))
    spread_p = float(spread_home_cover_prob) if spread_home_cover_prob is not None else 0.5
    market_p = float(market_home_prob) if market_home_prob is not None else 0.5
    gap = float(abs(market_gap)) if market_gap is not None else 0.0

    spread_p = float(np.clip(spread_p, 1e-6, 1 - 1e-6))
    market_p = float(np.clip(market_p, 1e-6, 1 - 1e-6))
    gap = float(np.clip(gap, 0.0, 1.0))
    spread_proxy = float(np.clip(0.5 + (spread_p - 0.5) * 0.60, 1e-6, 1 - 1e-6))

    return np.array(
        [
            [
                base_p,
                market_p,
                spread_p,
                spread_proxy,
                gap,
                base_p - market_p,
                base_p - spread_proxy,
                market_p - spread_proxy,
                base_p * gap,
                market_p * gap,
            ]
        ],
        dtype=float,
    )


def _compute_full_game_v2_fields(
    game_dict: Dict,
    home_team: str,
    away_team: str,
    v2_assets: Optional[Dict],
) -> None:
    full_pick = game_dict.get("full_game_pick")
    full_side = _pick_side_from_text(full_pick, home_team=home_team, away_team=away_team)
    full_pick_prob = _safe_prob(game_dict.get("full_game_calibrated_prob_pick"))
    if full_pick_prob is None:
        full_pick_prob = _safe_prob(game_dict.get("full_game_confidence"))
    if full_pick_prob is None:
        full_pick_prob = 0.5

    if full_side == "home":
        base_home_prob = float(full_pick_prob)
    elif full_side == "away":
        base_home_prob = float(1.0 - full_pick_prob)
    else:
        base_home_prob = 0.5

    spread_home_prob, _ = _resolve_spread_prob_home(game_dict, home_team=home_team, away_team=away_team)
    market_home_prob, _ = _resolve_market_home_prob_no_vig(game_dict)
    market_gap, _ = _resolve_market_gap(game_dict, market_home_prob)

    threshold = 0.5
    model_status = "fallback_blend"
    v2_home_prob = base_home_prob

    if v2_assets is not None:
        try:
            X_meta = _build_full_game_v2_meta_matrix(
                full_home_prob=base_home_prob,
                spread_home_cover_prob=spread_home_prob,
                market_home_prob=market_home_prob,
                market_gap=market_gap,
            )
            v2_home_prob = float(v2_assets["model"].predict_proba(X_meta)[0, 1])
            threshold = float(v2_assets.get("threshold", 0.5))
            model_status = "meta_model"
        except Exception:
            model_status = "fallback_blend"

    if model_status != "meta_model":
        spread_proxy = 0.5 if spread_home_prob is None else float(np.clip(0.5 + (spread_home_prob - 0.5) * 0.60, 0.0, 1.0))
        market_proxy = 0.5 if market_home_prob is None else float(np.clip(market_home_prob, 0.0, 1.0))
        v2_home_prob = float(np.clip((0.72 * base_home_prob) + (0.20 * market_proxy) + (0.08 * spread_proxy), 0.01, 0.99))

    v2_pred_bin = int(v2_home_prob >= threshold)
    v2_pick = home_team if v2_pred_bin == 1 else away_team
    v2_pick_prob = float(v2_home_prob if v2_pred_bin == 1 else (1.0 - v2_home_prob))
    v2_conf = int(round(v2_pick_prob * 100))

    game_dict["full_game_v2_pick"] = str(v2_pick)
    game_dict["full_game_v2_prob_home"] = round(v2_home_prob, 4)
    game_dict["full_game_v2_confidence"] = v2_conf
    game_dict["full_game_v2_threshold_used"] = round(float(threshold), 4)
    game_dict["full_game_v2_bucket"] = _meta_bucket_from_score(v2_pick_prob)
    game_dict["full_game_v2_reason"] = (
        f"status={model_status}; threshold={float(threshold):.3f}; "
        f"base_home_prob={base_home_prob:.4f}; v2_home_prob={v2_home_prob:.4f}"
    )


def _resolve_full_game_hybrid_pick(game_dict: Dict) -> Tuple[str, bool, str]:
    base_pick = str(game_dict.get("full_game_pick") or "")
    v2_pick = str(game_dict.get("full_game_v2_pick") or "")
    v2_bucket = _normalize_meta_bucket(game_dict.get("full_game_v2_bucket", "PASS"))
    alignment = str(game_dict.get("market_ml_alignment", "neutral") or "neutral").strip().lower()

    if not v2_pick:
        return base_pick, False, "v2_pick_missing"

    if NHL_FULL_GAME_HYBRID_BLOCK_CONFLICTED and alignment == "conflicted":
        return base_pick, False, "blocked_alignment_conflicted"

    if v2_bucket not in NHL_FULL_GAME_HYBRID_V2_BUCKETS:
        return base_pick, False, f"v2_bucket_not_enabled:{v2_bucket}"

    return v2_pick, True, f"v2_bucket_enabled:{v2_bucket}"


def _calculate_consensus_signal(
    full_side: Optional[str],
    spread_prob: Optional[float],
    full_conf: int,
    spread_conf: int,
) -> str:
    """Classify agreement strength between full-game and spread implied winner."""
    if full_side not in {"home", "away"}:
        return "NEUTRAL"

    spread_prob_value = _safe_prob(spread_prob)
    if spread_prob_value is None:
        return "NEUTRAL"

    implied_spread_winner: Optional[str] = None
    if spread_prob_value >= 0.55:
        implied_spread_winner = "home"
    elif spread_prob_value <= 0.45:
        implied_spread_winner = "away"

    if implied_spread_winner is None:
        return "NEUTRAL"

    spread_edge = abs(spread_prob_value - 0.5)

    if full_side == implied_spread_winner and full_conf >= 56 and spread_edge >= 0.08:
        return "STRONG"

    if full_side != implied_spread_winner and spread_conf >= 58:
        return "WEAK"

    return "NEUTRAL"


def _apply_handicap_override_full_game(
    game_dict: Dict,
    home_team: str,
    away_team: str,
    min_spread_conf: int = 58,
    min_conf_gap: int = 4,
) -> None:
    """Build full-game meta score from moneyline, spread and market without changing pick."""
    full_pick = game_dict.get("full_game_pick")
    full_conf = int(game_dict.get("full_game_confidence", 0) or 0)
    spread_conf = int(game_dict.get("spread_confidence", 0) or 0)

    game_dict["full_game_pick_base"] = full_pick
    game_dict["full_game_pick_final"] = full_pick

    full_side = _pick_side_from_text(full_pick, home_team, away_team) if full_pick else None

    full_prob_pick, full_prob_source = _resolve_pick_probability_with_source(
        game_dict,
        prob_key="full_game_calibrated_prob_pick",
        conf_key="full_game_confidence",
    )
    if full_prob_pick is None:
        full_prob_pick = 0.50
        full_prob_source = "default_0_50"

    spread_prob_home, spread_prob_source = _resolve_spread_prob_home(
        game_dict,
        home_team=home_team,
        away_team=away_team,
    )
    market_home_prob, market_source = _resolve_market_home_prob_no_vig(game_dict)

    spread_alignment = "neutral"
    spread_delta = 0.0
    spread_alignment_reason = "spread_signal_missing"
    spread_high = FULL_GAME_META_CFG["spread_prob_high"]
    spread_low = FULL_GAME_META_CFG["spread_prob_low"]
    if full_side in {"home", "away"} and spread_prob_home is not None:
        if full_side == "home" and spread_prob_home >= spread_high:
            spread_alignment = "aligned"
            spread_delta = FULL_GAME_META_CFG["spread_aligned_bonus"]
            spread_alignment_reason = f"home_pick_and_spread_prob_home>={spread_high:.2f}"
        elif full_side == "away" and spread_prob_home <= spread_low:
            spread_alignment = "aligned"
            spread_delta = FULL_GAME_META_CFG["spread_aligned_bonus"]
            spread_alignment_reason = f"away_pick_and_spread_prob_home<={spread_low:.2f}"
        elif full_side == "home" and spread_prob_home <= spread_low:
            spread_alignment = "conflicted"
            spread_delta = FULL_GAME_META_CFG["spread_conflicted_penalty"]
            spread_alignment_reason = f"home_pick_but_spread_prob_home<={spread_low:.2f}"
        elif full_side == "away" and spread_prob_home >= spread_high:
            spread_alignment = "conflicted"
            spread_delta = FULL_GAME_META_CFG["spread_conflicted_penalty"]
            spread_alignment_reason = f"away_pick_but_spread_prob_home>={spread_high:.2f}"
        else:
            spread_alignment_reason = (
                f"spread_prob_home_in_neutral_band({spread_low:.2f},{spread_high:.2f})"
            )

    market_gap, market_gap_source = _resolve_market_gap(game_dict, market_home_prob)
    market_ml_alignment, market_ml_alignment_reason = _resolve_market_ml_alignment(
        game_dict=game_dict,
        full_side=full_side,
        market_home_prob=market_home_prob,
        market_gap=market_gap,
    )

    market_delta = 0.0
    edge_component = 0.0
    min_gap = FULL_GAME_META_CFG["market_min_gap_for_signal"]
    if market_gap is not None and market_gap > min_gap:
        edge_component = min(0.20, market_gap - min_gap) * FULL_GAME_META_CFG["market_edge_bonus_scale"]

    if market_ml_alignment == "aligned":
        market_delta = FULL_GAME_META_CFG["market_aligned_bonus"] + edge_component
    elif market_ml_alignment == "conflicted":
        market_delta = FULL_GAME_META_CFG["market_conflicted_penalty"] - (edge_component * 1.2)

    meta_score_raw = float(full_prob_pick + spread_delta + market_delta)
    if full_side in {"home", "away"}:
        meta_score = float(
            np.clip(
                meta_score_raw,
                FULL_GAME_META_CFG["meta_score_floor"],
                FULL_GAME_META_CFG["meta_score_ceiling"],
            )
        )
    else:
        meta_score = float(np.clip(meta_score_raw, 0.0, 1.0))

    meta_confidence = int(round(meta_score * 100))
    base_meta_bucket = _meta_bucket_from_score(meta_score)
    meta_bucket = base_meta_bucket

    if market_ml_alignment == "conflicted":
        meta_bucket = _shift_meta_bucket(meta_bucket, -2)
    elif spread_alignment == "conflicted":
        meta_bucket = _shift_meta_bucket(meta_bucket, -1)

    if (
        market_ml_alignment == "aligned"
        and spread_alignment == "aligned"
        and meta_score >= FULL_GAME_BUCKET_THRESHOLDS["strong"]
    ):
        meta_bucket = _shift_meta_bucket(meta_bucket, 1)

    if full_side not in {"home", "away"}:
        meta_bucket = "PASS"

    publish_full_game, publish_full_game_reason = _should_publish_full_game(
        meta_bucket=meta_bucket,
        market_ml_alignment=market_ml_alignment,
        meta_confidence=meta_confidence,
    )

    consensus_signal = _calculate_consensus_signal(
        full_side=full_side,
        spread_prob=spread_prob_home,
        full_conf=full_conf,
        spread_conf=spread_conf,
    )

    game_dict["full_game_meta_score"] = round(meta_score, 4)
    game_dict["full_game_meta_confidence"] = meta_confidence
    game_dict["full_game_meta_bucket"] = meta_bucket
    game_dict["spread_prob_home"] = None if spread_prob_home is None else round(spread_prob_home, 4)
    game_dict["market_ml_alignment"] = market_ml_alignment
    game_dict["market_ml_alignment_reason"] = market_ml_alignment_reason
    game_dict["full_game_meta_alignment"] = spread_alignment
    game_dict["publish_full_game"] = bool(publish_full_game)
    game_dict["publish_full_game_reason"] = publish_full_game_reason
    game_dict["full_game_meta_reason"] = (
        f"base_prob={full_prob_pick:.4f}({full_prob_source}); "
        f"spread_prob_home={(f'{spread_prob_home:.4f}' if spread_prob_home is not None else 'na')}({spread_prob_source}); "
        f"spread_delta={spread_delta:+.4f}({spread_alignment}); "
        f"spread_reason={spread_alignment_reason}; "
        f"market_home_prob={(f'{market_home_prob:.4f}' if market_home_prob is not None else 'na')}({market_source}); "
        f"market_gap={(f'{market_gap:.4f}' if market_gap is not None else 'na')}({market_gap_source}); "
        f"market_delta={market_delta:+.4f}({market_ml_alignment}); "
        f"market_reason={market_ml_alignment_reason}; "
        f"bucket_base={base_meta_bucket}; bucket_final={meta_bucket}; "
        f"publish={int(bool(publish_full_game))}({publish_full_game_reason}); "
        f"meta_score={meta_score:.4f}"
    )

    # Legacy adjusted-confidence fields remain populated for backward compatibility.
    game_dict["full_game_calibrated_prob_adjusted"] = round(meta_score, 4)
    game_dict["full_game_confidence_adjusted"] = meta_confidence
    game_dict["consensus_signal"] = consensus_signal
    game_dict["full_game_pick_source"] = "full_game_model"
    game_dict["full_game_override_reason"] = (
        "meta_conf_adjustment_only"
        f"; consensus={consensus_signal}"
        f"; spread_delta={spread_delta:+.4f}"
        f"; market_delta={market_delta:+.4f}"
        f"; market_alignment={market_ml_alignment}"
        f"; publish={int(bool(publish_full_game))}"
        f"; spread_conf={spread_conf}"
        f"; min_spread_conf={min_spread_conf}"
        f"; min_conf_gap={min_conf_gap}"
    )


def predict_today_nhl():
    """Generate REAL predictions for upcoming NHL games."""
    print("[NHL] Live Predictions")
    print("=" * 60)
    print(f"[CFG] threshold_mode={NHL_LIVE_THRESHOLD_MODE}")
    print(
        "[CFG] full_game_publish_rule="
        f"{NHL_FULL_GAME_PUBLISH_RULE} min_conf={NHL_FULL_GAME_PUBLISH_MIN_META_CONF} "
        f"exclude_conflicted={int(NHL_FULL_GAME_PUBLISH_EXCLUDE_CONFLICTED)}"
    )
    print(f"[CFG] full_game_active_variant={NHL_FULL_GAME_ACTIVE_VARIANT}")
    print(
        "[CFG] full_game_hybrid_v2_buckets="
        f"{sorted(NHL_FULL_GAME_HYBRID_V2_BUCKETS)} block_conflicted={int(NHL_FULL_GAME_HYBRID_BLOCK_CONFLICTED)}"
    )

    full_game_v2_assets = _load_full_game_v2_meta_assets()
    print(f"[CFG] full_game_v2_model_loaded={int(full_game_v2_assets is not None)}")

    try:
        history_df = load_history()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    if not UPCOMING_FILE.exists():
        print(f"⚠️  No upcoming schedule found: {UPCOMING_FILE}")
        output_file = PREDICTIONS_DIR / datetime.now().strftime("%Y-%m-%d.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        return

    upcoming_df = pd.read_csv(UPCOMING_FILE, dtype={"game_id": str})
    if upcoming_df.empty:
        print("⚠️  No upcoming games today")
        output_file = PREDICTIONS_DIR / datetime.now().strftime("%Y-%m-%d.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        return

    print(f"[OK] Loaded {len(upcoming_df)} upcoming games")

    upcoming_df["date"] = pd.to_datetime(upcoming_df["date"], errors="coerce").astype(str)

    markets_available = {}
    for market in ["full_game", "handicap_1_5", "spread_2_5", "q1_over_15", "home_over_2_5"]:
        try:
            markets_available[market] = load_model_assets(market)
        except Exception as e:
            print(f"   [WARN] {market} models not available: {e}")

    if not markets_available:
        print("   [ERROR] No models available for prediction")
        return

    calibration_cfg = load_calibration_config(CALIBRATION_FILE)

    for date_str, group_df in upcoming_df.groupby("date"):
        print(f"\n[{date_str}] Predicting: {len(group_df)} games")

        games_predictions = []

        for _, row in group_df.iterrows():
            game_id = str(row.get("game_id", ""))
            home_team = row.get("home_team")
            away_team = row.get("away_team")
            time_str = row.get("time", "19:00")
            total_line_live = _resolve_nhl_total_line(row.get("odds_over_under"))
            total_line_txt = _goal_line_label(total_line_live)

            features = build_pregame_features(
                history_df,
                {"date": date_str, "home_team": home_team, "away_team": away_team},
            )

            if not features:
                print(f"   [WARN] Could not build features for game {game_id}: {home_team} vs {away_team}")
                continue

            game_dict = {
                "game_id": game_id,
                "date": date_str,
                "time": time_str,
                "home_team": home_team,
                "away_team": away_team,
                "odds_over_under": total_line_live,
                "closing_total_line": total_line_live,
                "closing_spread_line": 1.5,
                "closing_moneyline_odds": row.get("closing_moneyline_odds"),
                "home_moneyline_odds": row.get("home_moneyline_odds"),
                "away_moneyline_odds": row.get("away_moneyline_odds"),
                "closing_spread_odds": row.get("closing_spread_odds"),
                "closing_total_odds": row.get("closing_total_odds"),
                "ml_implied_home_prob_no_vig": row.get("ml_implied_home_prob_no_vig"),
                "ml_implied_away_prob_no_vig": row.get("ml_implied_away_prob_no_vig"),
                "ml_prob_gap_no_vig": row.get("ml_prob_gap_no_vig"),
                "ml_home_is_favorite_market": row.get("ml_home_is_favorite_market"),
                "odds_data_quality": str(row.get("odds_data_quality", "fallback") or "fallback"),
                "full_game_pick_source": "full_game_model",
                "consensus_signal": "NEUTRAL",
                "full_game_pick_base": None,
                "full_game_pick_final": None,
                "full_game_meta_score": None,
                "full_game_meta_confidence": None,
                "full_game_meta_bucket": "PASS",
                "full_game_meta_reason": "pending_full_game_meta",
                "spread_prob_home": None,
                "market_ml_alignment": "neutral",
                "market_ml_alignment_reason": "pending_market_alignment",
                "publish_full_game": False,
                "publish_full_game_reason": "pending_meta_rule",
                "full_game_v2_pick": None,
                "full_game_v2_prob_home": None,
                "full_game_v2_confidence": None,
                "full_game_v2_threshold_used": None,
                "full_game_v2_bucket": "PASS",
                "full_game_v2_reason": "pending_v2",
                "full_game_hybrid_pick": None,
                "full_game_hybrid_use_v2": False,
                "full_game_hybrid_reason": "pending_hybrid",
            }

            nhl_patterns = generate_nhl_patterns(features)
            pattern_edge = aggregate_pattern_edge(nhl_patterns)
            game_dict["detected_patterns"] = nhl_patterns
            game_dict["pattern_edge"] = round(pattern_edge, 4)

            confidences_list = []

            if "full_game" in markets_available:
                try:
                    pred, conf, probs = predict_market(
                        markets_available["full_game"],
                        features,
                        market="full_game",
                    )
                    full_labels = ["N/A", "Home Win", "Away Win"]
                    game_dict["full_game_pick"] = full_labels[int(pred)]
                    game_dict["full_game_pick_base"] = full_labels[int(pred)]
                    game_dict["full_game_pick_final"] = full_labels[int(pred)]

                    model_prob_pick = float(conf)
                    calibrated_prob_pick = calibrate_probability(
                        model_prob_pick,
                        "nhl",
                        "full_game",
                        calibration_cfg,
                    )
                    score = fuse_with_pattern_score(
                        recommendation_score(calibrated_prob_pick),
                        pattern_edge,
                    )

                    game_dict["full_game_confidence"] = int(calibrated_prob_pick * 100)
                    game_dict["full_game_model_prob_pick"] = round(model_prob_pick, 4)
                    game_dict["full_game_calibrated_prob_pick"] = round(calibrated_prob_pick, 4)
                    game_dict["full_game_recommended_score"] = round(score, 1)
                    if "handicap_1_5" not in markets_available:
                        spread_side = home_team if int(pred) == 1 else away_team
                        spread_sign = "-1.5" if int(pred) == 1 else "+1.5"
                        game_dict["spread_pick"] = f"{spread_side} {spread_sign}"
                        game_dict["spread_market"] = "Puck Line 1.5"
                        game_dict["spread_line"] = 1.5
                        game_dict["spread_confidence"] = max(0, int(calibrated_prob_pick * 100) - 4)
                        game_dict["spread_calibrated_prob_pick"] = round(
                            max(0.0, min(1.0, float(game_dict["spread_confidence"]) / 100.0)),
                            4,
                        )
                        game_dict["spread_recommended_score"] = round(max(0.0, score - 2.0), 1)

                    confidences_list.append(game_dict["full_game_confidence"])
                except Exception as e:
                    print(f"   [WARN] full_game error for {game_id}: {e}")
                    game_dict["full_game_pick"] = "N/A"
                    game_dict["full_game_confidence"] = 0
                    game_dict["full_game_recommended_score"] = 0.0

            if "handicap_1_5" in markets_available:
                try:
                    pred, conf, _ = predict_market(
                        markets_available["handicap_1_5"],
                        features,
                        market="handicap_1_5",
                    )
                    spread_labels = [f"{away_team} +1.5", f"{home_team} -1.5"]
                    spread_pick = spread_labels[int(pred)]
                    model_prob_pick = float(conf)
                    calibrated_prob_pick = calibrate_probability(
                        model_prob_pick,
                        "nhl",
                        "spread_1_5",
                        calibration_cfg,
                    )
                    score = fuse_with_pattern_score(
                        recommendation_score(calibrated_prob_pick),
                        pattern_edge,
                    )
                    game_dict["spread_pick"] = spread_pick
                    game_dict["spread_market"] = "Puck Line 1.5"
                    game_dict["spread_line"] = 1.5
                    game_dict["spread_confidence"] = int(calibrated_prob_pick * 100)
                    game_dict["spread_model_prob_pick"] = round(model_prob_pick, 4)
                    game_dict["spread_calibrated_prob_pick"] = round(calibrated_prob_pick, 4)
                    game_dict["spread_recommended_score"] = round(score, 1)
                    confidences_list.append(game_dict["spread_confidence"])
                except Exception as e:
                    print(f"   [WARN] handicap error for {game_id}: {e}")

            # TOTAL predictions (NHL usa totals, no spread real)
            if "spread_2_5" in markets_available:
                try:
                    pred, conf, probs = predict_market(
                        markets_available["spread_2_5"],
                        features,
                        market="spread_2_5",
                    )
                    total_labels = [f"Under {total_line_txt}", f"Over {total_line_txt}"]
                    total_pick = total_labels[int(pred)]

                    model_prob_pick = float(conf)
                    calibrated_prob_pick = calibrate_probability(
                        model_prob_pick,
                        "nhl",
                        "total",
                        calibration_cfg,
                    )
                    score = fuse_with_pattern_score(
                        recommendation_score(calibrated_prob_pick),
                        pattern_edge,
                    )

                    game_dict["total_pick"] = total_pick
                    game_dict["total_market"] = f"Total Goals O/U {total_line_txt}"
                    game_dict["total_line"] = total_line_live
                    game_dict["total_confidence"] = int(calibrated_prob_pick * 100)
                    game_dict["total_model_prob_pick"] = round(model_prob_pick, 4)
                    game_dict["total_calibrated_prob_pick"] = round(calibrated_prob_pick, 4)
                    game_dict["total_recommended_pick"] = total_pick
                    game_dict["total_recommended_score"] = round(score, 1)


                    over_55_prob = float(probs[1]) if hasattr(probs, "__len__") and len(probs) > 1 else float(conf)
                    if "q1_over_15" not in markets_available:
                        game_dict.update(derive_nhl_first_period_pick(over_55_prob))

                    confidences_list.append(game_dict["total_confidence"])
                except Exception as e:
                    print(f"   [WARN] total error for {game_id}: {e}")

                    game_dict["total_pick"] = "N/A"
                    game_dict["total_market"] = f"Total Goals O/U {total_line_txt}"
                    game_dict["total_line"] = total_line_live
                    game_dict["total_confidence"] = 0
                    game_dict["total_recommended_pick"] = "N/A"
                    game_dict["total_recommended_score"] = 0.0

            if "q1_over_15" in markets_available:
                try:
                    pred, conf, probs = predict_market(
                        markets_available["q1_over_15"],
                        features,
                        market="q1_over_15",
                    )
                    q1_labels = ["Under 1.5", "Over 1.5"]
                    q1_pick = q1_labels[int(pred)]
                    model_prob_pick = float(conf)
                    calibrated_prob_pick = calibrate_probability(
                        model_prob_pick,
                        "nhl",
                        "q1_over_15",
                        calibration_cfg,
                    )
                    game_dict["q1_pick"] = q1_pick
                    game_dict["q1_market"] = "1P Goals O/U 1.5"
                    game_dict["q1_line"] = 1.5
                    game_dict["q1_confidence"] = int(calibrated_prob_pick * 100)
                    game_dict["q1_action"] = "JUGAR" if game_dict["q1_confidence"] >= 56 else "PASS"
                    game_dict["q1_model_prob_yes"] = round(float(probs[1]) if hasattr(probs, "__len__") else float(conf), 4)
                    game_dict["q1_calibrated_prob_yes"] = round(calibrated_prob_pick, 4)
                    confidences_list.append(game_dict["q1_confidence"])
                except Exception as e:
                    print(f"   [WARN] q1 error for {game_id}: {e}")


            if "home_over_2_5" in markets_available:
                try:
                    pred, conf, probs = predict_market(
                        markets_available["home_over_2_5"],
                        features,
                        market="home_over_2_5",
                    )
                    home_labels = ["Home Under 2.5", "Home Over 2.5"]
                    game_dict["home_over_pick"] = home_labels[int(pred)]

                    model_prob_pick = float(conf)
                    calibrated_prob_pick = calibrate_probability(
                        model_prob_pick,
                        "nhl",
                        "home_over_25",
                        calibration_cfg,
                    )
                    score = fuse_with_pattern_score(
                        recommendation_score(calibrated_prob_pick),
                        pattern_edge,
                    )

                    game_dict["home_over_confidence"] = int(calibrated_prob_pick * 100)
                    game_dict["home_over_model_prob_pick"] = round(model_prob_pick, 4)
                    game_dict["home_over_calibrated_prob_pick"] = round(calibrated_prob_pick, 4)
                    game_dict["home_over_recommended_score"] = round(score, 1)

                    confidences_list.append(game_dict["home_over_confidence"])
                except Exception as e:
                    print(f"   [WARN] home_over error for {game_id}: {e}")
                    game_dict["home_over_pick"] = "N/A"
                    game_dict["home_over_confidence"] = 0
                    game_dict["home_over_recommended_score"] = 0.0

            _apply_handicap_override_full_game(
                game_dict,
                home_team=home_team,
                away_team=away_team,
            )

            if game_dict.get("full_game_pick") and game_dict.get("full_game_pick") != "N/A":
                _compute_full_game_v2_fields(
                    game_dict,
                    home_team=home_team,
                    away_team=away_team,
                    v2_assets=full_game_v2_assets,
                )

                hybrid_pick, hybrid_use_v2, hybrid_reason = _resolve_full_game_hybrid_pick(game_dict)
                game_dict["full_game_hybrid_pick"] = hybrid_pick
                game_dict["full_game_hybrid_use_v2"] = bool(hybrid_use_v2)
                game_dict["full_game_hybrid_reason"] = hybrid_reason

                if NHL_FULL_GAME_ACTIVE_VARIANT == "v2" and game_dict.get("full_game_v2_pick"):
                    game_dict["full_game_pick_final"] = game_dict.get("full_game_v2_pick")
                    game_dict["full_game_pick_source"] = "full_game_v2_meta"
                elif NHL_FULL_GAME_ACTIVE_VARIANT == "hybrid" and game_dict.get("full_game_hybrid_pick"):
                    game_dict["full_game_pick_final"] = game_dict.get("full_game_hybrid_pick")
                    game_dict["full_game_pick_source"] = (
                        "full_game_hybrid_v2" if game_dict.get("full_game_hybrid_use_v2") else "full_game_hybrid_base"
                    )

            avg_conf = sum(confidences_list) / len(confidences_list) if confidences_list else 0
            game_dict["recommended"] = avg_conf >= 55

            score_candidates = [
                float(game_dict.get("full_game_meta_confidence", game_dict.get("full_game_recommended_score", 0.0)) or 0.0),
                float(game_dict.get("spread_recommended_score", 0.0) or 0.0),
                float(game_dict.get("total_recommended_score", 0.0) or 0.0),
                float(game_dict.get("home_over_recommended_score", 0.0) or 0.0),
            ]
            best_score = max(score_candidates) if score_candidates else 0.0
            game_dict["recommended_score"] = round(best_score, 1)
            game_dict["recommended"] = best_score >= 56.0

            games_predictions.append(game_dict)

        output_file = PREDICTIONS_DIR / f"{date_str}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(games_predictions, f, indent=2, ensure_ascii=False)

        print(f"   [OK] Saved {len(games_predictions)} predictions to {output_file.name}")


if __name__ == "__main__":
    predict_today_nhl()
