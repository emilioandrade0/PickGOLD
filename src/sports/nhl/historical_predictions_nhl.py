import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import sys
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
INPUT_FILE = BASE_DIR / "data" / "nhl" / "processed" / "model_ready_features_nhl.csv"
RAW_FILE = BASE_DIR / "data" / "nhl" / "raw" / "nhl_advanced_history.csv"
HISTORICAL_DIR = BASE_DIR / "data" / "nhl" / "historical_predictions"
HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

NON_FEATURE_COLUMNS = {
    "game_id", "date", "date_dt", "time", "season", "home_team", "away_team",
    "home_score", "away_score", "total_goals", "is_draw", "completed",
    "venue_name", "odds_details", "odds_over_under", "odds_data_quality",
    "home_p1_goals", "away_p1_goals", "total_p1_goals",
    "home_goalie_name", "away_goalie_name", "home_goalie_id", "away_goalie_id",
    "goalie_data_quality",
    "TARGET_full_game", "TARGET_over_55", "TARGET_home_over_25", "TARGET_spread_1_5", "TARGET_p1_over_15",
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

NHL_FULL_GAME_HYBRID_V2_BUCKETS = {
    x.strip().upper()
    for x in os.getenv("NHL_FULL_GAME_HYBRID_V2_BUCKETS", "ELITE,STRONG,NORMAL").split(",")
    if x.strip()
}
NHL_FULL_GAME_HYBRID_BLOCK_CONFLICTED = _env_bool("NHL_FULL_GAME_HYBRID_BLOCK_CONFLICTED", False)


def build_xgb_binary() -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=2,
        random_state=42,
        n_jobs=-1,
    )


def build_lgbm_binary() -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=15,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def build_catboost_binary() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5.0,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        random_state=44,
        verbose=0,
        allow_writing_files=False,
        thread_count=-1,
    )


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    candidate_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]

    banned_keywords = ["correct", "actual", "winner", "result", "target"]
    safe_cols = []
    dropped_leaks = []

    for c in numeric_cols:
        if any(banned in c.lower() for banned in banned_keywords):
            dropped_leaks.append(c)
        else:
            safe_cols.append(c)

    if dropped_leaks:
        print(f"\n🛡️ Escudo activado: se bloquearon {len(dropped_leaks)} columnas sospechosas.")
        for c in dropped_leaks:
            print(f"   - {c}")

    return safe_cols


def derive_nhl_first_period_pick(prob_over_55: float) -> Dict:
    p = float(np.clip(prob_over_55, 0.01, 0.99))
    expected_total_goals = float(np.clip(5.5 + 1.4 * (p - 0.5), 4.6, 6.4))
    lambda_p1 = expected_total_goals * 0.30
    p_over_15 = float(np.clip(1.0 - float(np.exp(-lambda_p1) * (1.0 + lambda_p1)), 0.01, 0.99))

    pick_over = p_over_15 >= 0.53
    confidence = int((0.5 + abs(p_over_15 - 0.5)) * 100)

    return {
        "q1_pick": "Over 1.5" if pick_over else "Under 1.5",
        "q1_market": "1P Goals O/U 1.5",
        "q1_line": 1.5,
        "q1_confidence": confidence,
        "q1_action": "JUGAR" if confidence >= 56 else "PASS",
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


def _spread_cover_from_pick(pick_text: str, home_team: str, away_team: str, home_score: int, away_score: int, line: float = 1.5):
    pick = str(pick_text or "").upper()
    if not pick:
        return None
    if str(home_team).upper() in pick:
        return int((home_score - away_score) > line)
    if str(away_team).upper() in pick:
        return int((away_score - home_score) > line)
    return None


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


def _resolve_market_home_prob_no_vig_from_row(row: pd.Series) -> Optional[float]:
    for key in ["ml_implied_home_prob_no_vig", "moneyline_prob_home", "ml_implied_home_prob"]:
        value = _safe_prob(row.get(key))
        if value is not None:
            return value

    home_ml = _american_to_implied_prob(row.get("home_moneyline_odds"))
    away_ml = _american_to_implied_prob(row.get("away_moneyline_odds"))
    if home_ml is not None and away_ml is not None and (home_ml + away_ml) > 0:
        return float(np.clip(home_ml / (home_ml + away_ml), 0.0, 1.0))

    return None


def _resolve_market_gap_from_row(row: pd.Series, market_home_prob: Optional[float]) -> Optional[float]:
    gap = _safe_prob(row.get("ml_prob_gap_no_vig"))
    if gap is not None:
        return float(np.clip(abs(gap), 0.0, 1.0))

    away_prob = _safe_prob(row.get("ml_implied_away_prob_no_vig"))
    if market_home_prob is not None and away_prob is not None:
        return float(np.clip(abs(market_home_prob - away_prob), 0.0, 1.0))

    if market_home_prob is not None:
        return float(np.clip(abs(market_home_prob - 0.5) * 2.0, 0.0, 1.0))

    return None


def _build_full_game_v2_meta_matrix(
    full_home_prob: np.ndarray,
    spread_home_cover_prob: np.ndarray,
    market_home_prob: np.ndarray,
    market_gap: np.ndarray,
) -> np.ndarray:
    base_p = np.clip(np.asarray(full_home_prob, dtype=float), 1e-6, 1 - 1e-6)
    spread_p = np.asarray(spread_home_cover_prob, dtype=float)
    market_p = np.asarray(market_home_prob, dtype=float)
    gap = np.asarray(market_gap, dtype=float)

    spread_p = np.where(np.isnan(spread_p), 0.5, spread_p)
    market_p = np.where(np.isnan(market_p), 0.5, market_p)
    gap = np.where(np.isnan(gap), 0.0, gap)
    gap = np.clip(np.abs(gap), 0.0, 1.0)

    spread_proxy = np.clip(0.5 + (spread_p - 0.5) * 0.60, 1e-6, 1 - 1e-6)

    return np.column_stack(
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
    )


def _fit_full_game_v2_meta_model(
    y_train: np.ndarray,
    full_home_prob: np.ndarray,
    spread_home_cover_prob: np.ndarray,
    market_home_prob: np.ndarray,
    market_gap: np.ndarray,
) -> Tuple[Optional[LogisticRegression], float, str]:
    if len(y_train) < 60 or len(np.unique(y_train)) < 2:
        return None, 0.5, "insufficient_meta_data"

    X_meta = _build_full_game_v2_meta_matrix(
        full_home_prob=full_home_prob,
        spread_home_cover_prob=spread_home_cover_prob,
        market_home_prob=market_home_prob,
        market_gap=market_gap,
    )

    try:
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=400)
        model.fit(X_meta, y_train)
        train_probs = model.predict_proba(X_meta)[:, 1]
        threshold = choose_optimal_binary_threshold(y_true=y_train, probs=train_probs, market="full_game")
        return model, float(threshold), "trained"
    except Exception:
        return None, 0.5, "meta_fit_failed"


def _predict_full_game_v2_home_probs(
    model: Optional[LogisticRegression],
    full_home_prob: np.ndarray,
    spread_home_cover_prob: np.ndarray,
    market_home_prob: np.ndarray,
    market_gap: np.ndarray,
) -> np.ndarray:
    base = np.clip(np.asarray(full_home_prob, dtype=float), 1e-6, 1 - 1e-6)
    if model is None:
        return base

    X_meta = _build_full_game_v2_meta_matrix(
        full_home_prob=base,
        spread_home_cover_prob=spread_home_cover_prob,
        market_home_prob=market_home_prob,
        market_gap=market_gap,
    )
    try:
        return np.clip(model.predict_proba(X_meta)[:, 1], 1e-6, 1 - 1e-6)
    except Exception:
        return base


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


def _meta_score_range_label(meta_score: Optional[float]) -> str:
    if meta_score is None:
        return "missing"
    if meta_score >= 0.66:
        return ">=0.66"
    if meta_score >= 0.60:
        return "0.60-0.66"
    if meta_score >= 0.55:
        return "0.55-0.60"
    return "<0.55"


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
) -> bool:
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
    return full_side in {"home", "away"} and spread_prob_home is not None


def choose_optimal_binary_threshold(y_true: np.ndarray, probs: np.ndarray, market: str = "full_game") -> float:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)

    if market == "full_game":
        search_space = np.arange(0.35, 0.66, 0.01)
    else:
        search_space = np.arange(0.35, 0.66, 0.01)

    best_threshold = 0.50
    best_accuracy = -1.0

    for threshold in search_space:
        preds = (probs >= threshold).astype(int)
        acc = float((preds == y_true).mean())
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = float(threshold)

    return best_threshold


def _weighted_prob(probs_mat: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    return np.clip(np.dot(probs_mat, weights), 1e-6, 1 - 1e-6)


def _search_best_weights(y_true: np.ndarray, probs_mat: np.ndarray, market: str) -> np.ndarray:
    best_w = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    best_acc = -1.0
    best_ll = 1e9
    grid = np.arange(0.0, 1.01, 0.1)
    for w1 in grid:
        for w2 in grid:
            for w3 in grid:
                s = w1 + w2 + w3
                if s > 1.0:
                    continue
                w4 = round(1.0 - s, 10)
                if w4 < 0:
                    continue
                w = np.array([w1, w2, w3, w4], dtype=float)
                if np.sum(w) <= 0:
                    continue
                p = _weighted_prob(probs_mat, w)
                thr = choose_optimal_binary_threshold(y_true, p, market=market)
                pred = (p >= thr).astype(int)
                acc = float((pred == y_true).mean())
                ll = float(log_loss(y_true, p))
                if (acc > best_acc) or (acc == best_acc and ll < best_ll):
                    best_acc = acc
                    best_ll = ll
                    best_w = w.copy()
    return best_w / np.sum(best_w)


def fit_calibrated_ensemble(X_train: pd.DataFrame, y_train: pd.Series, market: str = "full_game"):
    n = len(X_train)
    if n < 40:
        base_X = X_train
        base_y = y_train
        calib_X = None
        calib_y = None
    else:
        split_idx = int(n * 0.8)
        split_idx = max(20, min(split_idx, n - 10))
        base_X = X_train.iloc[:split_idx]
        base_y = y_train.iloc[:split_idx]
        calib_X = X_train.iloc[split_idx:]
        calib_y = y_train.iloc[split_idx:]

    models = {
        "xgb": build_xgb_binary(),
        "lgbm": build_lgbm_binary(),
        "lgbm_sec": build_lgbm_binary(),
        "catboost": build_catboost_binary(),
    }

    for model in models.values():
        model.fit(base_X, base_y)

    calibrator = None
    blend_weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    use_calibrator = True
    chosen_threshold = 0.50

    if calib_X is not None and calib_y is not None and len(np.unique(calib_y)) > 1:
        probs_mat = np.column_stack(
            [
                models["xgb"].predict_proba(calib_X)[:, 1],
                models["lgbm"].predict_proba(calib_X)[:, 1],
                models["lgbm_sec"].predict_proba(calib_X)[:, 1],
                models["catboost"].predict_proba(calib_X)[:, 1],
            ]
        )
        y_cal = np.asarray(calib_y).astype(int)
        calib_raw_probs = _weighted_prob(probs_mat, blend_weights)
        try:
            calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            calibrator.fit(calib_raw_probs.reshape(-1, 1), calib_y)
            calib_final_probs = calibrator.predict_proba(calib_raw_probs.reshape(-1, 1))[:, 1]
        except Exception:
            calibrator = None
            use_calibrator = False
            calib_final_probs = calib_raw_probs
        chosen_threshold = choose_optimal_binary_threshold(
            y_true=y_cal,
            probs=calib_final_probs,
            market=market,
        )

    return models, calibrator, chosen_threshold, blend_weights, use_calibrator


def predict_calibrated_ensemble(
    models,
    calibrator,
    X_test: pd.DataFrame,
    threshold: float = 0.50,
    blend_weights: Optional[np.ndarray] = None,
    use_calibrator: bool = True,
):
    probs_mat = np.column_stack(
        [
            models["xgb"].predict_proba(X_test)[:, 1],
            models["lgbm"].predict_proba(X_test)[:, 1],
            models["lgbm_sec"].predict_proba(X_test)[:, 1],
            models["catboost"].predict_proba(X_test)[:, 1],
        ]
    )
    if blend_weights is None:
        blend_weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    raw_probs = _weighted_prob(probs_mat, blend_weights)

    if use_calibrator and calibrator is not None:
        final_probs = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    else:
        final_probs = raw_probs

    preds = (final_probs >= threshold).astype(int)
    confidences = np.maximum(final_probs, 1.0 - final_probs)
    return preds, confidences, final_probs


def _bad_team_value(v) -> bool:
    if pd.isna(v):
        return True
    s = str(v).strip().upper()
    return s in {"", "0", "0.0", "NAN", "NONE", "NULL"}


def recover_team_names_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    if not RAW_FILE.exists():
        return df

    try:
        raw_df = pd.read_csv(RAW_FILE, dtype={"game_id": str})
    except Exception:
        return df

    if "game_id" not in raw_df.columns:
        return df

    needed_cols = {"game_id", "home_team", "away_team"}
    if not needed_cols.issubset(set(raw_df.columns)):
        return df

    raw_map = raw_df[["game_id", "home_team", "away_team"]].drop_duplicates("game_id", keep="last")
    raw_map = raw_map.rename(
        columns={
            "home_team": "raw_home_team",
            "away_team": "raw_away_team",
        }
    )

    df = df.merge(raw_map, on="game_id", how="left")

    bad_home = df["home_team"].apply(_bad_team_value)
    bad_away = df["away_team"].apply(_bad_team_value)

    recovered_home = int(bad_home.sum())
    recovered_away = int(bad_away.sum())

    df["home_team"] = df["home_team"].astype("object")
    df["away_team"] = df["away_team"].astype("object")

    df.loc[bad_home, "home_team"] = df.loc[bad_home, "raw_home_team"]
    df.loc[bad_away, "away_team"] = df.loc[bad_away, "raw_away_team"]

    df["home_team"] = df["home_team"].fillna("UNK").astype(str)
    df["away_team"] = df["away_team"].fillna("UNK").astype(str)

    df = df.drop(columns=["raw_home_team", "raw_away_team"], errors="ignore")

    if recovered_home or recovered_away:
        print(
            f"[FIX] Team labels recovered from raw file | "
            f"home_team: {recovered_home}, away_team: {recovered_away}"
        )

    return df


def confidence_tier_from_pct(conf_pct: float) -> str:
    if conf_pct >= 70:
        return "ELITE"
    if conf_pct >= 62:
        return "PREMIUM"
    if conf_pct >= 55:
        return "STRONG"
    if conf_pct >= 52:
        return "NORMAL"
    return "PASS"


def generate_historical_predictions():
    print("[NHL] Historical Predictions - Block Walk-Forward Validation")
    print("=" * 60)
    print(
        "[CFG] full_game_publish_rule="
        f"{NHL_FULL_GAME_PUBLISH_RULE} min_conf={NHL_FULL_GAME_PUBLISH_MIN_META_CONF} "
        f"exclude_conflicted={int(NHL_FULL_GAME_PUBLISH_EXCLUDE_CONFLICTED)}"
    )
    print(
        "[CFG] full_game_hybrid_v2_buckets="
        f"{sorted(NHL_FULL_GAME_HYBRID_V2_BUCKETS)} block_conflicted={int(NHL_FULL_GAME_HYBRID_BLOCK_CONFLICTED)}"
    )

    if not INPUT_FILE.exists():
        print(f"[ERROR] Dataset not found: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE, dtype={"game_id": str})
    df["date"] = df["date"].astype(str)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt"]).sort_values("date_dt").reset_index(drop=True)

    df = recover_team_names_from_raw(df)

    feature_cols = get_feature_columns(df)
    unique_dates = sorted(df["date"].unique())

    print(f"[OK] Loaded {len(df)} games | {len(feature_cols)} features | {len(unique_dates)} dates")

    predictions_by_date = defaultdict(list)
    overall_stats = defaultdict(lambda: {"correct": 0, "total": 0, "total_conf": 0.0})
    premium_stats = defaultdict(lambda: {"correct": 0, "total": 0, "total_conf": 0.0})
    consensus_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    meta_bucket_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    meta_range_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    market_alignment_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    publish_stats = {True: {"correct": 0, "total": 0}, False: {"correct": 0, "total": 0}}
    top_subset_stats = {
        "elite_only": {"correct": 0, "total": 0},
        "elite_or_strong_non_conflicted": {"correct": 0, "total": 0},
    }
    v2_bucket_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    hybrid_bucket_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    STEP_DAYS = 7
    start_date_idx = min(20, len(unique_dates) // 10)

    print(f"\n[TRAIN] Executing Block Walk-Forward (Step: {STEP_DAYS} days)...")

    for i in range(start_date_idx, len(unique_dates), STEP_DAYS):
        test_start_date = unique_dates[i]
        test_end_date = unique_dates[min(i + STEP_DAYS - 1, len(unique_dates) - 1)]

        train_df = df[df["date"] < test_start_date].copy()
        test_df = df[(df["date"] >= test_start_date) & (df["date"] <= test_end_date)].copy()

        if len(train_df) < 20 or test_df.empty:
            continue

        print(
            f"   Training on {len(train_df)} games -> Predicting "
            f"{test_start_date} to {test_end_date} ({len(test_df)} games)"
        )

        X_train = train_df[feature_cols].fillna(0)
        X_test = test_df[feature_cols].fillna(0)

        # FULL GAME
        y_full_raw = pd.to_numeric(train_df["TARGET_full_game"], errors="coerce")
        valid_full = y_full_raw.isin([0, 1])
        y_full_train = y_full_raw.loc[valid_full].astype(int)

        models_full, calib_full, weights_full = None, None, None
        use_cal_full = False
        pred_full, conf_full, prob_full = None, None, None
        full_threshold = 0.50

        if valid_full.sum() >= 10 and len(np.unique(y_full_train)) > 1:
            models_full, calib_full, full_threshold, weights_full, use_cal_full = fit_calibrated_ensemble(
                X_train.loc[valid_full],
                y_full_train,
                market="full_game",
            )
            pred_full, conf_full, prob_full = predict_calibrated_ensemble(
                models_full,
                calib_full,
                X_test,
                threshold=full_threshold,
                blend_weights=weights_full,
                use_calibrator=use_cal_full,
            )

        # TOTALS 5.5
        y_totals_train = pd.to_numeric(train_df["TARGET_over_55"], errors="coerce").fillna(-1)
        valid_totals = y_totals_train >= 0
        pred_totals, conf_totals, prob_totals = None, None, None
        totals_threshold = 0.50

        if valid_totals.sum() >= 10 and len(np.unique(y_totals_train.loc[valid_totals])) > 1:
            models_totals, calib_totals, totals_threshold, weights_totals, use_cal_totals = fit_calibrated_ensemble(
                X_train.loc[valid_totals],
                y_totals_train.loc[valid_totals].astype(int),
                market="totals_5_5",
            )
            pred_totals, conf_totals, prob_totals = predict_calibrated_ensemble(
                models_totals,
                calib_totals,
                X_test,
                threshold=totals_threshold,
                blend_weights=weights_totals,
                use_calibrator=use_cal_totals,
            )

        # HANDICAP 1.5
        y_spread_train = pd.to_numeric(train_df["TARGET_spread_1_5"], errors="coerce").fillna(-1)
        valid_spread = y_spread_train >= 0
        models_spread, calib_spread, weights_spread = None, None, None
        use_cal_spread = False
        pred_spread, conf_spread, prob_spread = None, None, None
        spread_threshold = 0.50
        if valid_spread.sum() >= 10 and len(np.unique(y_spread_train.loc[valid_spread])) > 1:
            models_spread, calib_spread, spread_threshold, weights_spread, use_cal_spread = fit_calibrated_ensemble(
                X_train.loc[valid_spread],
                y_spread_train.loc[valid_spread].astype(int),
                market="spread_1_5",
            )
            pred_spread, conf_spread, prob_spread = predict_calibrated_ensemble(
                models_spread,
                calib_spread,
                X_test,
                threshold=spread_threshold,
                blend_weights=weights_spread,
                use_calibrator=use_cal_spread,
            )

        # FULL_GAME V2: lightweight meta model trained on split calibration rows.
        full_v2_prob_home = None
        full_v2_threshold = 0.50
        full_v2_status = "fallback_base"
        if pred_full is not None and prob_full is not None:
            if models_full is not None and weights_full is not None:
                X_full_valid = X_train.loc[valid_full].copy()
                y_full_valid = y_full_train.copy()

                split_meta_idx = int(len(X_full_valid) * 0.8)
                split_meta_idx = max(20, min(split_meta_idx, len(X_full_valid) - 10))

                if split_meta_idx < len(X_full_valid):
                    X_meta_train = X_full_valid.iloc[split_meta_idx:].copy()
                    y_meta_train = y_full_valid.iloc[split_meta_idx:].to_numpy(dtype=int)

                    _, _, full_meta_home_prob = predict_calibrated_ensemble(
                        models_full,
                        calib_full,
                        X_meta_train,
                        threshold=full_threshold,
                        blend_weights=weights_full,
                        use_calibrator=use_cal_full,
                    )

                    if models_spread is not None and weights_spread is not None:
                        _, _, spread_meta_home_prob = predict_calibrated_ensemble(
                            models_spread,
                            calib_spread,
                            X_meta_train,
                            threshold=spread_threshold,
                            blend_weights=weights_spread,
                            use_calibrator=use_cal_spread,
                        )
                    else:
                        spread_meta_home_prob = np.full(len(X_meta_train), np.nan, dtype=float)

                    meta_rows = train_df.loc[X_meta_train.index].copy()
                    market_home_meta = meta_rows.apply(_resolve_market_home_prob_no_vig_from_row, axis=1).to_numpy(dtype=float)
                    market_gap_meta = np.array(
                        [
                            _resolve_market_gap_from_row(row, market_home)
                            for row, market_home in zip(meta_rows.to_dict("records"), market_home_meta)
                        ],
                        dtype=float,
                    )

                    full_v2_model, full_v2_threshold, full_v2_status = _fit_full_game_v2_meta_model(
                        y_train=y_meta_train,
                        full_home_prob=np.asarray(full_meta_home_prob, dtype=float),
                        spread_home_cover_prob=np.asarray(spread_meta_home_prob, dtype=float),
                        market_home_prob=market_home_meta,
                        market_gap=market_gap_meta,
                    )
                else:
                    full_v2_model = None
                    full_v2_status = "meta_split_invalid"
            else:
                full_v2_model = None
                full_v2_status = "full_assets_missing"

            if models_spread is not None and prob_spread is not None:
                spread_home_test_prob = np.asarray(prob_spread, dtype=float)
            else:
                spread_home_test_prob = np.full(len(test_df), np.nan, dtype=float)

            market_home_test = test_df.apply(_resolve_market_home_prob_no_vig_from_row, axis=1).to_numpy(dtype=float)
            market_gap_test = np.array(
                [
                    _resolve_market_gap_from_row(row, market_home)
                    for row, market_home in zip(test_df.to_dict("records"), market_home_test)
                ],
                dtype=float,
            )

            full_v2_prob_home = _predict_full_game_v2_home_probs(
                model=full_v2_model,
                full_home_prob=np.asarray(prob_full, dtype=float),
                spread_home_cover_prob=spread_home_test_prob,
                market_home_prob=market_home_test,
                market_gap=market_gap_test,
            )

        # Q1 O/U 1.5
        y_q1_train = pd.to_numeric(train_df["TARGET_p1_over_15"], errors="coerce").fillna(-1)
        valid_q1 = y_q1_train >= 0
        pred_q1, conf_q1, prob_q1 = None, None, None
        q1_threshold = 0.50
        if valid_q1.sum() >= 10 and len(np.unique(y_q1_train.loc[valid_q1])) > 1:
            models_q1, calib_q1, q1_threshold, weights_q1, use_cal_q1 = fit_calibrated_ensemble(
                X_train.loc[valid_q1],
                y_q1_train.loc[valid_q1].astype(int),
                market="q1_over_15",
            )
            pred_q1, conf_q1, prob_q1 = predict_calibrated_ensemble(
                models_q1,
                calib_q1,
                X_test,
                threshold=q1_threshold,
                blend_weights=weights_q1,
                use_calibrator=use_cal_q1,
            )

        # HOME OVER 2.5
        y_home_train = pd.to_numeric(train_df["TARGET_home_over_25"], errors="coerce").fillna(-1)
        valid_home = y_home_train >= 0
        pred_home, conf_home, prob_home = None, None, None
        home_threshold = 0.50

        if valid_home.sum() >= 10 and len(np.unique(y_home_train.loc[valid_home])) > 1:
            models_home, calib_home, home_threshold, weights_home, use_cal_home = fit_calibrated_ensemble(
                X_train.loc[valid_home],
                y_home_train.loc[valid_home].astype(int),
                market="home_over_2_5",
            )
            pred_home, conf_home, prob_home = predict_calibrated_ensemble(
                models_home,
                calib_home,
                X_test,
                threshold=home_threshold,
                blend_weights=weights_home,
                use_calibrator=use_cal_home,
            )

        for idx, test_row in test_df.reset_index(drop=True).iterrows():
            total_line_live = _resolve_nhl_total_line(test_row.get("odds_over_under"))
            total_line_txt = _goal_line_label(total_line_live)
            home_team = str(test_row["home_team"])
            away_team = str(test_row["away_team"])
            game_dict = {
                "game_id": str(test_row["game_id"]),
                "date": test_row["date"],
                "time": str(test_row.get("time", "19:00")),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": None if pd.isna(test_row.get("home_score")) else int(test_row.get("home_score")),
                "away_score": None if pd.isna(test_row.get("away_score")) else int(test_row.get("away_score")),
                "odds_over_under": total_line_live,
                "closing_total_line": total_line_live,
                "closing_spread_line": 1.5,
                "closing_moneyline_odds": test_row.get("closing_moneyline_odds"),
                "home_moneyline_odds": test_row.get("home_moneyline_odds"),
                "away_moneyline_odds": test_row.get("away_moneyline_odds"),
                "closing_spread_odds": test_row.get("closing_spread_odds"),
                "closing_total_odds": test_row.get("closing_total_odds"),
                "ml_implied_home_prob_no_vig": test_row.get("ml_implied_home_prob_no_vig"),
                "ml_implied_away_prob_no_vig": test_row.get("ml_implied_away_prob_no_vig"),
                "ml_prob_gap_no_vig": test_row.get("ml_prob_gap_no_vig"),
                "ml_home_is_favorite_market": test_row.get("ml_home_is_favorite_market"),
                "odds_data_quality": str(test_row.get("odds_data_quality", "fallback")),
                "spread_pick": None,
                "spread_market": None,
                "spread_line": None,
                "spread_confidence": None,
                "correct_spread": None,
                "full_game_pick_source": "full_game_model",
                "consensus_signal": "NEUTRAL",
                "full_game_pick_base": None,
                "full_game_pick_final": None,
                "full_game_meta_score": None,
                "full_game_meta_confidence": None,
                "full_game_meta_bucket": "PASS",
                "full_game_meta_reason": None,
                "spread_prob_home": None,
                "market_ml_alignment": "neutral",
                "market_ml_alignment_reason": "pending_market_alignment",
                "full_game_meta_alignment": "neutral",
                "publish_full_game": False,
                "publish_full_game_reason": "pending_meta_rule",
                "full_game_v2_pick": None,
                "full_game_v2_prob_home": None,
                "full_game_v2_confidence": None,
                "full_game_v2_threshold_used": None,
                "full_game_v2_bucket": "PASS",
                "full_game_v2_hit": None,
                "full_game_v2_reason": "pending_v2",
                "full_game_hybrid_pick": None,
                "full_game_hybrid_use_v2": False,
                "full_game_hybrid_hit": None,
                "full_game_hybrid_reason": "pending_hybrid",
            }
            full_game_actual_bin: Optional[int] = None
            full_game_prob_home_win: Optional[float] = None

            # FULL GAME
            if pred_full is not None:
                actual_target = test_row.get("TARGET_full_game", np.nan)

                if pd.notna(actual_target) and int(actual_target) in (0, 1):
                    confidence = float(conf_full[idx])
                    prob_home_win = float(prob_full[idx])
                    pred_bin = int(pred_full[idx])  # 1=home, 0=away

                    pick_team = home_team if pred_bin == 1 else away_team
                    actual_bin = int(actual_target)
                    actual_winner = home_team if actual_bin == 1 else away_team
                    is_correct = int(pred_bin == actual_bin)
                    full_game_actual_bin = actual_bin
                    full_game_prob_home_win = prob_home_win

                    conf_pct = int(round(confidence * 100))
                    tier = confidence_tier_from_pct(conf_pct)

                    game_dict["moneyline_pick"] = str(pick_team)
                    game_dict["moneyline_confidence"] = conf_pct
                    game_dict["moneyline_recommended_score"] = round(confidence * 100, 1)
                    game_dict["moneyline_prob_home"] = round(prob_home_win, 4)
                    game_dict["moneyline_threshold_used"] = round(float(full_threshold), 4)
                    game_dict["moneyline_actual"] = str(actual_winner)
                    game_dict["moneyline_correct"] = is_correct
                    game_dict["full_game_pick"] = str(pick_team)
                    game_dict["full_game_pick_base"] = str(pick_team)
                    game_dict["full_game_pick_final"] = str(pick_team)
                    game_dict["full_game_confidence"] = conf_pct
                    game_dict["full_game_model_prob_pick"] = round(confidence, 4)
                    game_dict["full_game_calibrated_prob_pick"] = round(confidence, 4)
                    game_dict["full_game_recommended_score"] = round(confidence * 100, 1)
                    game_dict["full_game_hit"] = is_correct
                    game_dict["full_game_result_winner"] = str(actual_winner)

                    game_dict["market"] = "FULL_GAME"
                    game_dict["pick"] = str(pick_team)
                    game_dict["pick_team"] = str(pick_team)
                    game_dict["probability"] = round(
                        prob_home_win if pred_bin == 1 else (1.0 - prob_home_win),
                        4
                    )
                    game_dict["threshold_used"] = round(float(full_threshold), 4)
                    game_dict["confidence"] = conf_pct
                    game_dict["tier"] = tier
                    game_dict["actual_winner"] = str(actual_winner)
                    game_dict["correct"] = is_correct

                    v2_prob_home = float(full_v2_prob_home[idx]) if full_v2_prob_home is not None else float(prob_home_win)
                    v2_pred_bin = int(v2_prob_home >= full_v2_threshold)
                    v2_pick_team = home_team if v2_pred_bin == 1 else away_team
                    v2_pick_prob = float(v2_prob_home if v2_pred_bin == 1 else (1.0 - v2_prob_home))
                    v2_conf_pct = int(round(v2_pick_prob * 100))
                    v2_hit = int(v2_pred_bin == actual_bin)

                    game_dict["full_game_v2_pick"] = str(v2_pick_team)
                    game_dict["full_game_v2_prob_home"] = round(v2_prob_home, 4)
                    game_dict["full_game_v2_confidence"] = v2_conf_pct
                    game_dict["full_game_v2_threshold_used"] = round(float(full_v2_threshold), 4)
                    game_dict["full_game_v2_bucket"] = _meta_bucket_from_score(v2_pick_prob)
                    game_dict["full_game_v2_hit"] = v2_hit
                    game_dict["full_game_v2_reason"] = (
                        f"status={full_v2_status}; threshold={full_v2_threshold:.3f}; "
                        f"base_home_prob={prob_home_win:.4f}; v2_home_prob={v2_prob_home:.4f}"
                    )

                    hybrid_pick, hybrid_use_v2, hybrid_reason = _resolve_full_game_hybrid_pick(game_dict)
                    hybrid_hit = v2_hit if hybrid_use_v2 else is_correct
                    game_dict["full_game_hybrid_pick"] = str(hybrid_pick)
                    game_dict["full_game_hybrid_use_v2"] = bool(hybrid_use_v2)
                    game_dict["full_game_hybrid_hit"] = int(hybrid_hit)
                    game_dict["full_game_hybrid_reason"] = hybrid_reason

                    if pred_spread is None:
                        spread_side = test_row["home_team"] if pred_bin == 1 else test_row["away_team"]
                        spread_sign = "-1.5" if pred_bin == 1 else "+1.5"
                        spread_pick = f"{spread_side} {spread_sign}"
                        game_dict["spread_pick"] = spread_pick
                        game_dict["spread_market"] = "Puck Line 1.5"
                        game_dict["spread_line"] = 1.5
                        game_dict["spread_confidence"] = max(0, conf_pct - 4)
                        game_dict["spread_calibrated_prob_pick"] = round(
                            max(0.0, min(1.0, float(game_dict["spread_confidence"]) / 100.0)),
                            4,
                        )
                        spread_hit = _spread_cover_from_pick(
                            spread_pick,
                            home_team,
                            away_team,
                            int(test_row.get("home_score", 0) or 0),
                            int(test_row.get("away_score", 0) or 0),
                            1.5,
                        )
                        game_dict["correct_spread"] = spread_hit
                        if spread_hit is not None:
                            overall_stats["spread_1_5"]["correct"] += int(spread_hit)
                            overall_stats["spread_1_5"]["total"] += 1
                            overall_stats["spread_1_5"]["total_conf"] += confidence
                            if confidence >= 0.55:
                                premium_stats["spread_1_5"]["correct"] += int(spread_hit)
                                premium_stats["spread_1_5"]["total"] += 1
                                premium_stats["spread_1_5"]["total_conf"] += confidence

            # TOTALS 5.5
            if pred_totals is not None:
                confidence = float(conf_totals[idx])
                total_labels = [f"Under {total_line_txt}", f"Over {total_line_txt}"]
                actual = int(test_row["TARGET_over_55"])
                pred_val = int(pred_totals[idx])
                total_pick = total_labels[pred_val]
                is_correct = int(pred_val == actual)

                game_dict["total_pick"] = total_pick
                game_dict["total_market"] = f"Total Goals O/U {total_line_txt}"
                game_dict["total_line"] = total_line_live
                game_dict["total_confidence"] = int(round(confidence * 100))
                game_dict["total_recommended_pick"] = total_pick
                game_dict["total_recommended_score"] = round(confidence * 100, 1)
                game_dict["total_prob_over"] = round(float(prob_totals[idx]), 4)
                game_dict["total_threshold_used"] = round(float(totals_threshold), 4)
                game_dict["total_actual"] = total_labels[actual]
                game_dict["total_correct"] = is_correct
                game_dict["correct_total"] = is_correct
                if pred_q1 is None:
                    game_dict.update(derive_nhl_first_period_pick(float(prob_totals[idx])))

                overall_stats["totals_5_5"]["correct"] += is_correct
                overall_stats["totals_5_5"]["total"] += 1
                overall_stats["totals_5_5"]["total_conf"] += confidence

                if confidence >= 0.55:
                    premium_stats["totals_5_5"]["correct"] += is_correct
                    premium_stats["totals_5_5"]["total"] += 1
                    premium_stats["totals_5_5"]["total_conf"] += confidence

            # HOME OVER 2.5
            if pred_home is not None:
                confidence = float(conf_home[idx])
                home_labels = ["Home Under 2.5", "Home Over 2.5"]
                actual = int(test_row["TARGET_home_over_25"])
                pred_val = int(pred_home[idx])
                is_correct = int(pred_val == actual)

                game_dict["home_over_pick"] = home_labels[pred_val]
                game_dict["home_over_confidence"] = int(round(confidence * 100))
                game_dict["home_over_recommended_score"] = round(confidence * 100, 1)
                game_dict["home_over_prob_yes"] = round(float(prob_home[idx]), 4)
                game_dict["home_over_threshold_used"] = round(float(home_threshold), 4)
                game_dict["home_over_actual"] = home_labels[actual]
                game_dict["home_over_correct"] = is_correct

                overall_stats["home_over_2_5"]["correct"] += is_correct
                overall_stats["home_over_2_5"]["total"] += 1
                overall_stats["home_over_2_5"]["total_conf"] += confidence

                if confidence >= 0.55:
                    premium_stats["home_over_2_5"]["correct"] += is_correct
                    premium_stats["home_over_2_5"]["total"] += 1
                    premium_stats["home_over_2_5"]["total_conf"] += confidence

            # HANDICAP 1.5 (dedicated)
            if pred_spread is not None:
                confidence = float(conf_spread[idx])
                spread_labels = [f"{away_team} +1.5", f"{home_team} -1.5"]
                actual = int(test_row["TARGET_spread_1_5"])
                pred_val = int(pred_spread[idx])
                spread_pick = spread_labels[pred_val]
                is_correct = int(pred_val == actual)
                game_dict["spread_pick"] = spread_pick
                game_dict["spread_market"] = "Puck Line 1.5"
                game_dict["spread_line"] = 1.5
                game_dict["spread_confidence"] = int(round(confidence * 100))
                game_dict["spread_model_prob_pick"] = round(confidence, 4)
                game_dict["spread_calibrated_prob_pick"] = round(
                    float(prob_spread[idx]) if pred_val == 1 else float(1.0 - prob_spread[idx]),
                    4,
                )
                game_dict["spread_threshold_used"] = round(float(spread_threshold), 4)
                game_dict["correct_spread"] = is_correct
                overall_stats["spread_1_5"]["correct"] += is_correct
                overall_stats["spread_1_5"]["total"] += 1
                overall_stats["spread_1_5"]["total_conf"] += confidence
                if confidence >= 0.55:
                    premium_stats["spread_1_5"]["correct"] += is_correct
                    premium_stats["spread_1_5"]["total"] += 1
                    premium_stats["spread_1_5"]["total_conf"] += confidence

            if full_game_actual_bin is not None:
                _apply_handicap_override_full_game(
                    game_dict,
                    home_team=home_team,
                    away_team=away_team,
                )
                final_side = _pick_side_from_text(
                    game_dict.get("full_game_pick"),
                    home_team=home_team,
                    away_team=away_team,
                )
                final_pred_bin = 1 if final_side == "home" else 0
                final_is_correct = int(final_pred_bin == int(full_game_actual_bin))
                final_conf_pct = int(
                    game_dict.get(
                        "full_game_meta_confidence",
                        game_dict.get("full_game_confidence", 0),
                    )
                    or 0
                )
                final_confidence = float(final_conf_pct / 100.0)
                final_tier = confidence_tier_from_pct(int(round(final_confidence * 100)))
                actual_winner = home_team if int(full_game_actual_bin) == 1 else away_team
                prob_home_for_row = float(full_game_prob_home_win if full_game_prob_home_win is not None else 0.5)
                pick_prob_for_row = _safe_prob(game_dict.get("full_game_meta_score"))
                if pick_prob_for_row is None:
                    pick_prob_for_row = float(
                        prob_home_for_row if final_pred_bin == 1 else (1.0 - prob_home_for_row)
                    )

                game_dict["moneyline_actual"] = str(actual_winner)
                game_dict["moneyline_correct"] = final_is_correct
                game_dict["full_game_hit"] = final_is_correct
                game_dict["full_game_result_winner"] = str(actual_winner)
                game_dict["actual_winner"] = str(actual_winner)
                game_dict["correct"] = final_is_correct
                game_dict["tier"] = final_tier
                game_dict["confidence"] = final_conf_pct
                game_dict["probability"] = round(pick_prob_for_row, 4)

                overall_stats["full_game"]["correct"] += final_is_correct
                overall_stats["full_game"]["total"] += 1
                overall_stats["full_game"]["total_conf"] += final_confidence

                v2_hit_raw = game_dict.get("full_game_v2_hit")
                v2_hit = int(v2_hit_raw) if v2_hit_raw is not None else final_is_correct
                v2_conf_pct = int(game_dict.get("full_game_v2_confidence", final_conf_pct) or final_conf_pct)
                v2_confidence = float(v2_conf_pct / 100.0)
                v2_bucket_key = str(game_dict.get("full_game_v2_bucket", "PASS") or "PASS").upper()
                if v2_bucket_key == "LOW":
                    v2_bucket_key = "PASS"
                if v2_bucket_key not in {"ELITE", "STRONG", "NORMAL", "PASS"}:
                    v2_bucket_key = "PASS"

                overall_stats["full_game_v2"]["correct"] += v2_hit
                overall_stats["full_game_v2"]["total"] += 1
                overall_stats["full_game_v2"]["total_conf"] += v2_confidence
                v2_bucket_stats[v2_bucket_key]["correct"] += v2_hit
                v2_bucket_stats[v2_bucket_key]["total"] += 1

                hybrid_hit_raw = game_dict.get("full_game_hybrid_hit")
                hybrid_hit = int(hybrid_hit_raw) if hybrid_hit_raw is not None else final_is_correct
                hybrid_use_v2 = bool(game_dict.get("full_game_hybrid_use_v2", False))
                hybrid_confidence = v2_confidence if hybrid_use_v2 else final_confidence
                base_bucket_for_hybrid = str(game_dict.get("full_game_meta_bucket", "PASS") or "PASS").upper()
                if base_bucket_for_hybrid == "LOW":
                    base_bucket_for_hybrid = "PASS"
                if base_bucket_for_hybrid not in {"ELITE", "STRONG", "NORMAL", "PASS"}:
                    base_bucket_for_hybrid = "PASS"
                hybrid_bucket_key = v2_bucket_key if hybrid_use_v2 else base_bucket_for_hybrid

                overall_stats["full_game_hybrid"]["correct"] += hybrid_hit
                overall_stats["full_game_hybrid"]["total"] += 1
                overall_stats["full_game_hybrid"]["total_conf"] += hybrid_confidence
                hybrid_bucket_stats[hybrid_bucket_key]["correct"] += hybrid_hit
                hybrid_bucket_stats[hybrid_bucket_key]["total"] += 1

                consensus_key = str(game_dict.get("consensus_signal", "NEUTRAL") or "NEUTRAL").upper()
                consensus_stats[consensus_key]["correct"] += final_is_correct
                consensus_stats[consensus_key]["total"] += 1

                meta_bucket_key = str(game_dict.get("full_game_meta_bucket", "PASS") or "PASS").upper()
                if meta_bucket_key == "LOW":
                    meta_bucket_key = "PASS"
                if meta_bucket_key not in {"ELITE", "STRONG", "NORMAL", "PASS"}:
                    meta_bucket_key = "PASS"
                meta_bucket_stats[meta_bucket_key]["correct"] += final_is_correct
                meta_bucket_stats[meta_bucket_key]["total"] += 1

                meta_score_value = _safe_prob(game_dict.get("full_game_meta_score"))
                range_key = _meta_score_range_label(meta_score_value)
                meta_range_stats[range_key]["correct"] += final_is_correct
                meta_range_stats[range_key]["total"] += 1

                alignment_key = str(game_dict.get("market_ml_alignment", "neutral") or "neutral").lower()
                if alignment_key not in {"aligned", "neutral", "conflicted"}:
                    alignment_key = "neutral"
                market_alignment_stats[alignment_key]["correct"] += final_is_correct
                market_alignment_stats[alignment_key]["total"] += 1

                publish_flag = bool(game_dict.get("publish_full_game", False))
                publish_stats[publish_flag]["correct"] += final_is_correct
                publish_stats[publish_flag]["total"] += 1

                if meta_bucket_key == "ELITE":
                    top_subset_stats["elite_only"]["correct"] += final_is_correct
                    top_subset_stats["elite_only"]["total"] += 1

                if meta_bucket_key in {"ELITE", "STRONG"} and alignment_key != "conflicted":
                    top_subset_stats["elite_or_strong_non_conflicted"]["correct"] += final_is_correct
                    top_subset_stats["elite_or_strong_non_conflicted"]["total"] += 1

                if final_confidence >= 0.55:
                    premium_stats["full_game"]["correct"] += final_is_correct
                    premium_stats["full_game"]["total"] += 1
                    premium_stats["full_game"]["total_conf"] += final_confidence

                if v2_confidence >= 0.55:
                    premium_stats["full_game_v2"]["correct"] += v2_hit
                    premium_stats["full_game_v2"]["total"] += 1
                    premium_stats["full_game_v2"]["total_conf"] += v2_confidence

                if hybrid_confidence >= 0.55:
                    premium_stats["full_game_hybrid"]["correct"] += hybrid_hit
                    premium_stats["full_game_hybrid"]["total"] += 1
                    premium_stats["full_game_hybrid"]["total_conf"] += hybrid_confidence

            # Q1 O/U 1.5 (dedicated)
            if pred_q1 is not None:
                confidence = float(conf_q1[idx])
                q1_labels = ["Under 1.5", "Over 1.5"]
                actual = int(test_row["TARGET_p1_over_15"])
                pred_val = int(pred_q1[idx])
                is_correct = int(pred_val == actual)
                game_dict["q1_pick"] = q1_labels[pred_val]
                game_dict["q1_market"] = "1P Goals O/U 1.5"
                game_dict["q1_line"] = 1.5
                game_dict["q1_confidence"] = int(round(confidence * 100))
                game_dict["q1_action"] = "JUGAR" if game_dict["q1_confidence"] >= 56 else "PASS"
                game_dict["q1_model_prob_yes"] = round(float(prob_q1[idx]), 4)
                game_dict["q1_calibrated_prob_yes"] = round(float(confidence), 4)
                game_dict["q1_correct"] = is_correct
                game_dict["q1_hit"] = is_correct
                overall_stats["q1_over_15"]["correct"] += is_correct
                overall_stats["q1_over_15"]["total"] += 1
                overall_stats["q1_over_15"]["total_conf"] += confidence
                if confidence >= 0.55:
                    premium_stats["q1_over_15"]["correct"] += is_correct
                    premium_stats["q1_over_15"]["total"] += 1
                    premium_stats["q1_over_15"]["total_conf"] += confidence

            predictions_by_date[test_row["date"]].append(game_dict)

    print("\n[SAVE] Saving predictions by date...")
    for date_str in sorted(predictions_by_date.keys()):
        output_file = HISTORICAL_DIR / f"{date_str}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(predictions_by_date[date_str], f, indent=2, ensure_ascii=False)

    print("\n[STATS] OVERALL ACCURACY (Todos los picks):")
    for market in ["full_game", "full_game_v2", "full_game_hybrid", "spread_1_5", "totals_5_5", "q1_over_15", "home_over_2_5"]:
        stats = overall_stats[market]
        if stats["total"] > 0:
            accuracy = stats["correct"] / stats["total"]
            avg_conf = stats["total_conf"] / stats["total"]
            print(f"   {market.upper()}:")
            print(f"      Accuracy: {accuracy:.2%} ({stats['correct']}/{stats['total']})")
            print(f"      Avg Confidence: {avg_conf:.1%}")

    print("\n[STATS] PREMIUM PICKS ACCURACY (Confianza >= 55%):")
    for market in ["full_game", "full_game_v2", "full_game_hybrid", "spread_1_5", "totals_5_5", "q1_over_15", "home_over_2_5"]:
        stats = premium_stats[market]
        if stats["total"] > 0:
            accuracy = stats["correct"] / stats["total"]
            avg_conf = stats["total_conf"] / stats["total"]
            print(f"   {market.upper()}:")
            print(f"      Accuracy: {accuracy:.2%} ({stats['correct']}/{stats['total']})")
            print(f"      Avg Confidence: {avg_conf:.1%}")

    print("\n[STATS] FULL_GAME ACCURACY BY CONSENSUS_SIGNAL:")
    overall_full_stats = overall_stats["full_game"]
    overall_full_acc = (
        overall_full_stats["correct"] / overall_full_stats["total"]
        if overall_full_stats["total"] > 0
        else 0.0
    )
    for signal in ["STRONG", "NEUTRAL", "WEAK"]:
        stats = consensus_stats[signal]
        if stats["total"] == 0:
            print(f"   {signal}: N/A (0/0)")
            continue
        accuracy = stats["correct"] / stats["total"]
        print(f"   {signal}: {accuracy:.2%} ({stats['correct']}/{stats['total']})")

    strong_stats = consensus_stats["STRONG"]
    if strong_stats["total"] > 0 and overall_full_stats["total"] > 0:
        strong_acc = strong_stats["correct"] / strong_stats["total"]
        delta_pp = (strong_acc - overall_full_acc) * 100.0
        print(f"   DELTA STRONG vs GLOBAL: {delta_pp:+.2f} pp")

    print("\n[STATS] FULL_GAME BASE vs V2 vs HYBRID (GLOBAL):")
    v2_stats = overall_stats["full_game_v2"]
    hybrid_stats = overall_stats["full_game_hybrid"]
    if overall_full_stats["total"] > 0:
        base_acc = overall_full_stats["correct"] / overall_full_stats["total"]
        print(f"   BASE: {base_acc:.2%} ({overall_full_stats['correct']}/{overall_full_stats['total']})")
    else:
        base_acc = 0.0
        print("   BASE: N/A (0/0)")

    if v2_stats["total"] > 0:
        v2_acc = v2_stats["correct"] / v2_stats["total"]
        print(f"   V2  : {v2_acc:.2%} ({v2_stats['correct']}/{v2_stats['total']})")
        print(f"   DELTA V2 vs BASE: {(v2_acc - base_acc) * 100.0:+.2f} pp")
    else:
        print("   V2  : N/A (0/0)")

    if hybrid_stats["total"] > 0:
        hybrid_acc = hybrid_stats["correct"] / hybrid_stats["total"]
        print(f"   HYBRID: {hybrid_acc:.2%} ({hybrid_stats['correct']}/{hybrid_stats['total']})")
        print(f"   DELTA HYBRID vs BASE: {(hybrid_acc - base_acc) * 100.0:+.2f} pp")
    else:
        print("   HYBRID: N/A (0/0)")

    print("\n[STATS] FULL_GAME V2 ACCURACY BY BUCKET:")
    for bucket in ["ELITE", "STRONG", "NORMAL", "PASS"]:
        stats = v2_bucket_stats[bucket]
        if stats["total"] == 0:
            print(f"   {bucket}: N/A (0/0)")
            continue
        accuracy = stats["correct"] / stats["total"]
        print(f"   {bucket}: {accuracy:.2%} ({stats['correct']}/{stats['total']})")

    print("\n[STATS] FULL_GAME HYBRID ACCURACY BY BUCKET:")
    for bucket in ["ELITE", "STRONG", "NORMAL", "PASS"]:
        stats = hybrid_bucket_stats[bucket]
        if stats["total"] == 0:
            print(f"   {bucket}: N/A (0/0)")
            continue
        accuracy = stats["correct"] / stats["total"]
        print(f"   {bucket}: {accuracy:.2%} ({stats['correct']}/{stats['total']})")

    print("\n[STATS] FULL_GAME ACCURACY BY META_BUCKET:")
    for bucket in ["ELITE", "STRONG", "NORMAL", "PASS"]:
        stats = meta_bucket_stats[bucket]
        if stats["total"] == 0:
            print(f"   {bucket}: N/A (0/0)")
            continue
        accuracy = stats["correct"] / stats["total"]
        print(f"   {bucket}: {accuracy:.2%} ({stats['correct']}/{stats['total']})")

    print("\n[STATS] FULL_GAME ACCURACY BY META_SCORE RANGE:")
    for score_range in [">=0.66", "0.60-0.66", "0.55-0.60", "<0.55", "missing"]:
        stats = meta_range_stats[score_range]
        if stats["total"] == 0:
            print(f"   {score_range}: N/A (0/0)")
            continue
        accuracy = stats["correct"] / stats["total"]
        print(f"   {score_range}: {accuracy:.2%} ({stats['correct']}/{stats['total']})")

    print("\n[STATS] FULL_GAME ACCURACY BY MARKET_ML_ALIGNMENT:")
    for align_label in ["aligned", "neutral", "conflicted"]:
        stats = market_alignment_stats[align_label]
        if stats["total"] == 0:
            print(f"   {align_label.upper()}: N/A (0/0)")
            continue
        accuracy = stats["correct"] / stats["total"]
        print(f"   {align_label.upper()}: {accuracy:.2%} ({stats['correct']}/{stats['total']})")

    print("\n[STATS] FULL_GAME PUBLICATION FILTER:")
    global_total = overall_full_stats["total"]
    global_acc = (overall_full_stats["correct"] / global_total) if global_total > 0 else 0.0
    print(f"   GLOBAL: {global_acc:.2%} ({overall_full_stats['correct']}/{global_total})")

    published = publish_stats[True]
    if published["total"] > 0:
        published_acc = published["correct"] / published["total"]
        coverage = (published["total"] / global_total) if global_total > 0 else 0.0
        print(
            "   PUBLISHED (publish_full_game=True): "
            f"{published_acc:.2%} ({published['correct']}/{published['total']}) | coverage={coverage:.2%}"
        )
    else:
        print("   PUBLISHED (publish_full_game=True): N/A (0/0) | coverage=0.00%")

    not_published = publish_stats[False]
    if not_published["total"] > 0:
        not_pub_acc = not_published["correct"] / not_published["total"]
        coverage = (not_published["total"] / global_total) if global_total > 0 else 0.0
        print(
            "   NOT_PUBLISHED (publish_full_game=False): "
            f"{not_pub_acc:.2%} ({not_published['correct']}/{not_published['total']}) | coverage={coverage:.2%}"
        )
    else:
        print("   NOT_PUBLISHED (publish_full_game=False): N/A (0/0) | coverage=0.00%")

    elite_stats = top_subset_stats["elite_only"]
    if elite_stats["total"] > 0:
        elite_acc = elite_stats["correct"] / elite_stats["total"]
        elite_cov = (elite_stats["total"] / global_total) if global_total > 0 else 0.0
        print(
            "   TOP ELITE: "
            f"{elite_acc:.2%} ({elite_stats['correct']}/{elite_stats['total']}) | coverage={elite_cov:.2%}"
        )
    else:
        print("   TOP ELITE: N/A (0/0) | coverage=0.00%")

    strict_stats = top_subset_stats["elite_or_strong_non_conflicted"]
    if strict_stats["total"] > 0:
        strict_acc = strict_stats["correct"] / strict_stats["total"]
        strict_cov = (strict_stats["total"] / global_total) if global_total > 0 else 0.0
        print(
            "   TOP ELITE/STRONG + NON_CONFLICTED: "
            f"{strict_acc:.2%} ({strict_stats['correct']}/{strict_stats['total']}) | coverage={strict_cov:.2%}"
        )
    else:
        print("   TOP ELITE/STRONG + NON_CONFLICTED: N/A (0/0) | coverage=0.00%")

    if published["total"] > 0 and global_total > 0:
        published_acc = published["correct"] / published["total"]
        delta_pp = (published_acc - global_acc) * 100.0
        print(f"   DELTA PUBLISHED vs GLOBAL: {delta_pp:+.2f} pp")


if __name__ == "__main__":
    generate_historical_predictions()
