from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

# Global guard to disable all event-based adjustments quickly.
USE_EVENT_ADJUSTMENTS = True
MAX_ABS_ADJUSTMENT = 0.08


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _clip_prob(prob: float) -> float:
    return float(np.clip(_safe_float(prob, 0.5), 0.0, 1.0))


def _history_pre_match(history_df: pd.DataFrame, match_date: str) -> pd.DataFrame:
    if history_df is None or history_df.empty:
        return pd.DataFrame()

    if "date" not in history_df.columns:
        return pd.DataFrame()

    df = history_df.copy()
    df["date"] = df["date"].astype(str)
    return df[df["date"] < str(match_date)].copy()


def _team_games_before(history_df: pd.DataFrame, team: str, match_date: str) -> pd.DataFrame:
    df = _history_pre_match(history_df, match_date)
    if df.empty:
        return df

    m = (df["home_team"].astype(str) == str(team)) | (df["away_team"].astype(str) == str(team))
    out = df[m].copy()
    if out.empty:
        return out

    return out.sort_values(["date", "game_id"]).copy()


def _team_recent_metrics(team_games: pd.DataFrame, team: str, lookback: int = 5) -> Dict[str, float]:
    if team_games.empty:
        return {
            "sample_size": 0,
            "goals_scored_last5": 0.0,
            "goals_allowed_last5": 0.0,
            "over_2_5_rate_last5": 0.0,
            "btts_rate_last5": 0.0,
            "win_rate_last5": 0.0,
            "clean_sheet_rate_last5": 0.0,
            "failed_to_score_rate_last5": 0.0,
        }

    recent = team_games.tail(max(1, lookback)).copy()

    goals_scored = []
    goals_allowed = []
    over_flags = []
    btts_flags = []
    win_flags = []
    clean_sheet_flags = []
    failed_to_score_flags = []

    for _, row in recent.iterrows():
        home_team = str(row.get("home_team", ""))
        away_team = str(row.get("away_team", ""))
        home_score = _safe_int(row.get("home_score", 0))
        away_score = _safe_int(row.get("away_score", 0))

        if str(team) == home_team:
            gs, ga = home_score, away_score
        else:
            gs, ga = away_score, home_score

        goals_scored.append(gs)
        goals_allowed.append(ga)

        total_goals = gs + ga
        over_flags.append(1 if total_goals > 2.5 else 0)
        btts_flags.append(1 if (gs > 0 and ga > 0) else 0)
        win_flags.append(1 if gs > ga else 0)
        clean_sheet_flags.append(1 if ga == 0 else 0)
        failed_to_score_flags.append(1 if gs == 0 else 0)

    n = len(recent)
    return {
        "sample_size": int(n),
        "goals_scored_last5": float(np.mean(goals_scored)) if n else 0.0,
        "goals_allowed_last5": float(np.mean(goals_allowed)) if n else 0.0,
        "over_2_5_rate_last5": float(np.mean(over_flags)) if n else 0.0,
        "btts_rate_last5": float(np.mean(btts_flags)) if n else 0.0,
        "win_rate_last5": float(np.mean(win_flags)) if n else 0.0,
        "clean_sheet_rate_last5": float(np.mean(clean_sheet_flags)) if n else 0.0,
        "failed_to_score_rate_last5": float(np.mean(failed_to_score_flags)) if n else 0.0,
    }


def get_recent_team_form_features(
    history_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    match_date: str,
    lookback: int = 5,
) -> Dict[str, float]:
    home_games = _team_games_before(history_df, home_team, match_date)
    away_games = _team_games_before(history_df, away_team, match_date)

    home_metrics = _team_recent_metrics(home_games, home_team, lookback=lookback)
    away_metrics = _team_recent_metrics(away_games, away_team, lookback=lookback)

    out: Dict[str, float] = {
        "home_recent_games_count": int(home_metrics["sample_size"]),
        "away_recent_games_count": int(away_metrics["sample_size"]),
    }

    for key, val in home_metrics.items():
        if key == "sample_size":
            continue
        out[f"home_{key}"] = _safe_float(val)

    for key, val in away_metrics.items():
        if key == "sample_size":
            continue
        out[f"away_{key}"] = _safe_float(val)

    return out


def get_h2h_features(
    history_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    match_date: str,
) -> Dict[str, float]:
    df = _history_pre_match(history_df, match_date)
    if df.empty:
        return {
            "h2h_games_count": 0,
            "h2h_home_win_rate": 0.0,
            "h2h_away_win_rate": 0.0,
            "h2h_over_2_5_rate": 0.0,
            "h2h_btts_rate": 0.0,
            "h2h_avg_goals": 0.0,
        }

    m1 = (df["home_team"].astype(str) == str(home_team)) & (df["away_team"].astype(str) == str(away_team))
    m2 = (df["home_team"].astype(str) == str(away_team)) & (df["away_team"].astype(str) == str(home_team))

    h2h = df[m1 | m2].copy().sort_values(["date", "game_id"])
    if h2h.empty:
        return {
            "h2h_games_count": 0,
            "h2h_home_win_rate": 0.0,
            "h2h_away_win_rate": 0.0,
            "h2h_over_2_5_rate": 0.0,
            "h2h_btts_rate": 0.0,
            "h2h_avg_goals": 0.0,
        }

    n = len(h2h)
    home_wins = 0
    away_wins = 0
    over_flags = []
    btts_flags = []
    total_goals_list = []

    for _, row in h2h.iterrows():
        hs = _safe_int(row.get("home_score", 0))
        aw = _safe_int(row.get("away_score", 0))

        row_home = str(row.get("home_team", ""))
        row_away = str(row.get("away_team", ""))

        if hs > aw:
            winner = row_home
        elif aw > hs:
            winner = row_away
        else:
            winner = "DRAW"

        if winner == str(home_team):
            home_wins += 1
        elif winner == str(away_team):
            away_wins += 1

        tg = hs + aw
        total_goals_list.append(tg)
        over_flags.append(1 if tg > 2.5 else 0)
        btts_flags.append(1 if (hs > 0 and aw > 0) else 0)

    return {
        "h2h_games_count": int(n),
        "h2h_home_win_rate": float(home_wins / n) if n else 0.0,
        "h2h_away_win_rate": float(away_wins / n) if n else 0.0,
        "h2h_over_2_5_rate": float(np.mean(over_flags)) if n else 0.0,
        "h2h_btts_rate": float(np.mean(btts_flags)) if n else 0.0,
        "h2h_avg_goals": float(np.mean(total_goals_list)) if n else 0.0,
    }


def _make_event(event_name: str, direction: str, strength: float, explanation: str, category: str) -> Dict:
    return {
        "event_name": event_name,
        "direction": direction,
        "strength": float(np.clip(strength, 0.0, 0.04)),
        "explanation": explanation,
        "category": category,
    }


def detect_pre_match_events(
    recent_features: Dict[str, float],
    h2h_features: Dict[str, float],
    market_type: str,
) -> List[Dict]:
    events: List[Dict] = []

    hr = _safe_int(recent_features.get("home_recent_games_count", 0))
    ar = _safe_int(recent_features.get("away_recent_games_count", 0))
    min_recent = min(hr, ar)

    # Form events
    if _safe_float(recent_features.get("home_win_rate_last5")) >= 0.60:
        events.append(_make_event("home_good_form", "positive", 0.02, "Local con buena forma reciente.", "form"))
    if _safe_float(recent_features.get("away_win_rate_last5")) >= 0.60:
        events.append(_make_event("away_good_form", "positive", 0.02, "Visita con buena forma reciente.", "form"))

    # Attack/defense imbalance events
    if _safe_float(recent_features.get("home_goals_scored_last5")) >= 1.8:
        events.append(_make_event("home_team_hot_attack", "positive", 0.025, "Ataque local encendido.", "attack_defense"))
    if _safe_float(recent_features.get("away_goals_scored_last5")) >= 1.8:
        events.append(_make_event("away_team_hot_attack", "positive", 0.025, "Ataque visitante encendido.", "attack_defense"))

    if _safe_float(recent_features.get("home_goals_scored_last5")) <= 0.8:
        events.append(_make_event("home_team_cold_attack", "negative", 0.02, "Ataque local en baja.", "attack_defense"))
    if _safe_float(recent_features.get("away_goals_scored_last5")) <= 0.8:
        events.append(_make_event("away_team_cold_attack", "negative", 0.02, "Ataque visitante en baja.", "attack_defense"))

    if _safe_float(recent_features.get("home_goals_allowed_last5")) >= 1.6:
        events.append(_make_event("home_team_weak_defense", "negative", 0.02, "Defensa local vulnerable.", "attack_defense"))
    if _safe_float(recent_features.get("away_goals_allowed_last5")) >= 1.6:
        events.append(_make_event("away_team_weak_defense", "negative", 0.02, "Defensa visitante vulnerable.", "attack_defense"))

    # Trend events
    if (
        _safe_float(recent_features.get("home_over_2_5_rate_last5")) >= 0.60
        and _safe_float(recent_features.get("away_over_2_5_rate_last5")) >= 0.60
    ):
        events.append(_make_event("both_teams_over_trend", "positive", 0.03, "Ambos equipos con tendencia Over.", "trend"))

    if (
        _safe_float(recent_features.get("home_btts_rate_last5")) >= 0.60
        and _safe_float(recent_features.get("away_btts_rate_last5")) >= 0.60
    ):
        events.append(_make_event("both_teams_btts_trend", "positive", 0.03, "Ambos con tendencia BTTS.", "trend"))

    # Matchup events
    h2h_n = _safe_int(h2h_features.get("h2h_games_count", 0))
    if h2h_n >= 4 and _safe_float(h2h_features.get("h2h_over_2_5_rate")) >= 0.62:
        events.append(_make_event("strong_h2h_over_trend", "positive", 0.02, "H2H con sesgo Over.", "matchup"))
    if h2h_n >= 4 and _safe_float(h2h_features.get("h2h_btts_rate")) >= 0.62:
        events.append(_make_event("strong_h2h_btts_trend", "positive", 0.02, "H2H con sesgo BTTS.", "matchup"))

    # Rest/schedule events
    home_load = _safe_float(recent_features.get("home_recent_games_count"))
    away_load = _safe_float(recent_features.get("away_recent_games_count"))
    if home_load + 1.0 < away_load:
        events.append(_make_event("home_rest_advantage", "positive", 0.015, "Local llega con mejor descanso relativo.", "rest"))
    if away_load + 1.0 < home_load:
        events.append(_make_event("away_rest_advantage", "positive", 0.015, "Visitante llega con mejor descanso relativo.", "rest"))

    # Reliability tag as neutral event for transparency when sample is low.
    if min_recent < 4:
        events.append(_make_event("low_recent_sample", "neutral", 0.0, "Muestra reciente baja; ajustes moderados.", "reliability"))
    if h2h_n < 4:
        events.append(_make_event("low_h2h_sample", "neutral", 0.0, "Muestra H2H baja; peso H2H reducido.", "reliability"))

    return events


def _event_sign_for_market(event: Dict, market_type: str) -> float:
    name = str(event.get("event_name", ""))
    direction = str(event.get("direction", "neutral")).lower()

    if direction == "neutral":
        return 0.0

    # Binary markets where positive means higher YES probability.
    if market_type in {"over_25", "btts", "home_over_05", "away_over_05"}:
        # In these markets, weak defenses and hot attacks increase YES side.
        if market_type == "over_25":
            if "cold_attack" in name:
                return -1.0
            if "weak_defense" in name or "over_trend" in name:
                return 1.0
        if market_type == "btts":
            if "clean_sheet" in name or "cold_attack" in name:
                return -1.0
            if "btts" in name or "weak_defense" in name:
                return 1.0

        if direction == "positive":
            return 1.0
        return -1.0

    # Full game home/away/draw handling.
    if market_type == "full_game_home":
        if name.startswith("home_"):
            return 1.0 if direction == "positive" else -1.0
        if name.startswith("away_"):
            return -1.0 if direction == "positive" else 1.0
        if "draw_" in name or "parity" in name:
            return -0.5
        return 0.0

    if market_type == "full_game_away":
        if name.startswith("away_"):
            return 1.0 if direction == "positive" else -1.0
        if name.startswith("home_"):
            return -1.0 if direction == "positive" else 1.0
        if "draw_" in name or "parity" in name:
            return -0.5
        return 0.0

    if market_type == "full_game_draw":
        if "draw_" in name or "parity" in name:
            return 1.0
        if "hot_attack" in name:
            return -0.5
        return 0.0

    if market_type == "full_game":
        if direction == "positive":
            return 0.4
        if direction == "negative":
            return -0.4
        return 0.0

    return 0.0


def _market_multiplier(market_type: str) -> float:
    if market_type == "full_game_draw":
        return 0.8
    if market_type in {"full_game_home", "full_game_away", "full_game"}:
        return 0.9
    if market_type in {"over_25", "btts"}:
        return 1.0
    if market_type in {"home_over_05", "away_over_05"}:
        return 0.9
    return 0.85


def calculate_adjustment_score(
    events: List[Dict],
    market_type: str,
    reliability: float,
    max_cap: float = MAX_ABS_ADJUSTMENT,
) -> Dict:
    rel = float(np.clip(_safe_float(reliability, 0.0), 0.0, 1.0))
    market_mult = _market_multiplier(market_type)

    raw_adjustment = 0.0
    breakdown = []

    for event in events:
        base_strength = float(np.clip(_safe_float(event.get("strength", 0.0)), 0.0, 0.04))
        sign = _event_sign_for_market(event, market_type)
        contribution = sign * base_strength * rel * market_mult
        raw_adjustment += contribution

        breakdown.append({
            "event_name": event.get("event_name", "unknown"),
            "direction": event.get("direction", "neutral"),
            "strength": base_strength,
            "sign": sign,
            "contribution": float(contribution),
            "explanation": event.get("explanation", ""),
            "category": event.get("category", "general"),
        })

    capped_adjustment = float(np.clip(raw_adjustment, -abs(max_cap), abs(max_cap)))

    return {
        "raw_adjustment": float(raw_adjustment),
        "capped_adjustment": float(capped_adjustment),
        "breakdown": breakdown,
    }


def apply_probability_adjustment(
    base_prob: float,
    events: List[Dict],
    h2h_features: Dict[str, float],
    market_type: str,
) -> Dict:
    base = _clip_prob(base_prob)

    if not USE_EVENT_ADJUSTMENTS:
        return {
            "adjusted_prob": float(np.clip(base, 0.05, 0.95)),
            "adjustment_amount": 0.0,
            "adjustment_breakdown": [],
            "raw_adjustment": 0.0,
            "capped_adjustment": 0.0,
            "reliability": 0.0,
        }

    h2h_n = _safe_int(h2h_features.get("h2h_games_count", 0))
    h2h_reliability = float(min(1.0, h2h_n / 6.0))

    # Recent reliability is inferred from explicit low-sample events if present.
    low_recent = any(str(e.get("event_name", "")) == "low_recent_sample" for e in events)
    recent_reliability = 0.6 if low_recent else 1.0

    reliability = float(np.clip(0.7 * recent_reliability + 0.3 * h2h_reliability, 0.0, 1.0))

    if reliability < 0.25:
        return {
            "adjusted_prob": float(np.clip(base, 0.05, 0.95)),
            "adjustment_amount": 0.0,
            "adjustment_breakdown": [],
            "raw_adjustment": 0.0,
            "capped_adjustment": 0.0,
            "reliability": reliability,
        }

    score_info = calculate_adjustment_score(
        events=events,
        market_type=market_type,
        reliability=reliability,
        max_cap=MAX_ABS_ADJUSTMENT,
    )

    # Calibration safety: if near 0.50, avoid aggressive move.
    distance_from_mid = abs(base - 0.5)
    center_shrink = 0.5 + min(0.5, distance_from_mid / 0.45)

    # Additional shrinkage when evidence is weak.
    evidence_shrink = 0.55 + 0.45 * reliability

    final_adjustment = score_info["capped_adjustment"] * center_shrink * evidence_shrink
    adjusted_prob = float(np.clip(base + final_adjustment, 0.05, 0.95))

    return {
        "adjusted_prob": adjusted_prob,
        "adjustment_amount": float(adjusted_prob - base),
        "adjustment_breakdown": score_info["breakdown"],
        "raw_adjustment": float(score_info["raw_adjustment"]),
        "capped_adjustment": float(score_info["capped_adjustment"]),
        "reliability": reliability,
    }


def probability_to_confidence(prob: float, mode: str = "sym_50_100") -> int:
    p = _clip_prob(prob)

    if mode == "raw_0_100":
        return int(round(p * 100))

    # Default scheme used in the project: confidence on selected side.
    return int(round(max(p, 1.0 - p) * 100))


