from __future__ import annotations

from pathlib import Path
import json
import sys

import joblib
import pandas as pd


SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
MODELS_DIR = BASE_DIR / "data" / "tennis" / "models"
META_FILE = MODELS_DIR / "match_winner_model_meta.json"
MODEL_FILE = MODELS_DIR / "match_winner_model.pkl"

from sports.tennis.tennis_features import FEATURE_COLUMNS, implied_prob_from_decimal


def confidence_from_prob(prob: float | None) -> int | None:
    if prob is None or pd.isna(prob):
        return None
    return int(round(max(50.0, min(89.0, float(prob) * 100.0))))


def tier_from_conf(conf: int | None) -> str:
    if conf is None:
        return "SIN MODELO"
    if conf >= 74:
        return "ELITE"
    if conf >= 67:
        return "PREMIUM"
    if conf >= 60:
        return "STRONG"
    if conf >= 54:
        return "NORMAL"
    return "PASS"


def fallback_pick(row: pd.Series) -> tuple[str, float | None, str]:
    player_a = str(row.get("player_a") or "PLAYER A")
    player_b = str(row.get("player_b") or "PLAYER B")
    odds_a = pd.to_numeric(row.get("player_a_odds"), errors="coerce")
    odds_b = pd.to_numeric(row.get("player_b_odds"), errors="coerce")

    if pd.notna(odds_a) and pd.notna(odds_b):
        prob_a = implied_prob_from_decimal(odds_a)
        prob_b = implied_prob_from_decimal(odds_b)
        if pd.notna(prob_a) and pd.notna(prob_b):
            if float(prob_a) >= float(prob_b):
                return player_a, float(prob_a), "odds_favorite"
            return player_b, float(prob_b), "odds_favorite"

    rank_a = pd.to_numeric(row.get("player_a_rank"), errors="coerce")
    rank_b = pd.to_numeric(row.get("player_b_rank"), errors="coerce")
    if pd.notna(rank_a) and pd.notna(rank_b):
        if float(rank_a) < float(rank_b):
            return player_a, 0.56, "ranking_fallback"
        if float(rank_b) < float(rank_a):
            return player_b, 0.56, "ranking_fallback"

    return "Pendiente", None, "no_signal"


def load_model():
    if not MODEL_FILE.exists() or not META_FILE.exists():
        return None, FEATURE_COLUMNS
    try:
        model = joblib.load(MODEL_FILE)
        meta = json.loads(META_FILE.read_text(encoding="utf-8"))
        feature_columns = meta.get("feature_columns") or FEATURE_COLUMNS
        return model, feature_columns
    except Exception:
        return None, FEATURE_COLUMNS


def predict_match(row: pd.Series, feature_row: pd.Series | dict | None, model=None, feature_columns=None) -> tuple[str, int | None, str, float | None]:
    player_a = str(row.get("player_a") or "PLAYER A")
    player_b = str(row.get("player_b") or "PLAYER B")
    predicted_prob = None
    model_source = "fallback"
    feature_columns = feature_columns or FEATURE_COLUMNS

    if model is not None and feature_row is not None:
        try:
            X = pd.DataFrame([{col: (feature_row.get(col) if hasattr(feature_row, 'get') else None) for col in feature_columns}])
            predicted_prob = float(model.predict_proba(X)[0, 1])
            model_source = "logistic_regression"
        except Exception:
            predicted_prob = None
            model_source = "fallback"

    if predicted_prob is not None:
        if predicted_prob >= 0.5:
            return player_a, confidence_from_prob(predicted_prob), model_source, predicted_prob
        return player_b, confidence_from_prob(1.0 - predicted_prob), model_source, 1.0 - predicted_prob

    pick, fallback_prob, fallback_source = fallback_pick(row)
    return pick, confidence_from_prob(fallback_prob), fallback_source, fallback_prob
