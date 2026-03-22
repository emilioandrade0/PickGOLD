from __future__ import annotations

import re
from typing import Any, Dict, Optional


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        txt = str(value).strip()
        if txt == "":
            return None
        return float(txt)
    except Exception:
        return None


def _extract_price(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        for key in (
            "american",
            "us",
            "price",
            "odds",
            "value",
            "displayValue",
            "current",
            "closing",
            "decimal",
        ):
            if key in value:
                parsed = _extract_price(value.get(key))
                if parsed is not None:
                    return parsed
        return None

    if isinstance(value, (int, float)):
        return float(value)

    txt = str(value or "").strip()
    if not txt:
        return None

    # Strings like "+110", "-105", "1.91", "EVEN"
    if txt.upper() in {"EV", "EVEN", "EVS"}:
        return 100.0

    m = re.search(r"([+-]?\d+(?:\.\d+)?)", txt)
    if not m:
        return None

    return _safe_float(m.group(1))


def _coalesce_price(*values: Any) -> Optional[float]:
    for value in values:
        parsed = _extract_price(value)
        if parsed is not None:
            return parsed
    return None


def _get_nested(data: dict, *path: str):
    cur = data
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def extract_market_odds_fields(odds: dict) -> Dict[str, float]:
    """
    Extract normalized market odds fields from an ESPN odds object.

    Output keys are aligned with the keys consumed by best-picks/audit modules.
    """
    if not isinstance(odds, dict):
        return {}

    fields: Dict[str, float] = {}

    home_ml = _coalesce_price(
        odds.get("homeTeamOdds"),
        odds.get("homeOdds"),
        odds.get("homeMoneyLine"),
        _get_nested(odds, "moneyline", "home", "close", "odds"),
        _get_nested(odds, "moneyline", "home", "open", "odds"),
    )
    away_ml = _coalesce_price(
        odds.get("awayTeamOdds"),
        odds.get("awayOdds"),
        odds.get("awayMoneyLine"),
        _get_nested(odds, "moneyline", "away", "close", "odds"),
        _get_nested(odds, "moneyline", "away", "open", "odds"),
    )

    if home_ml is not None:
        fields["home_moneyline_odds"] = float(home_ml)
    if away_ml is not None:
        fields["away_moneyline_odds"] = float(away_ml)

    # Generic moneyline field: use available favorite-side proxy when both sides exist.
    if home_ml is not None and away_ml is not None:
        fields["closing_moneyline_odds"] = float(home_ml if home_ml <= away_ml else away_ml)
    elif home_ml is not None:
        fields["closing_moneyline_odds"] = float(home_ml)
    elif away_ml is not None:
        fields["closing_moneyline_odds"] = float(away_ml)

    spread_odds = _coalesce_price(
        odds.get("spreadOdds"),
        odds.get("homeSpreadOdds"),
        odds.get("awaySpreadOdds"),
        _get_nested(odds, "pointSpread", "home", "close", "odds"),
        _get_nested(odds, "pointSpread", "away", "close", "odds"),
        _get_nested(odds, "pointSpread", "home", "open", "odds"),
        _get_nested(odds, "pointSpread", "away", "open", "odds"),
    )
    if spread_odds is not None:
        fields["closing_spread_odds"] = float(spread_odds)

    total_odds = _coalesce_price(
        odds.get("overOdds"),
        odds.get("underOdds"),
        odds.get("totalOdds"),
        odds.get("overUnderOdds"),
        _get_nested(odds, "total", "over", "close", "odds"),
        _get_nested(odds, "total", "under", "close", "odds"),
        _get_nested(odds, "total", "over", "open", "odds"),
        _get_nested(odds, "total", "under", "open", "odds"),
    )
    if total_odds is not None:
        fields["closing_total_odds"] = float(total_odds)

    return fields


def odds_data_quality(fields: Dict[str, float]) -> str:
    return "real" if fields else "fallback"
