from __future__ import annotations

import json
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple


ESPN_ALL_LEAGUES_CODE = "__all__"


ESPN_LEAGUE_OPTIONS: Dict[str, str] = {
    "Todas las ligas ESPN (auto)": ESPN_ALL_LEAGUES_CODE,
    "Liga MX": "mex.1",
    "Liga Expansion MX": "mex.2",
    "Premier League": "eng.1",
    "LaLiga": "esp.1",
    "LaLiga 2": "esp.2",
    "Serie A": "ita.1",
    "Bundesliga": "ger.1",
    "Ligue 1": "fra.1",
    "MLS": "usa.1",
    "Brasil Serie A": "bra.1",
    "Liga Profesional Argentina": "arg.1",
    "Champions League": "uefa.champions",
    "Europa League": "uefa.europa",
}

DEFAULT_ESPN_LEAGUES: Tuple[str, ...] = (
    ESPN_ALL_LEAGUES_CODE,
)

_STOPWORDS = {
    "fc",
    "cf",
    "club",
    "de",
    "sc",
    "cd",
    "ac",
    "afc",
}

_TEAM_ALIAS_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bc\s*\.?\s*azul\b"), "cruz azul"),
    (re.compile(r"\baguilas\b"), "america"),
    (re.compile(r"\bfriburgo\b"), "freiburg"),
    (re.compile(r"\bescocia\b"), "scotland"),
    (re.compile(r"\bjapon\b"), "japan"),
    (re.compile(r"\bbelgica\b"), "belgium"),
    (re.compile(r"\bfrancia\b"), "france"),
    (re.compile(r"\bcolombia\b"), "colombia"),
    (re.compile(r"\bcorea\s*sur\b"), "south korea"),
    (re.compile(r"\bcosta\s*marf\b|\bcostamarf\b|\bcosta\s*marfil\b|\bcosta\s*de\s*marfil\b"), "ivory coast"),
    (re.compile(r"\beua\b|\beeuu\b|\bestados\s*unidos\b"), "usa"),
    (re.compile(r"\be\s*\.?\s*u\s*\.?\s*a\s*\.?\b"), "usa"),
    (re.compile(r"\bs\s*\.?\s*laguna\s*f\b|\bslagunaf\b"), "santos laguna"),
    (re.compile(r"\bsan\s*luis\s*f\b|\bsanluisf\b"), "atletico de san luis"),
    (re.compile(r"\blaspalmas\b"), "las palmas"),
    (re.compile(r"\boncecaldas\b"), "once caldas"),
    (re.compile(r"\bdepcoruna\b"), "deportivo la coruna"),
    (re.compile(r"\bint\s*\.?\s*bogota\b"), "bogota"),
    (re.compile(r"\bmunitedf\b"), "manchester united"),
    (re.compile(r"\bmancityf\b"), "manchester city"),
    (re.compile(r"\bsinaloa\b"), "dorados de sinaloa"),
    (re.compile(r"\bcorrecamino\b"), "correcaminos"),
    (re.compile(r"\bzacatecas\b"), "mineros de zacatecas"),
    (re.compile(r"\bvalladolid\b"), "real valladolid"),
    (re.compile(r"\bsp\s*\.?\s*gijon\b"), "sporting gijon"),
    (re.compile(r"\bdep\s*\.?\s*coruna\b"), "deportivo la coruna"),
    (re.compile(r"\bath\s*\.?\s*bilbao\b"), "athletic bilbao"),
    (re.compile(r"\bsanjose\b"), "san jose"),
    (re.compile(r"\bsandiego\b"), "san diego"),
    (re.compile(r"\bportland\s*timb\b|\bportlandtimb\b"), "portland timbers"),
    (re.compile(r"\bhouston\b"), "houston dynamo"),
    (re.compile(r"\bseattle\b"), "seattle sounders"),
    (re.compile(r"\bmarsella\b"), "marseille"),
    (re.compile(r"\bstuttgart\b"), "stuttgart"),
    (re.compile(r"\brosario\s*cen\b"), "rosario central"),
    (re.compile(r"\bl\s*\.?\s*a\s*\.?\s*galaxy\b"), "la galaxy"),
)

_MIN_PAIR_SCORE = 0.66
_MIN_SIDE_SCORE = 0.42
_HISTORICAL_MIN_PAIR_SCORE = 0.78
_HISTORICAL_MIN_SIDE_SCORE = 0.55
_PRIMARY_LOOKBACK_DAYS = 7
_PRIMARY_LOOKAHEAD_DAYS = 2
_MAX_LOOKBACK_DAYS = 35
_MAX_LOOKAHEAD_DAYS = 3
_DATE_DISTANCE_PENALTY = 0.02


@dataclass(frozen=True)
class EspnEvent:
    event_id: str
    league: str
    event_date: str
    home_team: str
    away_team: str
    home_score: int | None
    away_score: int | None
    completed: bool
    status_name: str


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _normalize_team_name(team: str) -> str:
    lowered = _strip_accents(str(team or "").lower())
    for pattern, replacement in _TEAM_ALIAS_PATTERNS:
        lowered = pattern.sub(replacement, lowered)

    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    tokens = [
        token
        for token in lowered.split()
        if token and token not in _STOPWORDS
    ]
    return " ".join(tokens)


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0

    ratio = SequenceMatcher(None, left, right).ratio()
    left_compact = left.replace(" ", "")
    right_compact = right.replace(" ", "")
    compact_ratio = SequenceMatcher(None, left_compact, right_compact).ratio()
    ratio = max(ratio, compact_ratio)

    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if left_tokens and right_tokens:
        overlap = len(left_tokens.intersection(right_tokens)) / max(1, len(left_tokens.union(right_tokens)))
        ratio = max(ratio, overlap)

    if left in right or right in left:
        ratio = max(ratio, 0.88)
    if left_compact in right_compact or right_compact in left_compact:
        ratio = max(ratio, 0.90)
    return ratio


def _pair_similarity(local: str, visitante: str, event: EspnEvent) -> Tuple[float, bool, float, float]:
    local_n = _normalize_team_name(local)
    visitante_n = _normalize_team_name(visitante)
    home_n = _normalize_team_name(event.home_team)
    away_n = _normalize_team_name(event.away_team)

    direct_local = _similarity(local_n, home_n)
    direct_visit = _similarity(visitante_n, away_n)
    direct = 0.5 * (direct_local + direct_visit)

    swapped_local = _similarity(local_n, away_n)
    swapped_visit = _similarity(visitante_n, home_n)
    swapped = 0.5 * (swapped_local + swapped_visit)

    if swapped > direct:
        return swapped, True, swapped_local, swapped_visit
    return direct, False, direct_local, direct_visit


def _parse_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _events_from_payload(payload: Dict[str, object], league_fallback: str) -> Tuple[EspnEvent, ...]:
    events: List[EspnEvent] = []
    for raw_event in payload.get("events", []):
        competitions = raw_event.get("competitions") or []
        if not competitions:
            continue

        competition = competitions[0] if competitions else {}
        competitors = competition.get("competitors") or []
        home = next((item for item in competitors if item.get("homeAway") == "home"), None)
        away = next((item for item in competitors if item.get("homeAway") == "away"), None)

        if not home or not away:
            continue

        status = raw_event.get("status") or {}
        status_type = status.get("type") or {}
        league_data = competition.get("league") or {}
        league_name = str(
            league_data.get("abbreviation")
            or league_data.get("shortName")
            or league_data.get("name")
            or league_fallback
        )

        events.append(
            EspnEvent(
                event_id=str(raw_event.get("id", "")),
                league=league_name,
                event_date=str(raw_event.get("date", "") or ""),
                home_team=str((home.get("team") or {}).get("displayName", "") or ""),
                away_team=str((away.get("team") or {}).get("displayName", "") or ""),
                home_score=_parse_int(home.get("score")),
                away_score=_parse_int(away.get("score")),
                completed=bool(status_type.get("completed", False)),
                status_name=str(status_type.get("name", "") or ""),
            )
        )

    return tuple(events)


@lru_cache(maxsize=512)
def _fetch_scoreboard(league: str, yyyymmdd: str) -> Tuple[EspnEvent, ...]:
    base = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard"
    query = urllib.parse.urlencode({"dates": yyyymmdd})
    url = f"{base}?{query}"

    with urllib.request.urlopen(url, timeout=18) as response:
        payload = json.loads(response.read().decode("utf-8"))

    return _events_from_payload(payload, league)


@lru_cache(maxsize=512)
def _fetch_scoreboard_all(yyyymmdd: str) -> Tuple[EspnEvent, ...]:
    base = "https://site.api.espn.com/apis/site/v2/sports/soccer/all/scoreboard"
    query = urllib.parse.urlencode({"dates": yyyymmdd})
    url = f"{base}?{query}"

    with urllib.request.urlopen(url, timeout=25) as response:
        payload = json.loads(response.read().decode("utf-8"))

    return _events_from_payload(payload, "all")


def clear_espn_cache() -> None:
    _fetch_scoreboard.cache_clear()
    _fetch_scoreboard_all.cache_clear()


def _result_symbol(event: EspnEvent, swapped: bool) -> str | None:
    if event.home_score is None or event.away_score is None:
        return None

    if event.home_score == event.away_score:
        symbol = "X"
    elif event.home_score > event.away_score:
        symbol = "1"
    else:
        symbol = "2"

    if not swapped:
        return symbol

    if symbol == "1":
        return "2"
    if symbol == "2":
        return "1"
    return "X"


def _score_text(event: EspnEvent, swapped: bool) -> str:
    if event.home_score is None or event.away_score is None:
        return "-"
    if swapped:
        return f"{event.away_score}-{event.home_score}"
    return f"{event.home_score}-{event.away_score}"


def _date_candidates(base_date: date, include_previous_day: bool) -> List[str]:
    if not include_previous_day:
        return [base_date.strftime("%Y%m%d")]

    days = [0]

    for offset in range(1, max(_PRIMARY_LOOKBACK_DAYS, _PRIMARY_LOOKAHEAD_DAYS) + 1):
        if offset <= _PRIMARY_LOOKBACK_DAYS:
            days.append(-offset)
        if offset <= _PRIMARY_LOOKAHEAD_DAYS:
            days.append(offset)

    for offset in range(_PRIMARY_LOOKBACK_DAYS + 1, _MAX_LOOKBACK_DAYS + 1):
        days.append(-offset)
    for offset in range(_PRIMARY_LOOKAHEAD_DAYS + 1, _MAX_LOOKAHEAD_DAYS + 1):
        days.append(offset)

    seen = set()
    ordered: List[str] = []
    for delta in days:
        code = (base_date + timedelta(days=delta)).strftime("%Y%m%d")
        if code in seen:
            continue
        seen.add(code)
        ordered.append(code)
    return ordered


def _is_hit_for_pick(actual_symbol: str, pick_symbol: str, double_pick: str) -> bool:
    _ = double_pick
    return actual_symbol == pick_symbol


def _event_day_distance(event: EspnEvent, base_date: date) -> int | None:
    raw = str(event.event_date or "")
    if len(raw) < 10:
        return None
    try:
        event_day = date.fromisoformat(raw[:10])
    except ValueError:
        return None
    return abs((event_day - base_date).days)


def _is_confident_match(
    pair_score: float,
    local_score: float,
    visit_score: float,
    min_pair_score: float,
    min_side_score: float,
    distance_days: int | None,
) -> bool:
    min_side = min(local_score, visit_score)
    max_side = max(local_score, visit_score)

    strict_ok = pair_score >= min_pair_score and min_side >= min_side_score
    if strict_ok:
        return True

    if distance_days is not None and distance_days <= 1:
        if pair_score >= max(min_pair_score - 0.08, 0.68) and min_side >= max(min_side_score - 0.18, 0.32) and max_side >= 0.78:
            return True
        if pair_score >= 0.75 and min_side >= 0.36:
            return True

    return False


def lookup_results_for_rows(
    rows: Sequence[Dict[str, object]],
    match_date: date,
    league_codes: Sequence[str] | None = None,
    include_previous_day: bool = True,
) -> Tuple[Dict[int, Dict[str, object]], List[str]]:
    leagues = tuple(league_codes) if league_codes else DEFAULT_ESPN_LEAGUES
    has_all_endpoint = ESPN_ALL_LEAGUES_CODE in leagues
    specific_leagues = tuple(code for code in leagues if code != ESPN_ALL_LEAGUES_CODE)
    date_codes = _date_candidates(match_date, include_previous_day)

    collected_events: List[EspnEvent] = []
    notes: List[str] = []

    for date_code in date_codes:
        if has_all_endpoint:
            try:
                collected_events.extend(_fetch_scoreboard_all(date_code))
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    notes.append(f"ESPN all {date_code}: {exc.code}")
            except Exception as exc:  # noqa: BLE001
                notes.append(f"ESPN all {date_code}: {type(exc).__name__}")

        for league in specific_leagues:
            try:
                collected_events.extend(_fetch_scoreboard(league, date_code))
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    notes.append(f"ESPN {league} {date_code}: {exc.code}")
            except Exception as exc:  # noqa: BLE001
                notes.append(f"ESPN {league} {date_code}: {type(exc).__name__}")

    results_by_row: Dict[int, Dict[str, object]] = {}
    unresolved_debug: List[str] = []

    for row in rows:
        partido = int(row.get("partido", 0) or 0)
        local = str(row.get("local", "") or "")
        visitante = str(row.get("visitante", "") or "")
        pick_symbol = str(row.get("pick_symbol", "") or "").upper()
        double_pick = str(row.get("doble_oportunidad", "") or "").upper().replace(" ", "")

        if partido <= 0 or pick_symbol not in {"1", "X", "2"}:
            continue

        best_event: EspnEvent | None = None
        best_score = 0.0
        best_effective_score = float("-inf")
        best_swapped = False
        best_local_score = 0.0
        best_visit_score = 0.0
        best_distance_days: int | None = None

        for event in collected_events:
            score, swapped, local_score, visit_score = _pair_similarity(local, visitante, event)
            distance_days = _event_day_distance(event, match_date)
            distance_penalty = _DATE_DISTANCE_PENALTY * float(min(distance_days or 0, 30))
            effective_score = score - distance_penalty

            if (
                effective_score > best_effective_score
                or (
                    effective_score == best_effective_score
                    and score > best_score
                )
            ):
                best_event = event
                best_score = score
                best_effective_score = effective_score
                best_swapped = swapped
                best_local_score = local_score
                best_visit_score = visit_score
                best_distance_days = distance_days

        is_historical_fallback = bool(best_distance_days is not None and best_distance_days > (_PRIMARY_LOOKBACK_DAYS + _PRIMARY_LOOKAHEAD_DAYS))
        min_pair_score = _HISTORICAL_MIN_PAIR_SCORE if is_historical_fallback else _MIN_PAIR_SCORE
        min_side_score = _HISTORICAL_MIN_SIDE_SCORE if is_historical_fallback else _MIN_SIDE_SCORE

        is_confident_match = bool(
            best_event is not None
            and _is_confident_match(
                pair_score=best_score,
                local_score=best_local_score,
                visit_score=best_visit_score,
                min_pair_score=min_pair_score,
                min_side_score=min_side_score,
                distance_days=best_distance_days,
            )
        )

        if not is_confident_match or best_event is None:
            results_by_row[partido] = {
                "resultado_estado": "sin_resultado",
                "resultado_texto": "Sin resultado ESPN",
                "resultado_real": "-",
                "marcador_real": "-",
            }
            unresolved_debug.append(
                f"P{partido}:sim={best_score:.3f},side={min(best_local_score,best_visit_score):.3f},d={best_distance_days}"
            )
            continue

        actual_symbol = _result_symbol(best_event, best_swapped)
        score_text = _score_text(best_event, best_swapped)

        if not best_event.completed or actual_symbol is None:
            results_by_row[partido] = {
                "resultado_estado": "pendiente",
                "resultado_texto": "Pendiente",
                "resultado_real": "-",
                "marcador_real": score_text,
            }
            continue

        is_hit = _is_hit_for_pick(actual_symbol, pick_symbol, double_pick)
        results_by_row[partido] = {
            "resultado_estado": "acierto" if is_hit else "fallo",
            "resultado_texto": "Acierto" if is_hit else "Fallo",
            "resultado_real": actual_symbol,
            "marcador_real": score_text,
        }

    if unresolved_debug:
        notes.append(
            "ESPN unresolved debug: " + " | ".join(unresolved_debug[:8])
        )

    return results_by_row, notes
