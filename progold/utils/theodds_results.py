from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

from .espn_results import _normalize_team_name, _pair_similarity


DEFAULT_THEODDS_API_KEY = "b9102e6779f44a028ed3e71000e5ef9c"
THEODDS_CACHE_FILE = Path(__file__).resolve().parents[1] / "cache" / "theodds_ticket_cache.xlsx"

_THEODDS_BASE_URL = "https://api.the-odds-api.com/v4"
_THEODDS_DAYS_FROM = 3
# Query all currently available soccer keys (free tier observed ~51).
# Cache-by-ticket keeps this from repeating credit consumption for the same ticket.
_THEODDS_MAX_SPORTS_PER_LOOKUP = 40
_THEODDS_MATCHER_VERSION = "v3_direct_pick_only"
_CACHE_REFRESH_HOURS_UNRESOLVED = 3

_MIN_PAIR_SCORE = 0.66
_MIN_SIDE_SCORE = 0.42
_HISTORICAL_MIN_PAIR_SCORE = 0.76
_HISTORICAL_MIN_SIDE_SCORE = 0.54
_DATE_DISTANCE_PENALTY = 0.02

_TICKETS_COLUMNS = [
    "ticket_id",
    "lookup_date",
    "row_count",
    "created_at_utc",
    "provider",
    "api_requests_used",
    "api_requests_remaining",
    "api_requests_last_total",
]

_MATCHES_COLUMNS = [
    "ticket_id",
    "partido",
    "local",
    "visitante",
    "pick_symbol",
    "doble_oportunidad",
    "resultado_estado",
    "resultado_texto",
    "resultado_real",
    "marcador_real",
    "provider",
    "matched_sport_key",
    "matched_event_id",
    "matched_home_team",
    "matched_away_team",
    "matched_commence_time",
    "matched_similarity",
    "matched_side_similarity",
]

_PRIORITY_SPORT_KEYS = (
    "soccer_mexico_ligamx",
    "soccer_spain_segunda_division",
    "soccer_spain_la_liga",
    "soccer_fifa_world_cup",
    "soccer_usa_mls",
    "soccer_argentina_primera_division",
    "soccer_brazil_campeonato",
    "soccer_chile_campeonato",
    "soccer_japan_j_league",
    "soccer_korea_kleague1",
    "soccer_epl",
    "soccer_efl_champ",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "soccer_france_ligue_one",
    "soccer_spl",
)

_SPORT_HINTS: Tuple[Tuple[Tuple[str, ...], Tuple[str, ...]], ...] = (
    (("gijon", "coruna", "valladolid", "eibar", "las palmas", "burgos", "villarreal", "valencia", "betis", "osasuna"), ("soccer_spain_segunda_division", "soccer_spain_la_liga")),
    (("zacatecas", "tapatio", "sinaloa", "santos", "queretaro", "tigres", "atlas", "san luis", "pachuca", "mazatlan"), ("soccer_mexico_ligamx",)),
    (("japan", "escocia", "scotland", "belgium", "france", "usa", "south korea", "ivory coast", "colombia"), ("soccer_fifa_world_cup",)),
    (("junior", "once caldas", "llaneros", "bogota"), ("soccer_fifa_world_cup",)),
    (("fluminense", "flamengo", "santos", "palmeiras", "corinthians"), ("soccer_brazil_campeonato",)),
    (("river plate", "racing", "boca", "rosario central", "huracan", "atalanta", "juventus"), ("soccer_argentina_primera_division", "soccer_italy_serie_a")),
)


@dataclass(frozen=True)
class OddsEvent:
    sport_key: str
    event_id: str
    commence_time: str
    home_team: str
    away_team: str
    completed: bool
    scores: Tuple[Tuple[str, int | None], ...]


def clear_theodds_runtime_cache() -> None:
    _fetch_soccer_sport_keys.cache_clear()
    _fetch_scores_for_sport.cache_clear()


def _parse_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _is_hit_for_pick(actual_symbol: str, pick_symbol: str, double_pick: str) -> bool:
    _ = double_pick
    return actual_symbol == pick_symbol


def _score_text(home_score: int | None, away_score: int | None, swapped: bool) -> str:
    if home_score is None or away_score is None:
        return "-"
    if swapped:
        return f"{away_score}-{home_score}"
    return f"{home_score}-{away_score}"


def _result_symbol(home_score: int | None, away_score: int | None, swapped: bool) -> str | None:
    if home_score is None or away_score is None:
        return None

    if home_score == away_score:
        symbol = "X"
    elif home_score > away_score:
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


def _ticket_id(rows: Sequence[Dict[str, object]], lookup_date: date) -> str:
    components = [lookup_date.isoformat(), _THEODDS_MATCHER_VERSION]
    ordered_rows = sorted(rows, key=lambda item: int(item.get("partido", 0) or 0))
    for row in ordered_rows:
        partido = int(row.get("partido", 0) or 0)
        local = _normalize_team_name(str(row.get("local", "") or ""))
        visitante = _normalize_team_name(str(row.get("visitante", "") or ""))
        pick = str(row.get("pick_symbol", "") or "").upper()
        doble = str(row.get("doble_oportunidad", "") or "").upper().replace(" ", "")
        components.append(f"{partido}:{local}:{visitante}:{pick}:{doble}")

    payload = "|".join(components).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _empty_tickets_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_TICKETS_COLUMNS)


def _empty_matches_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_MATCHES_COLUMNS)


def _read_cache_tables(cache_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not cache_path.exists():
        return _empty_tickets_df(), _empty_matches_df()

    try:
        tickets_df = pd.read_excel(cache_path, sheet_name="tickets")
    except Exception:
        tickets_df = _empty_tickets_df()

    try:
        matches_df = pd.read_excel(cache_path, sheet_name="matches")
    except Exception:
        matches_df = _empty_matches_df()

    for column in _TICKETS_COLUMNS:
        if column not in tickets_df.columns:
            tickets_df[column] = None
    for column in _MATCHES_COLUMNS:
        if column not in matches_df.columns:
            matches_df[column] = None

    return tickets_df[_TICKETS_COLUMNS], matches_df[_MATCHES_COLUMNS]


def _write_cache_tables(cache_path: Path, tickets_df: pd.DataFrame, matches_df: pd.DataFrame) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(cache_path, engine="openpyxl", mode="w") as writer:
        tickets_df.to_excel(writer, sheet_name="tickets", index=False)
        matches_df.to_excel(writer, sheet_name="matches", index=False)


def _load_cached_results(
    ticket_id: str,
    row_count: int,
    cache_path: Path,
) -> Tuple[Dict[int, Dict[str, object]] | None, str | None]:
    tickets_df, matches_df = _read_cache_tables(cache_path)

    ticket_rows = tickets_df[tickets_df["ticket_id"].astype(str) == ticket_id]
    match_rows = matches_df[matches_df["ticket_id"].astype(str) == ticket_id]
    if ticket_rows.empty or match_rows.empty:
        return None, None

    expected_rows = int(ticket_rows.iloc[-1].get("row_count", 0) or 0)
    if expected_rows <= 0:
        expected_rows = row_count

    if len(match_rows) < expected_rows:
        return None, None

    results_by_row: Dict[int, Dict[str, object]] = {}
    for _, item in match_rows.iterrows():
        try:
            partido = int(item.get("partido", 0) or 0)
        except (TypeError, ValueError):
            continue
        if partido <= 0:
            continue
        results_by_row[partido] = {
            "resultado_estado": str(item.get("resultado_estado", "sin_resultado") or "sin_resultado"),
            "resultado_texto": str(item.get("resultado_texto", "Sin resultado API") or "Sin resultado API"),
            "resultado_real": str(item.get("resultado_real", "-") or "-"),
            "marcador_real": str(item.get("marcador_real", "-") or "-"),
        }

    if len(results_by_row) < expected_rows:
        return None, None

    created_at = str(ticket_rows.iloc[-1].get("created_at_utc", "") or "").strip()

    unresolved_states = {"sin_resultado", "pendiente"}
    unresolved_count = sum(
        1
        for item in results_by_row.values()
        if str(item.get("resultado_estado", "sin_resultado")) in unresolved_states
    )

    # Avoid sticky stale cache when previous lookup had unresolved matches.
    if unresolved_count > 0:
        try:
            created_dt = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
            age_hours = (datetime.utcnow() - created_dt).total_seconds() / 3600.0
        except Exception:
            age_hours = float("inf")

        if age_hours >= _CACHE_REFRESH_HOURS_UNRESOLVED:
            return None, None

    note = "TheOdds cache hit"
    if created_at:
        note += f" ({created_at})"
    return results_by_row, note


def _save_cache(
    ticket_id: str,
    rows: Sequence[Dict[str, object]],
    lookup_date: date,
    results_by_row: Dict[int, Dict[str, object]],
    metadata_by_row: Dict[int, Dict[str, object]],
    cache_path: Path,
    api_requests_used: int | None,
    api_requests_remaining: int | None,
    api_requests_last_total: int | None,
) -> None:
    tickets_df, matches_df = _read_cache_tables(cache_path)

    tickets_df = tickets_df[tickets_df["ticket_id"].astype(str) != ticket_id]
    matches_df = matches_df[matches_df["ticket_id"].astype(str) != ticket_id]

    ticket_row = {
        "ticket_id": ticket_id,
        "lookup_date": lookup_date.isoformat(),
        "row_count": len(rows),
        "created_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "provider": "theoddsapi",
        "api_requests_used": api_requests_used,
        "api_requests_remaining": api_requests_remaining,
        "api_requests_last_total": api_requests_last_total,
    }

    match_records: List[Dict[str, object]] = []
    for row in rows:
        partido = int(row.get("partido", 0) or 0)
        result = results_by_row.get(partido) or {
            "resultado_estado": "sin_resultado",
            "resultado_texto": "Sin resultado API",
            "resultado_real": "-",
            "marcador_real": "-",
        }
        meta = metadata_by_row.get(partido, {})

        match_records.append(
            {
                "ticket_id": ticket_id,
                "partido": partido,
                "local": str(row.get("local", "") or ""),
                "visitante": str(row.get("visitante", "") or ""),
                "pick_symbol": str(row.get("pick_symbol", "") or ""),
                "doble_oportunidad": str(row.get("doble_oportunidad", "") or ""),
                "resultado_estado": str(result.get("resultado_estado", "sin_resultado") or "sin_resultado"),
                "resultado_texto": str(result.get("resultado_texto", "Sin resultado API") or "Sin resultado API"),
                "resultado_real": str(result.get("resultado_real", "-") or "-"),
                "marcador_real": str(result.get("marcador_real", "-") or "-"),
                "provider": str(meta.get("provider", "theoddsapi") or "theoddsapi"),
                "matched_sport_key": str(meta.get("sport_key", "") or ""),
                "matched_event_id": str(meta.get("event_id", "") or ""),
                "matched_home_team": str(meta.get("home_team", "") or ""),
                "matched_away_team": str(meta.get("away_team", "") or ""),
                "matched_commence_time": str(meta.get("commence_time", "") or ""),
                "matched_similarity": meta.get("similarity"),
                "matched_side_similarity": meta.get("side_similarity"),
            }
        )

    tickets_df = pd.concat([tickets_df, pd.DataFrame([ticket_row])], ignore_index=True)
    matches_df = pd.concat([matches_df, pd.DataFrame(match_records)], ignore_index=True)

    _write_cache_tables(cache_path, tickets_df[_TICKETS_COLUMNS], matches_df[_MATCHES_COLUMNS])


def _event_day_distance(event: OddsEvent, base_date: date) -> int | None:
    raw = str(event.commence_time or "")
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

    # OCR fallback: allow slightly lower side similarity only when one side is very clear,
    # pair score is still solid, and event date is close.
    if distance_days is not None and distance_days <= 1:
        if pair_score >= max(min_pair_score - 0.08, 0.68) and min_side >= max(min_side_score - 0.18, 0.32) and max_side >= 0.78:
            return True
        if pair_score >= 0.75 and min_side >= 0.36:
            return True

    return False


@lru_cache(maxsize=8)
def _fetch_soccer_sport_keys(api_key: str) -> Tuple[str, ...]:
    query = urllib.parse.urlencode({"apiKey": api_key})
    url = f"{_THEODDS_BASE_URL}/sports?{query}"

    with urllib.request.urlopen(url, timeout=25) as response:
        payload = json.loads(response.read().decode("utf-8"))

    soccer_keys: List[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "") or "")
        if not key.startswith("soccer_"):
            continue
        if bool(item.get("has_outrights", False)):
            continue
        if not bool(item.get("active", True)):
            continue
        soccer_keys.append(key)

    unique = sorted(set(soccer_keys))
    ordered: List[str] = []
    for key in _PRIORITY_SPORT_KEYS:
        if key in unique:
            ordered.append(key)
    for key in unique:
        if key not in ordered:
            ordered.append(key)

    return tuple(ordered)


@lru_cache(maxsize=512)
def _fetch_scores_for_sport(
    api_key: str,
    sport_key: str,
    days_from: int,
) -> Tuple[Tuple[OddsEvent, ...], int | None, int | None, int | None]:
    query = urllib.parse.urlencode(
        {
            "apiKey": api_key,
            "daysFrom": days_from,
            "dateFormat": "iso",
        }
    )
    url = f"{_THEODDS_BASE_URL}/sports/{sport_key}/scores/?{query}"

    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
        headers = response.headers

    events: List[OddsEvent] = []
    for raw in payload:
        if not isinstance(raw, dict):
            continue

        raw_scores = raw.get("scores") or []
        parsed_scores: List[Tuple[str, int | None]] = []
        if isinstance(raw_scores, list):
            for item in raw_scores:
                if not isinstance(item, dict):
                    continue
                parsed_scores.append(
                    (
                        str(item.get("name", "") or ""),
                        _parse_int(item.get("score")),
                    )
                )

        events.append(
            OddsEvent(
                sport_key=sport_key,
                event_id=str(raw.get("id", "") or ""),
                commence_time=str(raw.get("commence_time", "") or ""),
                home_team=str(raw.get("home_team", "") or ""),
                away_team=str(raw.get("away_team", "") or ""),
                completed=bool(raw.get("completed", False)),
                scores=tuple(parsed_scores),
            )
        )

    used = _parse_int(headers.get("X-Requests-Used"))
    remaining = _parse_int(headers.get("X-Requests-Remaining"))
    last = _parse_int(headers.get("X-Requests-Last"))

    return tuple(events), used, remaining, last


def _select_sport_keys_for_rows(all_sport_keys: Sequence[str], rows: Sequence[Dict[str, object]]) -> List[str]:
    if not all_sport_keys:
        return []

    all_keys = list(dict.fromkeys(str(item) for item in all_sport_keys if item))
    selected: List[str] = []

    for key in _PRIORITY_SPORT_KEYS:
        if key in all_keys and key not in selected:
            selected.append(key)

    row_blob = " ".join(
        _normalize_team_name(str(row.get("local", "") or "") + " " + str(row.get("visitante", "") or ""))
        for row in rows
    )

    for keywords, hinted_keys in _SPORT_HINTS:
        if any(keyword in row_blob for keyword in keywords):
            for key in hinted_keys:
                if key in all_keys and key not in selected:
                    selected.append(key)

    for key in all_keys:
        if key not in selected:
            selected.append(key)

    return selected[:_THEODDS_MAX_SPORTS_PER_LOOKUP]


def _event_scores_for_teams(event: OddsEvent) -> Tuple[int | None, int | None]:
    if not event.scores:
        return None, None

    home_norm = _normalize_team_name(event.home_team)
    away_norm = _normalize_team_name(event.away_team)

    home_score: int | None = None
    away_score: int | None = None

    for name, score in event.scores:
        norm = _normalize_team_name(name)
        if not norm:
            continue
        if home_score is None and norm == home_norm:
            home_score = score
        elif away_score is None and norm == away_norm:
            away_score = score

    if (home_score is None or away_score is None) and len(event.scores) >= 2:
        if home_score is None:
            home_score = event.scores[0][1]
        if away_score is None:
            away_score = event.scores[1][1]

    return home_score, away_score


def lookup_results_for_rows_theodds_cached(
    rows: Sequence[Dict[str, object]],
    match_date: date,
    api_key: str,
    cache_excel_path: Path | None = None,
    force_refresh: bool = False,
) -> Tuple[Dict[int, Dict[str, object]], List[str], bool]:
    candidate_rows = [
        row
        for row in rows
        if int(row.get("partido", 0) or 0) > 0 and str(row.get("pick_symbol", "") or "").upper() in {"1", "X", "2"}
    ]
    if not candidate_rows:
        return {}, [], False

    cache_path = cache_excel_path or THEODDS_CACHE_FILE
    ticket_id = _ticket_id(candidate_rows, match_date)

    if not force_refresh:
        cached_results, cache_note = _load_cached_results(ticket_id, len(candidate_rows), cache_path)
        if cached_results is not None:
            notes = [cache_note] if cache_note else ["TheOdds cache hit"]
            return cached_results, notes, True

    notes: List[str] = []
    api_key = str(api_key or "").strip()
    if not api_key:
        notes.append("TheOdds: falta API key; no se realizo consulta.")
        return {}, notes, False

    if (date.today() - match_date) > timedelta(days=_THEODDS_DAYS_FROM + 1):
        notes.append(
            "TheOdds: la fecha solicitada esta fuera de la ventana de scores (aprox. 3 dias); se omite para ahorrar creditos."
        )
        return {}, notes, False

    try:
        all_sport_keys = _fetch_soccer_sport_keys(api_key)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"TheOdds: error al listar deportes ({type(exc).__name__}).")
        return {}, notes, False

    selected_sport_keys = _select_sport_keys_for_rows(all_sport_keys, candidate_rows)
    if not selected_sport_keys:
        notes.append("TheOdds: no hay ligas de futbol disponibles para consulta.")
        return {}, notes, False

    events_by_id: Dict[str, OddsEvent] = {}
    requests_last_total = 0
    requests_used: int | None = None
    requests_remaining: int | None = None

    for sport_key in selected_sport_keys:
        try:
            events, used, remaining, last = _fetch_scores_for_sport(api_key, sport_key, _THEODDS_DAYS_FROM)
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                notes.append(f"TheOdds {sport_key}: HTTP {exc.code}")
            continue
        except Exception as exc:  # noqa: BLE001
            notes.append(f"TheOdds {sport_key}: {type(exc).__name__}")
            continue

        if used is not None:
            requests_used = used
        if remaining is not None:
            requests_remaining = remaining
        if last is not None:
            requests_last_total += int(last)

        for event in events:
            key = event.event_id or f"{event.sport_key}:{event.home_team}:{event.away_team}:{event.commence_time}"
            if key not in events_by_id:
                events_by_id[key] = event

    collected_events = list(events_by_id.values())

    results_by_row: Dict[int, Dict[str, object]] = {}
    metadata_by_row: Dict[int, Dict[str, object]] = {}
    unresolved_debug: List[str] = []

    for row in candidate_rows:
        partido = int(row.get("partido", 0) or 0)
        local = str(row.get("local", "") or "")
        visitante = str(row.get("visitante", "") or "")
        pick_symbol = str(row.get("pick_symbol", "") or "").upper()
        double_pick = str(row.get("doble_oportunidad", "") or "").upper().replace(" ", "")

        best_event: OddsEvent | None = None
        best_score = 0.0
        best_effective_score = float("-inf")
        best_swapped = False
        best_local_score = 0.0
        best_visit_score = 0.0
        best_distance_days: int | None = None

        for event in collected_events:
            score, swapped, local_score, visit_score = _pair_similarity(local, visitante, event)  # type: ignore[arg-type]

            distance_days = _event_day_distance(event, match_date)
            distance_penalty = _DATE_DISTANCE_PENALTY * float(min(distance_days or 0, 10))
            effective_score = score - distance_penalty

            if (
                effective_score > best_effective_score
                or (effective_score == best_effective_score and score > best_score)
            ):
                best_event = event
                best_score = score
                best_effective_score = effective_score
                best_swapped = swapped
                best_local_score = local_score
                best_visit_score = visit_score
                best_distance_days = distance_days

        is_historical_fallback = bool(best_distance_days is not None and best_distance_days > (_THEODDS_DAYS_FROM + 1))
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
                "resultado_texto": "Sin resultado TheOdds",
                "resultado_real": "-",
                "marcador_real": "-",
            }
            unresolved_debug.append(
                f"P{partido}:sim={best_score:.3f},side={min(best_local_score,best_visit_score):.3f},d={best_distance_days}"
            )
            metadata_by_row[partido] = {
                "provider": "theoddsapi",
                "similarity": round(best_score, 4),
                "side_similarity": round(min(best_local_score, best_visit_score), 4),
            }
            continue

        home_score, away_score = _event_scores_for_teams(best_event)
        actual_symbol = _result_symbol(home_score, away_score, best_swapped)
        score_text = _score_text(home_score, away_score, best_swapped)

        if not best_event.completed or actual_symbol is None:
            results_by_row[partido] = {
                "resultado_estado": "pendiente",
                "resultado_texto": "Pendiente",
                "resultado_real": "-",
                "marcador_real": score_text,
            }
        else:
            is_hit = _is_hit_for_pick(actual_symbol, pick_symbol, double_pick)
            results_by_row[partido] = {
                "resultado_estado": "acierto" if is_hit else "fallo",
                "resultado_texto": "Acierto" if is_hit else "Fallo",
                "resultado_real": actual_symbol,
                "marcador_real": score_text,
            }

        metadata_by_row[partido] = {
            "provider": "theoddsapi",
            "sport_key": best_event.sport_key,
            "event_id": best_event.event_id,
            "home_team": best_event.home_team,
            "away_team": best_event.away_team,
            "commence_time": best_event.commence_time,
            "similarity": round(best_score, 4),
            "side_similarity": round(min(best_local_score, best_visit_score), 4),
        }

    _save_cache(
        ticket_id=ticket_id,
        rows=candidate_rows,
        lookup_date=match_date,
        results_by_row=results_by_row,
        metadata_by_row=metadata_by_row,
        cache_path=cache_path,
        api_requests_used=requests_used,
        api_requests_remaining=requests_remaining,
        api_requests_last_total=requests_last_total,
    )

    notes.append(
        "TheOdds: "
        f"consulta completada ({len(selected_sport_keys)} ligas, "
        f"{len(collected_events)} eventos, "
        f"creditos consumidos aprox. {requests_last_total})."
    )
    if unresolved_debug:
        notes.append("TheOdds unresolved debug: " + " | ".join(unresolved_debug[:8]))
    if requests_remaining is not None:
        notes.append(f"TheOdds: creditos restantes {requests_remaining}.")

    return results_by_row, notes, False
