from __future__ import annotations

from collections import Counter
from datetime import date
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from config import HeuristicConfig, load_rules, save_rules
from logic import analyze_match, validate_match_input
from logic.historical_patterns import HistoricalPatternEngine
from models import MatchAnalysis, MatchInput
from ui import (
    apply_custom_style,
    build_match_json_output,
    confidence_badge,
    render_pattern_glossary,
    render_jornada_panel,
    render_summary_cards,
    render_ticket_board,
    risk_label,
    semaforo_badge,
)
from utils import (
    DEFAULT_ESPN_LEAGUES,
    DEFAULT_THEODDS_API_KEY,
    DUMMY_MATCHES,
    ESPN_LEAGUE_OPTIONS,
    THEODDS_CACHE_FILE,
    build_export_dataframes,
    clear_espn_cache,
    clear_theodds_runtime_cache,
    dataframe_to_csv_bytes,
    dataframes_to_excel_bytes,
    extract_matches_with_date_from_capture,
    lookup_results_for_rows,
    lookup_results_for_rows_theodds_cached,
    ocr_runtime_status,
    parse_session_json,
    session_to_json_bytes,
)


st.set_page_config(
    page_title="Progol Contrarian Lab",
    page_icon="PL",
    layout="wide",
)
apply_custom_style()

MAX_MATCHES = 14
HISTORICAL_DB_PATH = Path(__file__).parent / "cache" / "historical_patterns.db"


RULE_FIELDS = [
    *HeuristicConfig().to_dict().keys(),
]


def _empty_ticket_rows() -> List[Dict[str, Any]]:
    return [
        {
            "partido": index,
            "local": "",
            "pct_local": None,
            "pct_empate": None,
            "pct_visita": None,
            "visitante": "",
        }
        for index in range(1, MAX_MATCHES + 1)
    ]


def init_state() -> None:
    if "ticket_rows" not in st.session_state:
        st.session_state.ticket_rows = _empty_ticket_rows()
    if "rules" not in st.session_state:
        st.session_state.rules = load_rules()
    if "jornada_nombre" not in st.session_state:
        st.session_state.jornada_nombre = "Concurso Analitico - Jornada Actual"
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "Vista boleto"
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    if "espn_results_enabled" not in st.session_state:
        st.session_state.espn_results_enabled = True
    if "espn_lookup_date" not in st.session_state:
        st.session_state.espn_lookup_date = date.today()
    if "espn_include_previous_day" not in st.session_state:
        st.session_state.espn_include_previous_day = True
    if "espn_selected_leagues" not in st.session_state:
        auto_labels = [label for label, code in ESPN_LEAGUE_OPTIONS.items() if code == "__all__"]
        st.session_state.espn_selected_leagues = auto_labels[:1] if auto_labels else list(ESPN_LEAGUE_OPTIONS.keys())
    if "theodds_api_key" not in st.session_state:
        st.session_state.theodds_api_key = DEFAULT_THEODDS_API_KEY
    if "historical_enabled" not in st.session_state:
        st.session_state.historical_enabled = True
    if "historical_min_required" not in st.session_state:
        st.session_state.historical_min_required = 8
    if "historical_max_distance" not in st.session_state:
        st.session_state.historical_max_distance = 0.33
    if "historical_top_k" not in st.session_state:
        st.session_state.historical_top_k = 40
    if "historical_autosave_from_api" not in st.session_state:
        st.session_state.historical_autosave_from_api = True
    if "theodds_force_refresh_once" not in st.session_state:
        st.session_state.theodds_force_refresh_once = False
    if "results_source_mode" not in st.session_state:
        st.session_state.results_source_mode = "ESPN gratis (recomendado)"


@st.cache_resource(show_spinner=False)
def _get_historical_engine(db_path: str, min_required: int, max_distance: float, top_k: int) -> HistoricalPatternEngine:
    return HistoricalPatternEngine(
        db_path=Path(db_path),
        min_required=min_required,
        max_distance=max_distance,
        top_k=top_k,
    )


def _resolve_historical_engine() -> HistoricalPatternEngine:
    engine = _get_historical_engine(
        db_path=str(HISTORICAL_DB_PATH),
        min_required=int(st.session_state.get("historical_min_required", 8)),
        max_distance=float(st.session_state.get("historical_max_distance", 0.33)),
        top_k=int(st.session_state.get("historical_top_k", 40)),
    )

    # Compat guard: if Streamlit reused an old cached instance, rebuild it once.
    if not hasattr(engine, "insert_rows_dedup"):
        _get_historical_engine.clear()
        engine = _get_historical_engine(
            db_path=str(HISTORICAL_DB_PATH),
            min_required=int(st.session_state.get("historical_min_required", 8)),
            max_distance=float(st.session_state.get("historical_max_distance", 0.33)),
            top_k=int(st.session_state.get("historical_top_k", 40)),
        )
    return engine


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def _rows_to_editor_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for row in rows:
        p1 = row.get("pct_local")
        px = row.get("pct_empate")
        p2 = row.get("pct_visita")

        total = None
        if p1 is not None and px is not None and p2 is not None:
            total = round(float(p1) + float(px) + float(p2), 2)

        records.append(
            {
                "Partido": int(row.get("partido", 0)),
                "Local": row.get("local", ""),
                "% Local": p1,
                "% Empate": px,
                "% Visita": p2,
                "Visitante": row.get("visitante", ""),
                "Total %": total,
            }
        )
    return pd.DataFrame(records)


def _editor_dataframe_to_rows(editor_df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for idx in range(MAX_MATCHES):
        if idx >= len(editor_df):
            rows.append(_empty_ticket_rows()[idx])
            continue

        source = editor_df.iloc[idx]
        row = {
            "partido": idx + 1,
            "local": str(source.get("Local", "") or "").strip(),
            "pct_local": _to_float(source.get("% Local")),
            "pct_empate": _to_float(source.get("% Empate")),
            "pct_visita": _to_float(source.get("% Visita")),
            "visitante": str(source.get("Visitante", "") or "").strip(),
        }
        rows.append(row)

    return rows


def _load_dummy_rows() -> None:
    rows = _empty_ticket_rows()
    for idx, item in enumerate(DUMMY_MATCHES[:MAX_MATCHES]):
        rows[idx] = {
            "partido": idx + 1,
            "local": str(item.get("local", "")).strip(),
            "pct_local": _to_float(item.get("pct_local")),
            "pct_empate": _to_float(item.get("pct_empate")),
            "pct_visita": _to_float(item.get("pct_visita")),
            "visitante": str(item.get("visitante", "")).strip(),
        }
    st.session_state.ticket_rows = rows


def _load_session_into_rows(payload: Dict[str, Any]) -> None:
    rows = _empty_ticket_rows()
    for idx, item in enumerate(payload.get("matches", [])[:MAX_MATCHES]):
        rows[idx] = {
            "partido": idx + 1,
            "local": str(item.get("local", "")).strip(),
            "pct_local": _to_float(item.get("pct_local")),
            "pct_empate": _to_float(item.get("pct_empate")),
            "pct_visita": _to_float(item.get("pct_visita")),
            "visitante": str(item.get("visitante", "")).strip(),
        }

    st.session_state.ticket_rows = rows
    if "rules" in payload:
        st.session_state.rules = HeuristicConfig.from_dict(payload["rules"])


def _row_has_data(row: Dict[str, Any]) -> bool:
    return bool(
        str(row.get("local", "")).strip()
        or str(row.get("visitante", "")).strip()
        or row.get("pct_local") is not None
        or row.get("pct_empate") is not None
        or row.get("pct_visita") is not None
    )


def _load_extracted_rows_to_ticket(
    extracted_rows: List[Dict[str, Any]],
    start_partido: int,
    only_empty_rows: bool,
) -> int:
    current_rows = [dict(item) for item in st.session_state.ticket_rows]

    cursor = max(0, start_partido - 1)
    inserted = 0

    for extracted in extracted_rows:
        while cursor < MAX_MATCHES and only_empty_rows and _row_has_data(current_rows[cursor]):
            cursor += 1

        if cursor >= MAX_MATCHES:
            break

        current_rows[cursor] = {
            "partido": cursor + 1,
            "local": str(extracted.get("local", "") or "").strip(),
            "pct_local": _to_float(extracted.get("pct_local")),
            "pct_empate": _to_float(extracted.get("pct_empate")),
            "pct_visita": _to_float(extracted.get("pct_visita")),
            "visitante": str(extracted.get("visitante", "") or "").strip(),
        }

        inserted += 1
        cursor += 1

    st.session_state.ticket_rows = current_rows
    return inserted


def _short_warning_text(warning: str) -> str:
    lower = warning.lower()
    if "suma de porcentajes" in lower:
        return "Suma fuera de tolerancia"
    return warning


def _symbol_to_pick_text(symbol: str, local: str, visitante: str) -> str:
    if symbol == "1":
        return local or "Local"
    if symbol == "2":
        return visitante or "Visitante"
    if symbol == "X":
        return "Empate"
    return "-"


def _fallback_double_from_symbol(
    direct_symbol: str,
    score_local: float,
    score_visita: float,
) -> str:
    if direct_symbol == "1":
        return "1X"
    if direct_symbol == "2":
        return "X2"
    if direct_symbol == "X":
        return "1X" if score_local >= score_visita else "X2"
    return "1X" if score_local >= score_visita else "X2"


def _normalize_double_pick(
    recomendacion: str,
    doble_oportunidad: str,
    direct_symbol: str,
    score_local: float,
    score_visita: float,
) -> str:
    valid_double = {"1X", "X2", "12"}
    if doble_oportunidad in valid_double:
        return doble_oportunidad
    if recomendacion in valid_double:
        return recomendacion
    return _fallback_double_from_symbol(direct_symbol, score_local, score_visita)


def _resolve_direct_pick_symbol(analysis: MatchAnalysis) -> str:
    recommendation = str(analysis.recomendacion_principal or "").upper()
    options = [token for token in recommendation if token in {"1", "X", "2"}]
    if not options:
        return "X"

    if len(options) == 1:
        return options[0]

    score_map = {
        "1": float(analysis.score_local),
        "X": float(analysis.score_empate),
        "2": float(analysis.score_visita),
    }

    best_option = options[0]
    best_score = score_map.get(best_option, float("-inf"))

    for option in options[1:]:
        option_score = score_map.get(option, float("-inf"))
        if option_score > best_score:
            best_option = option
            best_score = option_score

    return best_option


def _evaluate_ticket_rows(
    rows: List[Dict[str, Any]],
    cfg: HeuristicConfig,
    historical_engine: HistoricalPatternEngine | None = None,
    historical_enabled: bool = False,
    debug_mode: bool = False,
) -> Tuple[List[Dict[str, Any]], List[MatchAnalysis], Dict[int, MatchAnalysis], List[Dict[str, Any]]]:
    row_reports: List[Dict[str, Any]] = []
    analyses: List[MatchAnalysis] = []
    analysis_by_row: Dict[int, MatchAnalysis] = {}
    export_matches: List[Dict[str, Any]] = []

    for row in rows:
        partido = int(row.get("partido", 0) or 0)
        local = str(row.get("local", "") or "").strip()
        visitante = str(row.get("visitante", "") or "").strip()
        pct_local = _to_float(row.get("pct_local"))
        pct_empate = _to_float(row.get("pct_empate"))
        pct_visita = _to_float(row.get("pct_visita"))

        has_any_value = bool(local or visitante or pct_local is not None or pct_empate is not None or pct_visita is not None)

        report: Dict[str, Any] = {
            "partido": partido,
            "local": local,
            "pct_local": pct_local,
            "pct_empate": pct_empate,
            "pct_visita": pct_visita,
            "visitante": visitante,
            "pick_symbol": "-",
            "recomendacion": "-",
            "doble_oportunidad": "-",
            "confianza": "-",
            "resultado_estado": "sin_resultado",
            "resultado_texto": "Sin validar",
            "resultado_real": "-",
            "marcador_real": "-",
            "alerta": "Sin captura",
            "semaforo": "neutro",
            "tipo_partido": "-",
            "estado": "vacio",
            "pick_heuristico_symbol": "-",
            "pick_heuristico": "-",
            "confianza_heuristica": "-",
            "senal_historica": "-",
            "historical_distribution": "",
            "historical_sample": 0,
            "historical_confidence": "-",
            "decision_hibrida": "-",
        }

        if not has_any_value:
            row_reports.append(report)
            continue

        if not local or not visitante:
            report.update(
                {
                    "estado": "incompleto",
                    "semaforo": "amarillo",
                    "alerta": "Captura incompleta: falta local o visitante",
                    "tipo_partido": "Partido de cuidado",
                }
            )
            row_reports.append(report)
            continue

        if pct_local is None or pct_empate is None or pct_visita is None:
            report.update(
                {
                    "estado": "incompleto",
                    "semaforo": "amarillo",
                    "alerta": "Captura incompleta: faltan porcentajes",
                    "tipo_partido": "Partido de cuidado",
                }
            )
            row_reports.append(report)
            continue

        match = MatchInput(
            local=local,
            visitante=visitante,
            pct_local=float(pct_local),
            pct_empate=float(pct_empate),
            pct_visita=float(pct_visita),
        )

        errors, warnings = validate_match_input(match, cfg.sum_tolerance)
        if errors:
            report.update(
                {
                    "estado": "error",
                    "semaforo": "rojo",
                    "alerta": errors[0],
                    "tipo_partido": "Partido de cuidado",
                }
            )
            row_reports.append(report)
            continue

        analysis = analyze_match(match, cfg, debug_mode=debug_mode)
        analyses.append(analysis)
        analysis_by_row[partido] = analysis
        export_matches.append(match.to_dict())

        semaforo = risk_label(analysis.score_riesgo)
        if warnings and semaforo == "verde":
            semaforo = "amarillo"

        direct_pick_symbol = _resolve_direct_pick_symbol(analysis)
        direct_pick_text = _symbol_to_pick_text(direct_pick_symbol, local, visitante)
        double_pick = _normalize_double_pick(
            analysis.recomendacion_principal,
            analysis.doble_oportunidad,
            direct_pick_symbol,
            float(analysis.score_local),
            float(analysis.score_visita),
        )

        alerta_parts: List[str] = []
        if warnings:
            alerta_parts.append(_short_warning_text(warnings[0]))
        if analysis.banderas_alerta:
            alerta_parts.append(analysis.banderas_alerta[0])
        elif analysis.sesgos:
            alerta_parts.append(analysis.sesgos[0])
        else:
            alerta_parts.append("Lectura estable")

        final_pick_symbol = direct_pick_symbol
        final_pick_text = direct_pick_text
        final_double_pick = double_pick
        final_confidence = analysis.confianza
        historical_signal = None
        hybrid_decision = None

        if historical_enabled and historical_engine is not None:
            historical_signal = historical_engine.find_similar_signal(match, analysis.patrones_activados)
            hybrid_decision = historical_engine.build_hybrid_decision(
                heuristic_recommendation=str(analysis.recomendacion_principal),
                heuristic_direct_symbol=direct_pick_symbol,
                heuristic_double=double_pick,
                heuristic_confidence=str(analysis.confianza),
                historical_signal=historical_signal,
            )
            final_pick_symbol = hybrid_decision.final_direct_symbol
            final_pick_text = _symbol_to_pick_text(final_pick_symbol, local, visitante)
            final_double_pick = hybrid_decision.final_double if hybrid_decision.final_double in {"1X", "X2", "12"} else "-"
            final_confidence = hybrid_decision.final_confidence

            signal_label = "-"
            if historical_signal.recommended in {"1", "X", "2"}:
                signal_label = historical_signal.recommended

            if historical_signal.enough_samples:
                alert_note = (
                    f"Historico({historical_signal.sample_size}) "
                    f"{_format_distribution_label(historical_signal.distribution)}"
                )
                alerta_parts.append(alert_note)

            if debug_mode:
                analysis.debug_data["historico"] = {
                    "signal": historical_signal.to_dict(),
                    "hybrid_decision": hybrid_decision.to_dict(),
                }
        else:
            signal_label = "-"

        report.update(
            {
                "pick_symbol": final_pick_symbol,
                "recomendacion": final_pick_text,
                "doble_oportunidad": final_double_pick,
                "confianza": str(final_confidence).capitalize(),
                "alerta": " / ".join(alerta_parts),
                "semaforo": semaforo,
                "tipo_partido": analysis.tipo_partido,
                "estado": "ok",
                "pick_heuristico_symbol": direct_pick_symbol,
                "pick_heuristico": direct_pick_text,
                "confianza_heuristica": str(analysis.confianza).capitalize(),
                "senal_historica": signal_label,
                "historical_distribution": (
                    _format_distribution_label(historical_signal.distribution) if historical_signal else ""
                ),
                "historical_sample": int(historical_signal.sample_size) if historical_signal else 0,
                "historical_confidence": (
                    historical_signal.confidence_label.capitalize() if historical_signal else "-"
                ),
                "decision_hibrida": hybrid_decision.decision_note if hybrid_decision else "Solo heuristica",
            }
        )
        row_reports.append(report)

    return row_reports, analyses, analysis_by_row, export_matches


def _selected_espn_league_codes() -> Tuple[str, ...]:
    selected_labels = st.session_state.get("espn_selected_leagues", list(ESPN_LEAGUE_OPTIONS.keys()))

    auto_labels = [label for label, code in ESPN_LEAGUE_OPTIONS.items() if code == "__all__"]
    auto_label = auto_labels[0] if auto_labels else None
    non_auto_labels = [label for label, code in ESPN_LEAGUE_OPTIONS.items() if code != "__all__"]

    normalized_selected = [str(item) for item in selected_labels]
    if auto_label and auto_label in normalized_selected and len(normalized_selected) > 1:
        # If user selected "all" plus specific leagues, keep specific leagues to avoid
        # relying only on the ESPN "all" endpoint.
        normalized_selected = [label for label in normalized_selected if label != auto_label]
        st.session_state.espn_selected_leagues = normalized_selected
    elif auto_label and normalized_selected and set(normalized_selected) == set(non_auto_labels):
        normalized_selected = [auto_label]
        st.session_state.espn_selected_leagues = normalized_selected

    league_codes: List[str] = []
    for label in normalized_selected:
        code = ESPN_LEAGUE_OPTIONS.get(str(label))
        if code:
            league_codes.append(code)

    if not league_codes:
        return tuple(DEFAULT_ESPN_LEAGUES)

    return tuple(dict.fromkeys(league_codes))


def _apply_espn_results_to_reports(row_reports: List[Dict[str, Any]]) -> List[str]:
    if not bool(st.session_state.get("espn_results_enabled", True)):
        return []

    lookup_date = st.session_state.get("espn_lookup_date", date.today())
    if not isinstance(lookup_date, date):
        lookup_date = date.today()

    candidate_rows = [
        row
        for row in row_reports
        if row.get("estado") == "ok" and str(row.get("pick_symbol", "")).upper() in {"1", "X", "2"}
    ]
    if not candidate_rows:
        return []

    mode = str(st.session_state.get("results_source_mode", "ESPN gratis (recomendado)") or "")
    api_key = str(st.session_state.get("theodds_api_key", "") or "").strip()
    results_by_row: Dict[int, Dict[str, Any]] = {}
    notes: List[str] = [f"Modo resultados: {mode}"]

    selected_codes = _selected_espn_league_codes()
    include_previous = bool(st.session_state.get("espn_include_previous_day", True))

    def _unresolved(rows_source: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pending: List[Dict[str, Any]] = []
        for row in rows_source:
            partido = int(row.get("partido", 0) or 0)
            resolved = results_by_row.get(partido)
            if not resolved or str(resolved.get("resultado_estado", "sin_resultado")) == "sin_resultado":
                pending.append(row)
        return pending

    def _apply_espn(rows_source: List[Dict[str, Any]], leagues: Tuple[str, ...]) -> None:
        if not rows_source:
            return
        fallback_results, fallback_notes = lookup_results_for_rows(
            rows_source,
            match_date=lookup_date,
            league_codes=leagues,
            include_previous_day=include_previous,
        )
        notes.extend(fallback_notes)
        for row in rows_source:
            partido = int(row.get("partido", 0) or 0)
            resolved = fallback_results.get(partido)
            if not resolved:
                continue
            if str(resolved.get("resultado_estado", "sin_resultado")) == "sin_resultado":
                continue
            results_by_row[partido] = resolved

    def _apply_theodds(rows_source: List[Dict[str, Any]]) -> None:
        if not rows_source:
            return
        theodds_results, local_notes, _cache_hit = lookup_results_for_rows_theodds_cached(
            rows_source,
            match_date=lookup_date,
            api_key=api_key,
            force_refresh=bool(st.session_state.get("theodds_force_refresh_once", False)),
        )
        notes.extend(local_notes)
        for partido, payload in theodds_results.items():
            if str(payload.get("resultado_estado", "sin_resultado")) == "sin_resultado":
                continue
            results_by_row[int(partido)] = dict(payload)

    if mode == "TheOdds primero":
        _apply_theodds(candidate_rows)
        st.session_state.theodds_force_refresh_once = False
        pending = _unresolved(candidate_rows)
        _apply_espn(pending, selected_codes)
    elif mode == "Hibrido ahorro (ESPN -> TheOdds faltantes)":
        _apply_espn(candidate_rows, selected_codes)
        pending = _unresolved(candidate_rows)
        if pending:
            _apply_theodds(pending)
        st.session_state.theodds_force_refresh_once = False
    else:
        _apply_espn(candidate_rows, selected_codes)
        st.session_state.theodds_force_refresh_once = False

    # Safety net: ESPN all for unresolved (works even in ESPN-only mode).
    unresolved_after_fallback = _unresolved(candidate_rows)

    if unresolved_after_fallback and "__all__" not in selected_codes:
        all_results, all_notes = lookup_results_for_rows(
            unresolved_after_fallback,
            match_date=lookup_date,
            league_codes=("__all__",),
            include_previous_day=include_previous,
        )
        notes.extend([f"ESPN all fallback: {item}" for item in all_notes] if all_notes else ["ESPN all fallback aplicado."])
        for row in unresolved_after_fallback:
            partido = int(row.get("partido", 0) or 0)
            fallback = all_results.get(partido)
            if not fallback:
                continue
            if str(fallback.get("resultado_estado", "sin_resultado")) == "sin_resultado":
                continue
            results_by_row[partido] = fallback

    for report in row_reports:
        if report.get("estado") != "ok":
            continue

        partido = int(report.get("partido", 0) or 0)
        resolved = results_by_row.get(partido)
        if not resolved:
            report.update(
                {
                    "resultado_estado": "sin_resultado",
                    "resultado_texto": "Sin resultado API",
                    "resultado_real": "-",
                    "marcador_real": "-",
                }
            )
            continue

        resolved_state = str(resolved.get("resultado_estado", "sin_resultado") or "sin_resultado")
        if resolved_state == "sin_resultado":
            report.update(
                {
                    "resultado_estado": "sin_resultado",
                    "resultado_texto": "Sin resultado API",
                    "resultado_real": "-",
                    "marcador_real": "-",
                }
            )
            continue

        report.update(resolved)

    status_counts = {"acierto": 0, "fallo": 0, "pendiente": 0, "sin_resultado": 0}
    for report in row_reports:
        if report.get("estado") != "ok":
            continue
        status = str(report.get("resultado_estado", "sin_resultado"))
        if status in status_counts:
            status_counts[status] += 1

    total_checked = sum(status_counts.values())
    notes.append(
        "Resultados: "
        f"{status_counts['acierto']} aciertos, "
        f"{status_counts['fallo']} fallos, "
        f"{status_counts['pendiente']} pendientes, "
        f"{status_counts['sin_resultado']} sin resultado "
        f"(total {total_checked})."
    )

    return notes


def _build_analysis_dataframe(row_reports: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for report in row_reports:
        if report["estado"] == "vacio":
            continue
        rows.append(
            {
                "Partido": report["partido"],
                "Local": report["local"],
                "% Local": report["pct_local"],
                "% Empate": report["pct_empate"],
                "% Visita": report["pct_visita"],
                "Visitante": report["visitante"],
                "Pick": report["recomendacion"],
                "Pick simb": report["pick_symbol"],
                "Pick heuristico": report.get("pick_heuristico", "-"),
                "Pick heuristico simb": report.get("pick_heuristico_symbol", "-"),
                "Confianza heuristica": report.get("confianza_heuristica", "-"),
                "Senal historica": report.get("senal_historica", "-"),
                "Muestra historica": report.get("historical_sample", 0),
                "Distribucion historica": report.get("historical_distribution", ""),
                "Confianza historica": report.get("historical_confidence", "-"),
                "Decision hibrida": report.get("decision_hibrida", "-"),
                "Resultado": report["resultado_texto"],
                "Marcador": report["marcador_real"],
                "Real": report["resultado_real"],
                "Doble": report["doble_oportunidad"],
                "Confianza": report["confianza"],
                "Semaforo": report["semaforo"],
                "Tipo": report["tipo_partido"],
                "Alerta": report["alerta"],
                "Estado": report["estado"],
            }
        )
    return pd.DataFrame(rows)


def _render_rules_editor_sidebar() -> None:
    st.sidebar.subheader("Reglas heuristicas")
    st.sidebar.checkbox(
        "Modo debug por partido",
        key="debug_mode",
        help="Muestra patrones, scores intermedios, penalizaciones y decision final en vista analisis.",
    )
    with st.sidebar.form("rules_form"):
        current = st.session_state.rules.to_dict()
        edited = {}
        for field_name in RULE_FIELDS:
            edited[field_name] = st.number_input(
                label=field_name,
                value=float(current[field_name]),
                step=0.5,
            )

        apply_clicked = st.form_submit_button("Aplicar reglas")
        if apply_clicked:
            st.session_state.rules = HeuristicConfig.from_dict(edited)
            st.sidebar.success("Reglas actualizadas en memoria.")

    if st.sidebar.button("Guardar reglas en config/default_rules.json", use_container_width=True):
        save_rules(st.session_state.rules)
        st.sidebar.success("Reglas guardadas en archivo.")


def _format_distribution_label(distribution: Dict[str, float]) -> str:
    p1 = distribution.get("1", 0.0) * 100.0
    px = distribution.get("X", 0.0) * 100.0
    p2 = distribution.get("2", 0.0) * 100.0
    return f"1={p1:.1f}% | X={px:.1f}% | 2={p2:.1f}%"


def _render_historical_memory_panel(engine: HistoricalPatternEngine, cfg: HeuristicConfig) -> Dict[str, Any]:
    st.subheader("Memoria historica (patrones)")

    c1, c2, c3, c4, c5 = st.columns([1.0, 1.0, 1.0, 1.0, 1.2])
    with c1:
        st.checkbox(
            "Activar ajuste historico",
            key="historical_enabled",
            help="Combina heuristica base + patrones historicos similares.",
        )
    with c2:
        st.number_input(
            "Min. similares",
            min_value=3,
            max_value=120,
            step=1,
            key="historical_min_required",
        )
    with c3:
        st.number_input(
            "Distancia maxima",
            min_value=0.08,
            max_value=1.0,
            step=0.01,
            key="historical_max_distance",
            format="%.2f",
        )
    with c4:
        st.number_input(
            "Vecinos maximos",
            min_value=8,
            max_value=200,
            step=1,
            key="historical_top_k",
        )
    with c5:
        st.checkbox(
            "Aprender automatico de API",
            key="historical_autosave_from_api",
            help="Guarda historicos automaticamente cuando ya existe resultado real (1/X/2).",
        )

    uploaded_csv = st.file_uploader(
        "Cargar historicos CSV (pct_local, pct_empate, pct_visita, resultado_real)",
        type=["csv"],
        key="historical_csv_uploader",
    )
    if uploaded_csv is not None and st.button("Importar historicos", use_container_width=True):
        try:
            report = engine.import_csv_bytes(uploaded_csv.getvalue(), source="streamlit_csv_upload")
            st.success(
                "Historicos importados. "
                f"Recibidos: {report['received_rows']} | "
                f"Validos: {report['valid_rows']} | "
                f"Insertados: {report['inserted_rows']} | "
                f"Omitidos: {report['skipped_rows']}"
            )
        except Exception as exc:
            st.error(f"No se pudo importar el CSV: {exc}")

    total_cases = engine.total_matches()
    st.caption(f"Base historica activa: {total_cases} partidos en {HISTORICAL_DB_PATH.name}")

    if total_cases <= 0:
        st.info("Aun no hay historicos cargados. La app usara solo heuristica.")
        return {
            "total_cases": 0,
            "calibration": {},
            "bucket_accuracy": [],
            "frequent_patterns": [],
        }

    with st.expander("Metricas historicas", expanded=False):
        metrics = engine.metrics_report(cfg)
        calibration = metrics.get("calibration", {})

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Partidos historicos", int(metrics.get("total_cases", 0)))
        m2.metric("Accuracy pick directo", f"{100 * float(calibration.get('accuracy_pick_directo', 0.0)):.1f}%")
        m3.metric("Accuracy doble oportunidad", f"{100 * float(calibration.get('accuracy_doble_oportunidad', 0.0)):.1f}%")
        m4.metric("Rescatados por doble", int(calibration.get("double_rescued_cases", 0)))

        bucket_rows = metrics.get("bucket_accuracy", [])
        if bucket_rows:
            st.markdown("**Accuracy por bucket de porcentajes**")
            st.dataframe(pd.DataFrame(bucket_rows), hide_index=True, use_container_width=True, height=250)

        frequent_patterns = metrics.get("frequent_patterns", [])
        if frequent_patterns:
            st.markdown("**Patrones heuristicos mas frecuentes**")
            st.dataframe(pd.DataFrame(frequent_patterns), hide_index=True, use_container_width=True, height=230)

        pattern_hit_rate = calibration.get("pattern_hit_rate", {})
        if pattern_hit_rate:
            st.markdown("**Accuracy por patron heuristico**")
            rows = []
            for pattern_name, stats in pattern_hit_rate.items():
                rows.append(
                    {
                        "pattern": pattern_name,
                        "activations": int(stats.get("activations", 0.0)),
                        "direct_hit_rate": round(float(stats.get("direct_hit_rate", 0.0)), 4),
                        "double_hit_rate": round(float(stats.get("double_hit_rate", 0.0)), 4),
                    }
                )
            rows = sorted(rows, key=lambda item: (-item["activations"], -item["double_hit_rate"]))
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=280)

    return metrics


def _autosave_historical_from_reports(
    row_reports: List[Dict[str, Any]],
    analysis_by_row: Dict[int, MatchAnalysis],
    engine: HistoricalPatternEngine | None,
) -> str | None:
    if engine is None:
        return None
    if not bool(st.session_state.get("historical_autosave_from_api", True)):
        return None

    lookup_date = st.session_state.get("espn_lookup_date", date.today())
    if isinstance(lookup_date, date):
        date_text = lookup_date.isoformat()
    else:
        date_text = str(lookup_date or "")

    rows_to_save: List[Dict[str, Any]] = []
    for report in row_reports:
        if str(report.get("estado", "")) != "ok":
            continue

        real_symbol = str(report.get("resultado_real", "")).upper().strip()
        if real_symbol not in {"1", "X", "2"}:
            continue

        partido = int(report.get("partido", 0) or 0)
        analysis = analysis_by_row.get(partido)
        patterns = analysis.patrones_activados if analysis is not None else []

        rows_to_save.append(
            {
                "local": report.get("local", "LOCAL"),
                "visitante": report.get("visitante", "VISITA"),
                "pct_local": report.get("pct_local"),
                "pct_empate": report.get("pct_empate"),
                "pct_visita": report.get("pct_visita"),
                "resultado_real": real_symbol,
                "fecha": date_text,
                "concurso": str(st.session_state.get("jornada_nombre", "") or ""),
                "patrones_activados": patterns,
            }
        )

    if not rows_to_save:
        return None

    ingest = engine.insert_rows_dedup(rows_to_save, source="auto_api_results")
    inserted = int(ingest.get("inserted_rows", 0))
    dupes = int(ingest.get("skipped_duplicate", 0))
    invalid = int(ingest.get("skipped_invalid", 0))
    return f"Memoria historica auto: +{inserted} nuevos, {dupes} duplicados, {invalid} invalidos."


def _render_header() -> None:
    st.markdown(
        """
        <div class="ticket-hero">
            <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px;">
                <span class="badge badge-blue">Progol Lab</span>
                <span class="badge badge-green">En vivo</span>
                <span class="badge badge-yellow">Picks activos</span>
            </div>
            <div class="ticket-hero-title">Progol Contrarian Lab</div>
            <div class="ticket-hero-subtitle">
                Herramienta analitica basada en distribucion publica de pronosticos.
                No predice ganadores con certeza.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([0.64, 0.36])
    with c1:
        st.text_input(
            "Nombre de jornada o concurso",
            key="jornada_nombre",
            placeholder="Ejemplo: Jornada 14 - Concurso principal",
        )
    with c2:
        st.radio(
            "Modo de vista",
            options=["Vista boleto", "Vista analisis"],
            horizontal=True,
            key="view_mode",
        )

    with st.expander("Ver glosario de patrones", expanded=True):
        render_pattern_glossary()


def _render_action_controls() -> None:
    c1, c2, c3 = st.columns([1.0, 1.0, 2.2])

    with c1:
        if st.button("Cargar datos dummy", use_container_width=True):
            _load_dummy_rows()
            st.success("Se cargaron ejemplos de prueba.")

    with c2:
        if st.button("Limpiar jornada", use_container_width=True):
            st.session_state.ticket_rows = _empty_ticket_rows()
            st.warning("Jornada reiniciada.")

    with c3:
        uploaded = st.file_uploader("Cargar sesion JSON", type=["json"])
        if uploaded is not None and st.button("Aplicar sesion cargada", use_container_width=True):
            try:
                payload = parse_session_json(uploaded.getvalue())
                _load_session_into_rows(payload)
                st.success("Sesion cargada correctamente.")
            except Exception as exc:
                st.error(f"No se pudo cargar la sesion: {exc}")

    with st.expander("Importar desde captura (OCR)", expanded=False):
        ocr_available, ocr_message = ocr_runtime_status()
        if ocr_available:
            st.caption(ocr_message)
        else:
            st.warning(ocr_message)

        ocr_files = st.file_uploader(
            "Sube una o varias capturas de porcentajes",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="ocr_capture_uploader",
        )

        c4, c5, c6 = st.columns([1.3, 1.0, 1.0])
        with c4:
            section_label = st.selectbox(
                "Bloque a leer",
                options=["Progol (tabla superior)", "Revancha (tabla inferior)"],
                index=0,
            )
        with c5:
            start_partido = st.number_input(
                "Cargar desde partido",
                min_value=1,
                max_value=MAX_MATCHES,
                value=1,
                step=1,
            )
        with c6:
            only_empty = st.checkbox("Solo filas vacias", value=False)

        st.markdown("##### Verificacion automatica de resultados")
        c7, c8, c9 = st.columns([1.3, 1.3, 1.0])
        with c7:
            st.checkbox(
                "Verificar picks contra API",
                key="espn_results_enabled",
                help="Marca Acierto/Fallo/Pendiente usando fuentes configuradas.",
            )
            st.selectbox(
                "Fuente de resultados",
                options=[
                    "ESPN gratis (recomendado)",
                    "Hibrido ahorro (ESPN -> TheOdds faltantes)",
                    "TheOdds primero",
                ],
                key="results_source_mode",
            )
            st.checkbox(
                "Incluir dia anterior",
                key="espn_include_previous_day",
                help="Prioriza la semana de la fecha elegida (con respaldo historico estricto si no hay match cercano).",
            )
        with c8:
            st.text_input(
                "API Key TheOdds",
                key="theodds_api_key",
                type="password",
                help="Se usa para consulta de resultados y cache local de tickets.",
            )
            st.date_input(
                "Fecha de partidos",
                key="espn_lookup_date",
            )
            st.multiselect(
                "Ligas ESPN (fallback)",
                options=list(ESPN_LEAGUE_OPTIONS.keys()),
                key="espn_selected_leagues",
            )
        with c9:
            if st.button("Refrescar cache en memoria", use_container_width=True):
                clear_espn_cache()
                clear_theodds_runtime_cache()
                st.info(
                    "Cache en memoria limpiada. El cache persistente por ticket se guarda en "
                    f"{THEODDS_CACHE_FILE.name}."
                )
            if st.button("Forzar reconsulta TheOdds (sin cache ticket)", use_container_width=True):
                st.session_state.theodds_force_refresh_once = True
                st.info("Se activara una consulta directa a TheOdds en el siguiente calculo de resultados.")

        if st.button(
            "Leer captura(s) y cargar boleto",
            use_container_width=True,
            disabled=not ocr_available,
        ):
            if not ocr_files:
                st.warning("Sube al menos una captura para iniciar la lectura automatica.")
            else:
                section = "revancha" if section_label.startswith("Revancha") else "progol"
                extracted_rows: List[Dict[str, Any]] = []
                extraction_notes: List[str] = []
                detected_dates: List[date] = []

                for capture in ocr_files:
                    rows, notes, detected_date = extract_matches_with_date_from_capture(
                        image_bytes=capture.getvalue(),
                        section=section,
                        max_matches=MAX_MATCHES,
                    )
                    extracted_rows.extend(rows)
                    extraction_notes.extend([f"{capture.name}: {note}" for note in notes])
                    if detected_date is not None:
                        detected_dates.append(detected_date)

                if detected_dates:
                    date_counter = Counter(detected_dates)
                    selected_date = sorted(
                        date_counter.items(),
                        key=lambda item: (-item[1], abs((date.today() - item[0]).days)),
                    )[0][0]
                    st.session_state.espn_lookup_date = selected_date
                    extraction_notes.append(
                        f"Fecha de resultados autoajustada por OCR: {selected_date.strftime('%d/%m/%Y')}"
                    )

                if not extracted_rows:
                    st.error("No se detectaron partidos validos en la captura. Ajusta el recorte o la calidad de la imagen.")
                else:
                    inserted = _load_extracted_rows_to_ticket(
                        extracted_rows=extracted_rows,
                        start_partido=int(start_partido),
                        only_empty_rows=only_empty,
                    )
                    if inserted == 0:
                        st.warning("No se pudieron insertar filas: no hubo espacio disponible segun el modo elegido.")
                    else:
                        st.success(f"Importacion lista: se cargaron {inserted} partidos desde captura.")
                        if st.session_state.get("espn_results_enabled", True):
                            st.caption("Se activara validacion de resultados API en los picks calculados.")

                    preview_df = pd.DataFrame(extracted_rows).head(MAX_MATCHES)
                    if not preview_df.empty:
                        st.dataframe(
                            preview_df,
                            hide_index=True,
                            use_container_width=True,
                            height=220,
                        )

                if extraction_notes:
                    shown_notes = extraction_notes[:8]
                    for note in shown_notes:
                        st.caption(f"- {note}")
                    if len(extraction_notes) > len(shown_notes):
                        st.caption(f"... {len(extraction_notes) - len(shown_notes)} notas adicionales")


def _render_capture_table() -> None:
    st.subheader("Boleto editable (14 partidos)")
    st.caption(
        "Captura directamente en filas tipo quiniela. "
        "La recomendacion, doble oportunidad y alerta se calculan automaticamente."
    )

    source_df = _rows_to_editor_dataframe(st.session_state.ticket_rows)
    edited_df = st.data_editor(
        source_df,
        hide_index=True,
        num_rows="fixed",
        disabled=["Partido", "Total %"],
        height=470,
        use_container_width=True,
        column_config={
            "Partido": st.column_config.NumberColumn("Partido", width="small", format="%d"),
            "Local": st.column_config.TextColumn("Equipo local", width="medium"),
            "% Local": st.column_config.NumberColumn("% Local", min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
            "% Empate": st.column_config.NumberColumn("% Empate", min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
            "% Visita": st.column_config.NumberColumn("% Visita", min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
            "Visitante": st.column_config.TextColumn("Equipo visitante", width="medium"),
            "Total %": st.column_config.NumberColumn("Total", format="%.2f"),
        },
    )

    st.session_state.ticket_rows = _editor_dataframe_to_rows(edited_df)


def _render_export_buttons(
    main_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    export_matches: List[Dict[str, Any]],
) -> None:
    st.subheader("Exportacion")

    csv_source = main_df.drop(columns=["id"], errors="ignore") if not main_df.empty else main_df
    csv_bytes = dataframe_to_csv_bytes(csv_source)
    excel_bytes = dataframes_to_excel_bytes(main_df, detail_df)
    session_bytes = session_to_json_bytes(export_matches, st.session_state.rules.to_dict())

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            label="Exportar tabla a CSV",
            data=csv_bytes,
            file_name="progol_tabla.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=main_df.empty,
        )
    with c2:
        st.download_button(
            label="Exportar analisis a Excel",
            data=excel_bytes,
            file_name="progol_analisis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            disabled=main_df.empty,
        )
    with c3:
        st.download_button(
            label="Guardar sesion JSON",
            data=session_bytes,
            file_name="progol_sesion.json",
            mime="application/json",
            use_container_width=True,
        )


def _render_boleto_view(row_reports: List[Dict[str, Any]], analyses: List[MatchAnalysis]) -> None:
    st.subheader("Panel de resumen")
    render_summary_cards(analyses)
    render_jornada_panel(analyses)

    st.subheader(f"Vista boleto: {st.session_state.jornada_nombre}")

    aciertos = sum(1 for row in row_reports if str(row.get("resultado_estado", "")) == "acierto")
    fallos = sum(1 for row in row_reports if str(row.get("resultado_estado", "")) == "fallo")
    pendientes = sum(1 for row in row_reports if str(row.get("resultado_estado", "")) == "pendiente")
    sin_resultado = sum(1 for row in row_reports if str(row.get("resultado_estado", "")) == "sin_resultado")
    total_boleto = 14
    st.markdown(
        f"{aciertos}/{total_boleto} acertados | "
        f"{fallos} fallados | "
        f"{pendientes} pendientes | "
        f"{sin_resultado} sin resultado"
    )

    render_ticket_board(row_reports)


def _apply_filters_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    with st.expander("Filtros de analisis", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            tipos = sorted(df["Tipo"].dropna().unique().tolist())
            selected_tipos = st.multiselect("Tipo", options=tipos, default=tipos)
        with c2:
            confs = sorted(df["Confianza"].dropna().unique().tolist())
            selected_conf = st.multiselect("Confianza", options=confs, default=confs)
        with c3:
            semaforos = sorted(df["Semaforo"].dropna().unique().tolist())
            selected_semaforos = st.multiselect("Semaforo", options=semaforos, default=semaforos)
        with c4:
            search = st.text_input("Buscar equipo")

    filtered = df[
        df["Tipo"].isin(selected_tipos)
        & df["Confianza"].isin(selected_conf)
        & df["Semaforo"].isin(selected_semaforos)
    ]

    if search:
        needle = search.strip().lower()
        filtered = filtered[
            filtered["Local"].str.lower().str.contains(needle, na=False)
            | filtered["Visitante"].str.lower().str.contains(needle, na=False)
        ]

    return filtered


def _extract_selected_row(selection_event: Any) -> List[int]:
    if selection_event is None:
        return []

    if isinstance(selection_event, dict):
        return selection_event.get("selection", {}).get("rows", [])

    selection = getattr(selection_event, "selection", None)
    if selection is None:
        return []

    if isinstance(selection, dict):
        return selection.get("rows", [])

    rows = getattr(selection, "rows", None)
    return rows if isinstance(rows, list) else []


def _render_analysis_view(
    filtered_df: pd.DataFrame,
    analysis_by_row: Dict[int, MatchAnalysis],
    report_by_row: Dict[int, Dict[str, Any]],
) -> None:
    st.subheader("Vista analisis")
    if filtered_df.empty:
        st.info("No hay filas capturadas para analizar.")
        return

    selection_event: Any = None
    try:
        selection_event = st.dataframe(
            filtered_df,
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            height=420,
        )
    except TypeError:
        st.dataframe(filtered_df, hide_index=True, use_container_width=True, height=420)

    selected_rows = _extract_selected_row(selection_event)
    selected_partido: int | None = None

    if selected_rows:
        selected_partido = int(filtered_df.iloc[selected_rows[0]]["Partido"])

    if selected_partido is None:
        selected_partido = st.selectbox(
            "Detalle de partido",
            options=filtered_df["Partido"].tolist(),
            format_func=lambda p: f"Partido {int(p)} - {filtered_df.loc[filtered_df['Partido'] == p, 'Local'].iloc[0]} vs {filtered_df.loc[filtered_df['Partido'] == p, 'Visitante'].iloc[0]}",
        )

    analysis = analysis_by_row.get(int(selected_partido))
    if analysis is None:
        st.info("La fila seleccionada no tiene datos completos para analisis heuristico.")
        return
    report = report_by_row.get(int(selected_partido), {})

    direct_pick_symbol = _resolve_direct_pick_symbol(analysis)
    direct_pick_text = _symbol_to_pick_text(direct_pick_symbol, analysis.local, analysis.visitante)
    double_pick = _normalize_double_pick(
        analysis.recomendacion_principal,
        analysis.doble_oportunidad,
        direct_pick_symbol,
        float(analysis.score_local),
        float(analysis.score_visita),
    )

    semaforo = risk_label(analysis.score_riesgo)
    extra_badge = ""
    if double_pick in {"1X", "X2", "12"}:
        extra_badge = '<span class="badge badge-blue">Doble oportunidad sugerida</span>'

    st.markdown(
        confidence_badge(analysis.confianza) + semaforo_badge(semaforo) + extra_badge,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Score local", f"{analysis.score_local:.2f}")
    c2.metric("Score empate", f"{analysis.score_empate:.2f}")
    c3.metric("Score visita", f"{analysis.score_visita:.2f}")
    c4.metric("Score riesgo", f"{analysis.score_riesgo:.2f}")
    c5.metric("Score contrarian", f"{analysis.score_contrarian:.2f}")
    c6.metric("Score estabilidad", f"{analysis.score_estabilidad:.2f}")

    st.write(f"Pick directo: {direct_pick_text} ({direct_pick_symbol})")
    st.write(f"Doble oportunidad sugerida: {double_pick}")
    if report:
        st.markdown("#### Capa hibrida (heuristica + historico)")
        st.write(
            "Heuristica: "
            f"{report.get('pick_heuristico', '-')}"
            f" ({report.get('pick_heuristico_symbol', '-')})"
            f" | Confianza: {report.get('confianza_heuristica', '-')}"
        )
        st.write(
            "Historico: "
            f"{report.get('senal_historica', '-')}"
            f" | Muestra similares: {int(report.get('historical_sample', 0) or 0)}"
            f" | Confianza historica: {report.get('historical_confidence', '-')}"
        )
        historical_distribution = str(report.get("historical_distribution", "") or "").strip()
        if historical_distribution:
            st.write(f"Distribucion historica: {historical_distribution}")
        st.write(
            "Final: "
            f"{report.get('recomendacion', '-')}"
            f" ({report.get('pick_symbol', '-')})"
            f" | Doble final: {report.get('doble_oportunidad', '-')}"
            f" | Confianza final: {report.get('confianza', '-')}"
        )
        st.caption(str(report.get("decision_hibrida", "-")))

    selected_row = filtered_df.loc[filtered_df["Partido"] == int(selected_partido)].iloc[0]
    result_label = str(selected_row.get("Resultado", "Sin validar"))
    score_label = str(selected_row.get("Marcador", "-"))
    real_symbol = str(selected_row.get("Real", "-"))
    if real_symbol in {"1", "X", "2"}:
        st.write(f"Resultado API: {result_label} / Real: {real_symbol} / Marcador: {score_label}")
    else:
        st.write(f"Resultado API: {result_label} / Marcador: {score_label}")
    st.write(f"Tipo de partido: {analysis.tipo_partido}")
    st.write(f"Sesgos detectados: {' | '.join(analysis.sesgos)}")

    if analysis.banderas_alerta:
        st.warning("Alertas: " + " | ".join(analysis.banderas_alerta))

    st.info(analysis.explicacion)

    if st.session_state.get("debug_mode") and analysis.debug_data:
        st.markdown("#### Debug heuristico")
        st.json(analysis.debug_data)

    st.markdown("Salida JSON por partido")
    st.json(build_match_json_output(analysis))


def main() -> None:
    init_state()

    _render_rules_editor_sidebar()
    historical_engine = _resolve_historical_engine()

    _render_header()
    with st.expander("Controles de jornada y carga", expanded=False):
        _render_action_controls()

    with st.expander("Memoria historica (patrones)", expanded=True):
        _render_historical_memory_panel(historical_engine, st.session_state.rules)

    with st.expander("Editor de boleto (14 partidos)", expanded=True):
        _render_capture_table()

    row_reports, analyses, analysis_by_row, export_matches = _evaluate_ticket_rows(
        st.session_state.ticket_rows,
        st.session_state.rules,
        historical_engine=historical_engine,
        historical_enabled=bool(st.session_state.get("historical_enabled", True)),
        debug_mode=bool(st.session_state.get("debug_mode", False)),
    )
    espn_notes = _apply_espn_results_to_reports(row_reports)
    auto_hist_note = _autosave_historical_from_reports(row_reports, analysis_by_row, historical_engine)

    main_df, detail_df = build_export_dataframes(analyses)
    _render_export_buttons(main_df, detail_df, export_matches)

    if espn_notes:
        st.caption("API: " + " | ".join(espn_notes[:3]))
        if st.session_state.get("debug_mode", False):
            with st.expander("Debug API notas completas", expanded=False):
                for note in espn_notes:
                    st.write(f"- {note}")
    if auto_hist_note:
        st.caption(auto_hist_note)

    if st.session_state.view_mode == "Vista boleto":
        _render_boleto_view(row_reports, analyses)
        return

    summary_df = _build_analysis_dataframe(row_reports)
    filtered_df = _apply_filters_for_analysis(summary_df)
    report_by_row = {int(item.get("partido", 0) or 0): item for item in row_reports}
    _render_analysis_view(filtered_df, analysis_by_row, report_by_row)


if __name__ == "__main__":
    main()
