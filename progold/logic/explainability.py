from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from models import MatchInput

from .pattern_detector import PatternDetectionResult
from .recommendation_engine import RecommendationDecision
from .scoring_engine import ScoringEngineResult


@dataclass
class ExplainabilityPayload:
    sesgos: List[str]
    alertas: List[str]
    explicacion: str
    debug_payload: Dict[str, Any]


_PATTERN_LABELS = {
    "empate_vivo": "empate_vivo",
    "empate_ignorado_por_masa": "empate_ignorado_por_masa",
    "visita_sobrecomprada_moderada": "visita_sobrecomprada_moderada",
    "local_sobrepopular": "local_sobrepopular",
    "favorito_estable": "favorito_estable",
    "partido_caotico": "partido_caotico",
    "visita_viva": "visita_viva",
    "partido_abierto_con_empate_castigado": "partido_abierto_con_empate_castigado",
    "favorito_sobrejugado": "favorito_sobrejugado",
}


def _format_pattern_list(pattern_names: List[str]) -> str:
    if not pattern_names:
        return "ninguno"
    if len(pattern_names) == 1:
        return pattern_names[0]
    if len(pattern_names) == 2:
        return f"{pattern_names[0]} y {pattern_names[1]}"
    return ", ".join(pattern_names[:-1]) + f" y {pattern_names[-1]}"


def _build_explanation_lines(
    detection: PatternDetectionResult,
    scoring: ScoringEngineResult,
    recommendation: RecommendationDecision,
) -> List[str]:
    lines: List[str] = []

    active = [name for name in detection.active_pattern_names() if name in _PATTERN_LABELS]
    if active:
        lines.append(f"Se activaron patrones: {_format_pattern_list(active)}.")
    else:
        lines.append("No se activaron patrones fuertes de sesgo; lectura base por distribucion.")

    if detection.is_active("empate_ignorado_por_masa"):
        lines.append("Se detecto empate_ignorado_por_masa y se elevo score de X.")
    if detection.is_active("local_sobrepopular"):
        lines.append("Se detecto local_sobrepopular y se penalizo el score del local.")
    if detection.is_active("visita_sobrecomprada_moderada"):
        lines.append("La visita lidera, pero se considera sobrecomprada de forma moderada.")
    if detection.is_active("favorito_estable"):
        lines.append("El liderazgo principal cumple estructura de favorito_estable.")
    if detection.is_active("partido_caotico"):
        lines.append("No hay suficiente estabilidad para pick directo; se prioriza cobertura.")
    if detection.is_active("favorito_sobrejugado") and not detection.is_active("favorito_estable"):
        lines.append("Se detecto favorito_sobrejugado y se privilegio correccion contrarian con cobertura.")
    if recommendation.recomendacion_principal in {"1X", "X2", "12"}:
        lines.append("La jerarquia final evita pick directo por riesgo/contexto competitivo.")

    lines.append(
        "Scores finales -> "
        f"1: {scoring.scores.score_local:.2f}, "
        f"X: {scoring.scores.score_empate:.2f}, "
        f"2: {scoring.scores.score_visita:.2f}, "
        f"riesgo: {scoring.scores.score_riesgo:.2f}, "
        f"contrarian: {scoring.scores.score_contrarian:.2f}, "
        f"estabilidad: {scoring.scores.score_estabilidad:.2f}."
    )

    if recommendation.explicacion_decision:
        lines.append("Decision jerarquica: " + " ".join(recommendation.explicacion_decision))

    return lines


def build_explainability(
    match: MatchInput,
    detection: PatternDetectionResult,
    scoring: ScoringEngineResult,
    recommendation: RecommendationDecision,
) -> ExplainabilityPayload:
    active_patterns = detection.active_pattern_names()

    sesgos: List[str] = []
    for name in active_patterns:
        label = _PATTERN_LABELS.get(name)
        if label:
            sesgos.append(label)

    if not sesgos:
        sesgos = ["lectura_estable"]

    alertas = list(detection.alerts)
    if scoring.scores.score_riesgo >= 70:
        alertas.append("riesgo_alto_por_scoring")
    if recommendation.confianza == "baja":
        alertas.append("confianza_baja")

    explanation_lines = _build_explanation_lines(detection, scoring, recommendation)

    adjustment_sources = {adj.source for adj in scoring.adjustments}
    draw_without_structure = (
        scoring.scores.score_empate >= max(scoring.scores.score_local, scoring.scores.score_visita)
        and not detection.is_active("empate_vivo")
        and not detection.is_active("empate_ignorado_por_masa")
    )
    conflict_flags = {
        "stable_vs_overplayed": detection.is_active("favorito_estable") and detection.is_active("favorito_sobrejugado"),
        "draw_without_structure": draw_without_structure,
        "local_overprotection_guard": (
            "guard_visita_competitiva" in adjustment_sources
            or "local_sobrepopular" in adjustment_sources
        ),
    }

    debug_payload: Dict[str, Any] = {
        "input": {
            "local": match.local,
            "visitante": match.visitante,
            "pct_local": round(float(match.pct_local), 2),
            "pct_empate": round(float(match.pct_empate), 2),
            "pct_visita": round(float(match.pct_visita), 2),
        },
        "derived_metrics": {
            "top_mass_key": detection.top_key,
            "second_mass_key": detection.second_key,
            "top_mass_pct": detection.top_pct,
            "second_mass_pct": detection.second_pct,
            "gap_top2": detection.gap_top2,
            "side_gap": detection.side_gap,
        },
        "patterns": {
            name: {
                "active": activation.active,
                "strength": round(activation.strength, 4),
                "reason": activation.reason,
            }
            for name, activation in detection.patterns.items()
        },
        "scores": {
            "base": scoring.base_scores,
            "final": scoring.final_scores,
            "adjustments": [
                {
                    "source": adjustment.source,
                    "target": adjustment.target,
                    "delta": adjustment.delta,
                    "reason": adjustment.reason,
                }
                for adjustment in scoring.adjustments
            ],
            "conflict_flags": conflict_flags,
        },
        "decision": {
            "tipo_partido": recommendation.tipo_partido,
            "recomendacion_principal": recommendation.recomendacion_principal,
            "doble_oportunidad": recommendation.doble_oportunidad,
            "confianza": recommendation.confianza,
            "trace": recommendation.explicacion_decision,
        },
    }

    return ExplainabilityPayload(
        sesgos=sesgos,
        alertas=alertas,
        explicacion=" ".join(explanation_lines),
        debug_payload=debug_payload,
    )
