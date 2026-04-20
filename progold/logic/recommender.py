from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from config import HeuristicConfig
from models import MatchInput, ScoreBreakdown


@dataclass
class RecommendationResult:
    recomendacion_principal: str
    doble_oportunidad: str
    confianza: str
    explicacion: str
    posible_ganador_masa: str
    apto_pick_directo: bool
    apto_doble_oportunidad: bool
    sugerir_sorpresa: bool


def _top_mass_pick(match: MatchInput) -> str:
    percentages = {
        "1": match.pct_local,
        "X": match.pct_empate,
        "2": match.pct_visita,
    }
    return sorted(percentages.items(), key=lambda item: item[1], reverse=True)[0][0]


def _best_score_pick(scores: ScoreBreakdown) -> str:
    score_map = {
        "1": scores.score_local,
        "X": scores.score_empate,
        "2": scores.score_visita,
    }
    return sorted(score_map.items(), key=lambda item: item[1], reverse=True)[0][0]


def _confidence_label(
    recommendation: str,
    risk_score: float,
    cfg: HeuristicConfig,
    trap_signal: bool,
) -> str:
    if recommendation in {"1", "X", "2"} and risk_score <= cfg.confianza_alta_riesgo_max and not trap_signal:
        return "alta"
    if risk_score >= cfg.confianza_baja_riesgo_min or trap_signal:
        return "baja"
    return "media"


def build_recommendation(
    match: MatchInput,
    scores: ScoreBreakdown,
    tipo_partido: str,
    sesgos: List[str],
    alertas: List[str],
    scoring_notes: List[str],
    cfg: HeuristicConfig,
) -> RecommendationResult:
    top_mass = _top_mass_pick(match)
    best_pick = _best_score_pick(scores)

    percentages: Dict[str, float] = {
        "1": match.pct_local,
        "X": match.pct_empate,
        "2": match.pct_visita,
    }
    ordered = sorted(percentages.items(), key=lambda item: item[1], reverse=True)
    top_pct = ordered[0][1]
    second_pct = ordered[1][1]
    gap_top2 = top_pct - second_pct

    side_gap = abs(match.pct_local - match.pct_visita)
    draw_undervalued = side_gap <= cfg.draw_gap_alert_max and match.pct_empate <= cfg.empate_bajo_max

    trap_signal = (
        scores.score_riesgo >= cfg.confianza_baja_riesgo_min
        or "Posible visitante sobrecomprado" in sesgos
        or "Posible sesgo por nombre" in sesgos
        or "Concentracion de masa muy alta" in alertas
        or scores.score_contrarian >= cfg.contrarian_alto_min
    )

    recomendacion = best_pick
    doble_oportunidad = "-"

    explanation_lines: List[str] = []

    if gap_top2 <= cfg.diff_parejo_max:
        explanation_lines.append(
            "La diferencia entre la opcion mas jugada y la segunda es pequena, por eso se considera partido parejo."
        )
    elif gap_top2 <= cfg.diff_ligera_max:
        explanation_lines.append(
            "La diferencia entre primera y segunda opcion es moderada, por eso se clasifica como inclinacion ligera."
        )
    elif gap_top2 <= cfg.diff_favorito_claro_max:
        explanation_lines.append(
            "Hay favorito claro por brecha de porcentajes, pero aun existe riesgo tactico."
        )
    else:
        explanation_lines.append(
            "La brecha de porcentajes es muy grande, lo que puede indicar concentracion excesiva de masa."
        )

    if draw_undervalued:
        explanation_lines.append(
            "El empate esta bajo respecto a la cercania entre local y visita, se considera infravalorado."
        )

    if "Posible visitante sobrecomprado" in sesgos:
        explanation_lines.append("La visita parece inflada por sesgo de publico.")

    if match.pct_empate < cfg.empate_bajo_min and match.pct_local >= 34.0 and match.pct_visita >= 34.0:
        recomendacion = "12"
        doble_oportunidad = "12"
        explanation_lines.append(
            "Local y visita tienen soporte alto con empate castigado, se recomienda cubrir 12."
        )
    elif draw_undervalued and gap_top2 <= cfg.diff_parejo_max:
        empate_compite = scores.score_empate >= max(scores.score_local, scores.score_visita) - 1.5
        if empate_compite:
            recomendacion = "X"
            explanation_lines.append("La estructura favorece lectura de empate como jugada contrarian.")
        else:
            recomendacion = "1X" if match.pct_local >= match.pct_visita else "X2"
            doble_oportunidad = recomendacion
            explanation_lines.append(
                "Se recomienda doble oportunidad para cubrir empate en un cruce de soporte cerrado."
            )
    elif trap_signal and top_pct >= cfg.favorito_extremo_min:
        if top_mass == "1":
            recomendacion = "1X"
        elif top_mass == "2":
            recomendacion = "X2"
        else:
            recomendacion = "12"
        doble_oportunidad = recomendacion
        explanation_lines.append(
            "Se recomienda doble oportunidad por exceso de concentracion en una sola opcion."
        )
    elif gap_top2 <= cfg.diff_ligera_max:
        recomendacion = "1X" if match.pct_local >= match.pct_visita else "X2"
        doble_oportunidad = recomendacion
        explanation_lines.append(
            "La ventaja es corta y conviene cobertura para bajar varianza."
        )
    else:
        recomendacion = best_pick
        if recomendacion in {"1", "2"} and trap_signal:
            doble_oportunidad = "1X" if recomendacion == "1" else "X2"
            explanation_lines.append(
                "Aunque el pick principal es directo, se sugiere cobertura por senales de trampa."
            )

    confianza = _confidence_label(recomendacion, scores.score_riesgo, cfg, trap_signal)

    if scoring_notes:
        explanation_lines.append(scoring_notes[0])

    apto_pick_directo = recomendacion in {"1", "X", "2"} and confianza != "baja"
    apto_doble_oportunidad = recomendacion in {"1X", "X2", "12"} or doble_oportunidad in {"1X", "X2", "12"}

    top_in_recommendation = top_mass in recomendacion
    sugerir_sorpresa = (not top_in_recommendation) or (
        scores.score_contrarian >= cfg.contrarian_alto_min and top_pct >= cfg.favorito_extremo_min
    )

    return RecommendationResult(
        recomendacion_principal=recomendacion,
        doble_oportunidad=doble_oportunidad,
        confianza=confianza,
        explicacion=" ".join(explanation_lines),
        posible_ganador_masa=top_mass,
        apto_pick_directo=apto_pick_directo,
        apto_doble_oportunidad=apto_doble_oportunidad,
        sugerir_sorpresa=sugerir_sorpresa,
    )
