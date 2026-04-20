from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from config import HeuristicConfig
from models import MatchInput, ScoreBreakdown

from .pattern_detector import PatternDetectionResult

_DIRECT = {"1", "X", "2"}
_DOUBLE = {"1X", "X2", "12"}


@dataclass
class RecommendationDecision:
    recomendacion_principal: str
    doble_oportunidad: str
    confianza: str
    tipo_partido: str
    explicacion_decision: List[str]
    apto_pick_directo: bool
    apto_doble_oportunidad: bool
    sugerir_sorpresa: bool


def _as_score_map(scores: ScoreBreakdown) -> Dict[str, float]:
    return {
        "1": float(scores.score_local),
        "X": float(scores.score_empate),
        "2": float(scores.score_visita),
    }


def _ordered_scores(scores: ScoreBreakdown) -> List[tuple[str, float]]:
    score_map = _as_score_map(scores)
    return sorted(score_map.items(), key=lambda item: item[1], reverse=True)


def _double_from_pair(options: Sequence[str]) -> str:
    unique = set(options)
    if unique == {"1", "X"}:
        return "1X"
    if unique == {"X", "2"}:
        return "X2"
    if unique == {"1", "2"}:
        return "12"
    return "-"


def _confidence_label(
    recommendation: str,
    scores: ScoreBreakdown,
    cfg: HeuristicConfig,
    detection: PatternDetectionResult,
) -> str:
    high_conf = (
        recommendation in _DIRECT
        and scores.score_riesgo <= cfg.confianza_alta_riesgo_max
        and scores.score_estabilidad >= cfg.estabilidad_alta_min
        and scores.score_contrarian < cfg.contrarian_alto_min
        and detection.is_active("favorito_estable")
        and not detection.is_active("favorito_sobrejugado")
    )
    if high_conf:
        return "alta"

    low_conf = (
        scores.score_riesgo >= cfg.confianza_baja_riesgo_min
        or scores.score_contrarian >= cfg.contrarian_alto_min
        or scores.score_estabilidad <= cfg.estabilidad_baja_max
        or detection.is_active("partido_caotico")
    )
    if low_conf:
        return "baja"

    return "media"


def _smart_double_opportunity(
    match: MatchInput,
    scores: ScoreBreakdown,
    detection: PatternDetectionResult,
    cfg: HeuristicConfig,
    ordered: Sequence[tuple[str, float]],
) -> str:
    if detection.is_active("partido_abierto_con_empate_castigado"):
        return "12"

    if detection.is_active("favorito_estable") and scores.score_estabilidad >= cfg.estabilidad_alta_min:
        if (ordered[0][1] - ordered[1][1]) >= cfg.double_evitar_gap_dominante:
            if ordered[0][0] == "1":
                return "1X"
            if ordered[0][0] == "2":
                return "X2"
            return "1X"

    score_map = _as_score_map(scores)

    one_x_competitive = (
        abs(score_map["1"] - score_map["X"]) <= cfg.double_competitive_gap
        and score_map["1"] >= score_map["2"] - cfg.double_side_guard
    )
    x_two_competitive = (
        abs(score_map["2"] - score_map["X"]) <= cfg.double_competitive_gap
        and score_map["2"] >= score_map["1"] - cfg.double_side_guard
    )

    if one_x_competitive and not x_two_competitive:
        return "1X"
    if x_two_competitive and not one_x_competitive:
        return "X2"

    if one_x_competitive and x_two_competitive:
        return "1X" if score_map["1"] >= score_map["2"] else "X2"

    if score_map["X"] >= ordered[0][1] - cfg.x_compite_margin:
        return "1X" if score_map["1"] >= score_map["2"] else "X2"

    if match.pct_empate <= cfg.partido_abierto_empate_castigado_max and match.pct_local >= cfg.partido_abierto_extremos_min and match.pct_visita >= cfg.partido_abierto_extremos_min:
        return "12"

    fallback = _double_from_pair([ordered[0][0], ordered[1][0]])
    if fallback == "12" and match.pct_empate >= cfg.empate_bajo_min:
        return "1X" if score_map["1"] >= score_map["2"] else "X2"
    return fallback


def _match_type(detection: PatternDetectionResult) -> str:
    if detection.is_active("favorito_estable") and not detection.is_active("favorito_sobrejugado"):
        return "Favorito estable"
    if detection.is_active("favorito_sobrejugado"):
        return "Favorito sobrejugado"
    if detection.is_active("partido_caotico"):
        return "Partido caotico"
    if detection.is_active("empate_ignorado_por_masa"):
        return "Empate ignorado por masa"
    if detection.is_active("empate_vivo"):
        return "Empate vivo"
    if detection.is_active("visita_viva"):
        return "Visita viva"
    return "Partido de cuidado"


def choose_recommendation(
    match: MatchInput,
    scores: ScoreBreakdown,
    detection: PatternDetectionResult,
    cfg: HeuristicConfig,
) -> RecommendationDecision:
    ordered = _ordered_scores(scores)
    top_key, top_score = ordered[0]
    second_key, second_score = ordered[1]
    top_gap = top_score - second_score
    min_draw_gap = min(abs(match.pct_local - match.pct_empate), abs(match.pct_visita - match.pct_empate))

    mass_parejo = detection.gap_top2 <= cfg.diff_parejo_max
    mass_ligero = detection.gap_top2 <= cfg.diff_ligera_max
    draw_competitive = min_draw_gap <= cfg.draw_gap_alert_max
    contrarian_hot = scores.score_contrarian >= cfg.contrarian_alto_min

    decision_notes: List[str] = []

    recomendacion = top_key
    doble = "-"

    if detection.is_active("partido_abierto_con_empate_castigado"):
        recomendacion = "12"
        doble = "12"
        decision_notes.append("partido abierto con empate castigado: se recomienda 12.")

    elif detection.is_active("partido_caotico") or (mass_parejo and scores.score_riesgo >= cfg.confianza_alta_riesgo_max):
        doble = _smart_double_opportunity(match, scores, detection, cfg, ordered)
        if doble not in _DOUBLE:
            doble = _double_from_pair([top_key, second_key])
        recomendacion = doble
        decision_notes.append("partido muy cerrado/caotico: se evita pick directo.")

    elif detection.is_active("favorito_sobrejugado"):
        doble = _smart_double_opportunity(match, scores, detection, cfg, ordered)
        if top_key == "1" and doble == "1X":
            doble = "X2" if scores.score_visita >= scores.score_empate else "1X"
        if top_key == "2" and doble == "X2" and scores.score_local >= scores.score_empate - cfg.x_compite_margin:
            doble = "1X"
        recomendacion = doble if doble in _DOUBLE else _double_from_pair([top_key, second_key])
        decision_notes.append("favorito_sobrejugado detectado: se prioriza cobertura sobre pick directo.")

    elif detection.is_active("favorito_estable") and not detection.is_active("favorito_sobrejugado"):
        direct_gap_required = max(cfg.decision_gap_directo_min, cfg.diff_ligera_max - 1.0)
        can_direct = (
            scores.score_riesgo <= cfg.confianza_alta_riesgo_max
            and scores.score_estabilidad >= cfg.estabilidad_alta_min
            and not contrarian_hot
            and top_gap >= direct_gap_required
        )
        if can_direct:
            recomendacion = top_key
            decision_notes.append("favorito_estable fuerte, bajo riesgo y brecha amplia: se permite pick directo.")
        else:
            doble = _smart_double_opportunity(match, scores, detection, cfg, ordered)
            recomendacion = doble if doble in _DOUBLE else top_key
            decision_notes.append("favorito_estable pero con margen/riesgo no ideal: se sugiere cobertura.")

    elif detection.is_active("empate_vivo") or detection.is_active("empate_ignorado_por_masa"):
        score_map = _as_score_map(scores)
        max_side = max(score_map["1"], score_map["2"])
        if draw_competitive:
            if score_map["X"] >= max_side - cfg.x_compite_margin:
                if score_map["X"] - max_side >= cfg.decision_gap_empate_directo_min:
                    recomendacion = "X"
                    decision_notes.append("empate_vivo fuerte: se propone X directo.")
                else:
                    doble = "1X" if score_map["1"] >= score_map["2"] else "X2"
                    recomendacion = doble
                    decision_notes.append("empate competitivo sin ventaja amplia: cobertura con empate.")
            else:
                doble = "1X" if score_map["1"] >= score_map["2"] else "X2"
                recomendacion = doble
                decision_notes.append("empate con estructura activa: se mantiene cobertura con empate.")
        else:
            doble = _smart_double_opportunity(match, scores, detection, cfg, ordered)
            recomendacion = doble if doble in _DOUBLE else top_key
            decision_notes.append("empate en lectura activa, pero domina un lado: se balancea con doble.")

    elif detection.is_active("visita_sobrecomprada_moderada") and top_key == "2":
        doble = "X2"
        recomendacion = doble
        decision_notes.append("visita_sobrecomprada_moderada: se evita 2 directo, preferencia X2.")

    elif detection.is_active("local_sobrepopular") and top_key == "1":
        score_map = _as_score_map(scores)
        doble = "1X" if score_map["X"] >= score_map["2"] else "X2"
        recomendacion = doble
        decision_notes.append("local_sobrepopular: se reduce proteccion al 1 con cobertura.")

    elif top_gap <= max(cfg.decision_gap_doble_max, cfg.diff_parejo_max * 0.65):
        doble = _smart_double_opportunity(match, scores, detection, cfg, ordered)
        recomendacion = doble if doble in _DOUBLE else _double_from_pair([top_key, second_key])
        decision_notes.append("scores cercanos: se prioriza doble oportunidad.")

    elif mass_ligero and (scores.score_riesgo >= cfg.confianza_alta_riesgo_max or contrarian_hot):
        doble = _smart_double_opportunity(match, scores, detection, cfg, ordered)
        recomendacion = doble if doble in _DOUBLE else _double_from_pair([top_key, second_key])
        decision_notes.append("ventaja ligera con riesgo/contrarian elevado: mejor doble oportunidad.")

    else:
        recomendacion = top_key
        if scores.score_riesgo >= cfg.riesgo_alto_min or detection.is_active("favorito_sobrejugado") or contrarian_hot:
            doble = _smart_double_opportunity(match, scores, detection, cfg, ordered)
            if doble in _DOUBLE:
                decision_notes.append("pick principal directo con riesgo alto: se sugiere doble de respaldo.")
        decision_notes.append("sin conflictos fuertes de patrones: pick directo por score.")

    if doble == "-":
        doble = _smart_double_opportunity(match, scores, detection, cfg, ordered)

    confianza = _confidence_label(recomendacion, scores, cfg, detection)
    apto_pick_directo = recomendacion in _DIRECT and confianza != "baja"
    apto_doble = recomendacion in _DOUBLE or doble in _DOUBLE

    sorpresa = (
        detection.is_active("favorito_sobrejugado")
        or detection.is_active("local_sobrepopular")
        or detection.is_active("visita_sobrecomprada_moderada")
        or scores.score_contrarian >= cfg.contrarian_alto_min
        or (detection.top_key not in recomendacion)
    )

    return RecommendationDecision(
        recomendacion_principal=recomendacion,
        doble_oportunidad=doble if doble in _DOUBLE else "-",
        confianza=confianza,
        tipo_partido=_match_type(detection),
        explicacion_decision=decision_notes,
        apto_pick_directo=apto_pick_directo,
        apto_doble_oportunidad=apto_doble,
        sugerir_sorpresa=sorpresa,
    )
