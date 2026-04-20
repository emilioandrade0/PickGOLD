from __future__ import annotations

from typing import List

from config import HeuristicConfig
from models import MatchAnalysis, MatchInput

from .explainability import build_explainability
from .pattern_detector import detect_patterns
from .recommendation_engine import choose_recommendation
from .scoring_engine import compute_match_scores


def _safe_join(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in values:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def analyze_match(match: MatchInput, cfg: HeuristicConfig, debug_mode: bool = False) -> MatchAnalysis:
    percentages = {
        "1": match.pct_local,
        "X": match.pct_empate,
        "2": match.pct_visita,
    }
    ordered = sorted(percentages.items(), key=lambda item: item[1], reverse=True)

    top_pct = ordered[0][1]
    second_pct = ordered[1][1]
    gap_top2 = top_pct - second_pct

    detection = detect_patterns(match, cfg)
    scoring = compute_match_scores(match, cfg, detection)
    decision = choose_recommendation(
        match=match,
        scores=scoring.scores,
        detection=detection,
        cfg=cfg,
    )
    explainability = build_explainability(match, detection, scoring, decision)

    all_sesgos = _safe_join(explainability.sesgos)

    return MatchAnalysis(
        local=match.local,
        visitante=match.visitante,
        pct_local=round(match.pct_local, 2),
        pct_empate=round(match.pct_empate, 2),
        pct_visita=round(match.pct_visita, 2),
        porcentaje_mayor=round(top_pct, 2),
        segundo_porcentaje_mayor=round(second_pct, 2),
        diferencia_top2=round(gap_top2, 2),
        tipo_partido=decision.tipo_partido,
        sesgos=all_sesgos,
        recomendacion_principal=decision.recomendacion_principal,
        doble_oportunidad=decision.doble_oportunidad,
        confianza=decision.confianza,
        explicacion=explainability.explicacion,
        posible_ganador_masa=detection.top_key,
        banderas_alerta=explainability.alertas,
        patrones_activados=detection.active_pattern_names(),
        score_local=scoring.scores.score_local,
        score_empate=scoring.scores.score_empate,
        score_visita=scoring.scores.score_visita,
        score_riesgo=scoring.scores.score_riesgo,
        score_contrarian=scoring.scores.score_contrarian,
        score_estabilidad=scoring.scores.score_estabilidad,
        apto_pick_directo=decision.apto_pick_directo,
        apto_doble_oportunidad=decision.apto_doble_oportunidad,
        sugerir_sorpresa=decision.sugerir_sorpresa,
        debug_data=explainability.debug_payload if debug_mode else {},
    )


def analyze_matches(matches: List[MatchInput], cfg: HeuristicConfig, debug_mode: bool = False) -> List[MatchAnalysis]:
    return [analyze_match(match, cfg, debug_mode=debug_mode) for match in matches]
