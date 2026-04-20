from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from config import HeuristicConfig
from models import MatchInput, ScoreBreakdown

from .pattern_detector import PatternDetectionResult


@dataclass
class ScoreAdjustment:
    source: str
    target: str
    delta: float
    reason: str


@dataclass
class ScoringEngineResult:
    scores: ScoreBreakdown
    base_scores: Dict[str, float]
    final_scores: Dict[str, float]
    adjustments: List[ScoreAdjustment]


def _clamp(value: float, min_value: float = 0.0, max_value: float = 100.0) -> float:
    return max(min_value, min(max_value, value))


def _apply(
    score_map: Dict[str, float],
    adjustments: List[ScoreAdjustment],
    *,
    source: str,
    target: str,
    delta: float,
    reason: str,
) -> None:
    score_map[target] = score_map[target] + delta
    adjustments.append(
        ScoreAdjustment(
            source=source,
            target=target,
            delta=round(delta, 4),
            reason=reason,
        )
    )


def _pattern_strength(detection: PatternDetectionResult, pattern_name: str) -> float:
    activation = detection.patterns.get(pattern_name)
    if activation is None or not activation.active:
        return 0.0
    return activation.strength


def compute_match_scores(
    match: MatchInput,
    cfg: HeuristicConfig,
    detection: PatternDetectionResult,
) -> ScoringEngineResult:
    base = {
        "1": float(match.pct_local),
        "X": float(match.pct_empate),
        "2": float(match.pct_visita),
    }

    min_draw_gap = min(abs(match.pct_local - match.pct_empate), abs(match.pct_visita - match.pct_empate))

    # Riesgo base por cercania competitiva y concentracion de masa.
    base_risk = 22.0 + max(0.0, (cfg.diff_ligera_max - detection.gap_top2)) * 1.45
    if detection.gap_top2 <= cfg.diff_parejo_max:
        base_risk += 3.2
    if detection.top_pct >= cfg.favorito_extremo_min:
        base_risk += max(0.0, detection.top_pct - cfg.favorito_extremo_min) * 0.28
    if min_draw_gap <= cfg.draw_gap_alert_max and match.pct_empate <= cfg.empate_bajo_max:
        base_risk += 2.2

    # Contrarian base por dominancia y brecha desproporcionada.
    base_contrarian = 16.0 + max(0.0, detection.top_pct - 44.0) * 0.5
    if detection.gap_top2 >= cfg.sobrepopularidad_gap_min:
        base_contrarian += 3.0
    if detection.top_pct >= cfg.hiperpopular_min:
        base_contrarian += 4.0

    # Estabilidad base por brecha lider-segunda, moderada en partidos parejos.
    base_stability = 44.0 + (detection.gap_top2 * 1.35)
    if detection.gap_top2 <= cfg.diff_parejo_max:
        base_stability -= 5.0

    score_map: Dict[str, float] = {
        "1": base["1"],
        "X": base["X"],
        "2": base["2"],
        "riesgo": base_risk,
        "contrarian": base_contrarian,
        "estabilidad": base_stability,
    }

    adjustments: List[ScoreAdjustment] = []

    empate_vivo = _pattern_strength(detection, "empate_vivo")
    if empate_vivo > 0:
        empate_vivo_gain = cfg.peso_empate_vivo * empate_vivo
        if detection.is_active("partido_abierto_con_empate_castigado") or match.pct_empate < cfg.empate_bajo_min:
            empate_vivo_gain *= 0.55
        _apply(
            score_map,
            adjustments,
            source="empate_vivo",
            target="X",
            delta=empate_vivo_gain,
            reason="Empate con estructura real competitiva.",
        )
        _apply(
            score_map,
            adjustments,
            source="empate_vivo",
            target="riesgo",
            delta=1.9 * empate_vivo,
            reason="El empate vivo introduce mas varianza.",
        )

    empate_ignorado = _pattern_strength(detection, "empate_ignorado_por_masa")
    if empate_ignorado > 0:
        ignored_gain = cfg.peso_empate_ignorado * empate_ignorado
        if match.pct_empate < cfg.empate_bajo_min:
            ignored_gain *= 0.55
        if detection.is_active("partido_abierto_con_empate_castigado"):
            ignored_gain *= 0.35
        _apply(
            score_map,
            adjustments,
            source="empate_ignorado_por_masa",
            target="X",
            delta=ignored_gain,
            reason="La masa subestima el empate en juego parejo.",
        )
        _apply(
            score_map,
            adjustments,
            source="empate_ignorado_por_masa",
            target="contrarian",
            delta=3.8 * empate_ignorado,
            reason="Existe oportunidad contrarian por sesgo de masa.",
        )
        _apply(
            score_map,
            adjustments,
            source="empate_ignorado_por_masa",
            target="riesgo",
            delta=3.9 * empate_ignorado,
            reason="Ignorar empate en partido parejo eleva riesgo.",
        )

    visita_sobrecomprada = _pattern_strength(detection, "visita_sobrecomprada_moderada")
    if visita_sobrecomprada > 0:
        visit_penalty = cfg.peso_visita_sobrecomprada * visita_sobrecomprada
        if match.pct_visita >= cfg.visitante_sobrecomprado_min:
            visit_penalty *= 1.18
        _apply(
            score_map,
            adjustments,
            source="visita_sobrecomprada_moderada",
            target="2",
            delta=-visit_penalty,
            reason="La visita lidera pero con senales de inflacion de masa.",
        )
        _apply(
            score_map,
            adjustments,
            source="visita_sobrecomprada_moderada",
            target="X",
            delta=(cfg.peso_visita_sobrecomprada * 0.48 * visita_sobrecomprada),
            reason="Cobertura natural por empate util.",
        )
        _apply(
            score_map,
            adjustments,
            source="visita_sobrecomprada_moderada",
            target="1",
            delta=(cfg.peso_visita_sobrecomprada * 0.22 * visita_sobrecomprada),
            reason="Local vivo evita 2 automatico.",
        )
        _apply(
            score_map,
            adjustments,
            source="visita_sobrecomprada_moderada",
            target="riesgo",
            delta=5.2 * visita_sobrecomprada,
            reason="Escenario de trampa moderada en visita.",
        )
        _apply(
            score_map,
            adjustments,
            source="visita_sobrecomprada_moderada",
            target="estabilidad",
            delta=-4.6 * visita_sobrecomprada,
            reason="La estabilidad baja cuando el lider no domina.",
        )

    local_sobrepopular = _pattern_strength(detection, "local_sobrepopular")
    if local_sobrepopular > 0:
        local_penalty = cfg.peso_local_sobrepopular * local_sobrepopular * 1.08
        _apply(
            score_map,
            adjustments,
            source="local_sobrepopular",
            target="1",
            delta=-local_penalty,
            reason="Local sobrejugado frente a visita/empate vivos.",
        )
        _apply(
            score_map,
            adjustments,
            source="local_sobrepopular",
            target="2",
            delta=(cfg.peso_local_sobrepopular * 0.55 * local_sobrepopular),
            reason="Visita mantiene vida competitiva.",
        )
        _apply(
            score_map,
            adjustments,
            source="local_sobrepopular",
            target="X",
            delta=(cfg.peso_local_sobrepopular * 0.30 * local_sobrepopular),
            reason="Empate gana valor ante sobrepopularidad local.",
        )
        if detection.gap_top2 <= cfg.diff_parejo_max:
            _apply(
                score_map,
                adjustments,
                source="local_sobrepopular",
                target="2",
                delta=(cfg.peso_local_sobrepopular * 0.18 * local_sobrepopular),
                reason="Brecha corta: se reduce sesgo de proteccion local.",
            )
        _apply(
            score_map,
            adjustments,
            source="local_sobrepopular",
            target="contrarian",
            delta=5.0 * local_sobrepopular,
            reason="Se detecta sesgo de publico hacia local.",
        )
        _apply(
            score_map,
            adjustments,
            source="local_sobrepopular",
            target="riesgo",
            delta=2.4 * local_sobrepopular,
            reason="El sesgo local eleva la varianza real del partido.",
        )
        _apply(
            score_map,
            adjustments,
            source="local_sobrepopular",
            target="estabilidad",
            delta=-4.3 * local_sobrepopular,
            reason="La sobrepopularidad erosiona estabilidad real.",
        )

    visita_viva = _pattern_strength(detection, "visita_viva")
    if visita_viva > 0:
        visit_gain = cfg.peso_visita_viva * visita_viva
        if detection.top_key == "1" and detection.side_gap <= cfg.visita_viva_local_no_domina_max:
            visit_gain *= 1.2
        _apply(
            score_map,
            adjustments,
            source="visita_viva",
            target="2",
            delta=visit_gain,
            reason="La visita conserva opciones reales de competir.",
        )
        if match.pct_local > match.pct_visita:
            _apply(
                score_map,
                adjustments,
                source="visita_viva",
                target="1",
                delta=-(cfg.peso_visita_viva * 0.45 * visita_viva),
                reason="Se evita sobreproteger al local cuando visita esta viva.",
            )
        _apply(
            score_map,
            adjustments,
            source="visita_viva",
            target="X",
            delta=(cfg.peso_visita_viva * 0.18 * visita_viva),
            reason="Visita viva suele convivir con empate util.",
        )
        _apply(
            score_map,
            adjustments,
            source="visita_viva",
            target="riesgo",
            delta=1.2 * visita_viva,
            reason="Escenario competitivo de visita incrementa incertidumbre.",
        )

    partido_abierto = _pattern_strength(detection, "partido_abierto_con_empate_castigado")
    if partido_abierto > 0:
        _apply(
            score_map,
            adjustments,
            source="partido_abierto_con_empate_castigado",
            target="1",
            delta=(cfg.peso_partido_abierto_extremos * 0.75 * partido_abierto),
            reason="Partido abierto empuja extremos.",
        )
        _apply(
            score_map,
            adjustments,
            source="partido_abierto_con_empate_castigado",
            target="2",
            delta=(cfg.peso_partido_abierto_extremos * 0.75 * partido_abierto),
            reason="Partido abierto empuja extremos.",
        )
        _apply(
            score_map,
            adjustments,
            source="partido_abierto_con_empate_castigado",
            target="X",
            delta=-(cfg.peso_partido_abierto_extremos * 1.0 * partido_abierto),
            reason="Empate realmente castigado por estructura, no artificial.",
        )
        _apply(
            score_map,
            adjustments,
            source="partido_abierto_con_empate_castigado",
            target="riesgo",
            delta=3.4 * partido_abierto,
            reason="Cobertura extrema con volatilidad de marcador.",
        )

    favorito_estable = _pattern_strength(detection, "favorito_estable")
    if favorito_estable > 0:
        stable_gain = cfg.peso_favorito_estable * favorito_estable
        stable_risk_relief = 3.4 * favorito_estable
        if detection.is_active("favorito_sobrejugado"):
            stable_gain *= 0.55
            stable_risk_relief *= 0.45
        elif detection.gap_top2 < cfg.diff_favorito_claro_max:
            stable_risk_relief *= 0.75
        _apply(
            score_map,
            adjustments,
            source="favorito_estable",
            target=detection.top_key,
            delta=stable_gain,
            reason="Favorito con base estable y no solo de masa.",
        )
        _apply(
            score_map,
            adjustments,
            source="favorito_estable",
            target="riesgo",
            delta=-stable_risk_relief,
            reason="El contexto estable reduce riesgo.",
        )
        _apply(
            score_map,
            adjustments,
            source="favorito_estable",
            target="estabilidad",
            delta=(stable_gain * 0.9),
            reason="Mayor consistencia para pick directo.",
        )

    partido_caotico = _pattern_strength(detection, "partido_caotico")
    if partido_caotico > 0:
        _apply(
            score_map,
            adjustments,
            source="partido_caotico",
            target="riesgo",
            delta=(cfg.peso_partido_caotico * 0.75 * partido_caotico),
            reason="Sin liderazgo claro, aumenta la volatilidad.",
        )
        _apply(
            score_map,
            adjustments,
            source="partido_caotico",
            target="estabilidad",
            delta=-(cfg.peso_partido_caotico * 0.70 * partido_caotico),
            reason="Distribucion repartida reduce estabilidad.",
        )
        _apply(
            score_map,
            adjustments,
            source="partido_caotico",
            target="X",
            delta=(cfg.peso_partido_caotico * 0.2 * partido_caotico),
            reason="Empate gana relevancia en entornos abiertos.",
        )
        _apply(
            score_map,
            adjustments,
            source="partido_caotico",
            target="1",
            delta=(cfg.peso_partido_caotico * 0.1 * partido_caotico),
            reason="Caos reparte probabilidad tambien hacia extremos.",
        )
        _apply(
            score_map,
            adjustments,
            source="partido_caotico",
            target="2",
            delta=(cfg.peso_partido_caotico * 0.1 * partido_caotico),
            reason="Caos reparte probabilidad tambien hacia extremos.",
        )

    favorito_sobrejugado = _pattern_strength(detection, "favorito_sobrejugado")
    if favorito_sobrejugado > 0:
        overplayed_penalty = cfg.peso_favorito_sobrejugado * favorito_sobrejugado
        _apply(
            score_map,
            adjustments,
            source="favorito_sobrejugado",
            target=detection.top_key,
            delta=-overplayed_penalty,
            reason="Se penaliza confianza artificial de masa.",
        )
        if detection.top_key == "1":
            redistribution = {"2": 0.30, "X": 0.18}
        elif detection.top_key == "2":
            redistribution = {"1": 0.26, "X": 0.24}
        else:
            redistribution = {"1": 0.22, "2": 0.22}
        for key, factor in redistribution.items():
            _apply(
                score_map,
                adjustments,
                source="favorito_sobrejugado",
                target=key,
                delta=(cfg.peso_favorito_sobrejugado * factor * favorito_sobrejugado),
                reason="Alternativas ganan valor por correccion de sesgo.",
            )
        _apply(
            score_map,
            adjustments,
            source="favorito_sobrejugado",
            target="contrarian",
            delta=(4.8 + max(0.0, cfg.contrarian_alto_min - 50.0) * 0.03) * favorito_sobrejugado,
            reason="Escenario propicio para lectura contrarian.",
        )
        _apply(
            score_map,
            adjustments,
            source="favorito_sobrejugado",
            target="riesgo",
            delta=5.4 * favorito_sobrejugado,
            reason="La sobrepopularidad eleva riesgo de trampa.",
        )
        _apply(
            score_map,
            adjustments,
            source="favorito_sobrejugado",
            target="estabilidad",
            delta=-4.9 * favorito_sobrejugado,
            reason="Menos estabilidad por dependencia de masa.",
        )

    # Control anti-sesgo al empate: evita inflarlo cuando no hay estructura real.
    has_draw_support = detection.is_active("empate_vivo") or detection.is_active("empate_ignorado_por_masa")
    if match.pct_empate <= cfg.empate_bajo_max:
        penalty_scale = 0.0
        if not has_draw_support:
            penalty_scale = 1.0
        elif match.pct_empate < cfg.empate_bajo_min:
            penalty_scale = 0.55

        if penalty_scale > 0:
            raw_penalty = cfg.peso_empate_sin_estructura * (
                (cfg.empate_bajo_max - match.pct_empate)
                / max(1.0, cfg.empate_bajo_max - (cfg.empate_bajo_min - 4.0))
            )
            _apply(
                score_map,
                adjustments,
                source="control_empate_sin_estructura",
                target="X",
                delta=-(raw_penalty * penalty_scale),
                reason="Empate bajo sin soporte estructural suficiente.",
            )

    if detection.is_active("partido_abierto_con_empate_castigado"):
        open_game_penalty = cfg.peso_empate_sin_estructura * (
            0.7 + 0.6 * _clamp((cfg.empate_bajo_min - match.pct_empate) / max(1.0, cfg.empate_bajo_min))
        )
        _apply(
            score_map,
            adjustments,
            source="partido_abierto_control_empate",
            target="X",
            delta=-open_game_penalty,
            reason="Partido abierto: se evita sobre-forzar empate en lectura final.",
        )

    visita_competitiva = (
        detection.top_key == "1"
        and match.pct_visita >= cfg.visita_viva_min
        and (match.pct_local - match.pct_visita) <= cfg.visita_viva_local_no_domina_max
    )
    if visita_competitiva:
        rebalance = _clamp(
            (cfg.visita_viva_local_no_domina_max - (match.pct_local - match.pct_visita))
            / max(1.0, cfg.visita_viva_local_no_domina_max)
        )
        local_shift = cfg.peso_visita_viva * 0.35 * rebalance
        _apply(
            score_map,
            adjustments,
            source="guard_visita_competitiva",
            target="1",
            delta=-local_shift,
            reason="Visita competitiva: se corrige sobreproteccion local.",
        )
        _apply(
            score_map,
            adjustments,
            source="guard_visita_competitiva",
            target="2",
            delta=local_shift,
            reason="Visita competitiva: se reconoce opcion de sorpresa util.",
        )

    final_score_map = {
        "1": round(_clamp(score_map["1"]), 2),
        "X": round(_clamp(score_map["X"]), 2),
        "2": round(_clamp(score_map["2"]), 2),
        "riesgo": round(_clamp(score_map["riesgo"]), 2),
        "contrarian": round(_clamp(score_map["contrarian"]), 2),
        "estabilidad": round(_clamp(score_map["estabilidad"]), 2),
    }

    score_breakdown = ScoreBreakdown(
        score_local=final_score_map["1"],
        score_empate=final_score_map["X"],
        score_visita=final_score_map["2"],
        score_riesgo=final_score_map["riesgo"],
        score_contrarian=final_score_map["contrarian"],
        score_estabilidad=final_score_map["estabilidad"],
    )

    return ScoringEngineResult(
        scores=score_breakdown,
        base_scores={
            "1": round(base["1"], 2),
            "X": round(base["X"], 2),
            "2": round(base["2"], 2),
            "riesgo": round(base_risk, 2),
            "contrarian": round(base_contrarian, 2),
            "estabilidad": round(base_stability, 2),
        },
        final_scores=final_score_map,
        adjustments=adjustments,
    )
