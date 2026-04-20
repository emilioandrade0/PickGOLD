from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from config import HeuristicConfig
from models import MatchInput


@dataclass
class PatternActivation:
    name: str
    active: bool
    strength: float
    reason: str


@dataclass
class PatternDetectionResult:
    patterns: Dict[str, PatternActivation]
    alerts: List[str]
    notes: List[str]
    top_key: str
    second_key: str
    top_pct: float
    second_pct: float
    gap_top2: float
    side_gap: float

    def active_pattern_names(self) -> List[str]:
        return [name for name, activation in self.patterns.items() if activation.active]

    def is_active(self, pattern_name: str) -> bool:
        activation = self.patterns.get(pattern_name)
        return bool(activation and activation.active)


def _clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def _proximity_strength(diff: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    return _clamp(1.0 - (abs(diff) / threshold))


def _range_center_strength(value: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 0.0
    if value < min_value or value > max_value:
        return 0.0

    center = (min_value + max_value) / 2.0
    half_span = (max_value - min_value) / 2.0
    if half_span <= 0:
        return 1.0
    return _clamp(1.0 - abs(value - center) / half_span)


def _activation(name: str, active: bool, strength: float, reason: str) -> PatternActivation:
    return PatternActivation(
        name=name,
        active=active,
        strength=_clamp(strength),
        reason=reason,
    )


def detect_patterns(match: MatchInput, cfg: HeuristicConfig) -> PatternDetectionResult:
    percentages = {
        "1": float(match.pct_local),
        "X": float(match.pct_empate),
        "2": float(match.pct_visita),
    }
    ordered = sorted(percentages.items(), key=lambda item: item[1], reverse=True)

    top_key, top_pct = ordered[0]
    second_key, second_pct = ordered[1]
    gap_top2 = top_pct - second_pct
    side_gap = abs(match.pct_local - match.pct_visita)

    local_draw_gap = abs(match.pct_local - match.pct_empate)
    visit_draw_gap = abs(match.pct_visita - match.pct_empate)
    min_draw_side_gap = min(local_draw_gap, visit_draw_gap)

    gap_is_parejo = gap_top2 <= cfg.diff_parejo_max
    gap_is_ligero = gap_top2 <= cfg.diff_ligera_max
    draw_near_any_side = min_draw_side_gap <= cfg.draw_gap_alert_max
    open_12_structure = (
        match.pct_empate <= cfg.partido_abierto_empate_castigado_max
        and match.pct_local >= cfg.partido_abierto_extremos_min
        and match.pct_visita >= cfg.partido_abierto_extremos_min
    )

    close_sides = side_gap <= cfg.umbral_lados_cercanos
    close_local_draw = local_draw_gap <= cfg.umbral_empate_cercano
    close_visit_draw = visit_draw_gap <= cfg.umbral_empate_cercano

    empate_window = cfg.empate_vivo_min <= match.pct_empate <= cfg.empate_vivo_max
    empate_real_structure = (close_sides and draw_near_any_side) or close_local_draw or close_visit_draw

    empate_vivo_strength = (
        (0.4 * _proximity_strength(side_gap, cfg.umbral_lados_cercanos))
        + (0.3 * _proximity_strength(local_draw_gap, cfg.umbral_empate_cercano))
        + (0.3 * _proximity_strength(visit_draw_gap, cfg.umbral_empate_cercano))
    )
    if not draw_near_any_side:
        empate_vivo_strength *= 0.5
    if not empate_window:
        empate_vivo_strength *= 0.4

    empate_vivo = _activation(
        name="empate_vivo",
        active=bool(empate_window and empate_real_structure),
        strength=empate_vivo_strength,
        reason=(
            "Empate en rango competitivo y estructura cercana entre lados/empate."
            if empate_window and empate_real_structure
            else "No hay estructura suficiente de empate real."
        ),
    )

    empate_ignorado_floor = cfg.empate_bajo_min - 2.0
    empate_ignorado_active = (
        (gap_is_parejo or gap_is_ligero)
        and side_gap <= cfg.umbral_diferencia_moderada
        and match.pct_empate <= cfg.empate_ignorado_max
        and match.pct_empate >= empate_ignorado_floor
        and draw_near_any_side
        and (top_key in {"1", "2"})
        and not open_12_structure
    )
    ignored_strength = (
        0.45 * _proximity_strength(side_gap, cfg.umbral_diferencia_moderada)
        + 0.3 * _proximity_strength(min_draw_side_gap, cfg.draw_gap_alert_max)
        + 0.25
        * _clamp(
            (match.pct_empate - empate_ignorado_floor)
            / max(1.0, cfg.empate_ignorado_max - empate_ignorado_floor)
        )
    )
    empate_ignorado = _activation(
        name="empate_ignorado_por_masa",
        active=empate_ignorado_active,
        strength=ignored_strength,
        reason=(
            "Juego parejo o moderado con empate castigado por la masa."
            if empate_ignorado_active
            else "No se observa castigo claro del empate en contexto parejo."
        ),
    )

    visita_sobrecomprada_alta = (
        match.pct_visita >= cfg.visitante_sobrecomprado_min
        and match.pct_visita > match.pct_local
        and match.pct_local >= (cfg.visita_sobrecomprada_local_min - 2.0)
        and match.pct_empate >= cfg.empate_bajo_min
    )
    visita_sobrecomprada_active = (
        (
            cfg.visita_sobrecomprada_moderada_min
            <= match.pct_visita
            <= cfg.visita_sobrecomprada_moderada_max
            and match.pct_visita > match.pct_local
            and match.pct_local >= cfg.visita_sobrecomprada_local_min
            and match.pct_empate >= cfg.visita_sobrecomprada_empate_min
        )
        or visita_sobrecomprada_alta
    )
    visita_sobrecomprada_strength = (
        0.5 * _range_center_strength(
            match.pct_visita,
            cfg.visita_sobrecomprada_moderada_min,
            cfg.visita_sobrecomprada_moderada_max,
        )
        + 0.2 * _clamp((match.pct_local - cfg.visita_sobrecomprada_local_min) / 16.0)
        + 0.2 * _clamp((match.pct_empate - cfg.visita_sobrecomprada_empate_min) / 10.0)
        + 0.1 * _clamp((match.pct_visita - cfg.visitante_sobrecomprado_min) / 12.0)
    )
    visita_sobrecomprada = _activation(
        name="visita_sobrecomprada_moderada",
        active=visita_sobrecomprada_active,
        strength=visita_sobrecomprada_strength,
        reason=(
            "La visita lidera sin dominar y local/empate siguen vivos."
            if visita_sobrecomprada_active
            else "No hay senales de visita sobrecomprada moderada."
        ),
    )

    local_sobrepopular_active = (
        top_key == "1"
        and match.pct_local >= cfg.local_sobrepopular_min
        and match.pct_visita >= cfg.local_sobrepopular_visita_min
        and match.pct_empate >= cfg.local_sobrepopular_empate_min
        and (gap_top2 <= cfg.diff_favorito_claro_max)
    )
    local_sobrepopular_strength = (
        0.6 * _clamp((match.pct_local - cfg.local_sobrepopular_min) / 20.0)
        + 0.2 * _clamp((match.pct_visita - cfg.local_sobrepopular_visita_min) / 14.0)
        + 0.2 * _clamp((match.pct_empate - cfg.local_sobrepopular_empate_min) / 8.0)
    )
    local_sobrepopular = _activation(
        name="local_sobrepopular",
        active=local_sobrepopular_active,
        strength=local_sobrepopular_strength,
        reason=(
            "Local muy cargado con visita y empate todavia competitivos."
            if local_sobrepopular_active
            else "No hay sobrepopularidad local estructural."
        ),
    )

    opposing_pct = match.pct_visita if top_key == "1" else match.pct_local if top_key == "2" else max(match.pct_local, match.pct_visita)
    stable_gap_min = max(cfg.favorito_estable_gap_min, cfg.diff_ligera_max)
    favorito_estable_active = (
        top_key in {"1", "2"}
        and gap_top2 >= stable_gap_min
        and match.pct_empate <= cfg.favorito_estable_empate_max
        and opposing_pct <= cfg.favorito_estable_opuesto_max
        and top_pct >= cfg.favorito_estable_min_pct
        and not local_sobrepopular_active
        and not visita_sobrecomprada_active
        and top_pct < cfg.hiperpopular_min
    )
    favorito_estable_strength = (
        0.45 * _clamp((gap_top2 - stable_gap_min) / 16.0)
        + 0.35 * _clamp((top_pct - cfg.favorito_estable_min_pct) / 20.0)
        + 0.2 * _proximity_strength(match.pct_empate, cfg.favorito_estable_empate_max)
    )
    favorito_estable = _activation(
        name="favorito_estable",
        active=favorito_estable_active,
        strength=favorito_estable_strength,
        reason=(
            "Lider claro, empate debil y lado opuesto sin soporte suficiente."
            if favorito_estable_active
            else "No cumple estructura de favorito realmente estable."
        ),
    )

    spread = max(percentages.values()) - min(percentages.values())
    partido_caotico_active = (
        gap_top2 <= min(cfg.partido_caotico_gap_max, cfg.diff_parejo_max)
        and spread <= cfg.partido_caotico_spread_max
        and match.pct_empate >= cfg.partido_caotico_empate_min
        and draw_near_any_side
        and side_gap <= cfg.umbral_diferencia_moderada
    )
    partido_caotico_strength = (
        0.45 * _proximity_strength(gap_top2, min(cfg.partido_caotico_gap_max, cfg.diff_parejo_max))
        + 0.3 * _proximity_strength(spread, cfg.partido_caotico_spread_max)
        + 0.25 * _proximity_strength(min_draw_side_gap, cfg.draw_gap_alert_max)
    )
    partido_caotico = _activation(
        name="partido_caotico",
        active=partido_caotico_active,
        strength=partido_caotico_strength,
        reason=(
            "Porcentajes repartidos, sin liderazgo claro y empate presente."
            if partido_caotico_active
            else "No hay caotizacion clara de porcentajes."
        ),
    )

    visita_viva_active = (
        cfg.visita_viva_min <= match.pct_visita <= cfg.visita_viva_max
        and match.pct_empate >= cfg.visita_viva_empate_min
        and (match.pct_local - match.pct_visita) <= min(cfg.visita_viva_local_no_domina_max, cfg.diff_ligera_max)
        and gap_top2 <= cfg.diff_favorito_claro_max
        and not (top_key == "2" and gap_top2 >= cfg.diff_favorito_claro_max and match.pct_empate <= cfg.empate_bajo_min)
    )
    visita_viva_strength = (
        0.5 * _range_center_strength(match.pct_visita, cfg.visita_viva_min, cfg.visita_viva_max)
        + 0.2 * _clamp((match.pct_empate - cfg.visita_viva_empate_min) / 10.0)
        + 0.3 * _proximity_strength(match.pct_local - match.pct_visita, cfg.visita_viva_local_no_domina_max)
    )
    visita_viva = _activation(
        name="visita_viva",
        active=visita_viva_active,
        strength=visita_viva_strength,
        reason=(
            "La visita mantiene competitividad con empate util y local no dominante."
            if visita_viva_active
            else "No hay estructura de visita viva competitiva."
        ),
    )

    partido_abierto_active = open_12_structure
    partido_abierto_strength = (
        0.4 * _clamp((cfg.partido_abierto_empate_castigado_max - match.pct_empate) / 8.0)
        + 0.3 * _clamp((match.pct_local - cfg.partido_abierto_extremos_min) / 18.0)
        + 0.3 * _clamp((match.pct_visita - cfg.partido_abierto_extremos_min) / 18.0)
    )
    partido_abierto = _activation(
        name="partido_abierto_con_empate_castigado",
        active=partido_abierto_active,
        strength=partido_abierto_strength,
        reason=(
            "Extremos fuertes con empate castigado; escenario de cobertura 12."
            if partido_abierto_active
            else "No hay partido abierto con empate castigado."
        ),
    )

    favorito_sobrejugado_active = (
        top_key in {"1", "2"}
        and (
            (top_key == "1" and local_sobrepopular_active)
            or (top_key == "2" and visita_sobrecomprada_active)
            or top_pct >= cfg.hiperpopular_min
            or (
                top_pct >= cfg.favorito_extremo_min
                and gap_top2 >= cfg.sobrepopularidad_gap_min
                and (second_pct >= cfg.empate_bajo_min or opposing_pct >= cfg.local_sobrepopular_visita_min)
            )
        )
        and not favorito_estable_active
    )
    favorito_sobrejugado_strength = max(
        local_sobrepopular_strength if top_key == "1" else 0.0,
        visita_sobrecomprada_strength if top_key == "2" else 0.0,
        _clamp((top_pct - cfg.favorito_extremo_min) / 18.0),
        _clamp((top_pct - cfg.hiperpopular_min) / 12.0),
    )
    favorito_sobrejugado = _activation(
        name="favorito_sobrejugado",
        active=favorito_sobrejugado_active,
        strength=favorito_sobrejugado_strength,
        reason=(
            "La masa infla al favorito mas alla de su estabilidad real."
            if favorito_sobrejugado_active
            else "No se observa favorito sobrejugado."
        ),
    )

    patterns = {
        empate_vivo.name: empate_vivo,
        empate_ignorado.name: empate_ignorado,
        visita_sobrecomprada.name: visita_sobrecomprada,
        local_sobrepopular.name: local_sobrepopular,
        favorito_estable.name: favorito_estable,
        partido_caotico.name: partido_caotico,
        visita_viva.name: visita_viva,
        partido_abierto.name: partido_abierto,
        favorito_sobrejugado.name: favorito_sobrejugado,
    }

    alerts: List[str] = []
    notes: List[str] = []

    for activation in patterns.values():
        if activation.active:
            notes.append(f"{activation.name}: {activation.reason}")

    if empate_ignorado.active:
        alerts.append("Empate ignorado por masa")
    if local_sobrepopular.active:
        alerts.append("Local sobrepopular")
    if visita_sobrecomprada.active:
        alerts.append("Visita sobrecomprada moderada")
    if partido_caotico.active:
        alerts.append("Partido caotico")
    if favorito_sobrejugado.active:
        alerts.append("Favorito sobrejugado")

    return PatternDetectionResult(
        patterns=patterns,
        alerts=alerts,
        notes=notes,
        top_key=top_key,
        second_key=second_key,
        top_pct=round(top_pct, 2),
        second_pct=round(second_pct, 2),
        gap_top2=round(gap_top2, 2),
        side_gap=round(side_gap, 2),
    )
