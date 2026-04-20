from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


DEFAULT_RULES_PATH = Path(__file__).with_name("default_rules.json")


@dataclass
class HeuristicConfig:
    sum_tolerance: float = 2.0
    empate_bajo_min: float = 18.0
    empate_bajo_max: float = 24.0
    diff_parejo_max: float = 7.0
    diff_ligera_max: float = 15.0
    diff_favorito_claro_max: float = 25.0
    favorito_extremo_min: float = 60.0
    hiperpopular_min: float = 70.0
    visitante_sobrecomprado_min: float = 50.0
    draw_gap_alert_max: float = 10.0
    sobrepopularidad_gap_min: float = 26.0
    contrarian_alto_min: float = 62.0
    confianza_alta_riesgo_max: float = 42.0
    confianza_baja_riesgo_min: float = 68.0

    # Umbrales estructurales de patrones.
    umbral_lados_cercanos: float = 8.5
    umbral_empate_cercano: float = 6.5
    umbral_diferencia_moderada: float = 12.0

    empate_vivo_min: float = 20.0
    empate_vivo_max: float = 28.0
    empate_ignorado_max: float = 22.0

    visita_sobrecomprada_moderada_min: float = 44.0
    visita_sobrecomprada_moderada_max: float = 54.0
    visita_sobrecomprada_local_min: float = 28.0
    visita_sobrecomprada_empate_min: float = 20.0

    local_sobrepopular_min: float = 45.0
    local_sobrepopular_visita_min: float = 30.0
    local_sobrepopular_empate_min: float = 22.0

    favorito_estable_min_pct: float = 48.0
    favorito_estable_gap_min: float = 12.0
    favorito_estable_empate_max: float = 22.0
    favorito_estable_opuesto_max: float = 34.0

    partido_caotico_gap_max: float = 6.0
    partido_caotico_spread_max: float = 16.0
    partido_caotico_empate_min: float = 22.0

    visita_viva_min: float = 30.0
    visita_viva_max: float = 46.0
    visita_viva_empate_min: float = 20.0
    visita_viva_local_no_domina_max: float = 12.0

    partido_abierto_empate_castigado_max: float = 19.0
    partido_abierto_extremos_min: float = 35.0

    # Jerarquia de decision.
    decision_gap_directo_min: float = 8.0
    decision_gap_doble_max: float = 4.5
    decision_gap_empate_directo_min: float = 2.0
    x_compite_margin: float = 1.7

    riesgo_alto_min: float = 60.0
    estabilidad_alta_min: float = 68.0
    estabilidad_baja_max: float = 44.0

    # Control de doble oportunidad inteligente.
    double_competitive_gap: float = 6.5
    double_side_guard: float = 2.5
    double_evitar_gap_dominante: float = 10.0

    # Pesos de patrones (calibrables por historico).
    peso_empate_vivo: float = 8.0
    peso_empate_ignorado: float = 10.0
    peso_visita_sobrecomprada: float = 9.0
    peso_local_sobrepopular: float = 9.0
    peso_favorito_estable: float = 7.0
    peso_partido_caotico: float = 8.0
    peso_visita_viva: float = 7.0
    peso_partido_abierto_extremos: float = 8.0
    peso_favorito_sobrejugado: float = 9.0
    peso_empate_sin_estructura: float = 4.5

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "HeuristicConfig":
        current = cls().to_dict()
        current.update({k: v for k, v in payload.items() if k in current})
        return cls(**current)


def load_rules(path: Path | None = None) -> HeuristicConfig:
    source = path or DEFAULT_RULES_PATH
    if not source.exists():
        return HeuristicConfig()

    with source.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return HeuristicConfig.from_dict(payload)


def save_rules(config: HeuristicConfig, path: Path | None = None) -> None:
    target = path or DEFAULT_RULES_PATH
    with target.open("w", encoding="utf-8") as fp:
        json.dump(config.to_dict(), fp, indent=2)
