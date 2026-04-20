from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class MatchInput:
    local: str
    visitante: str
    pct_local: float
    pct_empate: float
    pct_visita: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScoreBreakdown:
    score_local: float
    score_empate: float
    score_visita: float
    score_riesgo: float
    score_contrarian: float
    score_estabilidad: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MatchAnalysis:
    local: str
    visitante: str
    pct_local: float
    pct_empate: float
    pct_visita: float
    porcentaje_mayor: float
    segundo_porcentaje_mayor: float
    diferencia_top2: float
    tipo_partido: str
    sesgos: List[str]
    recomendacion_principal: str
    doble_oportunidad: str
    confianza: str
    explicacion: str
    posible_ganador_masa: str
    banderas_alerta: List[str] = field(default_factory=list)
    patrones_activados: List[str] = field(default_factory=list)
    score_local: float = 0.0
    score_empate: float = 0.0
    score_visita: float = 0.0
    score_riesgo: float = 0.0
    score_contrarian: float = 0.0
    score_estabilidad: float = 0.0
    apto_pick_directo: bool = False
    apto_doble_oportunidad: bool = False
    sugerir_sorpresa: bool = False
    debug_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
