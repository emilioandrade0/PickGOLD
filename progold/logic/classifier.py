from __future__ import annotations

from typing import List, Tuple

from config import HeuristicConfig
from models import MatchInput, ScoreBreakdown


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def classify_match(
    match: MatchInput,
    scores: ScoreBreakdown,
    cfg: HeuristicConfig,
) -> Tuple[str, List[str], List[str]]:
    percentages = {
        "1": match.pct_local,
        "X": match.pct_empate,
        "2": match.pct_visita,
    }
    ordered = sorted(percentages.items(), key=lambda item: item[1], reverse=True)

    top_key, top_pct = ordered[0]
    second_pct = ordered[1][1]
    gap_top2 = top_pct - second_pct
    side_gap = abs(match.pct_local - match.pct_visita)

    sesgos: List[str] = []
    alertas: List[str] = []

    if top_pct >= cfg.hiperpopular_min:
        tipo = "Favorito extremo"
        sesgos.append("Hiperpopular")
        alertas.append("Concentracion de masa muy alta")
    elif top_pct >= cfg.favorito_extremo_min:
        tipo = "Favorito extremo"
        sesgos.append("Favorito extremo")
    elif gap_top2 <= cfg.diff_parejo_max:
        tipo = "Partido parejo"
    elif gap_top2 <= cfg.diff_ligera_max:
        tipo = "Partido de cuidado"
    elif gap_top2 <= cfg.diff_favorito_claro_max:
        tipo = "Favorito claro"
    else:
        tipo = "Favorito extremo"

    if side_gap <= cfg.draw_gap_alert_max and match.pct_empate <= cfg.empate_bajo_max:
        sesgos.append("Empate infravalorado")
        alertas.append("Empate ignorado por masa")
        if tipo in {"Partido parejo", "Partido de cuidado"}:
            tipo = "Empate infravalorado"

    if top_key == "2" and match.pct_visita >= cfg.visitante_sobrecomprado_min:
        sesgos.append("Posible visitante sobrecomprado")
        alertas.append("Visita con posible sesgo de publico")

    if top_pct >= 55.0 and gap_top2 >= cfg.sobrepopularidad_gap_min:
        sesgos.append("Posible sesgo por nombre")
        alertas.append("Posible sobrepopularidad")

    if scores.score_contrarian >= cfg.contrarian_alto_min:
        sesgos.append("Posible contrarian")

    if scores.score_riesgo >= cfg.confianza_baja_riesgo_min:
        alertas.append("Partido de cuidado por riesgo alto")

    if not sesgos:
        sesgos.append("Lectura estable")

    return tipo, _dedupe(sesgos), _dedupe(alertas)
