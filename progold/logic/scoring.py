from __future__ import annotations

from typing import Dict, List, Tuple

from config import HeuristicConfig
from models import MatchInput, ScoreBreakdown


def _clamp(value: float, min_value: float = 0.0, max_value: float = 100.0) -> float:
    return max(min_value, min(max_value, value))


def _apply_penalty(
    scores: Dict[str, float],
    option_key: str,
    penalty: float,
) -> None:
    scores[option_key] = scores[option_key] - penalty


def compute_scores(match: MatchInput, cfg: HeuristicConfig) -> Tuple[ScoreBreakdown, List[str]]:
    notes: List[str] = []

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

    raw_scores = {
        "1": 35.0 + (match.pct_local * 0.70),
        "X": 34.0 + (match.pct_empate * 0.95),
        "2": 35.0 + (match.pct_visita * 0.70),
    }

    risk_score = 28.0
    contrarian_score = 22.0

    if gap_top2 <= cfg.diff_parejo_max:
        risk_score += 9.0
        contrarian_score += 4.0
        raw_scores["X"] += 5.0
        notes.append(
            "La diferencia entre primera y segunda opcion es baja, se sube el riesgo y el peso del empate."
        )
    elif gap_top2 <= cfg.diff_ligera_max:
        risk_score += 6.0
        contrarian_score += 2.0
        notes.append("La inclinacion es ligera, por eso no conviene una lectura totalmente directa.")
    elif gap_top2 <= cfg.diff_favorito_claro_max:
        risk_score += 4.0
        notes.append("Existe favorito claro, pero todavia hay espacio para cobertura.")
    else:
        risk_score += 7.0
        contrarian_score += 4.0
        notes.append("La concentracion de masa es alta y puede esconder sobrepopularidad.")

    if top_pct >= cfg.favorito_extremo_min:
        extreme_penalty = ((top_pct - cfg.favorito_extremo_min) * 0.55) + 6.0
        _apply_penalty(raw_scores, top_key, extreme_penalty)
        risk_score += 10.0
        contrarian_score += 8.0
        notes.append("Se penaliza al favorito extremo para evitar sesgo de masa.")

    if top_pct >= cfg.hiperpopular_min:
        hyper_penalty = ((top_pct - cfg.hiperpopular_min) * 0.85) + 5.0
        _apply_penalty(raw_scores, top_key, hyper_penalty)
        risk_score += 8.0
        contrarian_score += 10.0
        notes.append("Se detecta hiperpopularidad y se ajusta con enfoque contrarian.")

    draw_undervalued_window = cfg.empate_bajo_min <= match.pct_empate <= cfg.empate_bajo_max
    if side_gap <= cfg.draw_gap_alert_max and match.pct_empate <= cfg.empate_bajo_max:
        raw_scores["X"] += 10.0 if draw_undervalued_window else 6.0
        risk_score += 7.0
        contrarian_score += 6.0
        notes.append(
            "Local y visita estan cerca y el empate no domina; se eleva score del empate por posible infravaloracion."
        )

    if top_key == "2" and match.pct_visita >= cfg.visitante_sobrecomprado_min:
        raw_scores["2"] -= 8.0
        raw_scores["X"] += 4.0
        raw_scores["1"] += 3.0
        risk_score += 7.0
        contrarian_score += 9.0
        notes.append("La visita lidera con carga alta; se activa alerta de visitante sobrecomprado.")

    if top_pct >= 55.0 and gap_top2 >= cfg.sobrepopularidad_gap_min:
        risk_score += 6.0
        contrarian_score += 7.0
        notes.append("La brecha luce desproporcionada, posible sobrepopularidad por percepcion publica.")

    if match.pct_local >= 34.0 and match.pct_visita >= 34.0 and match.pct_empate < cfg.empate_bajo_min:
        raw_scores["1"] += 2.0
        raw_scores["2"] += 2.0
        raw_scores["X"] -= 4.0
        contrarian_score += 5.0
        notes.append("Con empate castigado y ambos lados altos, gana fuerza la doble oportunidad 12.")

    score_breakdown = ScoreBreakdown(
        score_local=round(_clamp(raw_scores["1"]), 2),
        score_empate=round(_clamp(raw_scores["X"]), 2),
        score_visita=round(_clamp(raw_scores["2"]), 2),
        score_riesgo=round(_clamp(risk_score), 2),
        score_contrarian=round(_clamp(contrarian_score), 2),
    )

    return score_breakdown, notes
