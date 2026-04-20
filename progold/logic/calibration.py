from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from config import HeuristicConfig, load_rules
from models import MatchInput

from .analyzer import analyze_match

_DIRECT = {"1", "X", "2"}
_DOUBLE = {"1X", "X2", "12"}


@dataclass
class HistoricalCase:
    pct_local: float
    pct_empate: float
    pct_visita: float
    resultado_real: str
    local: str = "LOCAL"
    visitante: str = "VISITA"


def _combo_hits_result(combo: str, resultado_real: str) -> bool:
    if combo in _DOUBLE:
        return resultado_real in combo
    if combo in _DIRECT:
        return combo == resultado_real
    return False


def _pattern_combo_key(patterns: Sequence[str]) -> str:
    if not patterns:
        return "sin_patrones"
    return " + ".join(sorted(patterns))


def evaluate_historical_cases(
    cases: Iterable[HistoricalCase | Dict[str, Any]],
    cfg: HeuristicConfig | None = None,
) -> Dict[str, Any]:
    config = cfg or load_rules()

    total = 0
    direct_total = 0
    direct_hits = 0
    double_total = 0
    double_hits = 0

    pattern_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "activations": 0,
        "direct_hits": 0,
        "double_hits": 0,
    })

    combo_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "activations": 0,
        "direct_hits": 0,
        "double_hits": 0,
    })

    pattern_matrix_counter: Counter[Tuple[str, str]] = Counter()
    recommendation_mode_counter: Counter[str] = Counter()
    confidence_counter: Counter[str] = Counter()
    tipo_counter: Counter[str] = Counter()
    rescued_by_double = 0

    for raw_case in cases:
        case = HistoricalCase(**raw_case) if isinstance(raw_case, dict) else raw_case
        resultado_real = str(case.resultado_real).upper().strip()
        if resultado_real not in _DIRECT:
            continue

        total += 1

        analysis = analyze_match(
            MatchInput(
                local=case.local,
                visitante=case.visitante,
                pct_local=float(case.pct_local),
                pct_empate=float(case.pct_empate),
                pct_visita=float(case.pct_visita),
            ),
            config,
            debug_mode=True,
        )

        pick = str(analysis.recomendacion_principal).upper().strip()
        doble = str(analysis.doble_oportunidad).upper().strip()

        if pick in _DIRECT:
            recommendation_mode_counter["principal_directo"] += 1
        elif pick in _DOUBLE:
            recommendation_mode_counter["principal_doble"] += 1
        else:
            recommendation_mode_counter["principal_otro"] += 1

        if doble in _DOUBLE:
            recommendation_mode_counter["doble_sugerido"] += 1

        confidence_counter[str(analysis.confianza or "-").lower()] += 1
        tipo_counter[str(analysis.tipo_partido or "-")] += 1

        if pick in _DIRECT:
            direct_total += 1
            if pick == resultado_real:
                direct_hits += 1

        if doble in _DOUBLE:
            double_total += 1
            if _combo_hits_result(doble, resultado_real):
                double_hits += 1
                if pick in _DIRECT and pick != resultado_real:
                    rescued_by_double += 1

        active_patterns = list(getattr(analysis, "patrones_activados", []) or [])
        combo_key = _pattern_combo_key(active_patterns)

        for pattern in active_patterns:
            pattern_stats[pattern]["activations"] += 1
            if pick in _DIRECT and pick == resultado_real:
                pattern_stats[pattern]["direct_hits"] += 1
            if doble in _DOUBLE and _combo_hits_result(doble, resultado_real):
                pattern_stats[pattern]["double_hits"] += 1

            pattern_matrix_counter[(pattern, resultado_real)] += 1

        combo_stats[combo_key]["activations"] += 1
        if pick in _DIRECT and pick == resultado_real:
            combo_stats[combo_key]["direct_hits"] += 1
        if doble in _DOUBLE and _combo_hits_result(doble, resultado_real):
            combo_stats[combo_key]["double_hits"] += 1

    accuracy_pick_directo = (direct_hits / direct_total) if direct_total else 0.0
    accuracy_doble_oportunidad = (double_hits / double_total) if double_total else 0.0

    pattern_matrix: Dict[str, Dict[str, int]] = {}
    for (pattern_name, real_result), count in pattern_matrix_counter.items():
        pattern_matrix.setdefault(pattern_name, {"1": 0, "X": 0, "2": 0})
        pattern_matrix[pattern_name][real_result] = count

    pattern_hit_rate: Dict[str, Dict[str, float]] = {}
    for pattern_name, stats in pattern_stats.items():
        activations = stats["activations"]
        pattern_hit_rate[pattern_name] = {
            "activations": float(activations),
            "direct_hit_rate": (stats["direct_hits"] / activations) if activations else 0.0,
            "double_hit_rate": (stats["double_hits"] / activations) if activations else 0.0,
        }

    combo_hit_rate: Dict[str, Dict[str, float]] = {}
    for combo_name, stats in combo_stats.items():
        activations = stats["activations"]
        combo_hit_rate[combo_name] = {
            "activations": float(activations),
            "direct_hit_rate": (stats["direct_hits"] / activations) if activations else 0.0,
            "double_hit_rate": (stats["double_hits"] / activations) if activations else 0.0,
        }

    return {
        "total_cases": total,
        "accuracy_pick_directo": round(accuracy_pick_directo, 4),
        "accuracy_doble_oportunidad": round(accuracy_doble_oportunidad, 4),
        "direct_evaluated": direct_total,
        "double_evaluated": double_total,
        "double_rescued_cases": rescued_by_double,
        "recommendation_mode_distribution": dict(recommendation_mode_counter),
        "confidence_distribution": dict(confidence_counter),
        "tipo_partido_distribution": dict(tipo_counter),
        "pattern_activation_matrix": pattern_matrix,
        "pattern_hit_rate": pattern_hit_rate,
        "pattern_combo_hit_rate": combo_hit_rate,
    }
