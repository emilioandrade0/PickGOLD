from __future__ import annotations

import math
from typing import List, Tuple

from models import MatchInput


def total_percentage(match: MatchInput) -> float:
    return match.pct_local + match.pct_empate + match.pct_visita


def validate_match_input(match: MatchInput, tolerance: float) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    if not match.local.strip():
        errors.append("El nombre del equipo local es obligatorio.")
    if not match.visitante.strip():
        errors.append("El nombre del equipo visitante es obligatorio.")

    pct_values = {
        "pct_local": match.pct_local,
        "pct_empate": match.pct_empate,
        "pct_visita": match.pct_visita,
    }

    for key, value in pct_values.items():
        if not isinstance(value, (int, float)):
            errors.append(f"{key} debe ser numerico.")
            continue
        if not math.isfinite(float(value)):
            errors.append(f"{key} no es un numero valido.")
            continue
        if float(value) < 0 or float(value) > 100:
            errors.append(f"{key} debe estar entre 0 y 100.")

    total = total_percentage(match)
    if abs(total - 100.0) > tolerance:
        warnings.append(
            "La suma de porcentajes no esta cerca de 100. "
            f"Total actual: {total:.2f}."
        )

    if abs(match.pct_local - match.pct_visita) <= 0.01 and match.pct_empate < 10:
        warnings.append(
            "Local y visita estan casi iguales, pero el empate es muy bajo. "
            "Revisar captura por consistencia."
        )

    return errors, warnings
