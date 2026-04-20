from __future__ import annotations

import base64
import binascii
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
PROGOLD_ROOT = REPO_ROOT / "progold"
if str(PROGOLD_ROOT) not in sys.path:
    sys.path.insert(0, str(PROGOLD_ROOT))

# PROGOLD modules use top-level imports (config/models/logic) because they were
# originally executed from the `progold/` directory.
from config import HeuristicConfig, load_rules  # type: ignore
from logic import analyze_match, validate_match_input  # type: ignore
from models import MatchInput  # type: ignore


def get_rules_dict() -> dict[str, float]:
    return load_rules().to_dict()


def get_ocr_status() -> dict[str, Any]:
    try:
        from utils import ocr_runtime_status  # type: ignore
    except Exception:
        return {
            "ok": True,
            "available": False,
            "message": "OCR no disponible en este entorno.",
        }

    available, message = ocr_runtime_status()
    return {
        "ok": True,
        "available": bool(available),
        "message": str(message or ""),
    }


def extract_rows_from_capture(
    *,
    image_base64: str,
    section: str = "progol",
    max_matches: int = 14,
) -> dict[str, Any]:
    if not str(image_base64 or "").strip():
        return {
            "ok": False,
            "available": False,
            "error": "No se recibio imagen para OCR.",
            "rows": [],
            "notes": [],
            "detected_date": None,
        }

    raw_b64 = str(image_base64 or "").strip()
    if raw_b64.startswith("data:") and "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(raw_b64, validate=False)
    except (ValueError, binascii.Error):
        return {
            "ok": False,
            "available": False,
            "error": "Imagen en formato base64 invalido.",
            "rows": [],
            "notes": [],
            "detected_date": None,
        }

    try:
        from utils import extract_matches_with_date_from_capture, ocr_runtime_status  # type: ignore
    except Exception:
        return {
            "ok": False,
            "available": False,
            "error": "OCR no disponible en este entorno.",
            "rows": [],
            "notes": [],
            "detected_date": None,
        }

    available, message = ocr_runtime_status()
    if not available:
        return {
            "ok": False,
            "available": False,
            "error": str(message or "OCR no disponible."),
            "rows": [],
            "notes": [str(message or "OCR no disponible.")],
            "detected_date": None,
        }

    safe_section = "revancha" if str(section or "").strip().lower() == "revancha" else "progol"
    safe_max = max(1, min(int(max_matches or 14), 14))

    rows, notes, detected_date = extract_matches_with_date_from_capture(
        image_bytes=image_bytes,
        section=safe_section,
        max_matches=safe_max,
    )

    return {
        "ok": True,
        "available": True,
        "rows": rows or [],
        "notes": notes or [],
        "detected_date": detected_date.isoformat() if detected_date else None,
        "count": len(rows or []),
    }


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if numeric != numeric:  # NaN
        return None
    return numeric


def _short_warning_text(warning: str) -> str:
    lower = str(warning or "").lower()
    if "suma de porcentajes" in lower:
        return "Suma fuera de tolerancia"
    return str(warning or "").strip()


def _symbol_to_pick_text(symbol: str, local: str, visitante: str) -> str:
    if symbol == "1":
        return local or "Local"
    if symbol == "2":
        return visitante or "Visitante"
    if symbol == "X":
        return "Empate"
    return "-"


def _fallback_double_from_symbol(direct_symbol: str, score_local: float, score_visita: float) -> str:
    if direct_symbol == "1":
        return "1X"
    if direct_symbol == "2":
        return "X2"
    if direct_symbol == "X":
        return "1X" if score_local >= score_visita else "X2"
    return "1X" if score_local >= score_visita else "X2"


def _normalize_double_pick(
    recomendacion: str,
    doble_oportunidad: str,
    direct_symbol: str,
    score_local: float,
    score_visita: float,
) -> str:
    valid_double = {"1X", "X2", "12"}
    if doble_oportunidad in valid_double:
        return doble_oportunidad
    if recomendacion in valid_double:
        return recomendacion
    return _fallback_double_from_symbol(direct_symbol, score_local, score_visita)


def _resolve_direct_pick_symbol(analysis: Any) -> str:
    recommendation = str(getattr(analysis, "recomendacion_principal", "") or "").upper()
    options = [token for token in recommendation if token in {"1", "X", "2"}]
    if not options:
        return "X"

    if len(options) == 1:
        return options[0]

    score_local = float(getattr(analysis, "score_local", 0.0))
    score_empate = float(getattr(analysis, "score_empate", 0.0))
    score_visita = float(getattr(analysis, "score_visita", 0.0))
    score_map = {
        "1": score_local,
        "X": score_empate,
        "2": score_visita,
    }

    best_option = options[0]
    best_score = score_map.get(best_option, float("-inf"))
    for option in options[1:]:
        option_score = score_map.get(option, float("-inf"))
        if option_score > best_score:
            best_option = option
            best_score = option_score
    return best_option


def _risk_label(score_riesgo: float) -> str:
    if score_riesgo >= 68:
        return "rojo"
    if score_riesgo >= 45:
        return "amarillo"
    return "verde"


def _empty_report(partido: int) -> dict[str, Any]:
    return {
        "partido": partido,
        "local": "",
        "visitante": "",
        "pct_local": None,
        "pct_empate": None,
        "pct_visita": None,
        "estado": "vacio",
        "pick_symbol": "-",
        "recomendacion": "-",
        "doble_oportunidad": "-",
        "confianza": "-",
        "tipo_partido": "-",
        "semaforo": "neutro",
        "alerta": "Sin captura",
        "score_riesgo": 0.0,
        "score_contrarian": 0.0,
        "apto_pick_directo": False,
        "apto_doble_oportunidad": False,
        "sugerir_sorpresa": False,
        "patrones_activados": [],
        "explicacion": "",
        "analisis": None,
    }


def analyze_ticket_rows(
    rows: list[dict[str, Any]],
    *,
    rules_override: dict[str, Any] | None = None,
    debug_mode: bool = False,
) -> dict[str, Any]:
    config = HeuristicConfig.from_dict(rules_override or {}) if isinstance(rules_override, dict) else load_rules()
    ticket_rows = (rows or [])[:14]
    reports: list[dict[str, Any]] = []
    analyses: list[dict[str, Any]] = []

    for idx in range(14):
        source = ticket_rows[idx] if idx < len(ticket_rows) else {}
        partido = int(source.get("partido") or (idx + 1))
        local = str(source.get("local") or "").strip()
        visitante = str(source.get("visitante") or "").strip()
        pct_local = _to_float(source.get("pct_local"))
        pct_empate = _to_float(source.get("pct_empate"))
        pct_visita = _to_float(source.get("pct_visita"))

        has_any_value = bool(local or visitante or pct_local is not None or pct_empate is not None or pct_visita is not None)
        report = _empty_report(partido)
        report.update({
            "local": local,
            "visitante": visitante,
            "pct_local": pct_local,
            "pct_empate": pct_empate,
            "pct_visita": pct_visita,
        })

        if not has_any_value:
            reports.append(report)
            continue

        if not local or not visitante:
            report.update({
                "estado": "incompleto",
                "semaforo": "amarillo",
                "alerta": "Captura incompleta: falta local o visitante",
                "tipo_partido": "Partido de cuidado",
            })
            reports.append(report)
            continue

        if pct_local is None or pct_empate is None or pct_visita is None:
            report.update({
                "estado": "incompleto",
                "semaforo": "amarillo",
                "alerta": "Captura incompleta: faltan porcentajes",
                "tipo_partido": "Partido de cuidado",
            })
            reports.append(report)
            continue

        match = MatchInput(
            local=local,
            visitante=visitante,
            pct_local=float(pct_local),
            pct_empate=float(pct_empate),
            pct_visita=float(pct_visita),
        )

        errors, warnings = validate_match_input(match, config.sum_tolerance)
        if errors:
            report.update({
                "estado": "error",
                "semaforo": "rojo",
                "alerta": str(errors[0]),
                "tipo_partido": "Partido de cuidado",
            })
            reports.append(report)
            continue

        analysis = analyze_match(match, config, debug_mode=debug_mode)
        analysis_dict = analysis.to_dict()
        analyses.append(analysis_dict)

        direct_pick_symbol = _resolve_direct_pick_symbol(analysis)
        direct_pick_text = _symbol_to_pick_text(direct_pick_symbol, local, visitante)
        double_pick = _normalize_double_pick(
            str(getattr(analysis, "recomendacion_principal", "")),
            str(getattr(analysis, "doble_oportunidad", "")),
            direct_pick_symbol,
            float(getattr(analysis, "score_local", 0.0)),
            float(getattr(analysis, "score_visita", 0.0)),
        )

        score_riesgo = float(getattr(analysis, "score_riesgo", 0.0))
        semaforo = _risk_label(score_riesgo)
        if warnings and semaforo == "verde":
            semaforo = "amarillo"

        alert_messages: list[str] = []
        if warnings:
            alert_messages.append(_short_warning_text(str(warnings[0])))
        flags = list(getattr(analysis, "banderas_alerta", []) or [])
        sesgos = list(getattr(analysis, "sesgos", []) or [])
        if flags:
            alert_messages.append(str(flags[0]))
        elif sesgos:
            alert_messages.append(str(sesgos[0]))
        else:
            alert_messages.append("Lectura estable")

        report.update({
            "estado": "ok",
            "pick_symbol": direct_pick_symbol,
            "recomendacion": direct_pick_text,
            "doble_oportunidad": double_pick,
            "confianza": str(getattr(analysis, "confianza", "-") or "-").capitalize(),
            "tipo_partido": str(getattr(analysis, "tipo_partido", "-") or "-"),
            "semaforo": semaforo,
            "alerta": " / ".join(alert_messages),
            "score_riesgo": score_riesgo,
            "score_contrarian": float(getattr(analysis, "score_contrarian", 0.0)),
            "apto_pick_directo": bool(getattr(analysis, "apto_pick_directo", False)),
            "apto_doble_oportunidad": bool(getattr(analysis, "apto_doble_oportunidad", False)),
            "sugerir_sorpresa": bool(getattr(analysis, "sugerir_sorpresa", False)),
            "patrones_activados": list(getattr(analysis, "patrones_activados", []) or []),
            "explicacion": str(getattr(analysis, "explicacion", "") or ""),
            "analisis": analysis_dict,
        })
        reports.append(report)

    ok_reports = [r for r in reports if r.get("estado") == "ok"]
    summary = {
        "total_rows": len(reports),
        "evaluated_rows": len(ok_reports),
        "picks_directos": sum(1 for r in ok_reports if bool(r.get("apto_pick_directo"))),
        "dobles": sum(1 for r in ok_reports if bool(r.get("apto_doble_oportunidad"))),
        "trampa": sum(1 for r in ok_reports if str(r.get("semaforo")) == "rojo"),
        "contrarian": sum(
            1
            for r in ok_reports
            if bool(r.get("sugerir_sorpresa")) or "Posible contrarian" in str(r.get("alerta") or "")
        ),
    }

    return {
        "ok": True,
        "rules": config.to_dict(),
        "reports": reports,
        "analyses": analyses,
        "summary": summary,
    }
