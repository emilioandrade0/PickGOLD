from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Tuple

import pandas as pd

from models import MatchAnalysis


def build_export_dataframes(analyses: List[MatchAnalysis]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    main_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []

    for idx, analysis in enumerate(analyses, start=1):
        partido = f"{analysis.local} vs {analysis.visitante}"
        semaforo = "verde"
        if analysis.score_riesgo >= 68:
            semaforo = "rojo"
        elif analysis.score_riesgo >= 45:
            semaforo = "amarillo"

        main_rows.append(
            {
                "id": idx,
                "partido": partido,
                "local": analysis.pct_local,
                "empate": analysis.pct_empate,
                "visita": analysis.pct_visita,
                "porcentaje_mayor": analysis.porcentaje_mayor,
                "segundo_porcentaje_mayor": analysis.segundo_porcentaje_mayor,
                "diferencia_top2": analysis.diferencia_top2,
                "tipo_partido": analysis.tipo_partido,
                "sesgo_detectado": " | ".join(analysis.sesgos),
                "recomendacion": analysis.recomendacion_principal,
                "nivel_confianza": analysis.confianza,
                "doble_oportunidad": analysis.doble_oportunidad,
                "semaforo": semaforo,
                "score_riesgo": analysis.score_riesgo,
                "score_contrarian": analysis.score_contrarian,
                "score_estabilidad": analysis.score_estabilidad,
            }
        )

        detail_rows.append(
            {
                "id": idx,
                "partido": partido,
                "posible_ganador_masa": analysis.posible_ganador_masa,
                "score_local": analysis.score_local,
                "score_empate": analysis.score_empate,
                "score_visita": analysis.score_visita,
                "score_riesgo": analysis.score_riesgo,
                "score_contrarian": analysis.score_contrarian,
                "score_estabilidad": analysis.score_estabilidad,
                "patrones_activados": " | ".join(analysis.patrones_activados),
                "sesgos": " | ".join(analysis.sesgos),
                "banderas_alerta": " | ".join(analysis.banderas_alerta),
                "explicacion": analysis.explicacion,
                "apto_pick_directo": analysis.apto_pick_directo,
                "apto_doble_oportunidad": analysis.apto_doble_oportunidad,
                "sugerir_sorpresa": analysis.sugerir_sorpresa,
            }
        )

    main_df = pd.DataFrame(main_rows)
    detail_df = pd.DataFrame(detail_rows)
    return main_df, detail_df


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dataframes_to_excel_bytes(main_df: pd.DataFrame, detail_df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        main_df.to_excel(writer, sheet_name="tabla_principal", index=False)
        detail_df.to_excel(writer, sheet_name="analisis_detallado", index=False)
    output.seek(0)
    return output.read()


def session_to_json_bytes(
    matches_payload: List[Dict[str, Any]],
    rules_payload: Dict[str, Any],
) -> bytes:
    payload = {
        "version": "1.0",
        "matches": matches_payload,
        "rules": rules_payload,
    }
    return json.dumps(payload, indent=2).encode("utf-8")


def parse_session_json(content: bytes) -> Dict[str, Any]:
    payload = json.loads(content.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("El JSON de sesion debe ser un objeto.")
    if "matches" not in payload or not isinstance(payload["matches"], list):
        raise ValueError("El JSON no incluye una lista valida de partidos.")
    if "rules" in payload and not isinstance(payload["rules"], dict):
        raise ValueError("El bloque rules debe ser un objeto JSON.")
    return payload
