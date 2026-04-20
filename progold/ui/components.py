from __future__ import annotations

import math
from html import escape
from typing import Dict, List

import streamlit as st

from models import MatchAnalysis


def apply_custom_style() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Outfit:wght@500;600;700&display=swap');

            :root {
                --bg-main: #020816;
                --bg-grad-1: #0b1427;
                --bg-grad-2: #041022;
                --card-bg: #111a2b;
                --card-bg-soft: #0f1728;
                --line: #263551;
                --line-soft: #1d2940;
                --text-main: #e8f1ff;
                --text-soft: #8ca0c1;
                --green: #25e7c2;
                --yellow: #f4cf4e;
                --red: #ff5f8a;
                --blue: #47b9ff;
            }

            html, body, [class*="css"] {
                font-family: 'Manrope', sans-serif;
            }

            h1, h2, h3, h4 {
                font-family: 'Outfit', sans-serif;
                letter-spacing: 0.01em;
            }

            /* Global readable text */
            .stApp, .stApp p, .stApp li, .stApp span, .stApp label, .stApp small {
                color: var(--text-main);
            }

            [data-testid="stHeading"] h1,
            [data-testid="stHeading"] h2,
            [data-testid="stHeading"] h3,
            [data-testid="stHeading"] h4 {
                color: var(--text-main) !important;
            }

            [data-testid="stMarkdownContainer"] p,
            [data-testid="stMarkdownContainer"] li,
            [data-testid="stCaptionContainer"] {
                color: var(--text-soft) !important;
            }

            [data-testid="stWidgetLabel"] p,
            [data-testid="stWidgetLabel"] span,
            [data-testid="stWidgetLabel"] label {
                color: #c8d7f0 !important;
                font-weight: 600 !important;
            }

            [data-testid="stTextInput"] input,
            [data-testid="stNumberInput"] input,
            [data-testid="stDateInput"] input,
            [data-testid="stSelectbox"] div,
            [data-testid="stMultiSelect"] div {
                color: #e7f1ff !important;
            }

            [data-baseweb="input"] {
                background: #0f1728 !important;
                border: 1px solid #2b3a58 !important;
                border-radius: 10px !important;
            }

            [data-baseweb="base-input"] {
                background: #0f1728 !important;
            }

            [data-testid="stFileUploaderDropzone"] {
                background: linear-gradient(180deg, #111b2e 0%, #0d1728 100%) !important;
                border: 1px dashed #2f4368 !important;
                border-radius: 12px !important;
            }

            [data-testid="stFileUploaderDropzone"] * {
                color: #cde0ff !important;
            }

            /* Sidebar panel */
            [data-testid="stSidebar"] {
                background:
                    linear-gradient(180deg, #080f1d 0%, #060c17 100%) !important;
                border-right: 1px solid rgba(84, 112, 153, 0.22);
            }

            [data-testid="stSidebar"] [data-testid="stHeading"] h1,
            [data-testid="stSidebar"] [data-testid="stHeading"] h2,
            [data-testid="stSidebar"] [data-testid="stHeading"] h3,
            [data-testid="stSidebar"] [data-testid="stHeading"] h4,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] small {
                color: #dce8ff !important;
            }

            [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
            [data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
            [data-testid="stSidebar"] [data-testid="stWidgetLabel"] span {
                color: #9db8de !important;
                font-weight: 600 !important;
            }

            [data-testid="stSidebar"] [data-baseweb="input"] {
                background: #111c31 !important;
                border: 1px solid rgba(76, 108, 158, 0.45) !important;
            }

            [data-testid="stSidebar"] [data-testid="stNumberInput"] input,
            [data-testid="stSidebar"] [data-testid="stTextInput"] input {
                color: #e8f2ff !important;
                font-weight: 600 !important;
            }

            [data-testid="stSidebar"] button {
                border-radius: 10px !important;
            }

            /* Section titles */
            [data-testid="stHeading"] h2 {
                color: #f0f6ff !important;
                font-weight: 760 !important;
                letter-spacing: 0.005em;
            }

            [data-testid="stHeading"] h3 {
                color: #deebff !important;
                font-weight: 700 !important;
            }

            /* Expander cards */
            [data-testid="stExpander"] {
                border: 1px solid #263754 !important;
                border-radius: 14px !important;
                background: linear-gradient(180deg, #111b2e 0%, #0e1828 100%) !important;
            }

            [data-testid="stFileUploaderDropzone"] {
                background: linear-gradient(180deg, #111b2e 0%, #0d1728 100%) !important;
                border: 1px dashed #2f4368 !important;
                border-radius: 12px !important;
            }

            .stButton > button {
                border-radius: 12px !important;
                border: 1px solid rgba(72, 102, 148, 0.35) !important;
                background: linear-gradient(180deg, #12213a 0%, #0d1728 100%) !important;
                color: #dff0ff !important;
            }

            .stApp {
                background:
                    radial-gradient(circle at 8% 0%, rgba(40, 209, 173, 0.12), transparent 30%),
                    radial-gradient(circle at 90% 10%, rgba(71, 185, 255, 0.10), transparent 30%),
                    linear-gradient(180deg, var(--bg-grad-1) 0%, var(--bg-main) 100%);
            }

            .block-container {
                max-width: 1320px;
                padding-top: 1.4rem;
                padding-bottom: 2.2rem;
            }

            .ticket-hero {
                border: 1px solid var(--line);
                border-radius: 18px;
                background:
                    linear-gradient(120deg, rgba(20,30,49,0.98) 0%, rgba(14,23,40,0.98) 100%);
                padding: 18px 20px;
                margin-bottom: 12px;
                box-shadow: 0 16px 34px rgba(2, 9, 20, 0.45);
            }

            .ticket-hero-title {
                font-size: 1.78rem;
                font-weight: 700;
                color: var(--text-main);
                margin: 0;
            }

            .ticket-hero-subtitle {
                color: var(--text-soft);
                font-size: 0.9rem;
                margin-top: 6px;
            }

            .summary-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(180px, 1fr));
                gap: 12px;
                margin: 12px 0;
            }

            .summary-card {
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 12px 13px;
                background: linear-gradient(180deg, #16233b 0%, #101a2d 100%);
                box-shadow: inset 0 0 0 1px rgba(113, 150, 202, 0.08), 0 8px 18px rgba(5, 12, 25, 0.35);
            }

            .summary-label {
                color: var(--text-soft);
                font-size: 0.72rem;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }

            .summary-value {
                color: var(--text-main);
                font-size: 1.45rem;
                font-weight: 700;
                line-height: 1.1;
                margin-top: 2px;
            }

            .summary-note {
                color: var(--text-soft);
                font-size: 0.78rem;
                margin-top: 2px;
            }

            .summary-green { border-color: rgba(37, 231, 194, 0.35); }
            .summary-blue { border-color: rgba(71, 185, 255, 0.35); }
            .summary-red { border-color: rgba(255, 95, 138, 0.35); }
            .summary-yellow { border-color: rgba(244, 207, 78, 0.35); }

            .strip-list {
                display: grid;
                grid-template-columns: 1fr;
                gap: 10px;
                margin-top: 12px;
            }

            .event-strip {
                border: 1px solid #2b3d5f;
                border-radius: 16px;
                background: linear-gradient(180deg, #121d32 0%, #0e1728 100%);
                box-shadow: 0 10px 20px rgba(2, 9, 20, 0.4);
                overflow: hidden;
            }

            .event-strip--win {
                border-color: rgba(37, 231, 194, 0.58);
                background: linear-gradient(180deg, rgba(14,40,52,0.95) 0%, rgba(10,24,38,0.95) 100%);
                box-shadow: 0 0 0 1px rgba(37, 231, 194, 0.15), 0 12px 24px rgba(6, 40, 34, 0.35);
            }

            .event-strip--loss {
                border-color: rgba(255, 95, 138, 0.58);
                background: linear-gradient(180deg, rgba(50,18,40,0.95) 0%, rgba(23,14,32,0.95) 100%);
                box-shadow: 0 0 0 1px rgba(255, 95, 138, 0.15), 0 12px 24px rgba(42, 12, 24, 0.35);
            }

            .event-strip--neutral {
                border-color: #31476d;
            }

            .event-strip-head {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 8px 12px;
                background: rgba(12, 20, 35, 0.6);
                border-bottom: 1px solid var(--line-soft);
            }

            .event-strip-title {
                font-family: 'Outfit', sans-serif;
                font-size: 0.9rem;
                color: #d5e6ff;
                font-weight: 700;
            }

            .event-strip-status {
                border: 1px solid rgba(67, 96, 139, 0.5);
                border-radius: 999px;
                font-size: 0.68rem;
                padding: 2px 8px;
                letter-spacing: 0.08em;
                color: #a9c3e8;
                text-transform: uppercase;
            }

            .event-strip-body {
                padding: 10px 12px 11px 12px;
                display: grid;
                grid-template-columns: 1.25fr 1.1fr 1.2fr;
                align-items: center;
                gap: 12px;
            }

            .strip-teams {
                display: grid;
                gap: 4px;
            }

            .strip-team-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 8px;
                color: #edf5ff;
                font-size: 0.98rem;
                font-weight: 700;
            }

            .strip-team-name {
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }

            .strip-pct {
                border: 1px solid rgba(77, 110, 158, 0.48);
                border-radius: 999px;
                padding: 2px 8px;
                font-family: 'Outfit', sans-serif;
                color: #a5c9f7;
                font-size: 0.75rem;
                background: rgba(14, 24, 40, 0.88);
            }

            .strip-pick-core {
                display: flex;
                align-items: center;
                gap: 6px;
                justify-content: center;
                font-size: 0.76rem;
                border: 1px solid rgba(71, 185, 255, 0.42);
                color: #9fdfff;
                background: rgba(71, 185, 255, 0.12);
                border-radius: 999px;
                padding: 4px 8px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.03em;
            }

            .score-big {
                text-align: center;
                font-family: 'Outfit', sans-serif;
                font-size: 1.55rem;
                font-weight: 800;
                line-height: 1.05;
                color: #f2f8ff;
                letter-spacing: 0.02em;
                margin: 2px 0 8px 0;
                text-shadow: 0 0 18px rgba(71, 185, 255, 0.25);
            }

            .strip-meta {
                display: grid;
                gap: 6px;
            }

            .strip-chip-row {
                display: flex;
                gap: 6px;
                flex-wrap: wrap;
                justify-content: flex-end;
                align-items: center;
            }

            .fg-status {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                border-radius: 999px;
                padding: 4px 10px;
                font-size: 0.74rem;
                font-weight: 800;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                border: 1px solid transparent;
            }

            .fg-status::before {
                content: "";
                width: 8px;
                height: 8px;
                border-radius: 999px;
                display: inline-block;
            }

            .fg-win {
                color: #9dffe7;
                background: rgba(37, 231, 194, 0.18);
                border-color: rgba(37, 231, 194, 0.45);
            }
            .fg-win::before { background: #25e7c2; }

            .fg-loss {
                color: #ffabc0;
                background: rgba(255, 95, 138, 0.18);
                border-color: rgba(255, 95, 138, 0.45);
            }
            .fg-loss::before { background: #ff5f8a; }

            .fg-pending {
                color: #ffdf82;
                background: rgba(244, 207, 78, 0.16);
                border-color: rgba(244, 207, 78, 0.4);
            }
            .fg-pending::before { background: #f4cf4e; }

            .fg-none {
                color: #bfd3ef;
                background: rgba(90, 120, 168, 0.14);
                border-color: rgba(90, 120, 168, 0.42);
            }
            .fg-none::before { background: #8aa6cc; }

            .strip-line {
                text-align: right;
                font-size: 0.8rem;
                color: #94abcc;
            }

            .ticket-index {
                font-family: 'Outfit', sans-serif;
                font-weight: 700;
                font-size: 1rem;
                color: #cce0ff;
            }

            .ticket-team {
                color: #213127;
                font-weight: 700;
                letter-spacing: 0.015em;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }

            .ticket-pct {
                font-family: 'Space Grotesk', sans-serif;
                color: #35473c;
                font-size: 0.96rem;
                text-align: center;
            }

            .pick-chip,
            .double-chip,
            .alert-chip {
                display: inline-block;
                border-radius: 999px;
                padding: 3px 10px;
                font-size: 0.76rem;
                border: 1px solid transparent;
            }

            .pick-chip {
                color: #1f6f7a;
                border-color: rgba(31, 111, 122, 0.32);
                background: rgba(31, 111, 122, 0.10);
                font-weight: 700;
            }

            .pick-hit {
                color: #2f8f68;
                border-color: rgba(47, 143, 104, 0.35);
                background: rgba(47, 143, 104, 0.13);
            }

            .pick-miss {
                color: #b24b3f;
                border-color: rgba(178, 75, 63, 0.38);
                background: rgba(178, 75, 63, 0.12);
            }

            .pick-pending {
                color: #a87a22;
                border-color: rgba(168, 122, 34, 0.36);
                background: rgba(168, 122, 34, 0.14);
            }

            .result-chip {
                display: inline-block;
                margin-left: 6px;
                border-radius: 999px;
                padding: 2px 8px;
                font-size: 0.72rem;
                border: 1px solid rgba(126, 145, 134, 0.3);
                color: #566a5e;
                background: rgba(126, 145, 134, 0.1);
            }

            .result-hit {
                color: #2f8f68;
                border-color: rgba(47, 143, 104, 0.32);
                background: rgba(47, 143, 104, 0.11);
            }

            .result-miss {
                color: #b24b3f;
                border-color: rgba(178, 75, 63, 0.35);
                background: rgba(178, 75, 63, 0.11);
            }

            .result-pending {
                color: #a87a22;
                border-color: rgba(168, 122, 34, 0.35);
                background: rgba(168, 122, 34, 0.11);
            }

            .double-chip {
                color: #1f6f7a;
                border-color: rgba(31, 111, 122, 0.28);
                background: rgba(31, 111, 122, 0.08);
            }

            .alert-chip {
                color: #54665b;
                border-color: rgba(126, 145, 134, 0.35);
                background: rgba(126, 145, 134, 0.10);
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                max-width: 100%;
            }

            .alert-green {
                color: #2f8f68;
                border-color: rgba(47, 143, 104, 0.32);
                background: rgba(47, 143, 104, 0.11);
            }

            .alert-yellow {
                color: #a87a22;
                border-color: rgba(168, 122, 34, 0.35);
                background: rgba(168, 122, 34, 0.11);
            }

            .alert-red {
                color: #b24b3f;
                border-color: rgba(178, 75, 63, 0.35);
                background: rgba(178, 75, 63, 0.11);
            }

            .panel {
                border: 1px solid var(--line);
                border-radius: 16px;
                background: linear-gradient(180deg, #15233a 0%, #101a2d 100%);
                padding: 12px 13px;
                height: 100%;
                margin-top: 8px;
                box-shadow: 0 8px 20px rgba(6, 13, 28, 0.35);
            }

            .panel h4 {
                margin: 0 0 6px 0;
                font-size: 0.98rem;
            }

            .panel-item {
                color: #d8e8ff;
                font-size: 0.84rem;
                margin-bottom: 4px;
            }

            .badge {
                display: inline-block;
                border-radius: 999px;
                padding: 4px 10px;
                font-size: 0.78rem;
                border: 1px solid transparent;
                margin-right: 6px;
                margin-bottom: 6px;
            }

            .badge-green { background: rgba(37, 231, 194, 0.12); color: #7cf2d9; border-color: rgba(37, 231, 194, 0.35); }
            .badge-yellow { background: rgba(244, 207, 78, 0.14); color: #ffd76a; border-color: rgba(244, 207, 78, 0.35); }
            .badge-red { background: rgba(255, 95, 138, 0.12); color: #ff86ab; border-color: rgba(255, 95, 138, 0.35); }
            .badge-blue { background: rgba(71, 185, 255, 0.12); color: #91d8ff; border-color: rgba(71, 185, 255, 0.30); }

            .compact-muted {
                color: var(--text-soft);
                font-size: 0.83rem;
            }

            .pattern-glossary {
                border: 1px solid var(--line);
                border-radius: 18px;
                background: linear-gradient(180deg, #15233a 0%, #101a2d 100%);
                box-shadow: 0 10px 24px rgba(6, 13, 28, 0.35);
                padding: 14px;
                margin: 10px 0 16px 0;
            }

            .pattern-glossary-title {
                margin: 0 0 9px 0;
                color: #ecf4ff;
                font-size: 1rem;
                font-weight: 700;
                letter-spacing: 0.01em;
            }

            .pattern-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(220px, 1fr));
                gap: 10px;
            }

            .pattern-item {
                border: 1px solid var(--line);
                border-radius: 13px;
                background: #111b2e;
                padding: 10px;
            }

            .pattern-name {
                display: inline-block;
                font-family: 'Outfit', sans-serif;
                font-size: 0.8rem;
                color: #7cd7ff;
                background: rgba(71, 185, 255, 0.14);
                border: 1px solid rgba(71, 185, 255, 0.3);
                border-radius: 999px;
                padding: 2px 8px;
                margin-bottom: 6px;
            }

            .pattern-desc {
                color: #d6e7ff;
                font-size: 0.82rem;
                line-height: 1.35;
            }

            .pattern-tip {
                margin-top: 6px;
                color: #90a7ca;
                font-size: 0.78rem;
            }

            @media (max-width: 1100px) {
                .summary-grid {
                    grid-template-columns: repeat(2, minmax(180px, 1fr));
                }

                .event-strip-body {
                    grid-template-columns: 1fr;
                }

                .pattern-grid {
                    grid-template-columns: repeat(2, minmax(180px, 1fr));
                }
            }

            @media (max-width: 760px) {
                .summary-grid {
                    grid-template-columns: repeat(1, minmax(180px, 1fr));
                }

                .pattern-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def risk_label(score_riesgo: float) -> str:
    if score_riesgo >= 68:
        return "rojo"
    if score_riesgo >= 45:
        return "amarillo"
    return "verde"


def confidence_badge(confianza: str) -> str:
    if confianza == "alta":
        return '<span class="badge badge-green">Confianza alta</span>'
    if confianza == "media":
        return '<span class="badge badge-yellow">Confianza media</span>'
    return '<span class="badge badge-red">Confianza baja</span>'


def semaforo_badge(semaforo: str) -> str:
    if semaforo == "verde":
        return '<span class="badge badge-green">Lectura estable</span>'
    if semaforo == "amarillo":
        return '<span class="badge badge-yellow">Partido de cuidado</span>'
    if semaforo == "rojo":
        return '<span class="badge badge-red">Posible trampa / sesgo</span>'
    return '<span class="badge badge-blue">Sin datos</span>'


def _summary_card(title: str, value: int, note: str, variant: str) -> str:
    return (
        f'<div class="summary-card {variant}">'
        f'<div class="summary-label">{title}</div>'
        f'<div class="summary-value">{value}</div>'
        f'<div class="summary-note">{note}</div>'
        "</div>"
    )


def _format_pct(value: object) -> str:
    if value is None:
        return "--"
    if isinstance(value, float) and math.isnan(value):
        return "--"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "--"


def render_summary_cards(analyses: List[MatchAnalysis]) -> None:
    picks_directos = sum(1 for item in analyses if item.apto_pick_directo)
    dobles = sum(1 for item in analyses if item.apto_doble_oportunidad)
    trampa = sum(1 for item in analyses if risk_label(item.score_riesgo) == "rojo")
    contrarian = sum(
        1 for item in analyses if item.sugerir_sorpresa or "Posible contrarian" in item.sesgos
    )

    html = "".join(
        [
            _summary_card("Picks directos", picks_directos, "Partidos aptos para 1, X o 2", "summary-green"),
            _summary_card("Doble oportunidad", dobles, "Partidos de cobertura sugerida", "summary-blue"),
            _summary_card("Partidos trampa", trampa, "Riesgo alto por sesgo de masa", "summary-red"),
            _summary_card("Partidos contrarian", contrarian, "Posible valor en sorpresa", "summary-yellow"),
        ]
    )
    st.markdown(f'<div class="summary-grid">{html}</div>', unsafe_allow_html=True)


def render_jornada_panel(analyses: List[MatchAnalysis]) -> None:
    directos = [item for item in analyses if item.apto_pick_directo]
    dobles = [item for item in analyses if item.apto_doble_oportunidad]
    trampa = [item for item in analyses if risk_label(item.score_riesgo) == "rojo"]
    contrarian = [item for item in analyses if item.sugerir_sorpresa or "Posible contrarian" in item.sesgos]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="panel"><h4>Picks directos</h4>', unsafe_allow_html=True)
        if directos:
            for item in directos[:4]:
                st.markdown(
                    f'<div class="panel-item">{escape(item.local)} vs {escape(item.visitante)}: '
                    f"{escape(item.recomendacion_principal)}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div class="compact-muted">Sin picks directos fuertes.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel"><h4>Doble oportunidad</h4>', unsafe_allow_html=True)
        if dobles:
            for item in dobles[:4]:
                suggestion = item.doble_oportunidad if item.doble_oportunidad != "-" else item.recomendacion_principal
                st.markdown(
                    f'<div class="panel-item">{escape(item.local)} vs {escape(item.visitante)}: '
                    f"{escape(suggestion)}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div class="compact-muted">No se detectaron coberturas claras.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="panel"><h4>Partidos trampa</h4>', unsafe_allow_html=True)
        if trampa:
            for item in trampa[:4]:
                st.markdown(
                    f'<div class="panel-item">{escape(item.local)} vs {escape(item.visitante)} '
                    f"({item.score_riesgo:.1f})</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div class="compact-muted">No hay riesgo extremo detectado.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="panel"><h4>Partidos contrarian</h4>', unsafe_allow_html=True)
        if contrarian:
            for item in contrarian[:4]:
                st.markdown(
                    f'<div class="panel-item">{escape(item.local)} vs {escape(item.visitante)}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div class="compact-muted">Sin lectura contrarian marcada.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_pattern_glossary() -> None:
    entries = [
        {
            "name": "partido_caotico",
            "desc": "Los tres porcentajes estan muy juntos; no hay liderazgo real.",
            "tip": "Suele convenir doble oportunidad y bajar expectativa de pick directo.",
        },
        {
            "name": "empate_vivo",
            "desc": "El empate compite de verdad contra local/visita por estructura de porcentajes.",
            "tip": "No ignores la X; puede entrar en pick final o en cobertura.",
        },
        {
            "name": "empate_ignorado_por_masa",
            "desc": "La masa castiga el empate, pero la geometria del partido lo mantiene viable.",
            "tip": "Buena senal contrarian para considerar X, 1X o X2.",
        },
        {
            "name": "visita_viva",
            "desc": "La visita tiene soporte competitivo y el local no domina claramente.",
            "tip": "Evita descartar al 2; muchas veces termina en X2.",
        },
        {
            "name": "favorito_estable",
            "desc": "Un lado lidera con margen sano y estructura consistente.",
            "tip": "Permite pick directo con mayor confianza si el riesgo no sube.",
        },
        {
            "name": "favorito_sobrejugado",
            "desc": "El favorito esta inflado por masa mas de lo que respalda la estructura.",
            "tip": "Mejor proteger con doble oportunidad en vez de ir all-in.",
        },
        {
            "name": "local_sobrepopular",
            "desc": "El local concentra jugadas, pero visita/empate siguen con vida.",
            "tip": "Desconfia del 1 directo; revisa X2 o 1X segun scores.",
        },
        {
            "name": "visita_sobrecomprada_moderada",
            "desc": "La visita lidera, pero sin dominio total; el mercado puede estar sesgado.",
            "tip": "Evita el 2 ciego; cobertura X2 suele ser mas estable.",
        },
        {
            "name": "partido_abierto_con_empate_castigado",
            "desc": "Local y visita altos con empate bajo: escenario de extremos.",
            "tip": "Normalmente favorece cobertura 12.",
        },
    ]

    cards = []
    for item in entries:
        cards.append(
            "<div class='pattern-item'>"
            f"<div class='pattern-name'>{escape(item['name'])}</div>"
            f"<div class='pattern-desc'>{escape(item['desc'])}</div>"
            f"<div class='pattern-tip'>{escape(item['tip'])}</div>"
            "</div>"
        )

    st.markdown(
        "<div class='pattern-glossary'>"
        "<div class='pattern-glossary-title'>Guia de lectura de patrones</div>"
        "<div class='pattern-grid'>"
        + "".join(cards)
        + "</div></div>",
        unsafe_allow_html=True,
    )


def render_ticket_board(rows: List[Dict[str, object]]) -> None:
    chunks: List[str] = ['<div class="strip-list">']

    for row in rows:
        if str(row.get("estado", "vacio")) == "vacio":
            continue

        semaforo = str(row.get("semaforo", "neutro"))
        if semaforo not in {"verde", "amarillo", "rojo"}:
            semaforo = "neutro"

        result_state = str(row.get("resultado_estado", "sin_resultado") or "sin_resultado")
        if result_state not in {"acierto", "fallo", "pendiente", "sin_resultado"}:
            result_state = "sin_resultado"

        pick_state_class = "pick-chip"
        if result_state == "acierto":
            pick_state_class += " pick-hit"
            strip_state = "win"
            status_badge = "FG ACIERTO"
            fg_status_class = "fg-status fg-win"
        elif result_state == "fallo":
            pick_state_class += " pick-miss"
            strip_state = "loss"
            status_badge = "FG FALLO"
            fg_status_class = "fg-status fg-loss"
        elif result_state == "pendiente":
            pick_state_class += " pick-pending"
            strip_state = "neutral"
            status_badge = "FG PENDIENTE"
            fg_status_class = "fg-status fg-pending"
        else:
            strip_state = "neutral"
            status_badge = "FG SIN RESULTADO"
            fg_status_class = "fg-status fg-none"

        local = escape(str(row.get("local", "")).strip() or "Local")
        visitante = escape(str(row.get("visitante", "")).strip() or "Visitante")
        pick = escape(str(row.get("recomendacion", "-") or "-"))
        doble = escape(str(row.get("doble_oportunidad", "-") or "-"))
        alerta = escape(str(row.get("alerta", "Sin captura") or "Sin captura"))
        result_text = escape(str(row.get("resultado_texto", "Sin validar") or "Sin validar"))
        partido = int(row.get("partido", 0) or 0)
        confidence = escape(str(row.get("confianza", "-") or "-").upper())
        tipo = escape(str(row.get("tipo_partido", "Partido de cuidado") or "Partido de cuidado"))

        pct_local = _format_pct(row.get("pct_local"))
        pct_empate = _format_pct(row.get("pct_empate"))
        pct_visita = _format_pct(row.get("pct_visita"))
        marcador = escape(str(row.get("marcador_real", "-") or "-"))
        resultado_real = escape(str(row.get("resultado_real", "-") or "-"))

        chunks.append(
            f'<div class="event-strip event-strip--{strip_state}">'
            '<div class="event-strip-head">'
            f'<div class="event-strip-title">Partido {partido:02d}</div>'
            f'<div class="event-strip-status">{confidence}</div>'
            "</div>"
            '<div class="event-strip-body">'
            '<div class="strip-teams">'
            f'<div class="strip-team-row"><span class="strip-team-name">{local}</span><span class="strip-pct">{pct_local}%</span></div>'
            f'<div class="strip-team-row"><span class="strip-team-name">Empate</span><span class="strip-pct">{pct_empate}%</span></div>'
            f'<div class="strip-team-row"><span class="strip-team-name">{visitante}</span><span class="strip-pct">{pct_visita}%</span></div>'
            f'<div class="strip-line">Resultado real: {resultado_real}</div>'
            "</div>"
            f'<div class="score-big">{marcador}</div>'
            f'<div class="strip-pick-core">Pick principal: {pick}</div>'
            '<div class="strip-meta">'
            '<div class="strip-chip-row">'
            f'<span class="{fg_status_class}">{status_badge}</span>'
            f'<span class="result-chip">{tipo}</span>'
            f'<span class="double-chip">Doble: {doble}</span>'
            "</div>"
            f'<div class="strip-line"><span class="alert-chip alert-{semaforo}">{alerta}</span></div>'
            "</div>"
            "</div>"
            "</div>"
        )

    chunks.append("</div>")
    st.markdown("".join(chunks), unsafe_allow_html=True)


def build_match_json_output(analysis: MatchAnalysis) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "local": analysis.local,
        "visitante": analysis.visitante,
        "pct_local": analysis.pct_local,
        "pct_empate": analysis.pct_empate,
        "pct_visita": analysis.pct_visita,
        "tipo_partido": analysis.tipo_partido,
        "sesgos": analysis.sesgos,
        "patrones_activados": analysis.patrones_activados,
        "recomendacion_principal": analysis.recomendacion_principal,
        "doble_oportunidad": analysis.doble_oportunidad,
        "confianza": analysis.confianza,
        "score_estabilidad": analysis.score_estabilidad,
        "explicacion": analysis.explicacion,
    }
    if analysis.debug_data:
        payload["debug_data"] = analysis.debug_data
    return payload
