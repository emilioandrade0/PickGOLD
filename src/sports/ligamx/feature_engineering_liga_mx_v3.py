from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering_liga_mx import OUTPUT_FILE as BASELINE_FEATURES_FILE
from feature_engineering_liga_mx import build_features

import sys
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
PROCESSED_DIR = BASE_DIR / "data" / "liga_mx" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = PROCESSED_DIR / "model_ready_features_liga_mx_v3.csv"


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_v3_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    preserve_nan_cols = [
        c
        for c in [
            "home_ht_score",
            "away_ht_score",
            "total_ht_goals",
            "TARGET_ht_result",
            "TARGET_h1_over_15",
            "TARGET_corners_over_95",
        ]
        if c in out.columns
    ]
    preserved = {c: out[c].copy() for c in preserve_nan_cols}

    # Rolling xG-like proxies from attack-defense interaction (no leakage: all inputs are shifted windows).
    out["home_xg_proxy_l10"] = (
        0.65 * out["home_goals_scored_L10"] + 0.35 * out["away_goals_allowed_L10"]
    )
    out["away_xg_proxy_l10"] = (
        0.65 * out["away_goals_scored_L10"] + 0.35 * out["home_goals_allowed_L10"]
    )
    out["diff_xg_proxy_l10"] = out["home_xg_proxy_l10"] - out["away_xg_proxy_l10"]

    out["home_xga_proxy_l10"] = (
        0.60 * out["home_goals_allowed_L10"] + 0.40 * out["away_goals_scored_L10"]
    )
    out["away_xga_proxy_l10"] = (
        0.60 * out["away_goals_allowed_L10"] + 0.40 * out["home_goals_scored_L10"]
    )
    out["diff_xga_proxy_l10"] = out["home_xga_proxy_l10"] - out["away_xga_proxy_l10"]

    # LA CREAMOS AQUÍ:
    out["xg_balance_index"] = out["diff_xg_proxy_l10"] - 0.75 * out["diff_xga_proxy_l10"]

    # ==========================================================
    # NUEVAS VARIABLES: ALTITUD Y CIERRE DE TORNEO
    # ==========================================================
    ALTITUDES = {
        "TOL": 2660, "PUM": 2240, "AME": 2240, "CAZ": 2240, "PAC": 2400,
        "PUE": 2135, "QRO": 1820, "GDL": 1566, "LEO": 1815, "SAN": 1120,
        "MTY": 540, "TIG": 540, "JUA": 1137, "NEC": 1880, "SLP": 1860,
        "MAZ": 5, "TIJ": 20, "ATS": 1566
    }
    
    out['home_alt'] = out['home_team'].map(ALTITUDES).fillna(1000)
    out['away_alt'] = out['away_team'].map(ALTITUDES).fillna(1000)
    
    # Impacto físico: El visitante sube o baja drásticamente de altura
    out['diff_altitude_visitor'] = out['home_alt'] - out['away_alt']
    
    # Motivación por Fase de Torneo (Jornadas Finales)
    if 'date' in out.columns:
        date_series = pd.to_datetime(out['date'])
        # Proxy de "presión de cierre": Abril/Mayo o Noviembre/Diciembre
        out['is_tournament_closing'] = date_series.dt.month.isin([4, 5, 11, 12]).astype(int)
    # ==========================================================

    # Rest and locality-context proxies.
    out["home_locality_proxy"] = (
        0.45 * out["home_surface_edge"]
        + 0.30 * out["home_rest_days"]
        - 0.25 * out["home_games_last_5_days"]
    )
    out["away_locality_proxy"] = (
        0.45 * out["away_surface_edge"]
        + 0.30 * out["away_rest_days"]
        - 0.25 * out["away_games_last_5_days"]
    )
    out["diff_locality_proxy"] = out["home_locality_proxy"] - out["away_locality_proxy"]

    # Cards-discipline pressure proxy from schedule load + draw pressure + volatility.
    out["home_cards_pressure_proxy"] = (
        0.40 * out["home_schedule_load_exp"]
        + 0.35 * out["home_goal_diff_std_L10"]
        + 0.25 * out["home_draw_pct_L10"]
    )
    out["away_cards_pressure_proxy"] = (
        0.40 * out["away_schedule_load_exp"]
        + 0.35 * out["away_goal_diff_std_L10"]
        + 0.25 * out["away_draw_pct_L10"]
    )
    out["diff_cards_pressure_proxy"] = (
        out["home_cards_pressure_proxy"] - out["away_cards_pressure_proxy"]
    )

    # Corners and BTTS proxies.
    out["home_corners_proxy"] = (
        2.4
        + 0.95 * out["home_goals_scored_L10"]
        + 0.55 * out["home_home_only_over_25_rate_L10"]
        + 0.30 * out["home_home_only_btts_rate_L10"]
    )
    out["away_corners_proxy"] = (
        2.4
        + 0.95 * out["away_goals_scored_L10"]
        + 0.55 * out["away_away_only_over_25_rate_L10"]
        + 0.30 * out["away_away_only_btts_rate_L10"]
    )
    out["diff_corners_proxy"] = out["home_corners_proxy"] - out["away_corners_proxy"]
    out["total_corners_proxy"] = out["home_corners_proxy"] + out["away_corners_proxy"]

    out["btts_proxy"] = (
        0.50 * (out["home_btts_rate_L10"] + out["away_btts_rate_L10"])
        + 0.30 * _safe_div(
            out["home_goals_scored_L10"] + out["away_goals_scored_L10"],
            pd.Series(2.0, index=out.index),
        )
        + 0.20 * _safe_div(
            out["home_goals_allowed_L10"] + out["away_goals_allowed_L10"],
            pd.Series(2.0, index=out.index),
        )
    )

    # Market-implied context from available total line.
    out["implied_total_line"] = out["odds_over_under"].clip(lower=0.0)
    expected_total_l10 = (
        out["home_goals_scored_L10"]
        + out["away_goals_scored_L10"]
        + out["home_goals_allowed_L10"]
        + out["away_goals_allowed_L10"]
    ) / 2.0
    out["expected_total_l10"] = expected_total_l10
    out["market_total_gap_l10"] = out["implied_total_line"] - out["expected_total_l10"]

    out["draw_tension_v3"] = (
        out["draw_equilibrium_index"]
        * np.exp(-np.abs(out["xg_balance_index"]) / 1.8)
        * (1.0 + 0.25 * out["draw_pressure_avg"])
    )

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for c, series in preserved.items():
        out[c] = series
    return out

def main() -> None:
    print("=" * 72)
    print("LIGA MX FEATURE ENGINEERING V3")
    print("=" * 72)

    print("[1/3] Building baseline features (source-of-truth pipeline)...")
    baseline_df = build_features()

    print("[2/3] Adding V3 feature block...")
    v3_df = add_v3_features(baseline_df)

    print("[3/3] Writing outputs...")
    baseline_df.to_csv(BASELINE_FEATURES_FILE, index=False)
    v3_df.to_csv(OUTPUT_FILE, index=False)

    print(f"[OK] Baseline features refreshed: {BASELINE_FEATURES_FILE}")
    print(f"[OK] V3 features saved       : {OUTPUT_FILE}")
    print(f"[OK] Rows: {len(v3_df)} | Columns baseline={len(baseline_df.columns)} v3={len(v3_df.columns)}")


if __name__ == "__main__":
    main()
