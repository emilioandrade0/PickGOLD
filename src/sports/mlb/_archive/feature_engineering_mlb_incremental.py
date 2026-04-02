from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from datetime import datetime, timedelta
import pandas as pd
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = SRC_ROOT
RAW_DATA = BASE_DIR / "data" / "mlb" / "raw" / "mlb_advanced_history.csv"
OUTPUT_FILE = BASE_DIR / "data" / "mlb" / "processed" / "model_ready_features_mlb.csv"

print("ℹ️ Iniciando builder incremental de features MLB...")

def main(window_days: int = 60):
    if not RAW_DATA.exists():
        print(f"⚠️ Raw data no encontrada: {RAW_DATA}. Ejecuta primero el ingest.")
        return

    df_raw = pd.read_csv(RAW_DATA)
    df_raw["date"] = df_raw["date"].astype(str)

    if not OUTPUT_FILE.exists():
        print("⚠️ No existe CSV procesado previo. Ejecutando generación completa (fallback).")
        # fallback: call full builder if available
        try:
            from sports.mlb.feature_engineering_mlb_core import build_features
            build_features()
        except Exception as e:
            print(f"Error ejecutando build_features(): {e}")
        return

    try:
        df_prev = pd.read_csv(OUTPUT_FILE)
    except Exception as e:
        print(f"⚠️ No se pudo leer {OUTPUT_FILE}: {e}")
        return

    if df_prev.empty:
        last_date = None
    else:
        last_date = pd.to_datetime(df_prev['date']).max()

    if last_date is None:
        print("ℹ️ No hay fecha previa; se hará generación completa.")
        try:
            from sports.mlb.feature_engineering_mlb_core import build_features
            build_features()
        except Exception as e:
            print(f"Error build_features(): {e}")
        return

    # Determine affected window
    window_start_dt = last_date - pd.Timedelta(days=window_days)
    window_start = window_start_dt.strftime('%Y-%m-%d')

    # rows to recompute: those with date >= window_start
    to_recompute = df_raw[pd.to_datetime(df_raw['date']) >= pd.to_datetime(window_start)].copy()
    if to_recompute.empty:
        print("✅ No hay filas nuevas en la ventana incremental; nada que hacer.")
        return

    print(f"🔁 Recalculando features para ventana desde {window_start} (rows={len(to_recompute)})")

    # Use existing functions from feature_engineering_mlb_core to rebuild the affected slice
    try:
        from sports.mlb.feature_engineering_mlb_core import (
            calculate_elo_ratings,
            calculate_team_rolling_features,
            calculate_surface_split_features,
            build_pitcher_game_table,
            build_bullpen_proxy_features,
            ensure_market_columns,
            add_league_relative_features,
        )
    except Exception as e:
        print(f"❌ No se pudieron importar funciones de feature_engineering_mlb_core: {e}")
        return

    # Build full context: we need history including events before window_start to compute shifted rolls
    hist_needed = df_raw[pd.to_datetime(df_raw['date']) < pd.to_datetime(window_start)].copy()
    context_df = pd.concat([hist_needed, to_recompute], ignore_index=True).sort_values(['date', 'game_id'])

    # Recompute building blocks
    context_df = calculate_elo_ratings(context_df)
    rolling = calculate_team_rolling_features(context_df)
    home_surf, away_surf = calculate_surface_split_features(context_df)
    pitcher_tbl = build_pitcher_game_table(context_df)
    bullpen_tbl = build_bullpen_proxy_features(context_df)

    # Re-assemble model-ready rows for the window (dates >= window_start)
    rec_df = context_df[context_df['date'] >= window_start].copy()

    # Merge rolling features and others similar to build_features flow
    rec_df = pd.merge(
        rec_df,
        rolling,
        left_on=["game_id", "home_team"],
        right_on=["game_id", "team"],
        how="left",
    )
    rec_df = rec_df.rename(columns={c: f"home_{c}" for c in rolling.columns if c not in ["game_id", "team"]}).drop(columns=["team"])

    rec_df = pd.merge(
        rec_df,
        rolling,
        left_on=["game_id", "away_team"],
        right_on=["game_id", "team"],
        how="left",
    )
    rec_df = rec_df.rename(columns={c: f"away_{c}" for c in rolling.columns if c not in ["game_id", "team"]}).drop(columns=["team"])

    # surface
    rec_df = pd.merge(
        rec_df,
        home_surf,
        left_on=["game_id", "home_team"],
        right_on=["game_id", "team"],
        how="left",
    )
    rec_df = rec_df.rename(columns={c: f"home_{c}" for c in home_surf.columns if c not in ["game_id", "team"]}).drop(columns=["team"])

    rec_df = pd.merge(
        rec_df,
        away_surf,
        left_on=["game_id", "away_team"],
        right_on=["game_id", "team"],
        how="left",
    )
    rec_df = rec_df.rename(columns={c: f"away_{c}" for c in away_surf.columns if c not in ["game_id", "team"]}).drop(columns=["team"])

    # pitchers and bullpen
    rec_df = pd.merge(rec_df, pitcher_tbl, left_on=["game_id", "home_team"], right_on=["game_id", "team"], how="left")
    rec_df = rec_df.rename(columns={c: f"home_{c}" for c in pitcher_tbl.columns if c not in ["game_id", "team"]}).drop(columns=["team"])
    rec_df = pd.merge(rec_df, pitcher_tbl, left_on=["game_id", "away_team"], right_on=["game_id", "team"], how="left")
    rec_df = rec_df.rename(columns={c: f"away_{c}" for c in pitcher_tbl.columns if c not in ["game_id", "team"]}).drop(columns=["team"])

    rec_df = pd.merge(rec_df, bullpen_tbl, left_on=["game_id", "home_team"], right_on=["game_id", "team"], how="left")
    rec_df = rec_df.rename(columns={c: f"home_{c}" for c in bullpen_tbl.columns if c not in ["game_id", "team"]}).drop(columns=["team"])
    rec_df = pd.merge(rec_df, bullpen_tbl, left_on=["game_id", "away_team"], right_on=["game_id", "team"], how="left")
    rec_df = rec_df.rename(columns={c: f"away_{c}" for c in bullpen_tbl.columns if c not in ["game_id", "team"]}).drop(columns=["team"])

    # Postprocess: ensure market cols and league relative features
    rec_df = ensure_market_columns(rec_df)
    rec_df = add_league_relative_features(rec_df)

    # Select model columns via original module to ensure schema compatibility
    try:
        from sports.mlb.feature_engineering_mlb_core import build_features as _bf
        # reuse final column list by calling _bf and reading OUTPUT_FILE, but to avoid running full builder,
        # we'll load existing prev columns
        model_cols = list(df_prev.columns)
    except Exception:
        model_cols = None

    # Build final combined dataset: keep prev rows with date < window_start, and rec_df for >= window_start
    prev_keep = df_prev[pd.to_datetime(df_prev['date']) < pd.to_datetime(window_start)].copy()

    # From rec_df, select only columns present in prev_keep or all if prev empty
    if prev_keep.empty:
        combined = rec_df.copy()
    else:
        # align columns
        common_cols = [c for c in rec_df.columns if c in prev_keep.columns]
        rec_sel = rec_df[common_cols].copy()
        combined = pd.concat([prev_keep, rec_sel], ignore_index=True)

    # Deduplicate by game_id keeping last
    if 'game_id' in combined.columns:
        combined['game_id'] = combined['game_id'].astype(str)
        combined = combined.drop_duplicates(subset=['game_id'], keep='last').reset_index(drop=True)

    try:
        combined.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Incremental features guardadas en: {OUTPUT_FILE} (rows={len(combined)})")
    except Exception as e:
        print(f"❌ Error guardando OUTPUT_FILE: {e}")


if __name__ == '__main__':
    window = 60
    if len(sys.argv) > 1:
        try:
            window = int(sys.argv[1])
        except Exception:
            pass
    main(window)
