import sys
import tempfile
from datetime import date
from pathlib import Path
import re
import unicodedata

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
MLB_DIR = SRC_ROOT / "sports" / "mlb"
for p in (str(PROJECT_ROOT), str(SRC_ROOT), str(MLB_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from sports.mlb import predict_today_mlb as mlb_predict
except ImportError:
    from src.sports.mlb import predict_today_mlb as mlb_predict

BASE_DIR = SRC_ROOT
RAW_UPCOMING_FILE = BASE_DIR / "data" / "lmb" / "raw" / "lmb_upcoming_schedule.csv"
RAW_HISTORY_FILE = BASE_DIR / "data" / "lmb" / "raw" / "lmb_advanced_history.csv"
PREDICTIONS_DIR = BASE_DIR / "data" / "lmb" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

mlb_predict.FEATURES_FILE = BASE_DIR / "data" / "lmb" / "processed" / "model_ready_features_lmb.csv"
mlb_predict.MODELS_DIR = BASE_DIR / "data" / "lmb" / "models"
mlb_predict.PREDICTIONS_DIR = PREDICTIONS_DIR
mlb_predict.CALIBRATION_FILE = mlb_predict.MODELS_DIR / "calibration_params.json"

_original_predict_regression_market = mlb_predict.predict_regression_market


def _safe_predict_regression_market(df: pd.DataFrame, market_key: str):
    market_dir = mlb_predict.MODELS_DIR / market_key
    required = [
        market_dir / "xgb_model.pkl",
        market_dir / "lgbm_model.pkl",
        market_dir / "feature_columns.json",
        market_dir / "metadata.json",
    ]
    if all(path.exists() for path in required):
        return _original_predict_regression_market(df, market_key)
    fallback = pd.Series(0.0, index=df.index, dtype=float).to_numpy()
    return fallback, {"fallback": True, "market_key": market_key}


mlb_predict.predict_regression_market = _safe_predict_regression_market


LMB_TEAM_NAME_FALLBACKS = {
    "diablos rojos del mexico": "MEX",
    "piratas de campeche": "CAM",
    "caliente de durango": "DUR",
    "dorados de chihuahua": "CHI",
    "pericos de puebla": "PUE",
    "guerreros de oaxaca": "OAX",
    "el aguila de veracruz": "VER",
    "aguila de veracruz": "VER",
    "conspiradores de queretaro": "QRO",
    "rieleros de aguascalientes": "AGS",
    "tecos de los dos laredos": "LAR",
    "saraperos de saltillo": "SLT",
    "toros de tijuana": "TIJ",
    "algodoneros union laguna": "LAG",
    "algodoneros de union laguna": "LAG",
    "sultanes de monterrey": "MTY",
    "olmecas de tabasco": "TAB",
    "tigres de quintana roo": "TIG",
    "charros de jalisco": "JAL",
    "acereros del norte": "MVA",
    "leones de yucatan": "YUC",
    "bravos de leon": "LEO",
}


def _normalize_team_name(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_lmb_team_alias_map(history_df: pd.DataFrame, features_file: Path) -> dict[str, str]:
    alias_map: dict[str, str] = dict(LMB_TEAM_NAME_FALLBACKS)

    if history_df.empty or not features_file.exists():
        return alias_map
    required_cols = {"game_id", "home_team", "away_team"}
    if not required_cols.issubset(set(history_df.columns)):
        return alias_map

    try:
        feature_team_df = pd.read_csv(
            features_file,
            dtype={"game_id": str},
            usecols=["game_id", "home_team", "away_team"],
        )
    except Exception:
        return alias_map

    if feature_team_df.empty:
        return alias_map

    raw_pairs = pd.concat(
        [
            history_df[["game_id", "home_team"]].rename(columns={"home_team": "team_raw"}).assign(side="home"),
            history_df[["game_id", "away_team"]].rename(columns={"away_team": "team_raw"}).assign(side="away"),
        ],
        ignore_index=True,
    )
    feat_pairs = pd.concat(
        [
            feature_team_df[["game_id", "home_team"]].rename(columns={"home_team": "team_abbr"}).assign(side="home"),
            feature_team_df[["game_id", "away_team"]].rename(columns={"away_team": "team_abbr"}).assign(side="away"),
        ],
        ignore_index=True,
    )

    raw_pairs["game_id"] = raw_pairs["game_id"].astype(str).str.strip()
    feat_pairs["game_id"] = feat_pairs["game_id"].astype(str).str.strip()
    raw_pairs["team_raw"] = raw_pairs["team_raw"].astype(str).str.strip()
    feat_pairs["team_abbr"] = feat_pairs["team_abbr"].astype(str).str.strip().str.upper()

    merged = raw_pairs.merge(feat_pairs, on=["game_id", "side"], how="inner")
    if merged.empty:
        return alias_map

    merged["team_raw_norm"] = merged["team_raw"].map(_normalize_team_name)
    merged = merged[(merged["team_raw_norm"] != "") & (merged["team_abbr"] != "")]
    if merged.empty:
        return alias_map

    counts = (
        merged.groupby(["team_raw_norm", "team_abbr"]).size().rename("n").reset_index()
        .sort_values(["team_raw_norm", "n"], ascending=[True, False], kind="stable")
    )
    best = counts.drop_duplicates(subset=["team_raw_norm"], keep="first")
    inferred_map = dict(zip(best["team_raw_norm"], best["team_abbr"]))

    alias_map.update(inferred_map)
    return alias_map


def _apply_team_alias(value, alias_map: dict[str, str]):
    if pd.isna(value):
        return value
    text = str(value).strip()
    key = _normalize_team_name(text)
    return alias_map.get(key, text)


def main() -> None:
    if not RAW_UPCOMING_FILE.exists():
        print(f"LMB: no existe agenda upcoming en {RAW_UPCOMING_FILE}")
        return

    upcoming_df = pd.read_csv(RAW_UPCOMING_FILE, dtype={"game_id": str})
    history_df = (
        pd.read_csv(RAW_HISTORY_FILE, dtype={"game_id": str})
        if RAW_HISTORY_FILE.exists()
        else pd.DataFrame()
    )

    today_str = date.today().strftime("%Y-%m-%d")
    today_completed_df = (
        history_df.loc[history_df.get("date").astype(str) == today_str].copy()
        if not history_df.empty and "date" in history_df.columns
        else pd.DataFrame(columns=upcoming_df.columns)
    )

    board_df = pd.concat([upcoming_df, today_completed_df], ignore_index=True)
    board_df = board_df.drop_duplicates(subset=["game_id"], keep="last")
    if not board_df.empty:
        board_df = board_df.sort_values(["date", "time", "game_id"], kind="stable")

    if board_df.empty:
        print("LMB: agenda vacia; se omiten predicciones futuras.")
        return

    team_alias_map = _build_lmb_team_alias_map(history_df, mlb_predict.FEATURES_FILE)
    if team_alias_map:
        if "home_team" in board_df.columns:
            board_df["home_team"] = board_df["home_team"].map(lambda v: _apply_team_alias(v, team_alias_map))
        if "away_team" in board_df.columns:
            board_df["away_team"] = board_df["away_team"].map(lambda v: _apply_team_alias(v, team_alias_map))
        print(f"LMB: alias de equipos aplicados ({len(team_alias_map)} entradas)")

    features_file = mlb_predict.FEATURES_FILE
    if not features_file.exists():
        print(f"LMB: faltan features para predecir: {features_file}")
        return

    active_dates = {str(d) for d in board_df["date"].dropna().astype(str).unique()}
    for stale_file in PREDICTIONS_DIR.glob("*.json"):
        if stale_file.stem not in active_dates:
            stale_file.unlink(missing_ok=True)

    for date_str in active_dates:
        output_path = PREDICTIONS_DIR / f"{date_str}.json"
        if output_path.exists():
            output_path.unlink(missing_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="") as tmp:
        temp_path = Path(tmp.name)
        board_df.to_csv(temp_path, index=False)

    try:
        mlb_predict.UPCOMING_FILE = temp_path
        print(f"LMB: generando predicciones para {len(board_df)} juegos")
        mlb_predict.main()
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
