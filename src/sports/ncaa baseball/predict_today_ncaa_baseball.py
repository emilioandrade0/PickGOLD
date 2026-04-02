import sys
import tempfile
from pathlib import Path

import pandas as pd

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
RAW_UPCOMING_FILE = BASE_DIR / "data" / "ncaa_baseball" / "raw" / "ncaa_baseball_upcoming_schedule.csv"
PREDICTIONS_DIR = BASE_DIR / "data" / "ncaa_baseball" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

mlb_predict.FEATURES_FILE = BASE_DIR / "data" / "ncaa_baseball" / "processed" / "model_ready_features_ncaa_baseball.csv"
mlb_predict.MODELS_DIR = BASE_DIR / "data" / "ncaa_baseball" / "models"
mlb_predict.PREDICTIONS_DIR = PREDICTIONS_DIR
mlb_predict.CALIBRATION_FILE = mlb_predict.MODELS_DIR / "calibration_params.json"
NCAA_FEATURES_FILE = mlb_predict.FEATURES_FILE


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

    # NCAA Baseball can still surface board picks even when regression sidecars were never trained.
    fallback = pd.Series(0.0, index=df.index, dtype=float).to_numpy()
    return fallback, {"fallback": True, "market_key": market_key}


mlb_predict.predict_regression_market = _safe_predict_regression_market


def _clear_prediction_jsons() -> None:
    for json_file in PREDICTIONS_DIR.glob("*.json"):
        try:
            json_file.unlink()
        except Exception:
            pass


def _has_bettable_market(row: pd.Series) -> bool:
    details = str(row.get("odds_details", "") or "").strip().lower()
    total_line = pd.to_numeric(row.get("odds_over_under", 0), errors="coerce")

    if pd.notna(total_line) and float(total_line) > 0:
        return True

    if details and details not in {"no line", "n/a", "nan", "none", ""}:
        return True

    return False


def main() -> None:
    if not RAW_UPCOMING_FILE.exists():
        _clear_prediction_jsons()
        print(f"NCAA Baseball: no existe agenda upcoming en {RAW_UPCOMING_FILE}")
        return

    upcoming_df = pd.read_csv(RAW_UPCOMING_FILE, dtype={"game_id": str})
    if upcoming_df.empty:
        _clear_prediction_jsons()
        print("NCAA Baseball: agenda vacia; se omiten predicciones futuras.")
        return

    upcoming_df["date"] = upcoming_df["date"].astype(str)
    all_dates = set(upcoming_df["date"].dropna().unique().tolist())

    bettable_df = upcoming_df[upcoming_df.apply(_has_bettable_market, axis=1)].copy()

    if not NCAA_FEATURES_FILE.exists():
        _clear_prediction_jsons()
        print(f"NCAA Baseball: faltan features para predecir: {NCAA_FEATURES_FILE}")
        return

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="") as tmp:
        temp_path = Path(tmp.name)
        upcoming_df.to_csv(temp_path, index=False)

    try:
        mlb_predict.UPCOMING_FILE = temp_path
        print(
            f"NCAA Baseball: generando predicciones para {len(upcoming_df)} juegos | con linea detectable: {len(bettable_df)}"
        )
        mlb_predict.main()
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
