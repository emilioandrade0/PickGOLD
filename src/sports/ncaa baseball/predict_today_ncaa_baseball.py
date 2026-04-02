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

    filtered_df = upcoming_df[upcoming_df.apply(_has_bettable_market, axis=1)].copy()
    kept_dates = set(filtered_df["date"].dropna().unique().tolist())
    removed_dates = sorted(all_dates - kept_dates)

    for date_str in removed_dates:
        stale_file = PREDICTIONS_DIR / f"{date_str}.json"
        if stale_file.exists():
            stale_file.unlink()

    if filtered_df.empty:
        _clear_prediction_jsons()
        print("NCAA Baseball: no hay juegos con mercado real; se omiten predicciones futuras.")
        return

    if not NCAA_FEATURES_FILE.exists():
        _clear_prediction_jsons()
        print(f"NCAA Baseball: faltan features para predecir: {NCAA_FEATURES_FILE}")
        return

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="") as tmp:
        temp_path = Path(tmp.name)
        filtered_df.to_csv(temp_path, index=False)

    try:
        mlb_predict.UPCOMING_FILE = temp_path
        print(
            f"NCAA Baseball bettable filter: {len(filtered_df)}/{len(upcoming_df)} juegos con mercado utilizable"
        )
        mlb_predict.main()
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
