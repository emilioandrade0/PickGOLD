import sys
import tempfile
from datetime import date
from pathlib import Path

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
RAW_UPCOMING_FILE = BASE_DIR / "data" / "triple_a" / "raw" / "triple_a_upcoming_schedule.csv"
RAW_HISTORY_FILE = BASE_DIR / "data" / "triple_a" / "raw" / "triple_a_advanced_history.csv"
PREDICTIONS_DIR = BASE_DIR / "data" / "triple_a" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

mlb_predict.FEATURES_FILE = BASE_DIR / "data" / "triple_a" / "processed" / "model_ready_features_triple_a.csv"
mlb_predict.MODELS_DIR = BASE_DIR / "data" / "triple_a" / "models"
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


def main() -> None:
    if not RAW_UPCOMING_FILE.exists():
        print(f"Triple-A: no existe agenda upcoming en {RAW_UPCOMING_FILE}")
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
        print("Triple-A: agenda vacia; se omiten predicciones futuras.")
        return

    features_file = mlb_predict.FEATURES_FILE
    if not features_file.exists():
        print(f"Triple-A: faltan features para predecir: {features_file}")
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
        print(f"Triple-A: generando predicciones para {len(board_df)} juegos")
        mlb_predict.main()
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
