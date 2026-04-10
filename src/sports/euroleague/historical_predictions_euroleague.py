from datetime import datetime
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd

from sports.nba import historical_predictions as nba_hist

BASE_DIR = SRC_ROOT

RAW_DATA = BASE_DIR / "data" / "euroleague" / "raw" / "euroleague_advanced_history.csv"
PROCESSED_DATA = BASE_DIR / "data" / "euroleague" / "processed" / "model_ready_features.csv"
MODELS_DIR = BASE_DIR / "data" / "euroleague" / "models"
HIST_PRED_DIR = BASE_DIR / "data" / "euroleague" / "historical_predictions"


def _resolve_range():
    if not RAW_DATA.exists():
        raise FileNotFoundError(f"No existe raw data: {RAW_DATA}")

    df = pd.read_csv(RAW_DATA, usecols=["date"])
    if df.empty:
        raise ValueError("Raw data de Euroliga esta vacia")

    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        raise ValueError("No se pudieron parsear fechas en raw data de Euroliga")

    start = dates.min().date()
    end_data = dates.max().date()
    end = min(end_data, datetime.now().date())

    return str(start), str(end)


def main():
    nba_hist.RAW_DATA = RAW_DATA
    nba_hist.PROCESSED_DATA = PROCESSED_DATA
    nba_hist.MODELS_DIR = MODELS_DIR
    nba_hist.HIST_PRED_DIR = HIST_PRED_DIR
    nba_hist.HIST_PRED_DIR.mkdir(parents=True, exist_ok=True)

    start_date, end_date = _resolve_range()
    print(f"📚 Rebuild historical EuroLeague: {start_date} -> {end_date}")
    nba_hist.generate_range(start_date, end_date)


if __name__ == "__main__":
    main()
