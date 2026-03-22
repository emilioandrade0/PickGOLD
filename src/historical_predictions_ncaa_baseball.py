from pathlib import Path

import historical_predictions_mlb as mlb_hist

BASE_DIR = Path(__file__).resolve().parent

mlb_hist.FEATURES_FILE = BASE_DIR / "data" / "ncaa_baseball" / "processed" / "model_ready_features_ncaa_baseball.csv"
mlb_hist.MODELS_DIR = BASE_DIR / "data" / "ncaa_baseball" / "models"
mlb_hist.HIST_DIR = BASE_DIR / "data" / "ncaa_baseball" / "historical_predictions"
mlb_hist.HIST_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    mlb_hist.main()
