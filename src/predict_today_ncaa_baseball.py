from pathlib import Path

import predict_today_mlb as mlb_predict

BASE_DIR = Path(__file__).resolve().parent

mlb_predict.FEATURES_FILE = BASE_DIR / "data" / "ncaa_baseball" / "processed" / "model_ready_features_ncaa_baseball.csv"
mlb_predict.UPCOMING_FILE = BASE_DIR / "data" / "ncaa_baseball" / "raw" / "ncaa_baseball_upcoming_schedule.csv"
mlb_predict.MODELS_DIR = BASE_DIR / "data" / "ncaa_baseball" / "models"
mlb_predict.PREDICTIONS_DIR = BASE_DIR / "data" / "ncaa_baseball" / "predictions"
mlb_predict.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
mlb_predict.CALIBRATION_FILE = mlb_predict.MODELS_DIR / "calibration_params.json"


if __name__ == "__main__":
    mlb_predict.main()
