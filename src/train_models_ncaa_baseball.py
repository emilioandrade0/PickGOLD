from pathlib import Path

import train_models_mlb as mlb_train

BASE_DIR = Path(__file__).resolve().parent

mlb_train.INPUT_FILE = BASE_DIR / "data" / "ncaa_baseball" / "processed" / "model_ready_features_ncaa_baseball.csv"
mlb_train.MODELS_DIR = BASE_DIR / "data" / "ncaa_baseball" / "models"
mlb_train.REPORTS_DIR = BASE_DIR / "data" / "ncaa_baseball" / "reports"
mlb_train.MODELS_DIR.mkdir(parents=True, exist_ok=True)
mlb_train.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    mlb_train.train_all_models()
