from pathlib import Path

import train_models as nba_train

BASE_DIR = Path(__file__).resolve().parent

nba_train.PROCESSED_DATA = BASE_DIR / "data" / "euroleague" / "processed" / "model_ready_features.csv"
nba_train.MODELS_DIR = BASE_DIR / "data" / "euroleague" / "models"
nba_train.MODELS_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    nba_train.train_all_models()
