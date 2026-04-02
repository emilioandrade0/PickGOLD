from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import train_models as nba_train

BASE_DIR = SRC_ROOT

nba_train.PROCESSED_DATA = BASE_DIR / "data" / "euroleague" / "processed" / "model_ready_features.csv"
nba_train.MODELS_DIR = BASE_DIR / "data" / "euroleague" / "models"
nba_train.MODELS_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    nba_train.train_all_models()
