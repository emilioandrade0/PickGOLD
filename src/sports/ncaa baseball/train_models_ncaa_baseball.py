from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sports.mlb import train_models_mlb as mlb_train

BASE_DIR = SRC_ROOT

mlb_train.INPUT_FILE = BASE_DIR / "data" / "ncaa_baseball" / "processed" / "model_ready_features_ncaa_baseball.csv"
mlb_train.MODELS_DIR = BASE_DIR / "data" / "ncaa_baseball" / "models"
mlb_train.REPORTS_DIR = BASE_DIR / "data" / "ncaa_baseball" / "reports"
mlb_train.MODELS_DIR.mkdir(parents=True, exist_ok=True)
mlb_train.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    mlb_train.train_all_models()
