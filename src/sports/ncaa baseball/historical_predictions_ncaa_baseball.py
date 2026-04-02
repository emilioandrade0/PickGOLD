from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import historical_predictions_mlb as mlb_hist

BASE_DIR = SRC_ROOT

mlb_hist.FEATURES_FILE = BASE_DIR / "data" / "ncaa_baseball" / "processed" / "model_ready_features_ncaa_baseball.csv"
mlb_hist.MODELS_DIR = BASE_DIR / "data" / "ncaa_baseball" / "models"
mlb_hist.HIST_DIR = BASE_DIR / "data" / "ncaa_baseball" / "historical_predictions"
mlb_hist.HIST_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    mlb_hist.main()
