from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import feature_engineering_mlb as mlb_fe

BASE_DIR = SRC_ROOT

mlb_fe.RAW_DATA = BASE_DIR / "data" / "ncaa_baseball" / "raw" / "ncaa_baseball_advanced_history.csv"
mlb_fe.PROCESSED_DATA_DIR = BASE_DIR / "data" / "ncaa_baseball" / "processed"
mlb_fe.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
mlb_fe.OUTPUT_FILE = mlb_fe.PROCESSED_DATA_DIR / "model_ready_features_ncaa_baseball.csv"


if __name__ == "__main__":
    mlb_fe.build_features()
