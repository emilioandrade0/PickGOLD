from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sports.nba import feature_engineering as nba_fe

BASE_DIR = SRC_ROOT

nba_fe.RAW_DATA = BASE_DIR / "data" / "euroleague" / "raw" / "euroleague_advanced_history.csv"
nba_fe.PROCESSED_DATA_DIR = BASE_DIR / "data" / "euroleague" / "processed"
nba_fe.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
nba_fe.OUTPUT_FILE = nba_fe.PROCESSED_DATA_DIR / "model_ready_features.csv"


if __name__ == "__main__":
    nba_fe.build_features()
