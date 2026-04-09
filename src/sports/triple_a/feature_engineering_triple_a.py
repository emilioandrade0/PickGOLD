from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from sports.mlb import feature_engineering_mlb_core as mlb_fe

BASE_DIR = SRC_ROOT

mlb_fe.RAW_DATA = BASE_DIR / "data" / "triple_a" / "raw" / "triple_a_advanced_history.csv"
mlb_fe.PROCESSED_DATA_DIR = BASE_DIR / "data" / "triple_a" / "processed"
mlb_fe.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
mlb_fe.OUTPUT_FILE = mlb_fe.PROCESSED_DATA_DIR / "model_ready_features_triple_a.csv"


if __name__ == "__main__":
    mlb_fe.build_features()
