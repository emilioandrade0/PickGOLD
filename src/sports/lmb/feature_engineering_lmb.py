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
CACHE_DIR = BASE_DIR / "data" / "lmb" / "cache"
WEATHER_DIR = CACHE_DIR / "weather"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
WEATHER_DIR.mkdir(parents=True, exist_ok=True)

mlb_fe.RAW_DATA = BASE_DIR / "data" / "lmb" / "raw" / "lmb_advanced_history.csv"
mlb_fe.PROCESSED_DATA_DIR = BASE_DIR / "data" / "lmb" / "processed"
mlb_fe.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
mlb_fe.OUTPUT_FILE = mlb_fe.PROCESSED_DATA_DIR / "model_ready_features_lmb.csv"

mlb_fe.UMPIRE_STATS_FILE = CACHE_DIR / "umpire_stats.csv"
mlb_fe.LINEUP_STRENGTH_FILE = CACHE_DIR / "lineup_strength.csv"
mlb_fe.LINE_MOVEMENT_FILE = CACHE_DIR / "line_movement.csv"
mlb_fe.PARK_FACTORS_FILE = CACHE_DIR / "park_factors.csv"
mlb_fe.WEATHER_CACHE_DIR = WEATHER_DIR
mlb_fe.WEATHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    mlb_fe.build_features()
