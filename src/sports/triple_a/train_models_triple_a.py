from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from sports.mlb import train_models_mlb as mlb_train

BASE_DIR = SRC_ROOT

mlb_train.INPUT_FILE = BASE_DIR / "data" / "triple_a" / "processed" / "model_ready_features_triple_a.csv"
mlb_train.MODELS_DIR = BASE_DIR / "data" / "triple_a" / "models"
mlb_train.REPORTS_DIR = BASE_DIR / "data" / "triple_a" / "reports"
mlb_train.MODELS_DIR.mkdir(parents=True, exist_ok=True)
mlb_train.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
# Triple-A arranca sin mercados de regresion hasta integrar lineas/odds estables.
mlb_train.REGRESSION_TARGET_CONFIG = {}


if __name__ == "__main__":
    mlb_train.train_all_models()
