from pathlib import Path
import sys

SPORTS_ROOT = Path(__file__).resolve().parent.parent
if str(SPORTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SPORTS_ROOT))

from evaluate_baseline_common import evaluate_for_sport


if __name__ == "__main__":
    evaluate_for_sport("mlb")
