from pathlib import Path
import sys

SPORTS_ROOT = Path(__file__).resolve().parent.parent
if str(SPORTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SPORTS_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from evaluate_baseline_common import evaluate_for_sport


if __name__ == "__main__":
    evaluate_for_sport("triple_a")
