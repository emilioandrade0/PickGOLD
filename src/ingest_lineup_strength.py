from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sports.mlb.ingest_lineup_strength import *  # noqa: F401,F403
