from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
CACHE_DIR = BASE_DIR / "data" / "lmb" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
UMP_CACHE = CACHE_DIR / "umpire_stats.csv"


def main() -> None:
    if UMP_CACHE.exists():
        try:
            df = pd.read_csv(UMP_CACHE)
            required = {"umpire", "zone_rate"}
            if required.issubset(df.columns):
                print(f"LMB umpire cache ya existe: {UMP_CACHE}")
                return
        except Exception:
            pass

    pd.DataFrame(
        columns=[
            "umpire",
            "zone_rate",
            "games_sample",
            "consistency_rate",
            "favor_abs_mean",
            "weighted_score",
        ]
    ).to_csv(UMP_CACHE, index=False)
    print(f"LMB umpire cache inicializado: {UMP_CACHE}")


if __name__ == "__main__":
    main()
