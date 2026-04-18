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
LINEUP_CACHE = CACHE_DIR / "lineup_strength.csv"


def main() -> None:
    if LINEUP_CACHE.exists():
        try:
            df = pd.read_csv(LINEUP_CACHE)
            required = {"date", "team", "lineup_strength"}
            if required.issubset(df.columns):
                print(f"LMB lineup cache ya existe: {LINEUP_CACHE}")
                return
        except Exception:
            pass

    pd.DataFrame(columns=["date", "team", "lineup_strength"]).to_csv(LINEUP_CACHE, index=False)
    print(f"LMB lineup cache inicializado: {LINEUP_CACHE}")


if __name__ == "__main__":
    main()
