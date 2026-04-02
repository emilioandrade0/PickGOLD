from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
RAW_HISTORY_FILE = BASE_DIR / "data" / "tennis" / "raw" / "tennis_advanced_history.csv"
PROCESSED_DIR = BASE_DIR / "data" / "tennis" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = PROCESSED_DIR / "model_ready_features_tennis.csv"

from sports.tennis.tennis_features import OUTPUT_COLUMNS, build_history_features


def build_features() -> pd.DataFrame:
    if not RAW_HISTORY_FILE.exists():
        df = pd.DataFrame(columns=OUTPUT_COLUMNS)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Tennis features: no existe historico. Se creo archivo vacio en {OUTPUT_FILE}")
        return df

    raw = pd.read_csv(RAW_HISTORY_FILE, dtype={"match_id": str})
    df = build_history_features(raw)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Tennis features generadas: {len(df)} filas -> {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    build_features()
