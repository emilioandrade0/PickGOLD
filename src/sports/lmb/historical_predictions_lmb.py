from pathlib import Path
import sys
import json

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from sports.mlb import historical_predictions_mlb_walkforward as mlb_hist
from sports.mlb import train_models_mlb as mlb_train

BASE_DIR = SRC_ROOT

mlb_hist.FEATURES_FILE = BASE_DIR / "data" / "lmb" / "processed" / "model_ready_features_lmb.csv"
mlb_hist.MODELS_DIR = BASE_DIR / "data" / "lmb" / "models"
mlb_hist.HIST_DIR = BASE_DIR / "data" / "lmb" / "historical_predictions"
mlb_hist.HIST_DIR.mkdir(parents=True, exist_ok=True)
mlb_hist.OUTPUT_DIR = BASE_DIR / "data" / "lmb" / "walkforward"
mlb_hist.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# LMB inicia sin mercados de regresion hasta estabilizar lineas/odds.
mlb_hist.REGRESSION_TARGET_CONFIG = {}
# Ventanas mas compactas para ligas con menos historial.
mlb_hist.MIN_TRAIN_DATES = 28
mlb_hist.CALIBRATION_DATES = 7
mlb_hist.TEST_DATES = 1
mlb_hist.STEP_DATES = 3
mlb_train.INPUT_FILE = mlb_hist.FEATURES_FILE


def _tier_from_confidence(conf: float) -> str:
    if conf >= 0.65:
        return "ELITE"
    if conf >= 0.60:
        return "PREMIUM"
    return "NORMAL"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"game_id": str})


def _build_historical_jsons() -> int:
    walkforward_dir = mlb_hist.OUTPUT_DIR
    full_df = _safe_read_csv(walkforward_dir / "full_game" / "walkforward_predictions_detail.csv")
    yrfi_df = _safe_read_csv(walkforward_dir / "yrfi" / "walkforward_predictions_detail.csv")
    f5_df = _safe_read_csv(walkforward_dir / "f5" / "walkforward_predictions_detail.csv")
    if full_df.empty:
        return 0

    full_df = full_df.copy()
    full_df["pick_team"] = full_df.apply(
        lambda row: row["home_team"] if int(row.get("pred_label", 0)) == 1 else row["away_team"],
        axis=1,
    )
    full_df["full_game_hit"] = full_df.apply(
        lambda row: int(row.get("pred_label", 0)) == int(row.get("y_true", -1)),
        axis=1,
    )
    full_df["full_game_conf_pct"] = (full_df["confidence"].astype(float) * 100.0).round(1)
    full_df["full_game_tier"] = full_df["confidence"].astype(float).apply(_tier_from_confidence)

    yrfi_lookup = {}
    if not yrfi_df.empty:
        for _, row in yrfi_df.iterrows():
            gid = str(row.get("game_id"))
            yrfi_lookup[gid] = {
                "q1_pick": "YRFI" if int(row.get("pred_label", 0)) == 1 else "NRFI",
                "q1_hit": int(row.get("pred_label", 0)) == int(row.get("y_true", -1)),
                "q1_conf_pct": round(float(row.get("confidence", 0.0)) * 100.0, 1),
            }

    f5_lookup = {}
    if not f5_df.empty:
        for _, row in f5_df.iterrows():
            gid = str(row.get("game_id"))
            f5_lookup[gid] = {
                "f5_pick": row["home_team"] if int(row.get("pred_label", 0)) == 1 else row["away_team"],
                "correct_f5": int(row.get("pred_label", 0)) == int(row.get("y_true", -1)),
                "f5_conf_pct": round(float(row.get("confidence", 0.0)) * 100.0, 1),
            }

    by_date: dict[str, list[dict]] = {}
    for _, row in full_df.iterrows():
        date_value = str(row.get("date", ""))[:10]
        if not date_value:
            continue
        gid = str(row.get("game_id"))
        record = {
            "game_id": gid,
            "date": date_value,
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "full_game_pick": row.get("pick_team"),
            "recommended_pick": row.get("pick_team"),
            "full_game_tier": row.get("full_game_tier"),
            "confidence": row.get("full_game_conf_pct"),
            "correct_full_game_base": bool(row.get("full_game_hit")),
            "correct_full_game_adjusted": bool(row.get("full_game_hit")),
            "result_label": "ACIERTO" if bool(row.get("full_game_hit")) else "FALLO",
            "moneyline_odds": None,
            "pick_ml_odds": None,
            "odds_details": "No Line",
            "sport": "lmb",
            "league": "LMB",
        }
        if gid in yrfi_lookup:
            record.update(yrfi_lookup[gid])
        if gid in f5_lookup:
            record.update(f5_lookup[gid])
        by_date.setdefault(date_value, []).append(record)

    mlb_hist.HIST_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for date_value, rows in by_date.items():
        out_path = mlb_hist.HIST_DIR / f"{date_value}.json"
        out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        count += 1
    return count


if __name__ == "__main__":
    mlb_hist.main()
    files_built = _build_historical_jsons()
    print(f"LMB: snapshots historicos generados: {files_built}")
