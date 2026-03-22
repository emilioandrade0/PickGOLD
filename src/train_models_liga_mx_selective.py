import json
import shutil
from pathlib import Path

import pandas as pd

import train_models_liga_mx as baseline_train
from liga_mx_market_source import load_market_feature_sources

BASE_DIR = Path(__file__).resolve().parent
BASELINE_FEATURES_FILE = BASE_DIR / "data" / "liga_mx" / "processed" / "model_ready_features_liga_mx.csv"
V3_FEATURES_FILE = BASE_DIR / "data" / "liga_mx" / "processed" / "model_ready_features_liga_mx_v3.csv"
BASELINE_MODELS_DIR = BASE_DIR / "data" / "liga_mx" / "models"
SELECTIVE_MODELS_DIR = BASE_DIR / "data" / "liga_mx" / "models_selective"
REPORTS_DIR = BASE_DIR / "data" / "liga_mx" / "reports"
SELECTIVE_PLAN_FILE = REPORTS_DIR / "liga_mx_selective_upgrade_plan.csv"
SELECTIVE_SUMMARY_FILE = REPORTS_DIR / "liga_mx_selective_models_training_summary.json"


def _copy_market_dir(market_key: str) -> dict:
    src = BASELINE_MODELS_DIR / market_key
    dst = SELECTIVE_MODELS_DIR / market_key

    if not src.exists():
        raise FileNotFoundError(f"Missing baseline market models: {src}")

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    return {
        "market": market_key,
        "source": "baseline_copy",
        "path": str(dst),
    }


def _train_market_from_file(market_key: str, features_file: Path) -> dict:
    if not features_file.exists():
        raise FileNotFoundError(f"Missing features file: {features_file}")

    df = pd.read_csv(features_file, dtype={"game_id": str})
    df["date"] = df["date"].astype(str)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt"]).sort_values(["date_dt", "game_id"]).reset_index(drop=True)

    feature_cols = baseline_train.get_feature_columns(df)
    market_cfg = baseline_train.TARGET_CONFIG[market_key]

    result = baseline_train.train_single_market(
        df=df,
        market_key=market_key,
        target_col=market_cfg["target_col"],
        feature_cols=feature_cols,
        problem_type=market_cfg["problem_type"],
    )

    result["source"] = "v3_train"
    result["features_file"] = str(features_file)
    return result


def main() -> None:
    print("=" * 72)
    print("LIGA MX SELECTIVE MODELS TRAINING")
    print("=" * 72)

    SELECTIVE_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    market_sources = load_market_feature_sources(SELECTIVE_PLAN_FILE)
    print(f"[INFO] Selective market sources: {market_sources}")

    # Route model outputs to selective directory.
    baseline_train.MODELS_DIR = SELECTIVE_MODELS_DIR
    baseline_train.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}
    for market_key in ["full_game", "over_25", "btts", "corners_over_95"]:
        source = market_sources.get(market_key, "baseline")
        if source == "v3" and market_key in {"over_25", "btts", "full_game"}:
            print(f"\n[TRAIN] {market_key} from v3 features")
            summary[market_key] = _train_market_from_file(market_key, V3_FEATURES_FILE)
        else:
            print(f"\n[COPY] {market_key} from baseline models")
            summary[market_key] = _copy_market_dir(market_key)

    with open(SELECTIVE_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Selective models ready at: {SELECTIVE_MODELS_DIR}")
    print(f"[OK] Summary saved: {SELECTIVE_SUMMARY_FILE}")


if __name__ == "__main__":
    main()
