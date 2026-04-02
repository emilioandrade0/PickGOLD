import csv
from pathlib import Path
from typing import Dict


def load_market_feature_sources(selective_plan_file: Path) -> Dict[str, str]:
    sources: Dict[str, str] = {
        "full_game": "baseline",
        "over_25": "baseline",
        "btts": "baseline",
        "corners_over_95": "baseline",
    }

    if not selective_plan_file.exists():
        return sources

    try:
        with open(selective_plan_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                market = str(row.get("market", "")).strip()
                source = str(row.get("selected_feature_source", "baseline")).strip().lower()
                apply_change = str(row.get("apply_change", "false")).strip().lower() == "true"
                if market and market in sources:
                    sources[market] = source if apply_change else "baseline"
    except Exception:
        return sources

    return sources


def resolve_market_model_dir(
    market_key: str,
    market_sources: Dict[str, str],
    baseline_models_dir: Path,
    selective_models_dir: Path,
) -> Path:
    source = market_sources.get(market_key, "baseline")
    if source == "v3":
        selective_market_dir = selective_models_dir / market_key
        if selective_market_dir.exists():
            return selective_market_dir
    return baseline_models_dir / market_key
