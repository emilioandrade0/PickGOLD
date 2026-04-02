# src/audit_nhl_full_game.py

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


import sys
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
DATA_DIR = BASE_DIR / "data" / "nhl"
HIST_DIR = DATA_DIR / "historical_predictions"
REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, str) and not x.strip():
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, str) and not str(x).strip():
            return default
        return int(float(x))
    except Exception:
        return default


def normalize_team(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().upper()


def normalize_market(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()

    aliases = {
        "FULL_GAME": "FULL_GAME",
        "FULLGAME": "FULL_GAME",
        "MONEYLINE": "FULL_GAME",
        "ML": "FULL_GAME",
        "HOME_MONEYLINE": "FULL_GAME",
        "AWAY_MONEYLINE": "FULL_GAME",
        "GAME_WINNER": "FULL_GAME",
        "WINNER": "FULL_GAME",
    }
    return aliases.get(s, s)


def find_first(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def list_json_files_recursive(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(folder.rglob("*.json"))


def extract_prediction_list(payload: Any) -> List[dict]:
    """
    Intenta sacar una lista de picks desde varios esquemas posibles.
    """
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if not isinstance(payload, dict):
        return []

    candidate_keys = [
        "games",
        "predictions",
        "items",
        "data",
        "results",
        "historical_predictions",
        "bets",
        "picks",
    ]

    for key in candidate_keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]

    # Caso nested dict -> buscar una lista interna
    for _, value in payload.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return value

    return []


def load_historical_predictions_any_market() -> pd.DataFrame:
    files = list_json_files_recursive(HIST_DIR)

    print(f"📂 Carpeta auditada: {HIST_DIR}")
    print(f"📄 JSON encontrados (recursivo): {len(files)}")

    if not files:
        return pd.DataFrame()

    rows = []
    market_counter = {}
    opened_ok = 0
    extracted_lists = 0

    for fp in files:
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
            opened_ok += 1
        except Exception as e:
            print(f"⚠️ No se pudo leer: {fp.name} -> {e}")
            continue

        items = extract_prediction_list(payload)
        if items:
            extracted_lists += 1

        file_date = fp.stem

        for item in items:
            raw_market = find_first(
                item,
                ["market", "bet_type", "type", "prediction_type", "market_type"],
                "",
            )
            market = normalize_market(raw_market)
            market_counter[market] = market_counter.get(market, 0) + 1

            home_team = normalize_team(find_first(item, ["home_team", "team_home", "home"], ""))
            away_team = normalize_team(find_first(item, ["away_team", "team_away", "away"], ""))

            pick = str(find_first(item, ["pick", "prediction", "recommended_pick", "side"], "")).strip()
            pick_team = normalize_team(find_first(item, ["pick_team", "predicted_team", "team_pick"], ""))

            confidence = safe_float(find_first(item, ["confidence", "conf", "confidence_pct"], np.nan))
            probability = safe_float(find_first(item, ["probability", "prob", "predicted_probability"], np.nan))

            tier = str(find_first(item, ["tier", "label", "confidence_tier"], "")).strip().upper()

            correct = find_first(item, ["correct", "is_correct", "won"], None)
            if correct is None:
                actual_winner = normalize_team(find_first(item, ["actual_winner", "winner", "winning_team"], ""))
                if actual_winner and pick_team:
                    correct = int(actual_winner == pick_team)
                else:
                    correct = np.nan
            else:
                correct = safe_int(correct, default=np.nan)

            game_date = str(find_first(item, ["date", "game_date"], file_date))

            rows.append(
                {
                    "file_name": fp.name,
                    "file_path": str(fp),
                    "date": game_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "market": market,
                    "raw_market": raw_market,
                    "pick": pick,
                    "pick_team": pick_team,
                    "confidence": confidence,
                    "probability": probability,
                    "tier": tier,
                    "correct": correct,
                }
            )

    print(f"✅ JSON leídos correctamente: {opened_ok}")
    print(f"✅ JSON con listas extraíbles: {extracted_lists}")

    if market_counter:
        print("\n📊 Markets detectados:")
        for k, v in sorted(market_counter.items(), key=lambda x: (-x[1], x[0])):
            label = k if k else "(vacío)"
            print(f"   {label}: {v}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"])

    return df


def load_training_dataset_optional() -> Optional[pd.DataFrame]:
    try:
        from train_models_nhl import load_dataset
        df = load_dataset()
        if df is None or df.empty:
            return None
    except Exception:
        return None

    out = df.copy()

    date_col = next((c for c in ["date", "game_date", "GAME_DATE", "event_date"] if c in out.columns), None)
    home_col = next((c for c in ["home_team", "HOME_TEAM", "team_home"] if c in out.columns), None)
    away_col = next((c for c in ["away_team", "AWAY_TEAM", "team_away"] if c in out.columns), None)

    if date_col is None or home_col is None or away_col is None:
        return None

    out["date"] = pd.to_datetime(out[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    out["home_team"] = out[home_col].map(normalize_team)
    out["away_team"] = out[away_col].map(normalize_team)
    out = out.dropna(subset=["date"])

    return out


def summarize_accuracy(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    n = len(df)
    wins = int(df["correct"].fillna(0).sum()) if n else 0
    acc = float(df["correct"].mean()) if n else np.nan
    return {"segment": name, "n": n, "wins": wins, "accuracy": acc}


def confidence_bucket(x: float) -> str:
    if pd.isna(x):
        return "NA"
    if x < 54:
        return "<54"
    if x < 58:
        return "54-57.99"
    if x < 62:
        return "58-61.99"
    if x < 66:
        return "62-65.99"
    if x < 70:
        return "66-69.99"
    return "70+"


def probability_edge_bucket(x: float) -> str:
    if pd.isna(x):
        return "NA"
    edge = abs(x - 0.5)
    if edge < 0.02:
        return "0.00-0.02"
    if edge < 0.04:
        return "0.02-0.04"
    if edge < 0.06:
        return "0.04-0.06"
    if edge < 0.08:
        return "0.06-0.08"
    return "0.08+"


def make_group_report(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df.empty or group_col not in df.columns:
        return pd.DataFrame(columns=[group_col, "n", "wins", "accuracy"])

    out = (
        df.groupby(group_col, dropna=False)
        .agg(
            n=("correct", "size"),
            wins=("correct", "sum"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
        .sort_values(["n", "accuracy"], ascending=[False, False])
    )
    return out


def add_optional_context(preds: pd.DataFrame, train_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if train_df is None or train_df.empty or preds.empty:
        return preds

    merged = preds.merge(
        train_df,
        on=["date", "home_team", "away_team"],
        how="left",
        suffixes=("", "_train"),
    )

    def has_any(cols: List[str]) -> Optional[str]:
        for c in cols:
            if c in merged.columns:
                return c
        return None

    goalie_flag_col = has_any([
        "goalie_data_available",
        "both_goalies_available",
        "goalie_confirmed_flag",
        "goalie_available_flag",
        "fg_goalie_data_available",
    ])
    if goalie_flag_col is not None:
        merged["ctx_goalie_available"] = pd.to_numeric(merged[goalie_flag_col], errors="coerce").fillna(0)

    rest_diff_col = has_any([
        "diff_rest_days",
        "rest_days_diff",
        "fg_schedule_stress_gap",
    ])
    if rest_diff_col is not None:
        merged["ctx_rest_diff"] = pd.to_numeric(merged[rest_diff_col], errors="coerce")

    b2b_home_col = has_any(["home_back_to_back", "home_b2b", "is_home_b2b"])
    b2b_away_col = has_any(["away_back_to_back", "away_b2b", "is_away_b2b"])
    if b2b_home_col is not None and b2b_away_col is not None:
        merged["ctx_b2b_state"] = (
            pd.to_numeric(merged[b2b_home_col], errors="coerce").fillna(0).astype(int).astype(str)
            + "_"
            + pd.to_numeric(merged[b2b_away_col], errors="coerce").fillna(0).astype(int).astype(str)
        )

    strength_col = has_any([
        "diff_elo",
        "elo_diff",
        "fg_team_strength_gap_long",
    ])
    if strength_col is not None:
        merged["ctx_strength_gap"] = pd.to_numeric(merged[strength_col], errors="coerce")

    return merged


def main():
    all_preds = load_historical_predictions_any_market()

    if all_preds.empty:
        print("\n❌ Se encontraron JSON, pero no se pudo extraer ninguna predicción.")
        print("   Eso significa que la estructura real del JSON es distinta a la esperada.")
        return

    all_preds.to_csv(REPORTS_DIR / "nhl_all_detected_predictions_debug.csv", index=False, encoding="utf-8-sig")

    full = all_preds[all_preds["market"] == "FULL_GAME"].copy()

    if full.empty:
        print("\n❌ Sí se leyeron predicciones, pero ninguna fue reconocida como FULL_GAME.")
        print("   Revisa en reports/nhl_all_detected_predictions_debug.csv la columna raw_market.")
        return

    full = full.dropna(subset=["correct"]).copy()
    if full.empty:
        print("\n❌ Hay FULL_GAME detectado, pero ninguna fila trae resultado 'correct' usable.")
        print("   Revisa si el JSON guarda winner/actual_winner con otros nombres.")
        return

    full["is_premium"] = full["tier"].isin(["PREMIUM", "ELITE", "STRONG"])
    full["pick_side"] = np.where(
        full["pick_team"] == full["home_team"], "HOME",
        np.where(full["pick_team"] == full["away_team"], "AWAY", "UNKNOWN")
    )
    full["confidence_bucket"] = full["confidence"].map(confidence_bucket)
    full["prob_edge_bucket"] = full["probability"].map(probability_edge_bucket)

    train_df = load_training_dataset_optional()
    full = add_optional_context(full, train_df)

    summary_df = pd.DataFrame([
        summarize_accuracy(full, "overall"),
        summarize_accuracy(full[full["is_premium"]], "premium"),
        summarize_accuracy(full[full["pick_side"] == "HOME"], "pick_home"),
        summarize_accuracy(full[full["pick_side"] == "AWAY"], "pick_away"),
    ])

    conf_report = make_group_report(full, "confidence_bucket")
    prob_report = make_group_report(full, "prob_edge_bucket")
    side_report = make_group_report(full, "pick_side")
    raw_market_report = make_group_report(all_preds.assign(raw_market_clean=all_preds["raw_market"].astype(str)), "raw_market_clean")

    summary_df.to_csv(REPORTS_DIR / "nhl_full_game_summary.csv", index=False, encoding="utf-8-sig")
    conf_report.to_csv(REPORTS_DIR / "nhl_full_game_by_confidence.csv", index=False, encoding="utf-8-sig")
    prob_report.to_csv(REPORTS_DIR / "nhl_full_game_by_prob_edge.csv", index=False, encoding="utf-8-sig")
    side_report.to_csv(REPORTS_DIR / "nhl_full_game_by_side.csv", index=False, encoding="utf-8-sig")
    raw_market_report.to_csv(REPORTS_DIR / "nhl_detected_raw_markets.csv", index=False, encoding="utf-8-sig")

    if "ctx_goalie_available" in full.columns:
        goalie_report = make_group_report(
            full.assign(goalie_bucket=np.where(full["ctx_goalie_available"] >= 1, "goalie_available", "goalie_missing")),
            "goalie_bucket"
        )
        goalie_report.to_csv(REPORTS_DIR / "nhl_full_game_by_goalie_availability.csv", index=False, encoding="utf-8-sig")

    if "ctx_b2b_state" in full.columns:
        b2b_report = make_group_report(full, "ctx_b2b_state")
        b2b_report.to_csv(REPORTS_DIR / "nhl_full_game_by_b2b_state.csv", index=False, encoding="utf-8-sig")

    if "ctx_strength_gap" in full.columns:
        tmp = full.copy()
        tmp["strength_gap_q"] = pd.qcut(tmp["ctx_strength_gap"], 5, duplicates="drop")
        strength_report = make_group_report(tmp, "strength_gap_q")
        strength_report.to_csv(REPORTS_DIR / "nhl_full_game_by_strength_quintile.csv", index=False, encoding="utf-8-sig")

    if "ctx_rest_diff" in full.columns:
        tmp = full.copy()
        tmp["rest_diff_bucket"] = pd.cut(
            tmp["ctx_rest_diff"],
            bins=[-999, -2, -1, 0, 1, 2, 999],
            labels=["<=-2", "-1", "0", "1", "2", ">=3"],
        )
        rest_report = make_group_report(tmp, "rest_diff_bucket")
        rest_report.to_csv(REPORTS_DIR / "nhl_full_game_by_rest_diff.csv", index=False, encoding="utf-8-sig")

    print("\n=== NHL FULL_GAME AUDIT ===")
    print(summary_df.to_string(index=False))
    print("\n✅ Debug completo guardado en:")
    print(REPORTS_DIR)


if __name__ == "__main__":
    main()