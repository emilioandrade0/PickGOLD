import json
import re
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
import xgboost as xgb

# --- RUTAS ---
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA = BASE_DIR / "data" / "raw" / "nba_advanced_history.csv"
PROCESSED_DATA = BASE_DIR / "data" / "processed" / "model_ready_features.csv"
HIST_PRED_DIR = BASE_DIR / "data" / "historical_predictions"
HIST_PRED_DIR.mkdir(parents=True, exist_ok=True)

ESPN_TO_NBA = {
    "GS": "GSW",
    "NY": "NYK",
    "SA": "SAS",
    "NO": "NOP",
    "WSH": "WAS",
    "UTAH": "UTA",
    "CHA": "CHA",
    "BKN": "BKN",
}


def get_pick_tier(conf: float) -> str:
    if conf >= 70:
        return "ELITE"
    elif conf >= 65:
        return "PREMIUM"
    elif conf >= 60:
        return "STRONG"
    elif conf >= 57:
        return "NORMAL"
    return "PASS"


def calculate_elo_map(df: pd.DataFrame, k: float = 20, home_advantage: float = 100):
    elo_dict = {}
    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date_dt", "game_id"]).reset_index(drop=True)

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        if home not in elo_dict:
            elo_dict[home] = 1500.0
        if away not in elo_dict:
            elo_dict[away] = 1500.0

        home_elo_pre = elo_dict[home]
        away_elo_pre = elo_dict[away]

        elo_diff = away_elo_pre - (home_elo_pre + home_advantage)
        expected_home = 1 / (1 + 10 ** (elo_diff / 400))
        expected_away = 1 - expected_home

        actual_home = 1 if row["home_pts_total"] > row["away_pts_total"] else 0
        actual_away = 1 - actual_home

        elo_dict[home] = home_elo_pre + k * (actual_home - expected_home)
        elo_dict[away] = away_elo_pre + k * (actual_away - expected_away)

    return elo_dict


def get_current_team_stats(df_history: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    home_df = df_history[
        ["date", "game_id", "home_team", "home_pts_total", "away_pts_total", "home_q1", "away_q1"]
    ].copy()
    home_df.columns = [
        "date", "game_id", "team", "pts_scored", "pts_conceded", "q1_scored", "q1_conceded"
    ]
    home_df["is_home"] = 1

    away_df = df_history[
        ["date", "game_id", "away_team", "away_pts_total", "home_pts_total", "away_q1", "home_q1"]
    ].copy()
    away_df.columns = [
        "date", "game_id", "team", "pts_scored", "pts_conceded", "q1_scored", "q1_conceded"
    ]
    away_df["is_home"] = 0

    all_stats = pd.concat([home_df, away_df], ignore_index=True)
    all_stats["date_dt"] = pd.to_datetime(all_stats["date"])
    all_stats = all_stats.sort_values(["team", "date_dt", "game_id"]).reset_index(drop=True)

    all_stats["won_game"] = (all_stats["pts_scored"] > all_stats["pts_conceded"]).astype(int)
    all_stats["pts_diff"] = all_stats["pts_scored"] - all_stats["pts_conceded"]
    all_stats["won_q1"] = (all_stats["q1_scored"] > all_stats["q1_conceded"]).astype(int)
    all_stats["q1_diff"] = all_stats["q1_scored"] - all_stats["q1_conceded"]

    latest_stats = []

    for team, group in all_stats.groupby("team"):
        group = group.sort_values(["date_dt", "game_id"]).reset_index(drop=True)
        l5 = group.tail(5)
        l10 = group.tail(10)

        if l10.empty:
            continue

        last_game_dt = group["date_dt"].iloc[-1]
        rest_days = min(max((target_date - last_game_dt).days, 0), 7)
        is_b2b = int(rest_days == 1)

        day_diffs = (target_date - group["date_dt"]).dt.days
        games_last_3_days = int(day_diffs.between(1, 3).sum())
        games_last_5_days = int(day_diffs.between(1, 5).sum())
        games_last_7_days = int(day_diffs.between(1, 7).sum())

        home_only = group[group["is_home"] == 1].tail(5)
        away_only = group[group["is_home"] == 0].tail(5)

        latest_stats.append(
            {
                "team": team,
                "rest_days": rest_days,
                "is_b2b": is_b2b,
                "games_last_3_days": games_last_3_days,
                "games_last_5_days": games_last_5_days,
                "games_last_7_days": games_last_7_days,
                "win_pct_L5": l5["won_game"].mean(),
                "win_pct_L10": l10["won_game"].mean(),
                "pts_diff_L5": l5["pts_diff"].mean(),
                "pts_diff_L10": l10["pts_diff"].mean(),
                "q1_win_pct_L5": l5["won_q1"].mean(),
                "q1_diff_L5": l5["q1_diff"].mean(),
                "pts_scored_L5": l5["pts_scored"].mean(),
                "pts_conceded_L5": l5["pts_conceded"].mean(),
                "home_only_win_pct_L5": home_only["won_game"].mean() if not home_only.empty else 0.0,
                "home_only_pts_diff_L5": home_only["pts_diff"].mean() if not home_only.empty else 0.0,
                "home_only_q1_win_pct_L5": home_only["won_q1"].mean() if not home_only.empty else 0.0,
                "away_only_win_pct_L5": away_only["won_game"].mean() if not away_only.empty else 0.0,
                "away_only_pts_diff_L5": away_only["pts_diff"].mean() if not away_only.empty else 0.0,
                "away_only_q1_win_pct_L5": away_only["won_q1"].mean() if not away_only.empty else 0.0,
            }
        )

    return pd.DataFrame(latest_stats)


def build_prediction_features_for_games(train_history: pd.DataFrame, target_games: pd.DataFrame, target_date: pd.Timestamp):
    elo_map = calculate_elo_map(train_history)
    current_stats = get_current_team_stats(train_history, target_date)

    df_predict = target_games.copy()
    df_predict["home_elo_pre"] = df_predict["home_team"].map(lambda t: elo_map.get(t, 1500.0))
    df_predict["away_elo_pre"] = df_predict["away_team"].map(lambda t: elo_map.get(t, 1500.0))

    df_predict = pd.merge(df_predict, current_stats, left_on="home_team", right_on="team", how="left")
    df_predict = df_predict.rename(columns={c: f"home_{c}" for c in current_stats.columns if c != "team"}).drop(columns=["team"])

    df_predict = pd.merge(df_predict, current_stats, left_on="away_team", right_on="team", how="left")
    df_predict = df_predict.rename(columns={c: f"away_{c}" for c in current_stats.columns if c != "team"}).drop(columns=["team"])

    fill_cols = [
        "home_rest_days", "away_rest_days",
        "home_is_b2b", "away_is_b2b",
        "home_games_last_3_days", "away_games_last_3_days",
        "home_games_last_5_days", "away_games_last_5_days",
        "home_games_last_7_days", "away_games_last_7_days",
        "home_win_pct_L5", "away_win_pct_L5",
        "home_win_pct_L10", "away_win_pct_L10",
        "home_pts_diff_L5", "away_pts_diff_L5",
        "home_pts_diff_L10", "away_pts_diff_L10",
        "home_q1_win_pct_L5", "away_q1_win_pct_L5",
        "home_q1_diff_L5", "away_q1_diff_L5",
        "home_pts_scored_L5", "away_pts_scored_L5",
        "home_pts_conceded_L5", "away_pts_conceded_L5",
        "home_home_only_win_pct_L5", "away_away_only_win_pct_L5",
        "home_home_only_pts_diff_L5", "away_away_only_pts_diff_L5",
        "home_home_only_q1_win_pct_L5", "away_away_only_q1_win_pct_L5",
    ]

    for col in fill_cols:
        if col not in df_predict.columns:
            df_predict[col] = 0.0
        else:
            df_predict[col] = df_predict[col].fillna(0.0)

    df_predict["diff_elo"] = df_predict["home_elo_pre"] - df_predict["away_elo_pre"]
    df_predict["diff_rest_days"] = df_predict["home_rest_days"] - df_predict["away_rest_days"]
    df_predict["diff_is_b2b"] = df_predict["home_is_b2b"] - df_predict["away_is_b2b"]
    df_predict["diff_games_last_3_days"] = df_predict["home_games_last_3_days"] - df_predict["away_games_last_3_days"]
    df_predict["diff_games_last_5_days"] = df_predict["home_games_last_5_days"] - df_predict["away_games_last_5_days"]
    df_predict["diff_games_last_7_days"] = df_predict["home_games_last_7_days"] - df_predict["away_games_last_7_days"]
    df_predict["diff_win_pct_L5"] = df_predict["home_win_pct_L5"] - df_predict["away_win_pct_L5"]
    df_predict["diff_win_pct_L10"] = df_predict["home_win_pct_L10"] - df_predict["away_win_pct_L10"]
    df_predict["diff_pts_diff_L5"] = df_predict["home_pts_diff_L5"] - df_predict["away_pts_diff_L5"]
    df_predict["diff_pts_diff_L10"] = df_predict["home_pts_diff_L10"] - df_predict["away_pts_diff_L10"]
    df_predict["diff_q1_win_pct_L5"] = df_predict["home_q1_win_pct_L5"] - df_predict["away_q1_win_pct_L5"]
    df_predict["diff_q1_diff_L5"] = df_predict["home_q1_diff_L5"] - df_predict["away_q1_diff_L5"]
    df_predict["diff_pts_scored_L5"] = df_predict["home_pts_scored_L5"] - df_predict["away_pts_scored_L5"]
    df_predict["diff_pts_conceded_L5"] = df_predict["home_pts_conceded_L5"] - df_predict["away_pts_conceded_L5"]
    df_predict["diff_surface_win_pct_L5"] = df_predict["home_home_only_win_pct_L5"] - df_predict["away_away_only_win_pct_L5"]
    df_predict["diff_surface_pts_diff_L5"] = df_predict["home_home_only_pts_diff_L5"] - df_predict["away_away_only_pts_diff_L5"]
    df_predict["diff_surface_q1_win_pct_L5"] = df_predict["home_home_only_q1_win_pct_L5"] - df_predict["away_away_only_q1_win_pct_L5"]

    return df_predict


def train_models_from_past(train_df: pd.DataFrame):
    cols_to_drop = [
        "game_id", "date", "season", "home_team", "away_team",
        "TARGET_home_win", "TARGET_home_win_q1"
    ]
    X = train_df.drop(columns=cols_to_drop)
    y_game = train_df["TARGET_home_win"].astype(int)
    y_q1 = train_df["TARGET_home_win_q1"].astype(int)

    xgb_game = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=4,
        subsample=0.85, colsample_bytree=0.85,
        eval_metric="logloss", random_state=42
    )
    lgb_game = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=4,
        num_leaves=31, subsample=0.85, colsample_bytree=0.85,
        random_state=42, verbosity=-1
    )

    xgb_q1 = xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        subsample=0.85, colsample_bytree=0.85,
        eval_metric="logloss", random_state=42
    )
    lgb_q1 = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        num_leaves=15, subsample=0.85, colsample_bytree=0.85,
        random_state=42, verbosity=-1
    )

    xgb_game.fit(X, y_game)
    lgb_game.fit(X, y_game)
    xgb_q1.fit(X, y_q1)
    lgb_q1.fit(X, y_q1)

    return {
        "xgb_game": xgb_game,
        "lgb_game": lgb_game,
        "xgb_q1": xgb_q1,
        "lgb_q1": lgb_q1,
        "feature_names": X.columns.tolist(),
    }


def predict_for_date(target_date_str: str):
    df_raw = pd.read_csv(RAW_DATA)
    df_feat = pd.read_csv(PROCESSED_DATA)

    target_date = pd.Timestamp(target_date_str)

    train_history = df_raw[pd.to_datetime(df_raw["date"]) < target_date].copy()
    target_games = df_raw[pd.to_datetime(df_raw["date"]) == target_date].copy()
    train_features = df_feat[pd.to_datetime(df_feat["date"]) < target_date].copy()

    if train_history.empty or target_games.empty or train_features.empty:
        print(f"❌ No hay suficiente información para reconstruir {target_date_str}")
        return

    models = train_models_from_past(train_features)
    df_predict = build_prediction_features_for_games(train_history, target_games, target_date)
    X_pred = df_predict.reindex(columns=models["feature_names"], fill_value=0)

    prob_game = (
        models["xgb_game"].predict_proba(X_pred)[:, 1] +
        models["lgb_game"].predict_proba(X_pred)[:, 1]
    ) / 2

    prob_q1 = (
        models["xgb_q1"].predict_proba(X_pred)[:, 1] +
        models["lgb_q1"].predict_proba(X_pred)[:, 1]
    ) / 2

    output = []

    for i, row in df_predict.iterrows():
        prob_game_home = prob_game[i] * 100
        prob_game_away = 100 - prob_game_home
        full_pick = row["home_team"] if prob_game_home >= 50 else row["away_team"]
        full_conf = max(prob_game_home, prob_game_away)

        prob_q1_home = prob_q1[i] * 100
        prob_q1_away = 100 - prob_q1_home
        q1_pick = row["home_team"] if prob_q1_home >= 50 else row["away_team"]
        q1_conf = max(prob_q1_home, prob_q1_away)

        output.append({
            "game_id": str(row["game_id"]),
            "date": target_date_str,
            "time": "",
            "game_name": f"{row['away_team']} @ {row['home_team']}",
            "away_team": row["away_team"],
            "home_team": row["home_team"],
            "spread_market": row.get("odds_spread", "N/A"),
            "home_spread": float(row.get("home_spread", 0) or 0),
            "spread_abs": float(row.get("spread_abs", 0) or 0),
            "odds_over_under": float(row.get("odds_over_under", 0) or 0),
            "market_missing": int(0),
            "full_game_pick": full_pick,
            "full_game_confidence": round(full_conf, 1),
            "full_game_tier": get_pick_tier(full_conf),
            "q1_pick": q1_pick,
            "q1_confidence": round(q1_conf, 1),
            "q1_action": "JUGAR Q1" if q1_conf >= 62 else "PASAR Q1",
            "total_pick": "Reconstruido",
            "spread_pick": "Reconstruido",
            "assists_pick": "Reconstruido",
            "prediction_mode": "historical_rebuild",
            "trained_until": str((target_date - pd.Timedelta(days=1)).date()),
        })

    output_file = HIST_PRED_DIR / f"{target_date_str}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ Predicciones históricas guardadas en: {output_file}")


if __name__ == "__main__":
    # Cambia esta fecha cuando quieras probar otra
    predict_for_date("2026-03-10")