import json
import math
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

# --- RUTAS ---
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA = BASE_DIR / "data" / "raw" / "nba_advanced_history.csv"
PROCESSED_DATA = BASE_DIR / "data" / "processed" / "model_ready_features.csv"
MODELS_DIR = BASE_DIR / "models"
HIST_PRED_DIR = BASE_DIR / "data" / "historical_predictions"
HIST_PRED_DIR.mkdir(parents=True, exist_ok=True)

TEAM_CONF_DIV = {
    "ATL": ("EAST", "SOUTHEAST"), "BOS": ("EAST", "ATLANTIC"), "BKN": ("EAST", "ATLANTIC"),
    "CHA": ("EAST", "SOUTHEAST"), "CHI": ("EAST", "CENTRAL"), "CLE": ("EAST", "CENTRAL"),
    "DAL": ("WEST", "SOUTHWEST"), "DEN": ("WEST", "NORTHWEST"), "DET": ("EAST", "CENTRAL"),
    "GSW": ("WEST", "PACIFIC"), "HOU": ("WEST", "SOUTHWEST"), "IND": ("EAST", "CENTRAL"),
    "LAC": ("WEST", "PACIFIC"), "LAL": ("WEST", "PACIFIC"), "MEM": ("WEST", "SOUTHWEST"),
    "MIA": ("EAST", "SOUTHEAST"), "MIL": ("EAST", "CENTRAL"), "MIN": ("WEST", "NORTHWEST"),
    "NOP": ("WEST", "SOUTHWEST"), "NYK": ("EAST", "ATLANTIC"), "OKC": ("WEST", "NORTHWEST"),
    "ORL": ("EAST", "SOUTHEAST"), "PHI": ("EAST", "ATLANTIC"), "PHX": ("WEST", "PACIFIC"),
    "POR": ("WEST", "NORTHWEST"), "SAC": ("WEST", "PACIFIC"), "SAS": ("WEST", "SOUTHWEST"),
    "TOR": ("EAST", "ATLANTIC"), "UTA": ("WEST", "NORTHWEST"), "WAS": ("EAST", "SOUTHEAST"),
}

TEAM_TIMEZONE = {
    "ATL": "ET", "BOS": "ET", "BKN": "ET", "CHA": "ET", "CHI": "CT", "CLE": "ET",
    "DAL": "CT", "DEN": "MT", "DET": "ET", "GSW": "PT", "HOU": "CT", "IND": "ET",
    "LAC": "PT", "LAL": "PT", "MEM": "CT", "MIA": "ET", "MIL": "CT", "MIN": "CT",
    "NOP": "CT", "NYK": "ET", "OKC": "CT", "ORL": "ET", "PHI": "ET", "PHX": "MT",
    "POR": "PT", "SAC": "PT", "SAS": "CT", "TOR": "ET", "UTA": "MT", "WAS": "ET",
}

TZ_OFFSET = {"ET": 0, "CT": -1, "MT": -2, "PT": -3}


class ConstantBinaryModel:
    def __init__(self, prob_class_1: float):
        self.prob_class_1 = float(min(max(prob_class_1, 0.0), 1.0))

    def predict_proba(self, X):
        n = len(X)
        p1 = np.full(n, self.prob_class_1, dtype=float)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


def days_to_playoffs(game_date: pd.Timestamp) -> int:
    playoff_year = game_date.year + 1 if game_date.month >= 7 else game_date.year
    playoff_start = pd.Timestamp(playoff_year, 4, 15)
    return int(max((playoff_start - game_date).days, 0))


def add_context_features(df_predict: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    home_conf = df_predict["home_team"].map(lambda t: TEAM_CONF_DIV.get(str(t), ("UNK", "UNK"))[0])
    away_conf = df_predict["away_team"].map(lambda t: TEAM_CONF_DIV.get(str(t), ("UNK", "UNK"))[0])
    home_div = df_predict["home_team"].map(lambda t: TEAM_CONF_DIV.get(str(t), ("UNK", "UNK"))[1])
    away_div = df_predict["away_team"].map(lambda t: TEAM_CONF_DIV.get(str(t), ("UNK", "UNK"))[1])

    home_tz = df_predict["home_team"].map(lambda t: TEAM_TIMEZONE.get(str(t), "ET"))
    away_tz = df_predict["away_team"].map(lambda t: TEAM_TIMEZONE.get(str(t), "ET"))

    df_predict["same_conference"] = (home_conf == away_conf).astype(int)
    df_predict["same_division"] = (home_div == away_div).astype(int)
    df_predict["away_tz_diff"] = (home_tz.map(TZ_OFFSET) - away_tz.map(TZ_OFFSET)).abs().astype(int)
    df_predict["interconference_travel"] = (
        (df_predict["same_conference"] == 0) & (df_predict["away_tz_diff"] >= 2)
    ).astype(int)

    d2p = days_to_playoffs(pd.Timestamp(target_date))
    df_predict["days_to_playoffs"] = d2p
    df_predict["playoff_pressure"] = float(min(max(1 - (d2p / 120.0), 0), 1))

    return df_predict


def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
    return obj


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([float("inf"), float("-inf")], 0).fillna(0)


def get_pick_tier(conf: float) -> str:
    if conf >= 70:
        return "ELITE"
    if conf >= 65:
        return "PREMIUM"
    if conf >= 60:
        return "STRONG"
    if conf >= 57:
        return "NORMAL"
    return "PASS"


def load_pick_params(models_dir: Path) -> dict:
    default_params = {
        "game": {"xgb_weight": 0.5, "lgb_weight": 0.5, "threshold": 0.5},
        "q1": {"xgb_weight": 0.5, "lgb_weight": 0.5, "threshold": 0.5},
    }

    params_file = models_dir / "pick_params.pkl"
    if not params_file.exists():
        return default_params

    try:
        loaded = joblib.load(params_file)
        if not isinstance(loaded, dict):
            return default_params

        for key in ["game", "q1"]:
            if key not in loaded:
                loaded[key] = default_params[key]
            for p in ["xgb_weight", "lgb_weight", "threshold"]:
                if p not in loaded[key]:
                    loaded[key][p] = default_params[key][p]

        return loaded
    except Exception:
        return default_params


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
    home_df.columns = ["date", "game_id", "team", "pts_scored", "pts_conceded", "q1_scored", "q1_conceded"]
    home_df["is_home"] = 1

    away_df = df_history[
        ["date", "game_id", "away_team", "away_pts_total", "home_pts_total", "away_q1", "home_q1"]
    ].copy()
    away_df.columns = ["date", "game_id", "team", "pts_scored", "pts_conceded", "q1_scored", "q1_conceded"]
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

    return clean_dataframe(pd.DataFrame(latest_stats))


def get_matchup_stats(df_history: pd.DataFrame, home_team: str, away_team: str) -> dict:
    played = df_history[
        (
            ((df_history["home_team"] == home_team) & (df_history["away_team"] == away_team))
            | ((df_history["home_team"] == away_team) & (df_history["away_team"] == home_team))
        )
    ].copy()

    if played.empty:
        return {
            "matchup_home_win_pct_L5": 0.5,
            "matchup_home_win_pct_L10": 0.5,
            "matchup_home_q1_win_pct_L5": 0.5,
            "matchup_home_pts_diff_L5": 0.0,
        }

    played["date_dt"] = pd.to_datetime(played["date"])
    played = played.sort_values(["date_dt", "game_id"]).reset_index(drop=True)

    def _winner(r):
        if int(r["home_pts_total"]) > int(r["away_pts_total"]):
            return r["home_team"]
        if int(r["away_pts_total"]) > int(r["home_pts_total"]):
            return r["away_team"]
        return "TIE"

    def _q1_winner(r):
        if int(r["home_q1"]) > int(r["away_q1"]):
            return r["home_team"]
        if int(r["away_q1"]) > int(r["home_q1"]):
            return r["away_team"]
        return "TIE"

    played["full_winner"] = played.apply(_winner, axis=1)
    played["q1_winner"] = played.apply(_q1_winner, axis=1)
    played["home_team_pts_diff_from_current_home"] = np.where(
        played["home_team"] == home_team,
        played["home_pts_total"] - played["away_pts_total"],
        played["away_pts_total"] - played["home_pts_total"],
    )

    l5 = played.tail(5)
    l10 = played.tail(10)

    return {
        "matchup_home_win_pct_L5": float((l5["full_winner"] == home_team).mean()) if len(l5) else 0.5,
        "matchup_home_win_pct_L10": float((l10["full_winner"] == home_team).mean()) if len(l10) else 0.5,
        "matchup_home_q1_win_pct_L5": float((l5["q1_winner"] == home_team).mean()) if len(l5) else 0.5,
        "matchup_home_pts_diff_L5": float(l5["home_team_pts_diff_from_current_home"].mean()) if len(l5) else 0.0,
    }


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

    matchup_rows = []
    for _, row in df_predict.iterrows():
        matchup_rows.append(get_matchup_stats(train_history, row["home_team"], row["away_team"]))

    df_matchup = pd.DataFrame(matchup_rows)
    for col in df_matchup.columns:
        df_predict[col] = df_matchup[col].values

    df_predict["diff_matchup_home_edge_L5"] = (df_predict["matchup_home_win_pct_L5"] - 0.5) * 2
    df_predict["diff_matchup_home_edge_L10"] = (df_predict["matchup_home_win_pct_L10"] - 0.5) * 2

    league_win_pct_L10 = float(current_stats["win_pct_L10"].mean()) if not current_stats.empty else 0.5
    league_pts_diff_L10 = float(current_stats["pts_diff_L10"].mean()) if not current_stats.empty else 0.0
    league_q1_win_pct_L5 = float(current_stats["q1_win_pct_L5"].mean()) if not current_stats.empty else 0.5

    df_predict["home_win_pct_L10_vs_league"] = df_predict["home_win_pct_L10"] - league_win_pct_L10
    df_predict["away_win_pct_L10_vs_league"] = df_predict["away_win_pct_L10"] - league_win_pct_L10
    df_predict["diff_win_pct_L10_vs_league"] = (
        df_predict["home_win_pct_L10_vs_league"] - df_predict["away_win_pct_L10_vs_league"]
    )

    df_predict["home_pts_diff_L10_vs_league"] = df_predict["home_pts_diff_L10"] - league_pts_diff_L10
    df_predict["away_pts_diff_L10_vs_league"] = df_predict["away_pts_diff_L10"] - league_pts_diff_L10
    df_predict["diff_pts_diff_L10_vs_league"] = (
        df_predict["home_pts_diff_L10_vs_league"] - df_predict["away_pts_diff_L10_vs_league"]
    )

    df_predict["home_q1_win_pct_L5_vs_league"] = df_predict["home_q1_win_pct_L5"] - league_q1_win_pct_L5
    df_predict["away_q1_win_pct_L5_vs_league"] = df_predict["away_q1_win_pct_L5"] - league_q1_win_pct_L5
    df_predict["diff_q1_win_pct_L5_vs_league"] = (
        df_predict["home_q1_win_pct_L5_vs_league"] - df_predict["away_q1_win_pct_L5_vs_league"]
    )

    df_predict = add_context_features(df_predict, target_date)

    return clean_dataframe(df_predict)


def train_models_from_past(train_df: pd.DataFrame):
    cols_to_drop = [
        "game_id", "date", "season", "home_team", "away_team",
        "TARGET_home_win", "TARGET_home_win_q1"
    ]

    train_df = clean_dataframe(train_df)

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

    if y_game.nunique() < 2:
        fixed_game_prob = 1.0 if int(y_game.iloc[0]) == 1 else 0.0
        xgb_game = ConstantBinaryModel(fixed_game_prob)
        lgb_game = ConstantBinaryModel(fixed_game_prob)
    else:
        xgb_game.fit(X, y_game)
        lgb_game.fit(X, y_game)

    if y_q1.nunique() < 2:
        fixed_q1_prob = 1.0 if int(y_q1.iloc[0]) == 1 else 0.0
        xgb_q1 = ConstantBinaryModel(fixed_q1_prob)
        lgb_q1 = ConstantBinaryModel(fixed_q1_prob)
    else:
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
    if not RAW_DATA.exists() or not PROCESSED_DATA.exists():
        print("❌ Faltan archivos raw o processed.")
        return False

    df_raw = pd.read_csv(RAW_DATA)
    df_feat = pd.read_csv(PROCESSED_DATA)

    target_date = pd.Timestamp(target_date_str)
    raw_dates = pd.to_datetime(df_raw["date"])
    feat_dates = pd.to_datetime(df_feat["date"])

    train_history = df_raw[raw_dates < target_date].copy()
    target_games = df_raw[raw_dates == target_date].copy()
    train_features = df_feat[feat_dates < target_date].copy()

    if train_history.empty or target_games.empty or train_features.empty:
        print(f"⏭️ Saltando {target_date_str}: sin suficiente información")
        return False

    models = train_models_from_past(train_features)
    df_predict = build_prediction_features_for_games(train_history, target_games, target_date)

    X_pred = df_predict.reindex(columns=models["feature_names"], fill_value=0)
    X_pred = clean_dataframe(X_pred)

    pick_params = load_pick_params(MODELS_DIR)

    game_wx = float(pick_params["game"]["xgb_weight"])
    game_wl = float(pick_params["game"]["lgb_weight"])
    game_th = float(pick_params["game"]["threshold"])

    q1_wx = float(pick_params["q1"]["xgb_weight"])
    q1_wl = float(pick_params["q1"]["lgb_weight"])
    q1_th = float(pick_params["q1"]["threshold"])

    prob_game = (
        game_wx * models["xgb_game"].predict_proba(X_pred)[:, 1] +
        game_wl * models["lgb_game"].predict_proba(X_pred)[:, 1]
    )

    prob_q1 = (
        q1_wx * models["xgb_q1"].predict_proba(X_pred)[:, 1] +
        q1_wl * models["lgb_q1"].predict_proba(X_pred)[:, 1]
    )

    output = []

    for i, row in df_predict.iterrows():
        prob_game_home = float(prob_game[i] * 100)
        prob_game_away = float(100 - prob_game_home)
        full_pick = row["home_team"] if prob_game[i] >= game_th else row["away_team"]
        full_conf = float(max(prob_game_home, prob_game_away))

        prob_q1_home = float(prob_q1[i] * 100)
        prob_q1_away = float(100 - prob_q1_home)
        q1_pick = row["home_team"] if prob_q1[i] >= q1_th else row["away_team"]
        q1_conf = float(max(prob_q1_home, prob_q1_away))

        output.append({
            "game_id": str(row["game_id"]),
            "date": target_date_str,
            "time": "",
            "game_name": f"{row['away_team']} @ {row['home_team']}",
            "away_team": str(row["away_team"]),
            "home_team": str(row["home_team"]),
            "spread_market": str(row.get("odds_spread", "N/A")),
            "home_spread": float(row.get("home_spread", 0) or 0),
            "spread_abs": float(row.get("spread_abs", 0) or 0),
            "odds_over_under": float(row.get("odds_over_under", 0) or 0),
            "closing_moneyline_odds": row.get("closing_moneyline_odds"),
            "home_moneyline_odds": row.get("home_moneyline_odds"),
            "away_moneyline_odds": row.get("away_moneyline_odds"),
            "closing_spread_odds": row.get("closing_spread_odds"),
            "closing_total_odds": row.get("closing_total_odds"),
            "odds_data_quality": str(row.get("odds_data_quality", "fallback")),
            "market_missing": int(row.get("market_missing", 0) or 0),
            "full_game_pick": str(full_pick),
            "full_game_confidence": round(full_conf, 1),
            "full_game_tier": get_pick_tier(full_conf),
            "q1_pick": str(q1_pick),
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
        json.dump(clean_for_json(output), f, ensure_ascii=False, indent=2)

    print(f"✅ {target_date_str} -> {output_file.name}")
    return True


def generate_range(start_date: str, end_date: str):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if end < start:
        print("❌ end_date no puede ser menor que start_date")
        return

    current = start
    ok = 0
    skipped = 0

    while current <= end:
        success = predict_for_date(str(current.date()))
        if success:
            ok += 1
        else:
            skipped += 1
        current += pd.Timedelta(days=1)

    print(f"\n🏁 Rango terminado")
    print(f"   Generadas: {ok}")
    print(f"   Saltadas : {skipped}")


if __name__ == "__main__":
    # Ejemplo: temporada 2025-26 hasta marzo
    generate_range("2025-10-20", "2026-03-16")