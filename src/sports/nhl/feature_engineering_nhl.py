import numpy as np
import pandas as pd
from pathlib import Path
import unicodedata

import sys
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
RAW_DATA = BASE_DIR / "data" / "nhl" / "raw" / "nhl_advanced_history.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "nhl" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = PROCESSED_DATA_DIR / "model_ready_features_nhl.csv"


def normalize_text(text: str) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ASCII", "ignore").decode("ASCII")
    return text.strip().upper()


def build_goalie_key(goalie_id, goalie_name) -> str:
    goalie_id = "" if pd.isna(goalie_id) else str(goalie_id).strip()
    goalie_name = normalize_text(goalie_name)
    if goalie_id:
        return f"ID_{goalie_id}"
    if goalie_name:
        return f"NAME_{goalie_name}"
    return ""


def calculate_elo_ratings(df: pd.DataFrame, k: float = 25, home_advantage: float = 50) -> pd.DataFrame:
    print("📈 Calculating ELO ratings (no leakage)...")

    elo_dict = {}
    elo_home_before = []
    elo_away_before = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        if home not in elo_dict:
            elo_dict[home] = 1500.0
        if away not in elo_dict:
            elo_dict[away] = 1500.0

        home_elo_pre = elo_dict[home]
        away_elo_pre = elo_dict[away]

        elo_home_before.append(home_elo_pre)
        elo_away_before.append(away_elo_pre)

        elo_diff = away_elo_pre - (home_elo_pre + home_advantage)
        expected_home = 1 / (1 + 10 ** (elo_diff / 400))
        expected_away = 1 - expected_home

        if row["home_score"] > row["away_score"]:
            actual_home = 1
            actual_away = 0
        else:
            actual_home = 0
            actual_away = 1

        elo_dict[home] = home_elo_pre + k * (actual_home - expected_home)
        elo_dict[away] = away_elo_pre + k * (actual_away - expected_away)

    df["home_elo_pre"] = elo_home_before
    df["away_elo_pre"] = elo_away_before
    return df


def calculate_team_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    print("⚙️ Calculating team rolling features (with .shift(1))...")

    home_df = df[["date", "game_id", "home_team", "home_score", "away_score", "home_p1_goals", "away_p1_goals"]].copy()
    home_df.columns = ["date", "game_id", "team", "goals_scored", "goals_allowed", "goals_scored_p1", "goals_allowed_p1"]
    home_df["is_home"] = 1

    away_df = df[["date", "game_id", "away_team", "away_score", "home_score", "away_p1_goals", "home_p1_goals"]].copy()
    away_df.columns = ["date", "game_id", "team", "goals_scored", "goals_allowed", "goals_scored_p1", "goals_allowed_p1"]
    away_df["is_home"] = 0

    team_df = pd.concat([home_df, away_df], ignore_index=True)
    team_df["date_dt"] = pd.to_datetime(team_df["date"], errors="coerce")
    team_df = team_df.sort_values(["team", "date_dt", "game_id"]).reset_index(drop=True)

    team_df["goals_scored_l5"] = team_df.groupby("team")["goals_scored"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    team_df["goals_allowed_l5"] = team_df.groupby("team")["goals_allowed"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    team_df["goals_scored_l10"] = team_df.groupby("team")["goals_scored"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )
    team_df["goals_allowed_l10"] = team_df.groupby("team")["goals_allowed"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )
    team_df["goals_scored_p1_l5"] = team_df.groupby("team")["goals_scored_p1"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    team_df["goals_allowed_p1_l5"] = team_df.groupby("team")["goals_allowed_p1"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    team_df["goals_scored_p1_l10"] = team_df.groupby("team")["goals_scored_p1"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )
    team_df["goals_allowed_p1_l10"] = team_df.groupby("team")["goals_allowed_p1"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )

    team_df["goal_diff"] = team_df["goals_scored"] - team_df["goals_allowed"]
    team_df["won"] = (team_df["goal_diff"] > 0).astype(int)

    team_df["win_rate_l5"] = team_df.groupby("team")["won"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    team_df["win_rate_l10"] = team_df.groupby("team")["won"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )

    team_df["goal_diff_l5"] = team_df.groupby("team")["goal_diff"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    team_df["goal_diff_l10"] = team_df.groupby("team")["goal_diff"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )

    # volatilidad reciente
    team_df["goal_diff_std_l5"] = team_df.groupby("team")["goal_diff"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).std()
    )
    team_df["goals_scored_std_l5"] = team_df.groupby("team")["goals_scored"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).std()
    )
    team_df["goals_allowed_std_l5"] = team_df.groupby("team")["goals_allowed"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).std()
    )

    team_df["last_game_date"] = team_df.groupby("team")["date_dt"].shift(1)
    team_df["rest_days"] = (team_df["date_dt"] - team_df["last_game_date"]).dt.days.fillna(2)
    team_df["rest_days"] = team_df["rest_days"].clip(lower=1, upper=14)
    team_df["is_b2b"] = (team_df["rest_days"] <= 1).astype(int)

    team_df["games_last_3"] = team_df.groupby("team")["date_dt"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).count()
    )
    team_df["games_last_5"] = team_df.groupby("team")["date_dt"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).count()
    )

    # splits home / away
    team_df["goals_scored_home_l5"] = np.nan
    team_df["goals_allowed_home_l5"] = np.nan
    team_df["win_rate_home_l5"] = np.nan

    team_df["goals_scored_away_l5"] = np.nan
    team_df["goals_allowed_away_l5"] = np.nan
    team_df["win_rate_away_l5"] = np.nan

    for team, group_idx in team_df.groupby("team").groups.items():
        g = team_df.loc[group_idx].copy()

        home_mask = g["is_home"] == 1
        away_mask = g["is_home"] == 0

        g.loc[home_mask, "goals_scored_home_l5"] = (
            g.loc[home_mask, "goals_scored"].shift(1).rolling(5, min_periods=1).mean()
        )
        g.loc[home_mask, "goals_allowed_home_l5"] = (
            g.loc[home_mask, "goals_allowed"].shift(1).rolling(5, min_periods=1).mean()
        )
        g.loc[home_mask, "win_rate_home_l5"] = (
            g.loc[home_mask, "won"].shift(1).rolling(5, min_periods=1).mean()
        )

        g.loc[away_mask, "goals_scored_away_l5"] = (
            g.loc[away_mask, "goals_scored"].shift(1).rolling(5, min_periods=1).mean()
        )
        g.loc[away_mask, "goals_allowed_away_l5"] = (
            g.loc[away_mask, "goals_allowed"].shift(1).rolling(5, min_periods=1).mean()
        )
        g.loc[away_mask, "win_rate_away_l5"] = (
            g.loc[away_mask, "won"].shift(1).rolling(5, min_periods=1).mean()
        )

        team_df.loc[group_idx, "goals_scored_home_l5"] = g["goals_scored_home_l5"].values
        team_df.loc[group_idx, "goals_allowed_home_l5"] = g["goals_allowed_home_l5"].values
        team_df.loc[group_idx, "win_rate_home_l5"] = g["win_rate_home_l5"].values

        team_df.loc[group_idx, "goals_scored_away_l5"] = g["goals_scored_away_l5"].values
        team_df.loc[group_idx, "goals_allowed_away_l5"] = g["goals_allowed_away_l5"].values
        team_df.loc[group_idx, "win_rate_away_l5"] = g["win_rate_away_l5"].values

    home_features = team_df[team_df["is_home"] == 1][[
        "game_id",
        "goals_scored_l5", "goals_allowed_l5",
        "goals_scored_l10", "goals_allowed_l10",
        "win_rate_l5", "win_rate_l10",
        "goal_diff_l5", "goal_diff_l10",
        "goals_scored_p1_l5", "goals_allowed_p1_l5",
        "goals_scored_p1_l10", "goals_allowed_p1_l10",
        "goal_diff_std_l5", "goals_scored_std_l5", "goals_allowed_std_l5",
        "rest_days", "is_b2b", "games_last_3", "games_last_5",
        "goals_scored_home_l5", "goals_allowed_home_l5", "win_rate_home_l5",
    ]].copy()
    home_features.columns = [
        "game_id",
        "home_goals_scored_l5", "home_goals_allowed_l5",
        "home_goals_scored_l10", "home_goals_allowed_l10",
        "home_win_rate_l5", "home_win_rate_l10",
        "home_goal_diff_l5", "home_goal_diff_l10",
        "home_goals_scored_p1_l5", "home_goals_allowed_p1_l5",
        "home_goals_scored_p1_l10", "home_goals_allowed_p1_l10",
        "home_goal_diff_std_l5", "home_goals_scored_std_l5", "home_goals_allowed_std_l5",
        "home_rest_days", "home_is_b2b", "home_games_last_3", "home_games_last_5",
        "home_goals_scored_home_l5", "home_goals_allowed_home_l5", "home_win_rate_home_l5",
    ]

    away_features = team_df[team_df["is_home"] == 0][[
        "game_id",
        "goals_scored_l5", "goals_allowed_l5",
        "goals_scored_l10", "goals_allowed_l10",
        "win_rate_l5", "win_rate_l10",
        "goal_diff_l5", "goal_diff_l10",
        "goals_scored_p1_l5", "goals_allowed_p1_l5",
        "goals_scored_p1_l10", "goals_allowed_p1_l10",
        "goal_diff_std_l5", "goals_scored_std_l5", "goals_allowed_std_l5",
        "rest_days", "is_b2b", "games_last_3", "games_last_5",
        "goals_scored_away_l5", "goals_allowed_away_l5", "win_rate_away_l5",
    ]].copy()
    away_features.columns = [
        "game_id",
        "away_goals_scored_l5", "away_goals_allowed_l5",
        "away_goals_scored_l10", "away_goals_allowed_l10",
        "away_win_rate_l5", "away_win_rate_l10",
        "away_goal_diff_l5", "away_goal_diff_l10",
        "away_goals_scored_p1_l5", "away_goals_allowed_p1_l5",
        "away_goals_scored_p1_l10", "away_goals_allowed_p1_l10",
        "away_goal_diff_std_l5", "away_goals_scored_std_l5", "away_goals_allowed_std_l5",
        "away_rest_days", "away_is_b2b", "away_games_last_3", "away_games_last_5",
        "away_goals_scored_away_l5", "away_goals_allowed_away_l5", "away_win_rate_away_l5",
    ]

    df = df.merge(home_features, on="game_id", how="left")
    df = df.merge(away_features, on="game_id", how="left")

    return df


def calculate_goalie_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    print("🥅 Calculating goalie rolling features (with .shift(1))...")

    work_df = df.copy()
    work_df["date_dt"] = pd.to_datetime(work_df["date"], errors="coerce")

    home_goalie = work_df[[
        "game_id", "date", "date_dt", "home_team", "home_goalie_name", "home_goalie_id",
        "home_goalie_found", "home_goalie_confirmed", "away_score", "home_score"
    ]].copy()
    home_goalie.columns = [
        "game_id", "date", "date_dt", "team", "goalie_name", "goalie_id",
        "goalie_found", "goalie_confirmed", "goals_allowed", "goals_scored"
    ]
    home_goalie["is_home"] = 1

    away_goalie = work_df[[
        "game_id", "date", "date_dt", "away_team", "away_goalie_name", "away_goalie_id",
        "away_goalie_found", "away_goalie_confirmed", "home_score", "away_score"
    ]].copy()
    away_goalie.columns = [
        "game_id", "date", "date_dt", "team", "goalie_name", "goalie_id",
        "goalie_found", "goalie_confirmed", "goals_allowed", "goals_scored"
    ]
    away_goalie["is_home"] = 0

    goalie_df = pd.concat([home_goalie, away_goalie], ignore_index=True)
    goalie_df["goalie_key"] = goalie_df.apply(
        lambda r: build_goalie_key(r["goalie_id"], r["goalie_name"]), axis=1
    )
    goalie_df["won"] = (goalie_df["goals_scored"] > goalie_df["goals_allowed"]).astype(int)

    goalie_df = goalie_df.sort_values(["goalie_key", "date_dt", "game_id"]).reset_index(drop=True)
    valid_goalie = goalie_df["goalie_key"] != ""

    goalie_df["goalie_games_started_before"] = 0.0
    goalie_df.loc[valid_goalie, "goalie_games_started_before"] = (
        goalie_df.loc[valid_goalie].groupby("goalie_key").cumcount().astype(float)
    )

    goalie_df["goalie_goals_allowed_l3"] = 0.0
    goalie_df["goalie_goals_allowed_l5"] = 0.0
    goalie_df["goalie_goals_allowed_l10"] = 0.0
    goalie_df["goalie_win_rate_l3"] = 0.0
    goalie_df["goalie_win_rate_l5"] = 0.0
    goalie_df["goalie_win_rate_l10"] = 0.0
    goalie_df["goalie_rest_days"] = 7.0

    valid_group = goalie_df.loc[valid_goalie].groupby("goalie_key")

    goalie_df.loc[valid_goalie, "goalie_goals_allowed_l3"] = valid_group["goals_allowed"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    goalie_df.loc[valid_goalie, "goalie_goals_allowed_l5"] = valid_group["goals_allowed"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    goalie_df.loc[valid_goalie, "goalie_goals_allowed_l10"] = valid_group["goals_allowed"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )
    goalie_df.loc[valid_goalie, "goalie_win_rate_l3"] = valid_group["won"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    goalie_df.loc[valid_goalie, "goalie_win_rate_l5"] = valid_group["won"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    goalie_df.loc[valid_goalie, "goalie_win_rate_l10"] = valid_group["won"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )

    goalie_df.loc[valid_goalie, "prev_goalie_date"] = valid_group["date_dt"].shift(1)
    goalie_df.loc[valid_goalie, "goalie_rest_days"] = (
        goalie_df.loc[valid_goalie, "date_dt"] - goalie_df.loc[valid_goalie, "prev_goalie_date"]
    ).dt.days.fillna(7).clip(lower=1, upper=30)

    team_goalie_df = goalie_df.sort_values(["team", "date_dt", "game_id"]).reset_index(drop=True)
    team_goalie_df["team_prev_goalie_key"] = team_goalie_df.groupby("team")["goalie_key"].shift(1)
    team_goalie_df["same_goalie_as_prev_team_game"] = (
        (team_goalie_df["goalie_key"] != "") &
        (team_goalie_df["goalie_key"] == team_goalie_df["team_prev_goalie_key"])
    ).astype(int)

    goalie_df = goalie_df.merge(
        team_goalie_df[["game_id", "team", "is_home", "same_goalie_as_prev_team_game"]],
        on=["game_id", "team", "is_home"],
        how="left",
    )
    goalie_df["same_goalie_as_prev_team_game"] = goalie_df["same_goalie_as_prev_team_game"].fillna(0)

    goalie_df["goalie_found"] = goalie_df["goalie_found"].fillna(0).astype(int)
    goalie_df["goalie_confirmed"] = goalie_df["goalie_confirmed"].fillna(0).astype(int)

    home_features = goalie_df[goalie_df["is_home"] == 1][[
        "game_id",
        "goalie_games_started_before",
        "goalie_goals_allowed_l3", "goalie_goals_allowed_l5", "goalie_goals_allowed_l10",
        "goalie_win_rate_l3", "goalie_win_rate_l5", "goalie_win_rate_l10",
        "goalie_rest_days", "same_goalie_as_prev_team_game",
    ]].copy()
    home_features.columns = [
        "game_id",
        "home_goalie_games_started_before",
        "home_goalie_goals_allowed_l3", "home_goalie_goals_allowed_l5", "home_goalie_goals_allowed_l10",
        "home_goalie_win_rate_l3", "home_goalie_win_rate_l5", "home_goalie_win_rate_l10",
        "home_goalie_rest_days", "home_same_goalie_as_prev_team_game",
    ]

    away_features = goalie_df[goalie_df["is_home"] == 0][[
        "game_id",
        "goalie_games_started_before",
        "goalie_goals_allowed_l3", "goalie_goals_allowed_l5", "goalie_goals_allowed_l10",
        "goalie_win_rate_l3", "goalie_win_rate_l5", "goalie_win_rate_l10",
        "goalie_rest_days", "same_goalie_as_prev_team_game",
    ]].copy()
    away_features.columns = [
        "game_id",
        "away_goalie_games_started_before",
        "away_goalie_goals_allowed_l3", "away_goalie_goals_allowed_l5", "away_goalie_goals_allowed_l10",
        "away_goalie_win_rate_l3", "away_goalie_win_rate_l5", "away_goalie_win_rate_l10",
        "away_goalie_rest_days", "away_same_goalie_as_prev_team_game",
    ]

    df = df.merge(home_features, on="game_id", how="left")
    df = df.merge(away_features, on="game_id", how="left")

    goalie_numeric_cols = [
        "home_goalie_found", "away_goalie_found",
        "home_goalie_confirmed", "away_goalie_confirmed",
        "home_goalie_games_started_before", "away_goalie_games_started_before",
        "home_goalie_goals_allowed_l3", "home_goalie_goals_allowed_l5", "home_goalie_goals_allowed_l10",
        "away_goalie_goals_allowed_l3", "away_goalie_goals_allowed_l5", "away_goalie_goals_allowed_l10",
        "home_goalie_win_rate_l3", "home_goalie_win_rate_l5", "home_goalie_win_rate_l10",
        "away_goalie_win_rate_l3", "away_goalie_win_rate_l5", "away_goalie_win_rate_l10",
        "home_goalie_rest_days", "away_goalie_rest_days",
        "home_same_goalie_as_prev_team_game", "away_same_goalie_as_prev_team_game",
    ]
    for col in goalie_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def calculate_h2h_incremental(df: pd.DataFrame) -> pd.DataFrame:
    print("📊 Calculating H2H records (incremental, no leakage).")

    h2h_records = {}
    h2h_home_win_rate = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        key = tuple(sorted([home, away]))

        if key not in h2h_records:
            h2h_records[key] = {"home_wins": 0, "total": 0}

        record = h2h_records[key]
        total = record["total"]

        if key[0] == home:
            win_rate = record["home_wins"] / total if total > 0 else 0.5
        else:
            win_rate = (total - record["home_wins"]) / total if total > 0 else 0.5

        h2h_home_win_rate.append(win_rate)

        if row["home_score"] > row["away_score"]:
            record["home_wins"] += 1
        record["total"] += 1

    df["h2h_home_win_rate"] = h2h_home_win_rate
    return df


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    print("🎯 Creating target variables...")

    df["TARGET_full_game"] = np.where(
        df["home_score"] > df["away_score"],
        1,
        np.where(df["home_score"] < df["away_score"], 0, np.nan),
    )

    df["TARGET_over_55"] = (df["home_score"] + df["away_score"] > 5.5).astype(int)
    df["TARGET_home_over_25"] = (df["home_score"] > 2.5).astype(int)
    df["TARGET_spread_1_5"] = ((df["home_score"] - df["away_score"]) > 1.5).astype(int)
    total_p1 = pd.to_numeric(df.get("total_p1_goals"), errors="coerce")
    df["TARGET_p1_over_15"] = np.where(total_p1.notna(), (total_p1 > 1.5).astype(int), np.nan)

    return df


def add_matchup_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    print("🧮 Creating base diff features...")

    df["elo_diff"] = df["home_elo_pre"] - df["away_elo_pre"]
    df["rest_days_diff"] = df["home_rest_days"] - df["away_rest_days"]
    df["win_rate_l5_diff"] = df["home_win_rate_l5"] - df["away_win_rate_l5"]
    df["win_rate_l10_diff"] = df["home_win_rate_l10"] - df["away_win_rate_l10"]
    df["goals_scored_l5_diff"] = df["home_goals_scored_l5"] - df["away_goals_scored_l5"]
    df["goals_allowed_l5_diff"] = df["home_goals_allowed_l5"] - df["away_goals_allowed_l5"]
    df["goals_scored_l10_diff"] = df["home_goals_scored_l10"] - df["away_goals_scored_l10"]
    df["goals_allowed_l10_diff"] = df["home_goals_allowed_l10"] - df["away_goals_allowed_l10"]
    df["goal_diff_l5_diff"] = df["home_goal_diff_l5"] - df["away_goal_diff_l5"]
    df["goal_diff_l10_diff"] = df["home_goal_diff_l10"] - df["away_goal_diff_l10"]
    df["goal_diff_std_l5_diff"] = df["home_goal_diff_std_l5"] - df["away_goal_diff_std_l5"]

    # Explicit differential aliases used by FULL_GAME V2 selectors.
    df["fg_form_diff"] = df["win_rate_l5_diff"]
    df["fg_form_long_diff"] = df["win_rate_l10_diff"]
    df["fg_scoring_diff"] = df["goals_scored_l5_diff"] - df["goals_allowed_l5_diff"]
    df["fg_scoring_long_diff"] = df["goals_scored_l10_diff"] - df["goals_allowed_l10_diff"]
    df["fg_rest_diff"] = df["rest_days_diff"]
    df["fg_volatility_diff"] = df["goal_diff_std_l5_diff"]

    if "home_goalie_goals_allowed_l5" in df.columns and "away_goalie_goals_allowed_l5" in df.columns:
        df["goalie_goals_allowed_l3_diff"] = df["home_goalie_goals_allowed_l3"] - df["away_goalie_goals_allowed_l3"]
        df["goalie_goals_allowed_l5_diff"] = df["home_goalie_goals_allowed_l5"] - df["away_goalie_goals_allowed_l5"]
        df["goalie_goals_allowed_l10_diff"] = df["home_goalie_goals_allowed_l10"] - df["away_goalie_goals_allowed_l10"]
        df["goalie_win_rate_l3_diff"] = df["home_goalie_win_rate_l3"] - df["away_goalie_win_rate_l3"]
        df["goalie_win_rate_l5_diff"] = df["home_goalie_win_rate_l5"] - df["away_goalie_win_rate_l5"]
        df["goalie_win_rate_l10_diff"] = df["home_goalie_win_rate_l10"] - df["away_goalie_win_rate_l10"]
        df["goalie_rest_days_diff"] = df["home_goalie_rest_days"] - df["away_goalie_rest_days"]
        df["goalie_experience_diff"] = df["home_goalie_games_started_before"] - df["away_goalie_games_started_before"]
        df["both_goalies_found"] = (
            (df["home_goalie_found"] > 0) & (df["away_goalie_found"] > 0)
        ).astype(int)
        df["both_goalies_confirmed"] = (
            (df["home_goalie_confirmed"] > 0) & (df["away_goalie_confirmed"] > 0)
        ).astype(int)

        df["fg_goalie_diff"] = (
            1.10 * df["goalie_win_rate_l5_diff"]
            + 0.55 * df["goalie_win_rate_l10_diff"]
            - 0.85 * df["goalie_goals_allowed_l5_diff"]
            + 0.08 * df["goalie_experience_diff"]
        )
    else:
        df["goalie_goals_allowed_l3_diff"] = 0.0
        df["goalie_goals_allowed_l5_diff"] = 0.0
        df["goalie_goals_allowed_l10_diff"] = 0.0
        df["goalie_win_rate_l3_diff"] = 0.0
        df["goalie_win_rate_l5_diff"] = 0.0
        df["goalie_win_rate_l10_diff"] = 0.0
        df["goalie_rest_days_diff"] = 0.0
        df["goalie_experience_diff"] = 0.0
        df["both_goalies_found"] = 0
        df["both_goalies_confirmed"] = 0
        df["fg_goalie_diff"] = 0.0

    return df


def add_full_game_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    print("🎯 Creating FULL_GAME specific signal features...")

    def num(col, default=0.0):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
        return pd.Series(default, index=df.index, dtype="float64")

    # ataque/defensa cruzados
    df["fg_home_attack_vs_away_defense"] = (
        num("home_goals_scored_home_l5")
        - num("away_goals_allowed_away_l5")
    )

    df["fg_away_attack_vs_home_defense"] = (
        num("away_goals_scored_away_l5")
        - num("home_goals_allowed_home_l5")
    )

    # edge de forma local / visita
    df["fg_home_ice_form_edge"] = (
        1.4 * (num("home_win_rate_home_l5") - num("away_win_rate_away_l5"))
        + 0.8 * (num("home_goals_scored_home_l5") - num("away_goals_scored_away_l5"))
        - 0.6 * (num("home_goals_allowed_home_l5") - num("away_goals_allowed_away_l5"))
    )

    # penalización de road fatigue
    df["fg_road_penalty_edge"] = (
        -1.4 * (num("away_rest_days") < 2).astype(float)
        -0.9 * num("away_is_b2b")
        +0.5 * num("home_rest_days")
        -0.25 * (num("away_games_last_3") - num("home_games_last_3"))
        -0.15 * (num("away_games_last_5") - num("home_games_last_5"))
    )

    # volatilidad / consistencia
    df["fg_recent_volatility_gap"] = (
        num("away_goal_diff_std_l5") - num("home_goal_diff_std_l5")
    )

    # fortaleza base y forma corta
    df["fg_team_strength_gap_long"] = (
        0.65 * num("elo_diff")
        + 95.0 * (num("home_win_rate_l10") - num("away_win_rate_l10"))
        + 7.0 * (num("home_goal_diff_l10") - num("away_goal_diff_l10"))
        + 0.9 * num("h2h_home_win_rate")
    )

    df["fg_form_gap_short"] = (
        100.0 * (num("home_win_rate_l5") - num("away_win_rate_l5"))
        + 7.5 * (num("home_goal_diff_l5") - num("away_goal_diff_l5"))
        + 1.5 * (num("home_goals_scored_l5") - num("away_goals_scored_l5"))
        - 1.0 * (num("home_goals_allowed_l5") - num("away_goals_allowed_l5"))
    )

    df["fg_schedule_stress_gap"] = (
        1.8 * (num("home_rest_days") - num("away_rest_days"))
        - 2.2 * (num("home_is_b2b") - num("away_is_b2b"))
        - 0.8 * (num("home_games_last_3") - num("away_games_last_3"))
        - 0.5 * (num("home_games_last_5") - num("away_games_last_5"))
    )

    # goalie edge
    df["fg_goalie_edge"] = (
        -1.6 * num("goalie_goals_allowed_l5_diff")
        -0.8 * num("goalie_goals_allowed_l3_diff")
        +1.3 * num("goalie_win_rate_l5_diff")
        +0.7 * num("goalie_win_rate_l10_diff")
        +0.08 * num("goalie_experience_diff")
        +0.20 * num("goalie_rest_days_diff")
        +0.50 * (num("home_goalie_confirmed") - num("away_goalie_confirmed"))
        +0.25 * (num("home_goalie_found") - num("away_goalie_found"))
        +0.20 * (num("home_same_goalie_as_prev_team_game") - num("away_same_goalie_as_prev_team_game"))
    )

    # señal direccional para no castigar tanto away cuando sí trae edge real
    df["fg_directional_signal"] = (
        1.35 * df["fg_home_attack_vs_away_defense"]
        - 1.10 * df["fg_away_attack_vs_home_defense"]
        + 0.90 * df["fg_home_ice_form_edge"]
        + 0.65 * df["fg_road_penalty_edge"]
        + 0.55 * df["fg_goalie_edge"]
        + 0.40 * df["fg_schedule_stress_gap"]
        - 0.40 * df["fg_recent_volatility_gap"]
    )

    df["fg_strength_x_goalie"] = df["fg_team_strength_gap_long"] * df["fg_goalie_edge"]
    df["fg_form_x_schedule"] = df["fg_form_gap_short"] * df["fg_schedule_stress_gap"]

    df["fg_signal_total"] = (
        0.30 * df["fg_team_strength_gap_long"]
        + 0.22 * df["fg_form_gap_short"]
        + 0.28 * df["fg_directional_signal"]
        + 0.10 * df["fg_goalie_edge"]
        + 0.05 * df["fg_schedule_stress_gap"]
        + 0.03 * df["fg_strength_x_goalie"]
        + 0.02 * df["fg_form_x_schedule"]
    )

    # Explicit aliases requested for model interpretability and selective training.
    df["fg_strength_diff"] = df["fg_team_strength_gap_long"]
    df["fg_strength_x_goalie_diff"] = df["fg_strength_x_goalie"]

    abs_signal = df["fg_signal_total"].abs().fillna(0)
    threshold = float(abs_signal.quantile(0.35)) if len(abs_signal) else 0.0
    df["fg_coinflip_flag"] = (abs_signal < threshold).astype(int)

    return df


def add_moneyline_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    print("💸 Creating moneyline odds features...")

    def _to_num(col):
        return pd.to_numeric(df.get(col), errors="coerce")

    def _american_to_implied_prob(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        pos = s > 0
        neg = s < 0
        out.loc[pos] = 100.0 / (s.loc[pos] + 100.0)
        out.loc[neg] = (-s.loc[neg]) / ((-s.loc[neg]) + 100.0)
        return out.clip(lower=0.01, upper=0.99)

    home_ml = _to_num("home_moneyline_odds")
    away_ml = _to_num("away_moneyline_odds")
    closing_ml = _to_num("closing_moneyline_odds")

    df["home_moneyline_odds"] = home_ml
    df["away_moneyline_odds"] = away_ml
    df["closing_moneyline_odds"] = closing_ml

    home_prob = _american_to_implied_prob(home_ml)
    away_prob = _american_to_implied_prob(away_ml)

    margin = (home_prob + away_prob) - 1.0
    df["ml_implied_home_prob"] = home_prob
    df["ml_implied_away_prob"] = away_prob
    df["ml_implied_margin"] = margin.fillna(0.0)
    df["ml_implied_home_prob_no_vig"] = (home_prob / (home_prob + away_prob)).replace([np.inf, -np.inf], np.nan)
    df["ml_implied_away_prob_no_vig"] = (away_prob / (home_prob + away_prob)).replace([np.inf, -np.inf], np.nan)
    df["ml_prob_gap_no_vig"] = (
        df["ml_implied_home_prob_no_vig"] - df["ml_implied_away_prob_no_vig"]
    ).fillna(0.0)

    # Positive when market price favors home.
    abs_home = home_ml.abs()
    abs_away = away_ml.abs()
    df["ml_abs_price_gap"] = (abs_away - abs_home).fillna(0.0)
    df["ml_home_is_favorite_market"] = ((home_ml < away_ml) & home_ml.notna() & away_ml.notna()).astype(int)
    df["ml_odds_available"] = (
        home_ml.notna() & away_ml.notna() & (home_ml != 0) & (away_ml != 0)
    ).astype(int)

    # Blend model form signal with market signal to stabilize coinflip games.
    if "fg_signal_total" in df.columns:
        df["fg_x_market_prob_gap"] = df["fg_signal_total"] * df["ml_prob_gap_no_vig"]
        df["fg_market_vs_model_gap"] = df["ml_prob_gap_no_vig"] - (df["fg_signal_total"] / 100.0)
    else:
        df["fg_x_market_prob_gap"] = 0.0
        df["fg_market_vs_model_gap"] = 0.0

    df["fg_market_edge"] = df["ml_prob_gap_no_vig"]
    df["fg_market_favorite_flag"] = df["ml_home_is_favorite_market"].astype(float)
    df["fg_market_confidence"] = df["ml_prob_gap_no_vig"].abs()

    return df


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    print("🎯 Creating target variables...")

    df["TARGET_full_game"] = np.where(
        df["home_score"] > df["away_score"],
        1,
        np.where(df["home_score"] < df["away_score"], 0, np.nan),
    )

    df["TARGET_over_55"] = (df["home_score"] + df["away_score"] > 5.5).astype(int)
    df["TARGET_home_over_25"] = (df["home_score"] > 2.5).astype(int)
    df["TARGET_spread_1_5"] = ((df["home_score"] - df["away_score"]) > 1.5).astype(int)
    total_p1 = pd.to_numeric(df.get("total_p1_goals"), errors="coerce")
    df["TARGET_p1_over_15"] = np.where(total_p1.notna(), (total_p1 > 1.5).astype(int), np.nan)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["completed"] == 1].copy()
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt"]).sort_values(["date_dt", "game_id"]).reset_index(drop=True)

    print(f"📌 Using {len(df)} completed games (excluded non-completed)")

    for col in [
        "home_goalie_name", "away_goalie_name",
        "home_goalie_id", "away_goalie_id",
        "home_goalie_found", "away_goalie_found",
        "home_goalie_confirmed", "away_goalie_confirmed",
    ]:
        if col not in df.columns:
            df[col] = 0 if col.endswith(("_found", "_confirmed")) else ""

    df = create_targets(df)
    df = df.dropna(subset=["TARGET_full_game"]).copy()

    df = calculate_elo_ratings(df)
    df = calculate_team_rolling_features(df)
    df = calculate_goalie_rolling_features(df)
    df = calculate_h2h_incremental(df)
    df = add_matchup_diff_features(df)
    df = add_full_game_signal_features(df)
    df = add_moneyline_odds_features(df)

    numeric_fill_zero = [
        c for c in df.columns
        if c.startswith("home_")
        or c.startswith("away_")
        or c.startswith("fg_")
        or c.endswith("_diff")
    ]
    numeric_fill_zero += [
        "h2h_home_win_rate",
        "home_elo_pre",
        "away_elo_pre",
        "elo_diff",
        "both_goalies_found",
        "both_goalies_confirmed",
        "closing_moneyline_odds",
        "ml_implied_home_prob",
        "ml_implied_away_prob",
        "ml_implied_margin",
        "ml_implied_home_prob_no_vig",
        "ml_implied_away_prob_no_vig",
        "ml_prob_gap_no_vig",
        "ml_abs_price_gap",
        "ml_home_is_favorite_market",
        "ml_odds_available",
        "fg_x_market_prob_gap",
    ]
    numeric_fill_zero = [c for c in set(numeric_fill_zero) if c in df.columns]
    df[numeric_fill_zero] = df[numeric_fill_zero].apply(pd.to_numeric, errors="coerce").fillna(0)

    return df


def process_nhl_data():
    print("🏒 Feature Engineering NHL (FULL_GAME MONEYLINE UPGRADE)")
    print("=" * 60)

    if not RAW_DATA.exists():
        print(f"⚠️ Raw data not found: {RAW_DATA}")
        return None

    print(f"📂 Loading raw data from {RAW_DATA.name}...")
    df = pd.read_csv(RAW_DATA, dtype={"game_id": str})
    print(f"   Loaded {len(df)} total records")

    df = engineer_features(df)
    df = df.dropna(subset=["TARGET_full_game", "TARGET_over_55", "TARGET_home_over_25"])

    non_features = {
        "game_id", "date", "date_dt", "time", "season", "home_team", "away_team",
        "home_score", "away_score", "total_goals", "is_draw", "completed",
        "home_p1_goals", "away_p1_goals", "total_p1_goals",
        "venue_name", "odds_details", "odds_over_under", "odds_data_quality",
        "home_goalie_name", "away_goalie_name", "home_goalie_id", "away_goalie_id",
        "goalie_data_quality",
        "TARGET_full_game", "TARGET_over_55", "TARGET_home_over_25",
        "TARGET_spread_1_5", "TARGET_p1_over_15",
    }

    feature_cols = [c for c in df.columns if c not in non_features]
    print(f"✅ Final dataset: {len(df)} rows | {len(feature_cols)} features")

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"💾 Saved processed features to: {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    process_nhl_data()
