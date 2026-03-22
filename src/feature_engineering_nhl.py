"""
Feature engineering for NHL games - CLEAN VERSION.
No data leakage, no future games, proper shifting.
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA = BASE_DIR / "data" / "nhl" / "raw" / "nhl_advanced_history.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "nhl" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = PROCESSED_DATA_DIR / "model_ready_features_nhl.csv"


def calculate_elo_ratings(df: pd.DataFrame, k: float = 25, home_advantage: float = 50) -> pd.DataFrame:
    """
    Calculate ELO ratings for NHL teams.
    Uses only completed games, in chronological order.
    """
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
        
        if row["is_draw"]:
            actual_home = expected_home
            actual_away = expected_away
        elif row["home_score"] > row["away_score"]:
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
    """
    Calculate rolling team features using ONLY PAST games (with .shift(1)).
    No leakage, no future data.
    """
    print("⚙️ Calculating rolling features (with .shift(1))...")
    
    # Separate home and away stats
    home_df = df[
        ["date", "game_id", "home_team", "home_score", "away_score"]
    ].copy()
    home_df.columns = ["date", "game_id", "team", "goals_scored", "goals_allowed"]
    home_df["is_home"] = 1
    
    away_df = df[
        ["date", "game_id", "away_team", "away_score", "home_score"]
    ].copy()
    away_df.columns = ["date", "game_id", "team", "goals_scored", "goals_allowed"]
    away_df["is_home"] = 0
    
    team_df = pd.concat([home_df, away_df], ignore_index=True)
    team_df["date_dt"] = pd.to_datetime(team_df["date"])
    team_df = team_df.sort_values(["team", "date_dt"]).reset_index(drop=True)
    
    # Rolling statistics WITH SHIFT(1) - only past games
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
    
    # Win rate (past games only)
    team_df["goal_diff"] = team_df["goals_scored"] - team_df["goals_allowed"]
    team_df["won"] = (team_df["goal_diff"] > 0).astype(int)
    team_df["win_rate_l5"] = team_df.groupby("team")["won"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    team_df["win_rate_l10"] = team_df.groupby("team")["won"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )
    
    # Rest days (gap since last game)
    team_df["last_game_date"] = team_df.groupby("team")["date_dt"].shift(1)
    team_df["rest_days"] = (team_df["date_dt"] - team_df["last_game_date"]).dt.days.fillna(1)
    team_df["rest_days"] = team_df["rest_days"].clip(lower=1, upper=14)
    
    # Merge back to original dataframe
    home_features = team_df[team_df["is_home"] == 1][[
        "game_id", "goals_scored_l5", "goals_allowed_l5", "goals_scored_l10",
        "goals_allowed_l10", "win_rate_l5", "win_rate_l10", "rest_days"
    ]].copy()
    home_features.columns = [
        "game_id", "home_goals_scored_l5", "home_goals_allowed_l5",
        "home_goals_scored_l10", "home_goals_allowed_l10",
        "home_win_rate_l5", "home_win_rate_l10", "home_rest_days"
    ]
    
    away_features = team_df[team_df["is_home"] == 0][[
        "game_id", "goals_scored_l5", "goals_allowed_l5", "goals_scored_l10",
        "goals_allowed_l10", "win_rate_l5", "win_rate_l10", "rest_days"
    ]].copy()
    away_features.columns = [
        "game_id", "away_goals_scored_l5", "away_goals_allowed_l5",
        "away_goals_scored_l10", "away_goals_allowed_l10",
        "away_win_rate_l5", "away_win_rate_l10", "away_rest_days"
    ]
    
    df = df.merge(home_features, on="game_id", how="left")
    df = df.merge(away_features, on="game_id", how="left")
    df = df.fillna(0)
    
    return df


def calculate_h2h_incremental(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate H2H record incrementally (no future leakage).
    For each game, only use H2H record from games BEFORE this one.
    """
    print("📊 Calculating H2H records (incremental, no leakage)...")
    
    h2h_records = {}  # key = sorted tuple of teams, value = {home_wins, total}
    h2h_home_win_rate = []
    
    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        key = tuple(sorted([home, away]))
        
        if key not in h2h_records:
            h2h_records[key] = {"home_wins": 0, "total": 0}
        
        record = h2h_records[key]
        total = record["total"]
        
        # Calculate win rate as if home team is first in sorted tuple
        if key[0] == home:
            win_rate = record["home_wins"] / total if total > 0 else 0.5
        else:
            # Home team is second, so it's actually the away team in the record
            win_rate = (total - record["home_wins"]) / total if total > 0 else 0.5
        
        h2h_home_win_rate.append(win_rate)
        
        # Update record AFTER using it
        if row["home_score"] > row["away_score"]:
            record["home_wins"] += 1
        record["total"] += 1
    
    df["h2h_home_win_rate"] = h2h_home_win_rate
    return df


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create target variables for models."""
    print("🎯 Creating target variables...")
    
    # Full Game: 0=Draw, 1=Home Win, 2=Away Win
    df["TARGET_full_game"] = df.apply(
        lambda row: 1 if row["home_score"] > row["away_score"]
                   else (0 if row["home_score"] == row["away_score"] else 2),
        axis=1
    )
    
    # Over/Under 5.5 goals total
    df["TARGET_over_55"] = (df["home_score"] + df["away_score"] > 5.5).astype(int)
    
    # Home Over/Under 2.5 goals
    df["TARGET_home_over_25"] = (df["home_score"] > 2.5).astype(int)
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Main feature engineering pipeline."""
    
    # Only keep completed games
    df = df[df["completed"] == 1].copy()
    
    # Basic cleaning
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt"]).sort_values("date_dt").reset_index(drop=True)
    
    print(f"📌 Using {len(df)} completed games (excluded non-completed)")
    
    # Create targets BEFORE rolling features
    df = create_targets(df)
    
    # ELO ratings
    df = calculate_elo_ratings(df)
    
    # Rolling team features (with .shift to avoid leakage)
    df = calculate_team_rolling_features(df)
    
    # H2H incremental (no future leakage)
    df = calculate_h2h_incremental(df)
    
    # Feature: ELO difference (pregame stat, no leakage)
    df["elo_diff"] = df["home_elo_pre"] - df["away_elo_pre"]
    
    return df


def process_nhl_data():
    """Main processing function."""
    print("🏒 Feature Engineering NHL (CLEAN - NO LEAKAGE)")
    print("=" * 60)
    
    if not RAW_DATA.exists():
        print(f"⚠️ Raw data not found: {RAW_DATA}")
        return None
    
    print(f"📂 Loading raw data from {RAW_DATA.name}...")
    df = pd.read_csv(RAW_DATA, dtype={"game_id": str})
    print(f"   Loaded {len(df)} total records")
    
    # Engineer features
    df = engineer_features(df)
    
    # Drop rows with missing targets
    df = df.dropna(subset=["TARGET_full_game", "TARGET_over_55"])
    
    # Define non-feature columns (NO score-derived features like home_goals_diff)
    non_features = {
        "game_id", "date", "date_dt", "time", "season", "home_team", "away_team",
        "home_score", "away_score", "total_goals", "is_draw", "completed",
        "venue_name", "odds_details", "odds_over_under",
        "TARGET_full_game", "TARGET_over_55", "TARGET_home_over_25",
    }
    
    feature_cols = [c for c in df.columns if c not in non_features]
    print(f"\n✅ Final dataset:")
    print(f"   Games: {len(df)}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Feature list: {', '.join(sorted(feature_cols))}")
    
    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Saved: {OUTPUT_FILE.name}")
    
    return df


if __name__ == "__main__":
    process_nhl_data()

