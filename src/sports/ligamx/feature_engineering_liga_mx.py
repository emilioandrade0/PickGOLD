import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

import sys
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
RAW_DATA = BASE_DIR / "data" / "liga_mx" / "raw" / "liga_mx_advanced_history.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "liga_mx" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = PROCESSED_DATA_DIR / "model_ready_features_liga_mx.csv"

HAS_ADVANCED_FEATURES = True  # Inline functions below


def calculate_elo_ratings(df: pd.DataFrame, k: float = 16, home_advantage: float = 50) -> pd.DataFrame:
    """
    Calcula ratings ELO para todos los equipos usando histórico de partidos.
    Adaptado para fútbol con home_advantage = 50 (menor que en NBA/MLB).
    """
    print("[ELO] Calculando Sistema ELO para Liga MX...")

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


def ensure_market_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura que existan columnas de mercado."""
    market_cols = [
        "odds_over_under",
        "home_moneyline_odds",
        "away_moneyline_odds",
        "closing_moneyline_odds",
        "closing_spread_odds",
        "closing_total_odds",
    ]

    for col in market_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    df["market_missing"] = df[market_cols].isna().any(axis=1).astype(int)

    df["odds_over_under"] = df["odds_over_under"].fillna(2.5)

    # Conservative defaults for missing odds fields.
    for col in [
        "home_moneyline_odds",
        "away_moneyline_odds",
        "closing_moneyline_odds",
        "closing_spread_odds",
        "closing_total_odds",
    ]:
        df[col] = df[col].fillna(0.0)

    return df


def _american_to_implied_prob_series(odds: pd.Series) -> pd.Series:
    odds = pd.to_numeric(odds, errors="coerce").fillna(0.0)

    # Decimal odds (common in soccer feeds): 1.01 .. 19.99
    decimal = (odds > 1.0) & (odds < 20.0)
    pos = odds > 0
    neg = odds < 0

    out = pd.Series(np.zeros(len(odds), dtype=float), index=odds.index)
    out.loc[decimal] = 1.0 / odds.loc[decimal]
    american_pos = pos & ~decimal
    out.loc[american_pos] = 100.0 / (odds.loc[american_pos] + 100.0)
    out.loc[neg] = np.abs(odds.loc[neg]) / (np.abs(odds.loc[neg]) + 100.0)
    out.loc[~(american_pos | neg | decimal)] = 0.5
    return out.clip(0.01, 0.99)


def count_games_in_last_days(group: pd.DataFrame, days: int) -> pd.Series:
    """Cuenta cuántos partidos tuvo un equipo en los últimos N días."""
    dates = group["date_dt"].to_numpy(dtype="datetime64[D]")
    counts = np.zeros(len(dates), dtype=int)

    for i in range(len(dates)):
        current_date = dates[i]
        start_date = current_date - np.timedelta64(days, "D")
        counts[i] = np.sum((dates[:i] >= start_date) & (dates[:i] < current_date))

    return pd.Series(counts, index=group.index)


# ====== ADVANCED FEATURES FOR FULL_GAME ======
def calculate_home_away_streaks(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate home/away streaks and dominance."""
    print("[ADVANCED] Computing home/away streaks...")
    
    home_df = df[["date", "game_id", "home_team", "home_score", "away_score", "is_draw"]].copy()
    home_df.columns = ["date", "game_id", "team", "goals_for", "goals_against", "drew"]
    home_df["date_dt"] = pd.to_datetime(home_df["date"])
    home_df = home_df.sort_values(["team", "date_dt", "game_id"]).reset_index(drop=True)
    
    home_df["won"] = (home_df["goals_for"] > home_df["goals_against"]).astype(int)
    home_df["goal_diff"] = home_df["goals_for"] - home_df["goals_against"]
    
    home_df["home_win_streak_L5"] = home_df.groupby("team")["won"].shift(1).rolling(5, 1).max().values
    home_df["home_win_streak_L10"] = home_df.groupby("team")["won"].shift(1).rolling(10, 1).max().values
    home_df["home_dominance_L10"] = home_df.groupby("team")["goal_diff"].shift(1).rolling(10, 1).mean().values
    
    home_streaks = home_df[["game_id", "team", "home_win_streak_L5", "home_win_streak_L10", "home_dominance_L10"]].copy()
    
    away_df = df[["date", "game_id", "away_team", "away_score", "home_score", "is_draw"]].copy()
    away_df.columns = ["date", "game_id", "team", "goals_for", "goals_against", "drew"]
    away_df["date_dt"] = pd.to_datetime(away_df["date"])
    away_df = away_df.sort_values(["team", "date_dt", "game_id"]).reset_index(drop=True)
    
    away_df["won"] = (away_df["goals_for"] > away_df["goals_against"]).astype(int)
    away_df["goal_diff"] = away_df["goals_for"] - away_df["goals_against"]
    
    away_df["away_win_streak_L5"] = away_df.groupby("team")["won"].shift(1).rolling(5, 1).max().values
    away_df["away_win_streak_L10"] = away_df.groupby("team")["won"].shift(1).rolling(10, 1).max().values
    away_df["away_dominance_L10"] = away_df.groupby("team")["goal_diff"].shift(1).rolling(10, 1).mean().values
    
    away_streaks = away_df[["game_id", "team", "away_win_streak_L5", "away_win_streak_L10", "away_dominance_L10"]].copy()
    
    return home_streaks, away_streaks


def calculate_draw_parity_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Draw outcome strength based on parity."""
    print("[ADVANCED] Computing draw parity strength...")
    
    out = df.copy()
    
    home_draw_rate = out.get("home_draw_pct_L10", out.get("home_draw_pct_L5", 0.0)).fillna(0)
    away_draw_rate = out.get("away_draw_pct_L10", out.get("away_draw_pct_L5", 0.0)).fillna(0)
    out["draw_tendency_similarity"] = 1.0 - np.abs(home_draw_rate - away_draw_rate)
    
    elo_diff = np.abs(out.get("diff_elo", 0).fillna(0))
    out["elo_proximity"] = np.exp(-elo_diff / 150.0)
    
    gd_diff = np.abs(out.get("diff_goal_diff_L10", out.get("diff_goal_diff_L5", 0)).fillna(0))
    out["form_balance"] = np.exp(-gd_diff / 2.0)
    
    out["draw_parity_strength"] = (
        0.35 * out["draw_tendency_similarity"]
        + 0.30 * out["elo_proximity"]
        + 0.20 * out["form_balance"]
        + 0.15 * (1.0 / (1.0 + np.abs(out.get("diff_goal_diff_std_L10", 1)).fillna(1)))
    )
    
    return out[["draw_parity_strength"]]


def calculate_fatigue_differential(df: pd.DataFrame) -> pd.DataFrame:
    """Fatigue advantage: positive = home fresher."""
    print("[ADVANCED] Computing fatigue differential...")
    
    out = df.copy()
    
    rest_diff = out.get("diff_rest_days", 0).fillna(0)
    games_7d_home = out.get("home_games_last_7_days", 0).fillna(0)
    games_7d_away = out.get("away_games_last_7_days", 0).fillna(0)
    
    out["fatigue_advantage"] = 0.6 * rest_diff + 0.4 * (games_7d_away - games_7d_home)
    
    return out[["fatigue_advantage"]]


def calculate_opponent_quality_adjusted_form(df: pd.DataFrame) -> pd.DataFrame:
    """Recent form weighted by opponent strength."""
    print("[ADVANCED] Computing opponent-adjusted form...")
    
    out = df.copy()
    
    away_strength = np.abs(out.get("away_elo_pre", 1500).fillna(1500)) / 1500.0
    home_strength = np.abs(out.get("home_elo_pre", 1500).fillna(1500)) / 1500.0
    
    out["home_form_vs_strong"] = out.get("home_win_pct_L5", 0.5).fillna(0.5) * (0.8 + 0.2 * away_strength)
    out["away_form_vs_strong"] = out.get("away_win_pct_L5", 0.5).fillna(0.5) * (0.8 + 0.2 * home_strength)
    
    return out[["home_form_vs_strong", "away_form_vs_strong"]]


def calculate_team_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula variables generales de rolling por equipo."""
    print("[GENERAL] Generando variables generales por equipo (Liga MX)...")

    home_df = df[
        ["date", "game_id", "home_team", "home_score", "away_score", "total_goals", "is_draw"]
    ].copy()
    home_df.columns = ["date", "game_id", "team", "goals_scored", "goals_allowed", "total_goals", "drew"]
    home_df["is_home"] = 1

    away_df = df[
        ["date", "game_id", "away_team", "away_score", "home_score", "total_goals", "is_draw"]
    ].copy()
    away_df.columns = ["date", "game_id", "team", "goals_scored", "goals_allowed", "total_goals", "drew"]
    away_df["is_home"] = 0

    all_stats = pd.concat([home_df, away_df], ignore_index=True).copy()
    all_stats["date_dt"] = pd.to_datetime(all_stats["date"])
    all_stats = all_stats.sort_values(["team", "date_dt", "game_id"]).reset_index(drop=True)

    all_stats["won_game"] = (all_stats["goals_scored"] > all_stats["goals_allowed"]).astype(int)
    all_stats["goal_diff"] = all_stats["goals_scored"] - all_stats["goals_allowed"]
    all_stats["draw"] = all_stats["drew"].astype(int)

    # BTTS: Both Teams To Score
    all_stats["btts_yes"] = ((all_stats["goals_scored"] > 0) & (all_stats["goals_allowed"] > 0)).astype(int)

    # Over/Under 2.5 goals
    all_stats["over_25"] = (all_stats["total_goals"] > 2.5).astype(int)

    all_stats["last_game_date"] = all_stats.groupby("team")["date_dt"].shift(1)
    all_stats["rest_days"] = (
        (all_stats["date_dt"] - all_stats["last_game_date"]).dt.days.fillna(7).clip(lower=0, upper=14)
    )
    all_stats["is_b2b"] = (all_stats["rest_days"] <= 2).astype(int)

    all_stats["games_last_3_days"] = (
        all_stats.groupby("team", group_keys=False)
        .apply(lambda g: count_games_in_last_days(g, 3), include_groups=False)
        .sort_index()
    )
    all_stats["games_last_5_days"] = (
        all_stats.groupby("team", group_keys=False)
        .apply(lambda g: count_games_in_last_days(g, 5), include_groups=False)
        .sort_index()
    )
    all_stats["games_last_7_days"] = (
        all_stats.groupby("team", group_keys=False)
        .apply(lambda g: count_games_in_last_days(g, 7), include_groups=False)
        .sort_index()
    )
    all_stats["games_last_10_days"] = (
        all_stats.groupby("team", group_keys=False)
        .apply(lambda g: count_games_in_last_days(g, 10), include_groups=False)
        .sort_index()
    )
    all_stats["games_last_14_days"] = (
        all_stats.groupby("team", group_keys=False)
        .apply(lambda g: count_games_in_last_days(g, 14), include_groups=False)
        .sort_index()
    )

    def roll_mean(col: str, window: int) -> pd.Series:
        return all_stats.groupby("team")[col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

    def roll_std(col: str, window: int, min_periods: int = 3) -> pd.Series:
        return all_stats.groupby("team")[col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=min_periods).std().fillna(0)
        )

    all_stats["win_pct_L5"] = roll_mean("won_game", 5)
    all_stats["win_pct_L10"] = roll_mean("won_game", 10)
    all_stats["draw_pct_L5"] = roll_mean("draw", 5)
    all_stats["draw_pct_L10"] = roll_mean("draw", 10)

    all_stats["goal_diff_L5"] = roll_mean("goal_diff", 5)
    all_stats["goal_diff_L3"] = roll_mean("goal_diff", 3)
    all_stats["goal_diff_L10"] = roll_mean("goal_diff", 10)

    all_stats["goals_scored_L3"] = roll_mean("goals_scored", 3)
    all_stats["goals_scored_L5"] = roll_mean("goals_scored", 5)
    all_stats["goals_scored_L10"] = roll_mean("goals_scored", 10)
    all_stats["goals_allowed_L3"] = roll_mean("goals_allowed", 3)
    all_stats["goals_allowed_L5"] = roll_mean("goals_allowed", 5)
    all_stats["goals_allowed_L10"] = roll_mean("goals_allowed", 10)

    all_stats["goal_diff_std_L10"] = roll_std("goal_diff", 10)
    all_stats["goals_scored_std_L10"] = roll_std("goals_scored", 10)
    all_stats["goals_allowed_std_L10"] = roll_std("goals_allowed", 10)

    all_stats["btts_rate_L10"] = roll_mean("btts_yes", 10)
    all_stats["over_25_rate_L10"] = roll_mean("over_25", 10)

    return all_stats[
        [
            "game_id",
            "team",
            "rest_days",
            "is_b2b",
            "games_last_3_days",
            "games_last_5_days",
            "games_last_7_days",
            "games_last_10_days",
            "games_last_14_days",
            "win_pct_L5",
            "win_pct_L10",
            "draw_pct_L5",
            "draw_pct_L10",
            "goal_diff_L3",
            "goal_diff_L5",
            "goal_diff_L10",
            "goals_scored_L3",
            "goals_scored_L5",
            "goals_scored_L10",
            "goals_allowed_L3",
            "goals_allowed_L5",
            "goals_allowed_L10",
            "goal_diff_std_L10",
            "goals_scored_std_L10",
            "goals_allowed_std_L10",
            "btts_rate_L10",
            "over_25_rate_L10",
        ]
    ].copy()


def calculate_surface_split_features(df: pd.DataFrame):
    """Calcula features separados para HOME y AWAY surface."""
    print("[SURFACE] Generando splits Home/Away (Liga MX)...")

    # HOME ONLY
    home_only = df[
        ["date", "game_id", "home_team", "home_score", "away_score", "total_goals", "is_draw"]
    ].copy()
    home_only.columns = ["date", "game_id", "team", "goals_scored", "goals_allowed", "total_goals", "drew"]
    home_only["date_dt"] = pd.to_datetime(home_only["date"])
    home_only = home_only.sort_values(["team", "date_dt", "game_id"]).reset_index(drop=True)

    home_only["won_game"] = (home_only["goals_scored"] > home_only["goals_allowed"]).astype(int)
    home_only["goal_diff"] = home_only["goals_scored"] - home_only["goals_allowed"]
    home_only["draw"] = home_only["drew"].astype(int)
    home_only["btts_yes"] = ((home_only["goals_scored"] > 0) & (home_only["goals_allowed"] > 0)).astype(int)
    home_only["over_25"] = (home_only["total_goals"] > 2.5).astype(int)

    def roll_home(col: str, window: int) -> pd.Series:
        return home_only.groupby("team")[col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

    home_only["home_only_win_pct_L5"] = roll_home("won_game", 5)
    home_only["home_only_goal_diff_L5"] = roll_home("goal_diff", 5)
    home_only["home_only_draw_pct_L5"] = roll_home("draw", 5)
    home_only["home_only_btts_rate_L10"] = roll_home("btts_yes", 10)
    home_only["home_only_over_25_rate_L10"] = roll_home("over_25", 10)

    home_features = home_only[
        [
            "game_id",
            "team",
            "home_only_win_pct_L5",
            "home_only_goal_diff_L5",
            "home_only_draw_pct_L5",
            "home_only_btts_rate_L10",
            "home_only_over_25_rate_L10",
        ]
    ].copy()

    # AWAY ONLY
    away_only = df[
        ["date", "game_id", "away_team", "away_score", "home_score", "total_goals", "is_draw"]
    ].copy()
    away_only.columns = ["date", "game_id", "team", "goals_scored", "goals_allowed", "total_goals", "drew"]
    away_only["date_dt"] = pd.to_datetime(away_only["date"])
    away_only = away_only.sort_values(["team", "date_dt", "game_id"]).reset_index(drop=True)

    away_only["won_game"] = (away_only["goals_scored"] > away_only["goals_allowed"]).astype(int)
    away_only["goal_diff"] = away_only["goals_scored"] - away_only["goals_allowed"]
    away_only["draw"] = away_only["drew"].astype(int)
    away_only["btts_yes"] = ((away_only["goals_scored"] > 0) & (away_only["goals_allowed"] > 0)).astype(int)
    away_only["over_25"] = (away_only["total_goals"] > 2.5).astype(int)

    def roll_away(col: str, window: int) -> pd.Series:
        return away_only.groupby("team")[col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

    away_only["away_only_win_pct_L5"] = roll_away("won_game", 5)
    away_only["away_only_goal_diff_L5"] = roll_away("goal_diff", 5)
    away_only["away_only_draw_pct_L5"] = roll_away("draw", 5)
    away_only["away_only_btts_rate_L10"] = roll_away("btts_yes", 10)
    away_only["away_only_over_25_rate_L10"] = roll_away("over_25", 10)

    away_features = away_only[
        [
            "game_id",
            "team",
            "away_only_win_pct_L5",
            "away_only_goal_diff_L5",
            "away_only_draw_pct_L5",
            "away_only_btts_rate_L10",
            "away_only_over_25_rate_L10",
        ]
    ].copy()

    return home_features, away_features


def build_features() -> pd.DataFrame:
    """Construye el dataset final con todas las features para modelos."""
    print("\n[BUILD] Construyendo dataset de features para Liga MX...")

    df = pd.read_csv(RAW_DATA, dtype={"game_id": str})

    for col in ["home_corners", "away_corners", "total_corners", "home_ht_score", "away_ht_score", "total_ht_goals"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date_dt", "game_id"]).reset_index(drop=True)
    # NOTE: Keep date_dt for H2H/Venue features (will drop later)

    # Calcular ELO
    df = calculate_elo_ratings(df)

    # Targets
    df["TARGET_full_game"] = np.select(
        [df["home_score"] > df["away_score"], df["is_draw"] == 1],
        [1, 2],
        default=0,
    )  # 1=home_win, 2=draw, 0=away_win
    df["TARGET_over_25"] = (df["total_goals"] > 2.5).astype(int)
    df["TARGET_btts"] = ((df["home_score"] > 0) & (df["away_score"] > 0)).astype(int)
    df["TARGET_corners_over_95"] = np.where(
        df["total_corners"].notna(),
        (df["total_corners"] > 9.5).astype(int),
        np.nan,
    )
    df["TARGET_ht_result"] = np.where(
        df["home_ht_score"].notna() & df["away_ht_score"].notna(),
        np.select(
            [df["home_ht_score"] > df["away_ht_score"], df["home_ht_score"] == df["away_ht_score"]],
            [1, 2],
            default=0,
        ),
        np.nan,
    )

    # Calcular features de rolling
    rolling_features = calculate_team_rolling_features(df)
    home_surface_features, away_surface_features = calculate_surface_split_features(df)

    # Merge features generales para HOME
    df = pd.merge(
        df,
        rolling_features,
        left_on=["game_id", "home_team"],
        right_on=["game_id", "team"],
        how="left",
    )
    df = df.rename(
        columns={c: f"home_{c}" for c in rolling_features.columns if c not in ["game_id", "team"]}
    ).drop(columns=["team"])

    # Merge features generales para AWAY
    df = pd.merge(
        df,
        rolling_features,
        left_on=["game_id", "away_team"],
        right_on=["game_id", "team"],
        how="left",
    )
    df = df.rename(
        columns={c: f"away_{c}" for c in rolling_features.columns if c not in ["game_id", "team"]}
    ).drop(columns=["team"])

    # Merge surface HOME-only para HOME team
    df = pd.merge(
        df,
        home_surface_features,
        left_on=["game_id", "home_team"],
        right_on=["game_id", "team"],
        how="left",
    )
    df = df.rename(
        columns={c: f"home_{c}" for c in home_surface_features.columns if c not in ["game_id", "team"]}
    ).drop(columns=["team"])

    # Merge surface AWAY-only para AWAY team
    df = pd.merge(
        df,
        away_surface_features,
        left_on=["game_id", "away_team"],
        right_on=["game_id", "team"],
        how="left",
    )
    df = df.rename(
        columns={c: f"away_{c}" for c in away_surface_features.columns if c not in ["game_id", "team"]}
    ).drop(columns=["team"])

    # Drop juegos sin histórico previo suficiente
    df = df.dropna(subset=["home_win_pct_L10", "away_win_pct_L10"]).copy()

    # Rellenar surfaces con 0
    split_cols = [
        "home_home_only_win_pct_L5",
        "home_home_only_goal_diff_L5",
        "home_home_only_draw_pct_L5",
        "home_home_only_btts_rate_L10",
        "home_home_only_over_25_rate_L10",
        "away_away_only_win_pct_L5",
        "away_away_only_goal_diff_L5",
        "away_away_only_draw_pct_L5",
        "away_away_only_btts_rate_L10",
        "away_away_only_over_25_rate_L10",
    ]
    for col in split_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Validar columnas de mercado
    df = ensure_market_columns(df)

    # Market-implied probabilities (strong pre-match signal)
    df["home_ml_implied_prob"] = _american_to_implied_prob_series(df["home_moneyline_odds"])
    df["away_ml_implied_prob"] = _american_to_implied_prob_series(df["away_moneyline_odds"])
    df["market_fav_edge"] = df["home_ml_implied_prob"] - df["away_ml_implied_prob"]
    df["market_fav_abs_edge"] = np.abs(df["market_fav_edge"])
    df["market_total_line"] = pd.to_numeric(df["odds_over_under"], errors="coerce").fillna(2.5)

    # Feature de diferencia ELO
    df["diff_elo"] = df["home_elo_pre"] - df["away_elo_pre"]

    # Features de diferencia general
    df["diff_rest_days"] = df["home_rest_days"] - df["away_rest_days"]
    df["diff_is_b2b"] = df["home_is_b2b"] - df["away_is_b2b"]

    df["diff_games_last_3_days"] = df["home_games_last_3_days"] - df["away_games_last_3_days"]
    df["diff_games_last_5_days"] = df["home_games_last_5_days"] - df["away_games_last_5_days"]
    df["diff_games_last_7_days"] = df["home_games_last_7_days"] - df["away_games_last_7_days"]
    df["diff_games_last_10_days"] = df["home_games_last_10_days"] - df["away_games_last_10_days"]
    df["diff_games_last_14_days"] = df["home_games_last_14_days"] - df["away_games_last_14_days"]

    df["diff_win_pct_L5"] = df["home_win_pct_L5"] - df["away_win_pct_L5"]
    df["diff_win_pct_L10"] = df["home_win_pct_L10"] - df["away_win_pct_L10"]

    df["diff_draw_pct_L5"] = df["home_draw_pct_L5"] - df["away_draw_pct_L5"]
    df["diff_draw_pct_L10"] = df["home_draw_pct_L10"] - df["away_draw_pct_L10"]

    df["diff_goal_diff_L3"] = df["home_goal_diff_L3"] - df["away_goal_diff_L3"]
    df["diff_goal_diff_L5"] = df["home_goal_diff_L5"] - df["away_goal_diff_L5"]
    df["diff_goal_diff_L10"] = df["home_goal_diff_L10"] - df["away_goal_diff_L10"]

    df["diff_goals_scored_L3"] = df["home_goals_scored_L3"] - df["away_goals_scored_L3"]
    df["diff_goals_scored_L5"] = df["home_goals_scored_L5"] - df["away_goals_scored_L5"]
    df["diff_goals_scored_L10"] = df["home_goals_scored_L10"] - df["away_goals_scored_L10"]
    df["diff_goals_allowed_L3"] = df["home_goals_allowed_L3"] - df["away_goals_allowed_L3"]
    df["diff_goals_allowed_L5"] = df["home_goals_allowed_L5"] - df["away_goals_allowed_L5"]
    df["diff_goals_allowed_L10"] = df["home_goals_allowed_L10"] - df["away_goals_allowed_L10"]

    df["diff_goal_diff_std_L10"] = df["home_goal_diff_std_L10"] - df["away_goal_diff_std_L10"]
    df["diff_goals_scored_std_L10"] = df["home_goals_scored_std_L10"] - df["away_goals_scored_std_L10"]
    df["diff_goals_allowed_std_L10"] = df["home_goals_allowed_std_L10"] - df["away_goals_allowed_std_L10"]

    df["diff_btts_rate_L10"] = df["home_btts_rate_L10"] - df["away_btts_rate_L10"]
    df["diff_over_25_rate_L10"] = df["home_over_25_rate_L10"] - df["away_over_25_rate_L10"]

    df["diff_surface_win_pct_L5"] = df["home_home_only_win_pct_L5"] - df["away_away_only_win_pct_L5"]
    df["diff_surface_goal_diff_L5"] = df["home_home_only_goal_diff_L5"] - df["away_away_only_goal_diff_L5"]
    df["diff_surface_draw_pct_L5"] = df["home_home_only_draw_pct_L5"] - df["away_away_only_draw_pct_L5"]
    df["diff_surface_btts_rate_L10"] = df["home_home_only_btts_rate_L10"] - df["away_away_only_btts_rate_L10"]
    df["diff_surface_over_25_rate_L10"] = df["home_home_only_over_25_rate_L10"] - df["away_away_only_over_25_rate_L10"]

    # Derivadas de forma reciente y contexto competitivo (sin leakage; todo viene de ventanas shift(1))
    df["home_momentum_win"] = df["home_win_pct_L5"] - df["home_win_pct_L10"]
    df["away_momentum_win"] = df["away_win_pct_L5"] - df["away_win_pct_L10"]
    df["diff_momentum_win"] = df["home_momentum_win"] - df["away_momentum_win"]

    df["home_momentum_goal_diff"] = df["home_goal_diff_L5"] - df["home_goal_diff_L10"]
    df["away_momentum_goal_diff"] = df["away_goal_diff_L5"] - df["away_goal_diff_L10"]
    df["diff_momentum_goal_diff"] = df["home_momentum_goal_diff"] - df["away_momentum_goal_diff"]

    df["home_surface_edge"] = df["home_home_only_win_pct_L5"] - df["home_win_pct_L10"]
    df["away_surface_edge"] = df["away_away_only_win_pct_L5"] - df["away_win_pct_L10"]
    df["diff_surface_edge"] = df["home_surface_edge"] - df["away_surface_edge"]

    df["home_fatigue_index"] = df["home_games_last_7_days"] - df["home_rest_days"]
    df["away_fatigue_index"] = df["away_games_last_7_days"] - df["away_rest_days"]
    df["diff_fatigue_index"] = df["home_fatigue_index"] - df["away_fatigue_index"]

    df["home_form_power"] = df["home_win_pct_L10"] * df["home_goal_diff_L10"]
    df["away_form_power"] = df["away_win_pct_L10"] * df["away_goal_diff_L10"]
    df["diff_form_power"] = df["home_form_power"] - df["away_form_power"]

    # Señales de partido cerrado / tendencia al empate
    df["match_parity_elo"] = np.abs(df["diff_elo"])
    df["match_parity_goal_diff"] = np.abs(df["diff_goal_diff_L10"])
    df["draw_pressure_avg"] = (df["home_draw_pct_L10"] + df["away_draw_pct_L10"]) / 2.0
    df["draw_pressure_surface_avg"] = (
        df["home_home_only_draw_pct_L5"] + df["away_away_only_draw_pct_L5"]
    ) / 2.0

    # Carga de calendario ponderada (más peso a ventanas cortas)
    df["home_schedule_load_exp"] = (
        1.00 * df["home_games_last_3_days"]
        + 0.70 * df["home_games_last_5_days"]
        + 0.40 * df["home_games_last_7_days"]
        + 0.25 * df["home_games_last_10_days"]
        + 0.15 * df["home_games_last_14_days"]
    )
    df["away_schedule_load_exp"] = (
        1.00 * df["away_games_last_3_days"]
        + 0.70 * df["away_games_last_5_days"]
        + 0.40 * df["away_games_last_7_days"]
        + 0.25 * df["away_games_last_10_days"]
        + 0.15 * df["away_games_last_14_days"]
    )
    df["diff_schedule_load_exp"] = df["home_schedule_load_exp"] - df["away_schedule_load_exp"]

    # Tendencias de cierre (L3 vs L10)
    df["home_attack_closing_trend"] = df["home_goals_scored_L3"] - df["home_goals_scored_L10"]
    df["away_attack_closing_trend"] = df["away_goals_scored_L3"] - df["away_goals_scored_L10"]
    df["diff_attack_closing_trend"] = (
        df["home_attack_closing_trend"] - df["away_attack_closing_trend"]
    )

    df["home_defense_closing_trend"] = df["home_goals_allowed_L10"] - df["home_goals_allowed_L3"]
    df["away_defense_closing_trend"] = df["away_goals_allowed_L10"] - df["away_goals_allowed_L3"]
    df["diff_defense_closing_trend"] = (
        df["home_defense_closing_trend"] - df["away_defense_closing_trend"]
    )

    # Estabilidad por varianza reciente
    df["home_match_stability_L10"] = 1.0 / (1.0 + df["home_goal_diff_std_L10"])
    df["away_match_stability_L10"] = 1.0 / (1.0 + df["away_goal_diff_std_L10"])
    df["diff_match_stability_L10"] = (
        df["home_match_stability_L10"] - df["away_match_stability_L10"]
    )

    # Índice combinado de equilibrio orientado a empate
    parity_elo_score = np.exp(-np.abs(df["diff_elo"]) / 120.0)
    parity_goal_score = np.exp(-np.abs(df["diff_goal_diff_L10"]) / 1.5)
    draw_tendency_score = (
        df["home_draw_pct_L10"]
        + df["away_draw_pct_L10"]
        + df["home_home_only_draw_pct_L5"]
        + df["away_away_only_draw_pct_L5"]
    ) / 4.0
    df["draw_equilibrium_index"] = draw_tendency_score * ((parity_elo_score + parity_goal_score) / 2.0)

    # ====== ADVANCED FEATURES FOR FULL_GAME ======
    print("\n[ADVANCED] Integrando features avanzadas para full_game...")
    try:
        # 1. Home/Away streaks
        print("  1/4 Calculando rachas por local/visita...")
        home_streaks, away_streaks = calculate_home_away_streaks(df)
        df = pd.merge(df, home_streaks, left_on=["game_id", "home_team"], 
                     right_on=["game_id", "team"], how="left").drop(columns=["team"], errors="ignore")
        df = pd.merge(df, away_streaks, left_on=["game_id", "away_team"], 
                     right_on=["game_id", "team"], how="left").drop(columns=["team"], errors="ignore")
        
        # 2. Draw parity strength
        print("  2/4 Calculando fuerza de empate por paridad...")
        draw_features = calculate_draw_parity_strength(df)
        for col in draw_features.columns:
            df[col] = draw_features[col]
        
        # 3. Fatigue differential
        print("  3/4 Calculando ventaja de fatiga...")
        fatigue_features = calculate_fatigue_differential(df)
        for col in fatigue_features.columns:
            df[col] = fatigue_features[col]
        
        # 4. Opponent-adjusted form
        print("  4/4 Calculando forma ponderada por rival...")
        form_features = calculate_opponent_quality_adjusted_form(df)
        for col in form_features.columns:
            df[col] = form_features[col]

        print("  [OK] Advanced features agregadas exitosamente")
    except Exception as e:
        print(f"  [WARN] Error agregando advanced features: {e}")
        import traceback
        traceback.print_exc()
        print("     Continuando con features baseline...")

    # Drop helper date column before selecting final model columns.
    if "date_dt" in df.columns:
        df = df.drop(columns=["date_dt"])

    # Seleccionar columnas finales
    model_columns = [
        "game_id",
        "date",
        "season",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_ht_score",
        "away_ht_score",
        "total_ht_goals",
        "is_draw",
        "total_goals",
        "TARGET_full_game",
        "TARGET_ht_result",
        "TARGET_over_25",
        "TARGET_btts",
        "TARGET_corners_over_95",
        "home_elo_pre",
        "away_elo_pre",
        "diff_elo",
        "home_rest_days",
        "away_rest_days",
        "diff_rest_days",
        "home_is_b2b",
        "away_is_b2b",
        "diff_is_b2b",
        "home_games_last_3_days",
        "away_games_last_3_days",
        "diff_games_last_3_days",
        "home_games_last_5_days",
        "away_games_last_5_days",
        "diff_games_last_5_days",
        "home_games_last_7_days",
        "away_games_last_7_days",
        "diff_games_last_7_days",
        "home_games_last_10_days",
        "away_games_last_10_days",
        "diff_games_last_10_days",
        "home_games_last_14_days",
        "away_games_last_14_days",
        "diff_games_last_14_days",
        "home_win_pct_L5",
        "away_win_pct_L5",
        "diff_win_pct_L5",
        "home_win_pct_L10",
        "away_win_pct_L10",
        "diff_win_pct_L10",
        "home_draw_pct_L5",
        "away_draw_pct_L5",
        "diff_draw_pct_L5",
        "home_draw_pct_L10",
        "away_draw_pct_L10",
        "diff_draw_pct_L10",
        "home_goal_diff_L3",
        "away_goal_diff_L3",
        "diff_goal_diff_L3",
        "home_goal_diff_L5",
        "away_goal_diff_L5",
        "diff_goal_diff_L5",
        "home_goal_diff_L10",
        "away_goal_diff_L10",
        "diff_goal_diff_L10",
        "home_goals_scored_L3",
        "away_goals_scored_L3",
        "diff_goals_scored_L3",
        "home_goals_scored_L5",
        "away_goals_scored_L5",
        "diff_goals_scored_L5",
        "home_goals_scored_L10",
        "away_goals_scored_L10",
        "diff_goals_scored_L10",
        "home_goals_allowed_L3",
        "away_goals_allowed_L3",
        "diff_goals_allowed_L3",
        "home_goals_allowed_L5",
        "away_goals_allowed_L5",
        "diff_goals_allowed_L5",
        "home_goals_allowed_L10",
        "away_goals_allowed_L10",
        "diff_goals_allowed_L10",
        "home_goal_diff_std_L10",
        "away_goal_diff_std_L10",
        "diff_goal_diff_std_L10",
        "home_goals_scored_std_L10",
        "away_goals_scored_std_L10",
        "diff_goals_scored_std_L10",
        "home_goals_allowed_std_L10",
        "away_goals_allowed_std_L10",
        "diff_goals_allowed_std_L10",
        "home_btts_rate_L10",
        "away_btts_rate_L10",
        "diff_btts_rate_L10",
        "home_over_25_rate_L10",
        "away_over_25_rate_L10",
        "diff_over_25_rate_L10",
        "home_home_only_win_pct_L5",
        "home_home_only_goal_diff_L5",
        "home_home_only_draw_pct_L5",
        "home_home_only_btts_rate_L10",
        "home_home_only_over_25_rate_L10",
        "away_away_only_win_pct_L5",
        "away_away_only_goal_diff_L5",
        "away_away_only_draw_pct_L5",
        "away_away_only_btts_rate_L10",
        "away_away_only_over_25_rate_L10",
        "diff_surface_win_pct_L5",
        "diff_surface_goal_diff_L5",
        "diff_surface_draw_pct_L5",
        "diff_surface_btts_rate_L10",
        "diff_surface_over_25_rate_L10",

        "home_momentum_win",
        "away_momentum_win",
        "diff_momentum_win",

        "home_momentum_goal_diff",
        "away_momentum_goal_diff",
        "diff_momentum_goal_diff",

        "home_surface_edge",
        "away_surface_edge",
        "diff_surface_edge",

        "home_fatigue_index",
        "away_fatigue_index",
        "diff_fatigue_index",

        "home_form_power",
        "away_form_power",
        "diff_form_power",

        "match_parity_elo",
        "match_parity_goal_diff",
        "draw_pressure_avg",
        "draw_pressure_surface_avg",
        "home_schedule_load_exp",
        "away_schedule_load_exp",
        "diff_schedule_load_exp",
        "home_attack_closing_trend",
        "away_attack_closing_trend",
        "diff_attack_closing_trend",
        "home_defense_closing_trend",
        "away_defense_closing_trend",
        "diff_defense_closing_trend",
        "home_match_stability_L10",
        "away_match_stability_L10",
        "diff_match_stability_L10",
        "draw_equilibrium_index",

        # Advanced features for full_game
        "home_win_streak_L5",
        "home_win_streak_L10",
        "home_dominance_L10",
        "away_win_streak_L5",
        "away_win_streak_L10",
        "away_dominance_L10",
        "draw_parity_strength",
        "fatigue_advantage",
        "home_form_vs_strong",
        "away_form_vs_strong",

        # Market-driven full-game features
        "home_moneyline_odds",
        "away_moneyline_odds",
        "closing_moneyline_odds",
        "closing_spread_odds",
        "closing_total_odds",
        "home_ml_implied_prob",
        "away_ml_implied_prob",
        "market_fav_edge",
        "market_fav_abs_edge",
        "market_total_line",

        "odds_over_under",
        "market_missing",
    ]

    # Filter to only existing columns (advanced features might not be generated)
    model_columns = [c for c in model_columns if c in df.columns]
    
    df = df[model_columns].copy()

    # Validar que no haya NaN
    print(f"\n[CHECK] Revisando NaN antes de guardar...")
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        print(f"   Columnas con NaN: {nan_cols}")
        preserve_nan = {
            "home_ht_score",
            "away_ht_score",
            "total_ht_goals",
            "TARGET_ht_result",
            "TARGET_corners_over_95",
        }
        fill_cols = [c for c in df.columns if c not in preserve_nan]
        df.loc[:, fill_cols] = df[fill_cols].fillna(0)

    return df


def main():
    print("=" * 70)
    print("[FEATURE ENGINEERING] PARA LIGA MX")
    print("=" * 70)

    if not RAW_DATA.exists():
        raise FileNotFoundError(f"No existe el archivo: {RAW_DATA}\nPrimero corre data_ingest_liga_mx.py")

    df = build_features()

    print(f"\n[SUMMARY] Dataset generado:")
    print(f"   - Partidos: {len(df)}")
    print(f"   - Columnas: {len(df.columns)}")
    corners_n = int(df["TARGET_corners_over_95"].notna().sum()) if "TARGET_corners_over_95" in df.columns else 0
    ht_n = int(df["TARGET_ht_result"].notna().sum()) if "TARGET_ht_result" in df.columns else 0
    print(
        f"   - Targets: {df['TARGET_full_game'].nunique()} full_game, "
        f"{df['TARGET_over_25'].nunique()} over_25, {df['TARGET_btts'].nunique()} btts, "
        f"ht_result muestras={ht_n}, corners_over_95 muestras={corners_n}"
    )

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[SUCCESS] Features guardadas en: {OUTPUT_FILE}")

    print(f"\n[PREVIEW] Primeras filas:")
    print(df.head(3))


if __name__ == "__main__":
    main()
