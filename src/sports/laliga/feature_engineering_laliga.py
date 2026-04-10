import numpy as np
import pandas as pd
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = SRC_ROOT
RAW_DATA = BASE_DIR / "data" / "laliga" / "raw" / "laliga_advanced_history.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "laliga" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = PROCESSED_DATA_DIR / "model_ready_features_laliga.csv"


def calculate_elo_ratings(df: pd.DataFrame, k: float = 16, home_advantage: float = 50) -> pd.DataFrame:
    """
    Calcula ratings ELO para todos los equipos usando histÃ³rico de partidos.
    Adaptado para fÃºtbol con home_advantage = 50 (menor que en NBA/MLB).
    """
    print("ðŸ“ˆ Calculando Sistema ELO para LaLiga EA Sports...")

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
    """Asegura que existan columnas de mercado y deriva features de odds 1X2."""
    market_cols = ["odds_over_under", "home_price", "draw_price", "away_price"]

    for col in market_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    df["market_missing"] = df[market_cols].isna().any(axis=1).astype(int)
    df["odds_over_under"] = df["odds_over_under"].fillna(2.5)
    df["home_price"] = df["home_price"].fillna(3.0)
    df["draw_price"] = df["draw_price"].fillna(3.3)
    df["away_price"] = df["away_price"].fillna(3.0)

    def decimal_to_prob(series: pd.Series) -> pd.Series:
        safe = pd.to_numeric(series, errors="coerce")
        return np.where(safe > 1.0, 1.0 / safe, np.nan)

    home_raw = pd.Series(decimal_to_prob(df["home_price"]), index=df.index)
    draw_raw = pd.Series(decimal_to_prob(df["draw_price"]), index=df.index)
    away_raw = pd.Series(decimal_to_prob(df["away_price"]), index=df.index)
    overround = (home_raw + draw_raw + away_raw).replace(0, np.nan)

    df["home_implied_prob"] = (home_raw / overround).fillna(1.0 / 3.0)
    df["draw_implied_prob"] = (draw_raw / overround).fillna(1.0 / 3.0)
    df["away_implied_prob"] = (away_raw / overround).fillna(1.0 / 3.0)
    df["market_overround_1x2"] = overround.fillna(1.0)
    df["market_home_edge"] = df["home_implied_prob"] - df["away_implied_prob"]
    df["market_draw_bias"] = df["draw_implied_prob"]
    top_probs = np.sort(df[["home_implied_prob", "draw_implied_prob", "away_implied_prob"]].to_numpy(dtype=float), axis=1)
    df["market_favorite_prob"] = top_probs[:, 2]
    df["market_favorite_gap"] = top_probs[:, 2] - top_probs[:, 1]

    return df


def count_games_in_last_days(group: pd.DataFrame, days: int) -> pd.Series:
    """Cuenta cuÃ¡ntos partidos tuvo un equipo en los Ãºltimos N dÃ­as."""
    dates = group["date_dt"].to_numpy(dtype="datetime64[D]")
    counts = np.zeros(len(dates), dtype=int)

    for i in range(len(dates)):
        current_date = dates[i]
        start_date = current_date - np.timedelta64(days, "D")
        counts[i] = np.sum((dates[:i] >= start_date) & (dates[:i] < current_date))

    return pd.Series(counts, index=group.index)


def calculate_team_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula variables generales de rolling por equipo."""
    print("âš™ï¸ Generando variables generales por equipo (LaLiga EA Sports)...")

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
    print("ðŸ âœˆï¸ Generando splits Home/Away (LaLiga EA Sports)...")

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
    print("\nðŸ”¨ Construyendo dataset de features para LaLiga EA Sports...")

    df = pd.read_csv(RAW_DATA, dtype={"game_id": str})

    for col in ["home_corners", "away_corners", "total_corners"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date_dt", "game_id"]).reset_index(drop=True)
    df = df.drop(columns=["date_dt"])

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

    # Drop juegos sin histÃ³rico previo suficiente
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

    # SeÃ±ales de partido cerrado / tendencia al empate
    df["match_parity_elo"] = np.abs(df["diff_elo"])
    df["match_parity_goal_diff"] = np.abs(df["diff_goal_diff_L10"])
    df["draw_pressure_avg"] = (df["home_draw_pct_L10"] + df["away_draw_pct_L10"]) / 2.0
    df["draw_pressure_surface_avg"] = (
        df["home_home_only_draw_pct_L5"] + df["away_away_only_draw_pct_L5"]
    ) / 2.0

    # Carga de calendario ponderada (mÃ¡s peso a ventanas cortas)
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

    # Ãndice combinado de equilibrio orientado a empate
    parity_elo_score = np.exp(-np.abs(df["diff_elo"]) / 120.0)
    parity_goal_score = np.exp(-np.abs(df["diff_goal_diff_L10"]) / 1.5)
    draw_tendency_score = (
        df["home_draw_pct_L10"]
        + df["away_draw_pct_L10"]
        + df["home_home_only_draw_pct_L5"]
        + df["away_away_only_draw_pct_L5"]
    ) / 4.0
    df["draw_equilibrium_index"] = draw_tendency_score * ((parity_elo_score + parity_goal_score) / 2.0)

    # Seleccionar columnas finales
    model_columns = [
        "game_id",
        "date",
        "season",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "is_draw",
        "total_goals",
        "TARGET_full_game",
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

        "odds_over_under",
        "home_price",
        "draw_price",
        "away_price",
        "home_implied_prob",
        "draw_implied_prob",
        "away_implied_prob",
        "market_overround_1x2",
        "market_home_edge",
        "market_draw_bias",
        "market_favorite_prob",
        "market_favorite_gap",
        "market_missing",
    ]

    df = df[model_columns].copy()

    # Validar que no haya NaN
    print(f"\nâš ï¸ Revisando NaN antes de guardar...")
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        print(f"   Columnas con NaN: {nan_cols}")
        df = df.fillna(0)

    return df


def main():
    print("=" * 70)
    print("ðŸŽ¯ FEATURE ENGINEERING PARA LaLiga EA Sports")
    print("=" * 70)

    if not RAW_DATA.exists():
        raise FileNotFoundError(f"No existe el archivo: {RAW_DATA}\nPrimero corre data_ingest_laliga.py")

    df = build_features()

    print(f"\nðŸ“Š Dataset generado:")
    print(f"   - Partidos: {len(df)}")
    print(f"   - Columnas: {len(df.columns)}")
    corners_n = int(df["TARGET_corners_over_95"].notna().sum()) if "TARGET_corners_over_95" in df.columns else 0
    print(
        f"   - Targets: {df['TARGET_full_game'].nunique()} full_game, "
        f"{df['TARGET_over_25'].nunique()} over_25, {df['TARGET_btts'].nunique()} btts, "
        f"corners_over_95 muestras={corners_n}"
    )

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nâœ… Features guardadas en: {OUTPUT_FILE}")

    print(f"\nðŸ“‹ Primeras filas:")
    print(df.head(3))


if __name__ == "__main__":
    main()

