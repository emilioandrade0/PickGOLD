import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA = BASE_DIR / "data" / "kbo" / "raw" / "kbo_advanced_history.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "kbo" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = PROCESSED_DATA_DIR / "model_ready_features_kbo.csv"


def calculate_elo_ratings(df: pd.DataFrame, k: float = 16, home_advantage: float = 35) -> pd.DataFrame:
    print("ðŸ“ˆ Calculando Sistema ELO kbo...")

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

        actual_home = 1 if row["home_runs_total"] > row["away_runs_total"] else 0
        actual_away = 1 - actual_home

        elo_dict[home] = home_elo_pre + k * (actual_home - expected_home)
        elo_dict[away] = away_elo_pre + k * (actual_away - expected_away)

    df["home_elo_pre"] = elo_home_before
    df["away_elo_pre"] = elo_away_before
    return df


def ensure_market_columns(df: pd.DataFrame) -> pd.DataFrame:
    market_cols = ["home_is_favorite", "odds_over_under"]

    for col in market_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    df["market_missing"] = df[market_cols].isna().any(axis=1).astype(int)

    for col in market_cols:
        df[col] = df[col].fillna(0)

    return df


def count_games_in_last_days(group: pd.DataFrame, days: int) -> pd.Series:
    dates = group["date_dt"].to_numpy(dtype="datetime64[D]")
    counts = np.zeros(len(dates), dtype=int)

    for i in range(len(dates)):
        current_date = dates[i]
        start_date = current_date - np.timedelta64(days, "D")
        counts[i] = np.sum((dates[:i] >= start_date) & (dates[:i] < current_date))

    return pd.Series(counts, index=group.index)


def calculate_team_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    print("âš™ï¸ Generando variables generales kbo por equipo...")

    home_df = df[
        [
            "date",
            "game_id",
            "home_team",
            "home_runs_total",
            "away_runs_total",
            "home_r1",
            "away_r1",
            "home_runs_f5",
            "away_runs_f5",
            "home_hits",
            "away_hits",
        ]
    ].copy()
    home_df.columns = [
        "date",
        "game_id",
        "team",
        "runs_scored",
        "runs_allowed",
        "r1_scored",
        "r1_allowed",
        "runs_f5_scored",
        "runs_f5_allowed",
        "hits",
        "hits_allowed",
    ]
    home_df["is_home"] = 1

    away_df = df[
        [
            "date",
            "game_id",
            "away_team",
            "away_runs_total",
            "home_runs_total",
            "away_r1",
            "home_r1",
            "away_runs_f5",
            "home_runs_f5",
            "away_hits",
            "home_hits",
        ]
    ].copy()
    away_df.columns = [
        "date",
        "game_id",
        "team",
        "runs_scored",
        "runs_allowed",
        "r1_scored",
        "r1_allowed",
        "runs_f5_scored",
        "runs_f5_allowed",
        "hits",
        "hits_allowed",
    ]
    away_df["is_home"] = 0

    all_stats = pd.concat([home_df, away_df], ignore_index=True).copy()
    all_stats["date_dt"] = pd.to_datetime(all_stats["date"])
    all_stats = all_stats.sort_values(["team", "date_dt", "game_id"]).reset_index(drop=True)

    all_stats["won_game"] = (all_stats["runs_scored"] > all_stats["runs_allowed"]).astype(int)
    all_stats["run_diff"] = all_stats["runs_scored"] - all_stats["runs_allowed"]

    all_stats["yrfi_yes_team"] = ((all_stats["r1_scored"] + all_stats["r1_allowed"]) > 0).astype(int)
    all_stats["r1_scored_flag"] = (all_stats["r1_scored"] > 0).astype(int)
    all_stats["r1_allowed_flag"] = (all_stats["r1_allowed"] > 0).astype(int)

    all_stats["won_f5"] = (all_stats["runs_f5_scored"] > all_stats["runs_f5_allowed"]).astype(int)
    all_stats["f5_diff"] = all_stats["runs_f5_scored"] - all_stats["runs_f5_allowed"]

    all_stats["last_game_date"] = all_stats.groupby("team")["date_dt"].shift(1)
    all_stats["rest_days"] = (
        (all_stats["date_dt"] - all_stats["last_game_date"]).dt.days.fillna(3).clip(lower=0, upper=10)
    )
    all_stats["is_b2b"] = (all_stats["rest_days"] <= 1).astype(int)

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

    def roll_mean(col: str, window: int) -> pd.Series:
        return all_stats.groupby("team")[col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

    all_stats["win_pct_L5"] = roll_mean("won_game", 5)
    all_stats["win_pct_L10"] = roll_mean("won_game", 10)

    all_stats["run_diff_L5"] = roll_mean("run_diff", 5)
    all_stats["run_diff_L10"] = roll_mean("run_diff", 10)

    all_stats["runs_scored_L5"] = roll_mean("runs_scored", 5)
    all_stats["runs_allowed_L5"] = roll_mean("runs_allowed", 5)

    all_stats["yrfi_rate_L10"] = roll_mean("yrfi_yes_team", 10)
    all_stats["r1_scored_rate_L10"] = roll_mean("r1_scored_flag", 10)
    all_stats["r1_allowed_rate_L10"] = roll_mean("r1_allowed_flag", 10)

    all_stats["f5_win_pct_L5"] = roll_mean("won_f5", 5)
    all_stats["f5_diff_L5"] = roll_mean("f5_diff", 5)

    all_stats["hits_L5"] = roll_mean("hits", 5)
    all_stats["hits_allowed_L5"] = roll_mean("hits_allowed", 5)

    return all_stats[
        [
            "game_id",
            "team",
            "rest_days",
            "is_b2b",
            "games_last_3_days",
            "games_last_5_days",
            "games_last_7_days",
            "win_pct_L5",
            "win_pct_L10",
            "run_diff_L5",
            "run_diff_L10",
            "runs_scored_L5",
            "runs_allowed_L5",
            "yrfi_rate_L10",
            "r1_scored_rate_L10",
            "r1_allowed_rate_L10",
            "f5_win_pct_L5",
            "f5_diff_L5",
            "hits_L5",
            "hits_allowed_L5",
        ]
    ].copy()


def calculate_surface_split_features(df: pd.DataFrame):
    print("ðŸ âœˆï¸ Generando splits Home/Away kbo...")

    home_only = df[
        [
            "date",
            "game_id",
            "home_team",
            "home_runs_total",
            "away_runs_total",
            "home_r1",
            "away_r1",
            "home_runs_f5",
            "away_runs_f5",
        ]
    ].copy()
    home_only.columns = [
        "date",
        "game_id",
        "team",
        "runs_scored",
        "runs_allowed",
        "r1_scored",
        "r1_allowed",
        "runs_f5_scored",
        "runs_f5_allowed",
    ]
    home_only["date_dt"] = pd.to_datetime(home_only["date"])
    home_only = home_only.sort_values(["team", "date_dt", "game_id"]).reset_index(drop=True)

    home_only["won_game"] = (home_only["runs_scored"] > home_only["runs_allowed"]).astype(int)
    home_only["run_diff"] = home_only["runs_scored"] - home_only["runs_allowed"]
    home_only["yrfi_yes_team"] = ((home_only["r1_scored"] + home_only["r1_allowed"]) > 0).astype(int)
    home_only["won_f5"] = (home_only["runs_f5_scored"] > home_only["runs_f5_allowed"]).astype(int)

    def roll_home(col: str, window: int) -> pd.Series:
        return home_only.groupby("team")[col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

    home_only["home_only_win_pct_L5"] = roll_home("won_game", 5)
    home_only["home_only_run_diff_L5"] = roll_home("run_diff", 5)
    home_only["home_only_yrfi_rate_L10"] = roll_home("yrfi_yes_team", 10)
    home_only["home_only_f5_win_pct_L5"] = roll_home("won_f5", 5)

    home_features = home_only[
        [
            "game_id",
            "team",
            "home_only_win_pct_L5",
            "home_only_run_diff_L5",
            "home_only_yrfi_rate_L10",
            "home_only_f5_win_pct_L5",
        ]
    ].copy()

    away_only = df[
        [
            "date",
            "game_id",
            "away_team",
            "away_runs_total",
            "home_runs_total",
            "away_r1",
            "home_r1",
            "away_runs_f5",
            "home_runs_f5",
        ]
    ].copy()
    away_only.columns = [
        "date",
        "game_id",
        "team",
        "runs_scored",
        "runs_allowed",
        "r1_scored",
        "r1_allowed",
        "runs_f5_scored",
        "runs_f5_allowed",
    ]
    away_only["date_dt"] = pd.to_datetime(away_only["date"])
    away_only = away_only.sort_values(["team", "date_dt", "game_id"]).reset_index(drop=True)

    away_only["won_game"] = (away_only["runs_scored"] > away_only["runs_allowed"]).astype(int)
    away_only["run_diff"] = away_only["runs_scored"] - away_only["runs_allowed"]
    away_only["yrfi_yes_team"] = ((away_only["r1_scored"] + away_only["r1_allowed"]) > 0).astype(int)
    away_only["won_f5"] = (away_only["runs_f5_scored"] > away_only["runs_f5_allowed"]).astype(int)

    def roll_away(col: str, window: int) -> pd.Series:
        return away_only.groupby("team")[col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

    away_only["away_only_win_pct_L5"] = roll_away("won_game", 5)
    away_only["away_only_run_diff_L5"] = roll_away("run_diff", 5)
    away_only["away_only_yrfi_rate_L10"] = roll_away("yrfi_yes_team", 10)
    away_only["away_only_f5_win_pct_L5"] = roll_away("won_f5", 5)

    away_features = away_only[
        [
            "game_id",
            "team",
            "away_only_win_pct_L5",
            "away_only_run_diff_L5",
            "away_only_yrfi_rate_L10",
            "away_only_f5_win_pct_L5",
        ]
    ].copy()

    return home_features, away_features


def add_league_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    print("ðŸŒ Generando features kbo relativas a promedio de liga...")

    home_snapshot = df[
        [
            "date",
            "home_win_pct_L10",
            "home_run_diff_L10",
            "home_yrfi_rate_L10",
            "home_f5_win_pct_L5",
        ]
    ].rename(
        columns={
            "home_win_pct_L10": "win_pct_L10",
            "home_run_diff_L10": "run_diff_L10",
            "home_yrfi_rate_L10": "yrfi_rate_L10",
            "home_f5_win_pct_L5": "f5_win_pct_L5",
        }
    )

    away_snapshot = df[
        [
            "date",
            "away_win_pct_L10",
            "away_run_diff_L10",
            "away_yrfi_rate_L10",
            "away_f5_win_pct_L5",
        ]
    ].rename(
        columns={
            "away_win_pct_L10": "win_pct_L10",
            "away_run_diff_L10": "run_diff_L10",
            "away_yrfi_rate_L10": "yrfi_rate_L10",
            "away_f5_win_pct_L5": "f5_win_pct_L5",
        }
    )

    league_snapshot = pd.concat([home_snapshot, away_snapshot], ignore_index=True)
    league_means = (
        league_snapshot.groupby("date", as_index=False)
        .agg(
            league_win_pct_L10=("win_pct_L10", "mean"),
            league_run_diff_L10=("run_diff_L10", "mean"),
            league_yrfi_rate_L10=("yrfi_rate_L10", "mean"),
            league_f5_win_pct_L5=("f5_win_pct_L5", "mean"),
        )
    )

    df = df.merge(league_means, on="date", how="left")

    df["home_win_pct_L10_vs_league"] = df["home_win_pct_L10"] - df["league_win_pct_L10"]
    df["away_win_pct_L10_vs_league"] = df["away_win_pct_L10"] - df["league_win_pct_L10"]
    df["diff_win_pct_L10_vs_league"] = (
        df["home_win_pct_L10_vs_league"] - df["away_win_pct_L10_vs_league"]
    )

    df["home_run_diff_L10_vs_league"] = df["home_run_diff_L10"] - df["league_run_diff_L10"]
    df["away_run_diff_L10_vs_league"] = df["away_run_diff_L10"] - df["league_run_diff_L10"]
    df["diff_run_diff_L10_vs_league"] = (
        df["home_run_diff_L10_vs_league"] - df["away_run_diff_L10_vs_league"]
    )

    df["home_yrfi_rate_L10_vs_league"] = df["home_yrfi_rate_L10"] - df["league_yrfi_rate_L10"]
    df["away_yrfi_rate_L10_vs_league"] = df["away_yrfi_rate_L10"] - df["league_yrfi_rate_L10"]
    df["diff_yrfi_rate_L10_vs_league"] = (
        df["home_yrfi_rate_L10_vs_league"] - df["away_yrfi_rate_L10_vs_league"]
    )

    df["home_f5_win_pct_L5_vs_league"] = df["home_f5_win_pct_L5"] - df["league_f5_win_pct_L5"]
    df["away_f5_win_pct_L5_vs_league"] = df["away_f5_win_pct_L5"] - df["league_f5_win_pct_L5"]
    df["diff_f5_win_pct_L5_vs_league"] = (
        df["home_f5_win_pct_L5_vs_league"] - df["away_f5_win_pct_L5_vs_league"]
    )

    return df


def build_features() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA)

    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date_dt", "game_id"]).reset_index(drop=True)
    df = df.drop(columns=["date_dt"])

    df = calculate_elo_ratings(df)

    df["TARGET_home_win"] = (df["home_runs_total"] > df["away_runs_total"]).astype(int)

    detail_ok = pd.Series(True, index=df.index)
    if "detail_parsed" in df.columns:
        detail_ok = df["detail_parsed"].fillna(0).astype(int) == 1

    yrfi_known = detail_ok & df["home_r1"].notna() & df["away_r1"].notna()
    f5_known = detail_ok & df["home_runs_f5"].notna() & df["away_runs_f5"].notna()

    df["TARGET_yrfi"] = np.where(yrfi_known, ((df["home_r1"] + df["away_r1"]) > 0).astype(int), np.nan)
    df["TARGET_home_win_f5"] = np.where(
        f5_known,
        (df["home_runs_f5"] > df["away_runs_f5"]).astype(int),
        np.nan,
    )

    rolling_features = calculate_team_rolling_features(df)
    home_surface_features, away_surface_features = calculate_surface_split_features(df)

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

    df = df.dropna(subset=["home_win_pct_L10", "away_win_pct_L10"]).copy()

    split_cols = [
        "home_home_only_win_pct_L5",
        "home_home_only_run_diff_L5",
        "home_home_only_yrfi_rate_L10",
        "home_home_only_f5_win_pct_L5",
        "away_away_only_win_pct_L5",
        "away_away_only_run_diff_L5",
        "away_away_only_yrfi_rate_L10",
        "away_away_only_f5_win_pct_L5",
    ]
    for col in split_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df = ensure_market_columns(df)
    df = add_league_relative_features(df)

    df["diff_elo"] = df["home_elo_pre"] - df["away_elo_pre"]

    df["diff_rest_days"] = df["home_rest_days"] - df["away_rest_days"]
    df["diff_is_b2b"] = df["home_is_b2b"] - df["away_is_b2b"]

    df["diff_games_last_3_days"] = df["home_games_last_3_days"] - df["away_games_last_3_days"]
    df["diff_games_last_5_days"] = df["home_games_last_5_days"] - df["away_games_last_5_days"]
    df["diff_games_last_7_days"] = df["home_games_last_7_days"] - df["away_games_last_7_days"]

    df["diff_win_pct_L5"] = df["home_win_pct_L5"] - df["away_win_pct_L5"]
    df["diff_win_pct_L10"] = df["home_win_pct_L10"] - df["away_win_pct_L10"]

    df["diff_run_diff_L5"] = df["home_run_diff_L5"] - df["away_run_diff_L5"]
    df["diff_run_diff_L10"] = df["home_run_diff_L10"] - df["away_run_diff_L10"]

    df["diff_runs_scored_L5"] = df["home_runs_scored_L5"] - df["away_runs_scored_L5"]
    df["diff_runs_allowed_L5"] = df["home_runs_allowed_L5"] - df["away_runs_allowed_L5"]

    df["diff_yrfi_rate_L10"] = df["home_yrfi_rate_L10"] - df["away_yrfi_rate_L10"]
    df["diff_r1_scored_rate_L10"] = df["home_r1_scored_rate_L10"] - df["away_r1_scored_rate_L10"]
    df["diff_r1_allowed_rate_L10"] = df["home_r1_allowed_rate_L10"] - df["away_r1_allowed_rate_L10"]

    df["diff_f5_win_pct_L5"] = df["home_f5_win_pct_L5"] - df["away_f5_win_pct_L5"]
    df["diff_f5_diff_L5"] = df["home_f5_diff_L5"] - df["away_f5_diff_L5"]

    df["diff_hits_L5"] = df["home_hits_L5"] - df["away_hits_L5"]
    df["diff_hits_allowed_L5"] = df["home_hits_allowed_L5"] - df["away_hits_allowed_L5"]

    df["diff_surface_win_pct_L5"] = df["home_home_only_win_pct_L5"] - df["away_away_only_win_pct_L5"]
    df["diff_surface_run_diff_L5"] = df["home_home_only_run_diff_L5"] - df["away_away_only_run_diff_L5"]
    df["diff_surface_yrfi_rate_L10"] = df["home_home_only_yrfi_rate_L10"] - df["away_away_only_yrfi_rate_L10"]
    df["diff_surface_f5_win_pct_L5"] = df["home_home_only_f5_win_pct_L5"] - df["away_away_only_f5_win_pct_L5"]

    # Momentum de corto plazo vs base reciente (sin leakage: todo viene de ventanas shift(1))
    df["home_momentum_win"] = df["home_win_pct_L5"] - df["home_win_pct_L10"]
    df["away_momentum_win"] = df["away_win_pct_L5"] - df["away_win_pct_L10"]
    df["diff_momentum_win"] = df["home_momentum_win"] - df["away_momentum_win"]

    df["home_momentum_run_diff"] = df["home_run_diff_L5"] - df["home_run_diff_L10"]
    df["away_momentum_run_diff"] = df["away_run_diff_L5"] - df["away_run_diff_L10"]
    df["diff_momentum_run_diff"] = df["home_momentum_run_diff"] - df["away_momentum_run_diff"]

    # Ventaja de superficie relativa al nivel base del equipo
    df["home_surface_edge"] = df["home_home_only_win_pct_L5"] - df["home_win_pct_L10"]
    df["away_surface_edge"] = df["away_away_only_win_pct_L5"] - df["away_win_pct_L10"]
    df["diff_surface_edge"] = df["home_surface_edge"] - df["away_surface_edge"]

    # Ãndice de carga reciente: mÃ¡s juegos y menos descanso => mayor fatiga
    df["home_fatigue_index"] = df["home_games_last_5_days"] - df["home_rest_days"]
    df["away_fatigue_index"] = df["away_games_last_5_days"] - df["away_rest_days"]
    df["diff_fatigue_index"] = df["home_fatigue_index"] - df["away_fatigue_index"]

    # SeÃ±al combinada de forma + diferencial de carreras
    df["home_form_power"] = df["home_win_pct_L10"] * df["home_run_diff_L10"]
    df["away_form_power"] = df["away_win_pct_L10"] * df["away_run_diff_L10"]
    df["diff_form_power"] = df["home_form_power"] - df["away_form_power"]

    model_columns = [
        "game_id",
        "date",
        "season",
        "home_team",
        "away_team",

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

        "home_win_pct_L5",
        "away_win_pct_L5",
        "diff_win_pct_L5",

        "home_win_pct_L10",
        "away_win_pct_L10",
        "diff_win_pct_L10",

        "home_run_diff_L5",
        "away_run_diff_L5",
        "diff_run_diff_L5",

        "home_run_diff_L10",
        "away_run_diff_L10",
        "diff_run_diff_L10",

        "home_runs_scored_L5",
        "away_runs_scored_L5",
        "diff_runs_scored_L5",

        "home_runs_allowed_L5",
        "away_runs_allowed_L5",
        "diff_runs_allowed_L5",

        "home_yrfi_rate_L10",
        "away_yrfi_rate_L10",
        "diff_yrfi_rate_L10",

        "home_r1_scored_rate_L10",
        "away_r1_scored_rate_L10",
        "diff_r1_scored_rate_L10",

        "home_r1_allowed_rate_L10",
        "away_r1_allowed_rate_L10",
        "diff_r1_allowed_rate_L10",

        "home_f5_win_pct_L5",
        "away_f5_win_pct_L5",
        "diff_f5_win_pct_L5",

        "home_f5_diff_L5",
        "away_f5_diff_L5",
        "diff_f5_diff_L5",

        "home_hits_L5",
        "away_hits_L5",
        "diff_hits_L5",

        "home_hits_allowed_L5",
        "away_hits_allowed_L5",
        "diff_hits_allowed_L5",

        "home_home_only_win_pct_L5",
        "away_away_only_win_pct_L5",
        "diff_surface_win_pct_L5",

        "home_home_only_run_diff_L5",
        "away_away_only_run_diff_L5",
        "diff_surface_run_diff_L5",

        "home_home_only_yrfi_rate_L10",
        "away_away_only_yrfi_rate_L10",
        "diff_surface_yrfi_rate_L10",

        "home_home_only_f5_win_pct_L5",
        "away_away_only_f5_win_pct_L5",
        "diff_surface_f5_win_pct_L5",

        "home_win_pct_L10_vs_league",
        "away_win_pct_L10_vs_league",
        "diff_win_pct_L10_vs_league",

        "home_run_diff_L10_vs_league",
        "away_run_diff_L10_vs_league",
        "diff_run_diff_L10_vs_league",

        "home_yrfi_rate_L10_vs_league",
        "away_yrfi_rate_L10_vs_league",
        "diff_yrfi_rate_L10_vs_league",

        "home_f5_win_pct_L5_vs_league",
        "away_f5_win_pct_L5_vs_league",
        "diff_f5_win_pct_L5_vs_league",

        "home_momentum_win",
        "away_momentum_win",
        "diff_momentum_win",

        "home_momentum_run_diff",
        "away_momentum_run_diff",
        "diff_momentum_run_diff",

        "home_surface_edge",
        "away_surface_edge",
        "diff_surface_edge",

        "home_fatigue_index",
        "away_fatigue_index",
        "diff_fatigue_index",

        "home_form_power",
        "away_form_power",
        "diff_form_power",

        "home_is_favorite",
        "odds_over_under",
        "market_missing",

        "TARGET_home_win",
        "TARGET_yrfi",
        "TARGET_home_win_f5",
    ]

    final_df = df[model_columns].copy()
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"âœ… Features kbo generadas. Partidos listos: {len(final_df)}")
    print(f"ðŸ’¾ Archivo guardado en: {OUTPUT_FILE}")
    return final_df


if __name__ == "__main__":
    build_features()
