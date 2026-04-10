import numpy as np
import pandas as pd
from pathlib import Path
import re
import sys
import os

# Ensure project `src` root is on sys.path so imports of shared modules work
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Base directory should point to src root so data remains centralized
BASE_DIR = SRC_ROOT

# Advanced feature cache files
UMPIRE_STATS_FILE = BASE_DIR / "data" / "mlb" / "cache" / "umpire_stats.csv"
LINEUP_STRENGTH_FILE = BASE_DIR / "data" / "mlb" / "cache" / "lineup_strength.csv"
LINE_MOVEMENT_FILE = BASE_DIR / "data" / "mlb" / "cache" / "line_movement.csv"

RAW_DATA = BASE_DIR / "data" / "mlb" / "raw" / "mlb_advanced_history.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "mlb" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = PROCESSED_DATA_DIR / "model_ready_features_mlb.csv"

# Park factors and weather caches
PARK_FACTORS_FILE = BASE_DIR / "data" / "mlb" / "cache" / "park_factors.csv"
WEATHER_CACHE_DIR = BASE_DIR / "data" / "mlb" / "cache" / "weather"
WEATHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_park_factors() -> dict:
    if not PARK_FACTORS_FILE.exists():
        return {}
    try:
        pf = pd.read_csv(PARK_FACTORS_FILE)
        if "team" in pf.columns and "park_factor" in pf.columns:
            return dict(zip(pf["team"].astype(str), pf["park_factor"].astype(float)))
    except Exception:
        pass
    return {}


def _weather_cache_path(date_str: str, team: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]", "", f"{date_str}_{team}")
    return WEATHER_CACHE_DIR / f"{safe}.json"


def load_weather_cache(date_str: str, team: str) -> dict | None:
    path = _weather_cache_path(date_str, team)
    if not path.exists():
        return None
    try:
        import json

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_weather_cache(date_str: str, team: str, data: dict):
    path = _weather_cache_path(date_str, team)
    try:
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def fetch_weather_for_game(date_str: str, home_team: str) -> dict | None:
    # Best-effort: prefer cached values. If no cache, no external call by default.
    w = load_weather_cache(date_str, home_team)
    if w is not None:
        return w
    # External API integration could be added here if API key and mapping are provided.
    return None


def ensure_park_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    pf_map = load_park_factors()
    df["home_park_factor"] = df["home_team"].map(lambda t: float(pf_map.get(str(t), 1.0)))
    df["away_park_factor"] = df["away_team"].map(lambda t: float(pf_map.get(str(t), 1.0)))
    df["avg_park_factor"] = (df["home_park_factor"] + df["away_park_factor"]) / 2.0

    # weather placeholders
    df["weather_temp"] = np.nan
    df["weather_wind"] = np.nan
    df["weather_humidity"] = np.nan

    # Try to fill from cache
    if "date" in df.columns and "home_team" in df.columns:
        for idx, row in df[["date", "home_team"]].iterrows():
            d = str(row["date"]) if row["date"] is not None else ""
            t = str(row["home_team"]) if row["home_team"] is not None else ""
            w = fetch_weather_for_game(d, t)
            if w:
                try:
                    df.at[idx, "weather_temp"] = float(w.get("temp"))
                    df.at[idx, "weather_wind"] = float(w.get("wind"))
                    df.at[idx, "weather_humidity"] = float(w.get("humidity"))
                except Exception:
                    pass

    return df


# =========================
# Helpers
# =========================
def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_numeric(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _safe_divide(a, b, default=0.0):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    out = np.where((b.notna()) & (b != 0), a / b, default)
    return pd.Series(out, index=a.index if hasattr(a, "index") else None)


def _signed_squash(series):
    vals = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return vals / (1.0 + np.abs(vals))


def _rolling_shifted_mean(grouped_series, window: int):
    return grouped_series.transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )


def _rolling_shifted_sum(grouped_series, window: int):
    return grouped_series.transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).sum()
    )


def _rolling_shifted_std(grouped_series, window: int):
    return grouped_series.transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=2).std()
    ).fillna(0.0)


def _compute_signed_streak(result_series: pd.Series) -> pd.Series:
    values = pd.to_numeric(result_series, errors="coerce").fillna(0).astype(int).tolist()
    out = []
    streak = 0
    for val in values:
        out.append(streak)
        if val == 1:
            streak = streak + 1 if streak > 0 else 1
        else:
            streak = streak - 1 if streak < 0 else -1
    return pd.Series(out, index=result_series.index, dtype=float)


# =========================
# Core team features
# =========================
def calculate_elo_ratings(df: pd.DataFrame, k: float = 16, home_advantage: float = 35) -> pd.DataFrame:
    print("📈 Calculando Sistema ELO MLB...")

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
    print("⚙️ Generando variables generales MLB por equipo...")

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
    all_stats["signed_win_streak"] = (
        all_stats.groupby("team", group_keys=False)["won_game"]
        .apply(_compute_signed_streak)
        .sort_index()
    )

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

    def roll_std(col: str, window: int) -> pd.Series:
        return all_stats.groupby("team")[col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=2).std()
        ).fillna(0.0)

    all_stats["win_pct_L5"] = roll_mean("won_game", 5)
    all_stats["win_pct_L10"] = roll_mean("won_game", 10)

    all_stats["run_diff_L5"] = roll_mean("run_diff", 5)
    all_stats["run_diff_L10"] = roll_mean("run_diff", 10)

    all_stats["runs_scored_L5"] = roll_mean("runs_scored", 5)
    all_stats["runs_allowed_L5"] = roll_mean("runs_allowed", 5)

    all_stats["runs_scored_std_L10"] = roll_std("runs_scored", 10)
    all_stats["runs_allowed_std_L10"] = roll_std("runs_allowed", 10)

    all_stats["yrfi_rate_L10"] = roll_mean("yrfi_yes_team", 10)
    all_stats["r1_scored_rate_L10"] = roll_mean("r1_scored_flag", 10)
    all_stats["r1_scored_rate_L5"] = roll_mean("r1_scored_flag", 5)
    all_stats["r1_allowed_rate_L10"] = roll_mean("r1_allowed_flag", 10)
    all_stats["r1_allowed_rate_L5"] = roll_mean("r1_allowed_flag", 5)
    all_stats["r1_scored_std_L10"] = roll_std("r1_scored_flag", 10)
    all_stats["r1_allowed_std_L10"] = roll_std("r1_allowed_flag", 10)

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
            "signed_win_streak",
            "runs_scored_L5",
            "runs_allowed_L5",
            "runs_scored_std_L10",
            "runs_allowed_std_L10",
            "yrfi_rate_L10",
            "r1_scored_rate_L10",
            "r1_scored_rate_L5",
            "r1_allowed_rate_L10",
            "r1_allowed_rate_L5",
            "r1_scored_std_L10",
            "r1_allowed_std_L10",
            "f5_win_pct_L5",
            "f5_diff_L5",
            "hits_L5",
            "hits_allowed_L5",
        ]
    ].copy()


def calculate_surface_split_features(df: pd.DataFrame):
    print("🏠✈️ Generando splits Home/Away MLB...")

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
    print("🌐 Generando features MLB relativas a promedio de liga...")

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
    df["diff_win_pct_L10_vs_league"] = df["home_win_pct_L10_vs_league"] - df["away_win_pct_L10_vs_league"]

    df["home_run_diff_L10_vs_league"] = df["home_run_diff_L10"] - df["league_run_diff_L10"]
    df["away_run_diff_L10_vs_league"] = df["away_run_diff_L10"] - df["league_run_diff_L10"]
    df["diff_run_diff_L10_vs_league"] = df["home_run_diff_L10_vs_league"] - df["away_run_diff_L10_vs_league"]

    df["home_yrfi_rate_L10_vs_league"] = df["home_yrfi_rate_L10"] - df["league_yrfi_rate_L10"]
    df["away_yrfi_rate_L10_vs_league"] = df["away_yrfi_rate_L10"] - df["league_yrfi_rate_L10"]
    df["diff_yrfi_rate_L10_vs_league"] = df["home_yrfi_rate_L10_vs_league"] - df["away_yrfi_rate_L10_vs_league"]

    df["home_f5_win_pct_L5_vs_league"] = df["home_f5_win_pct_L5"] - df["league_f5_win_pct_L5"]
    df["away_f5_win_pct_L5_vs_league"] = df["away_f5_win_pct_L5"] - df["league_f5_win_pct_L5"]
    df["diff_f5_win_pct_L5_vs_league"] = df["home_f5_win_pct_L5_vs_league"] - df["away_f5_win_pct_L5_vs_league"]

    return df


# =========================
# Pitcher features
# =========================
def build_pitcher_game_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye tabla por pitcher-juego.
    Usa columnas si existen; si faltan, crea defaults para no romper el pipeline.
    """
    print("🎯 Generando features de pitchers MLB...")

    home_pitcher_col = _first_existing(df, ["home_starting_pitcher", "home_pitcher", "home_starter"])
    away_pitcher_col = _first_existing(df, ["away_starting_pitcher", "away_pitcher", "away_starter"])

    home_ip_col = _first_existing(df, ["home_starting_pitcher_ip", "home_pitcher_ip", "home_starter_ip"])
    away_ip_col = _first_existing(df, ["away_starting_pitcher_ip", "away_pitcher_ip", "away_starter_ip"])

    home_er_col = _first_existing(df, ["home_starting_pitcher_er", "home_pitcher_er", "home_starter_er"])
    away_er_col = _first_existing(df, ["away_starting_pitcher_er", "away_pitcher_er", "away_starter_er"])

    home_h_col = _first_existing(df, ["home_starting_pitcher_hits", "home_pitcher_hits", "home_starter_hits"])
    away_h_col = _first_existing(df, ["away_starting_pitcher_hits", "away_pitcher_hits", "away_starter_hits"])

    home_bb_col = _first_existing(df, ["home_starting_pitcher_bb", "home_pitcher_bb", "home_starter_bb"])
    away_bb_col = _first_existing(df, ["away_starting_pitcher_bb", "away_pitcher_bb", "away_starter_bb"])

    home_k_col = _first_existing(df, ["home_starting_pitcher_k", "home_pitcher_k", "home_starter_k"])
    away_k_col = _first_existing(df, ["away_starting_pitcher_k", "away_pitcher_k", "away_starter_k"])

    home_hr_col = _first_existing(df, ["home_starting_pitcher_hr", "home_pitcher_hr", "home_starter_hr"])
    away_hr_col = _first_existing(df, ["away_starting_pitcher_hr", "away_pitcher_hr", "away_starter_hr"])

    home_name = df[home_pitcher_col] if home_pitcher_col else pd.Series(["UNKNOWN_HOME"] * len(df), index=df.index)
    away_name = df[away_pitcher_col] if away_pitcher_col else pd.Series(["UNKNOWN_AWAY"] * len(df), index=df.index)

    # Neutral defaults avoid leaking same-game outcomes when starter boxscore columns are missing.
    default_er = 2.0
    default_r1_allowed = 0
    default_f5_runs_allowed = 2.5

    if "away_r1" in df.columns:
        home_r1_allowed_flag_series = (_safe_numeric(df["away_r1"], default=0.0) > 0).astype(int)
    else:
        home_r1_allowed_flag_series = pd.Series(default_r1_allowed, index=df.index, dtype=int)

    if "home_r1" in df.columns:
        away_r1_allowed_flag_series = (_safe_numeric(df["home_r1"], default=0.0) > 0).astype(int)
    else:
        away_r1_allowed_flag_series = pd.Series(default_r1_allowed, index=df.index, dtype=int)

    if "away_runs_f5" in df.columns:
        home_f5_runs_allowed_series = _safe_numeric(df["away_runs_f5"], default=default_f5_runs_allowed)
    else:
        home_f5_runs_allowed_series = pd.Series(default_f5_runs_allowed, index=df.index, dtype=float)

    if "home_runs_f5" in df.columns:
        away_f5_runs_allowed_series = _safe_numeric(df["home_runs_f5"], default=default_f5_runs_allowed)
    else:
        away_f5_runs_allowed_series = pd.Series(default_f5_runs_allowed, index=df.index, dtype=float)

    home_tbl = pd.DataFrame(
        {
            "date": df["date"],
            "game_id": df["game_id"],
            "team": df["home_team"],
            "pitcher": home_name.astype(str).fillna("UNKNOWN_HOME"),
            "ip": _safe_numeric(df[home_ip_col], default=5.0) if home_ip_col else 5.0,
            "er": _safe_numeric(df[home_er_col], default=default_er) if home_er_col else float(default_er),
            "hits_allowed": _safe_numeric(df[home_h_col], default=5.0) if home_h_col else 5.0,
            "bb_allowed": _safe_numeric(df[home_bb_col], default=2.0) if home_bb_col else 2.0,
            "k": _safe_numeric(df[home_k_col], default=4.0) if home_k_col else 4.0,
            "hr_allowed": _safe_numeric(df[home_hr_col], default=1.0) if home_hr_col else 1.0,
            "r1_allowed_flag": home_r1_allowed_flag_series,
            "f5_runs_allowed": home_f5_runs_allowed_series.astype(float),
        }
    )

    away_tbl = pd.DataFrame(
        {
            "date": df["date"],
            "game_id": df["game_id"],
            "team": df["away_team"],
            "pitcher": away_name.astype(str).fillna("UNKNOWN_AWAY"),
            "ip": _safe_numeric(df[away_ip_col], default=5.0) if away_ip_col else 5.0,
            "er": _safe_numeric(df[away_er_col], default=default_er) if away_er_col else float(default_er),
            "hits_allowed": _safe_numeric(df[away_h_col], default=5.0) if away_h_col else 5.0,
            "bb_allowed": _safe_numeric(df[away_bb_col], default=2.0) if away_bb_col else 2.0,
            "k": _safe_numeric(df[away_k_col], default=4.0) if away_k_col else 4.0,
            "hr_allowed": _safe_numeric(df[away_hr_col], default=1.0) if away_hr_col else 1.0,
            "r1_allowed_flag": away_r1_allowed_flag_series,
            "f5_runs_allowed": away_f5_runs_allowed_series.astype(float),
        }
    )

    p = pd.concat([home_tbl, away_tbl], ignore_index=True).copy()
    p["date_dt"] = pd.to_datetime(p["date"])
    p = p.sort_values(["pitcher", "date_dt", "game_id"]).reset_index(drop=True)

    p["era_game"] = np.where(p["ip"] > 0, (p["er"] * 9.0) / p["ip"], np.nan)
    p["whip_game"] = np.where(p["ip"] > 0, (p["hits_allowed"] + p["bb_allowed"]) / p["ip"], np.nan)
    p["k_bb_game"] = np.where(p["bb_allowed"] > 0, p["k"] / p["bb_allowed"], p["k"])
    p["hr9_game"] = np.where(p["ip"] > 0, (p["hr_allowed"] * 9.0) / p["ip"], np.nan)
    p["quality_start_flag"] = ((p["ip"] >= 5.0) & (p["er"] <= 2.0)).astype(int)
    p["blowup_start_flag"] = ((p["er"] >= 4.0) | (p["hr_allowed"] >= 2.0)).astype(int)

    p["last_pitch_date"] = p.groupby("pitcher")["date_dt"].shift(1)
    p["pitcher_rest_days"] = (p["date_dt"] - p["last_pitch_date"]).dt.days.fillna(5).clip(lower=0, upper=20)

    by_pitcher = p.groupby("pitcher")
    p["pitcher_era_L5"] = _rolling_shifted_mean(by_pitcher["era_game"], 5)
    p["pitcher_whip_L5"] = _rolling_shifted_mean(by_pitcher["whip_game"], 5)
    p["pitcher_k_bb_L5"] = _rolling_shifted_mean(by_pitcher["k_bb_game"], 5)
    p["pitcher_hr9_L5"] = _rolling_shifted_mean(by_pitcher["hr9_game"], 5)
    p["pitcher_ip_L5"] = _rolling_shifted_mean(by_pitcher["ip"], 5)
    p["pitcher_r1_allowed_rate_L10"] = _rolling_shifted_mean(by_pitcher["r1_allowed_flag"], 10)
    p["pitcher_r1_allowed_rate_L5"] = _rolling_shifted_mean(by_pitcher["r1_allowed_flag"], 5)
    p["pitcher_f5_runs_allowed_L5"] = _rolling_shifted_mean(by_pitcher["f5_runs_allowed"], 5)
    p["pitcher_era_L3"] = _rolling_shifted_mean(by_pitcher["era_game"], 3)
    p["pitcher_era_L10"] = _rolling_shifted_mean(by_pitcher["era_game"], 10)
    p["pitcher_whip_L3"] = _rolling_shifted_mean(by_pitcher["whip_game"], 3)
    p["pitcher_whip_L10"] = _rolling_shifted_mean(by_pitcher["whip_game"], 10)
    p["pitcher_k_bb_L3"] = _rolling_shifted_mean(by_pitcher["k_bb_game"], 3)
    p["pitcher_quality_start_rate_L10"] = _rolling_shifted_mean(by_pitcher["quality_start_flag"], 10)
    p["pitcher_blowup_rate_L10"] = _rolling_shifted_mean(by_pitcher["blowup_start_flag"], 10)
    p["pitcher_era_trend"] = p["pitcher_era_L3"] - p["pitcher_era_L10"]
    p["pitcher_whip_trend"] = p["pitcher_whip_L3"] - p["pitcher_whip_L10"]
    p["pitcher_recent_quality_score"] = (
        (-0.45 * p["pitcher_era_L3"])
        + (-0.35 * p["pitcher_whip_L3"])
        + (0.20 * p["pitcher_k_bb_L3"])
        + (0.60 * p["pitcher_quality_start_rate_L10"])
        + (-0.50 * p["pitcher_blowup_rate_L10"])
    )

    pitcher_features = p[
        [
            "game_id",
            "team",
            "pitcher",
            "pitcher_rest_days",
            "pitcher_era_L5",
            "pitcher_whip_L5",
            "pitcher_k_bb_L5",
            "pitcher_hr9_L5",
            "pitcher_ip_L5",
            "pitcher_r1_allowed_rate_L10",
            "pitcher_r1_allowed_rate_L5",
            "pitcher_f5_runs_allowed_L5",
            "pitcher_quality_start_rate_L10",
            "pitcher_blowup_rate_L10",
            "pitcher_era_trend",
            "pitcher_whip_trend",
            "pitcher_recent_quality_score",
        ]
    ].copy()

    for c in [
        "pitcher_rest_days",
        "pitcher_era_L5",
        "pitcher_whip_L5",
        "pitcher_k_bb_L5",
        "pitcher_hr9_L5",
        "pitcher_ip_L5",
        "pitcher_r1_allowed_rate_L10",
        "pitcher_r1_allowed_rate_L5",
        "pitcher_f5_runs_allowed_L5",
        "pitcher_quality_start_rate_L10",
        "pitcher_blowup_rate_L10",
        "pitcher_era_trend",
        "pitcher_whip_trend",
        "pitcher_recent_quality_score",
    ]:
        pitcher_features[c] = pitcher_features[c].fillna(
            {
                "pitcher_rest_days": 5.0,
                "pitcher_era_L5": 4.25,
                "pitcher_whip_L5": 1.30,
                "pitcher_k_bb_L5": 2.0,
                "pitcher_hr9_L5": 1.0,
                "pitcher_ip_L5": 5.0,
                "pitcher_r1_allowed_rate_L10": 0.30,
                "pitcher_r1_allowed_rate_L5": 0.30,
                "pitcher_f5_runs_allowed_L5": 2.5,
                "pitcher_quality_start_rate_L10": 0.45,
                "pitcher_blowup_rate_L10": 0.30,
                "pitcher_era_trend": 0.0,
                "pitcher_whip_trend": 0.0,
                "pitcher_recent_quality_score": -1.8,
            }[c]
        )

    return pitcher_features


# =========================
# Bullpen proxy
# =========================
def build_bullpen_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    print("🧱 Generando proxies de bullpen MLB...")

    home_tbl = pd.DataFrame(
        {
            "date": df["date"],
            "game_id": df["game_id"],
            "team": df["home_team"],
            "bullpen_runs_allowed": np.maximum(_safe_numeric(df["away_runs_total"], 0) - _safe_numeric(df["away_runs_f5"], 0), 0),
        }
    )

    away_tbl = pd.DataFrame(
        {
            "date": df["date"],
            "game_id": df["game_id"],
            "team": df["away_team"],
            "bullpen_runs_allowed": np.maximum(_safe_numeric(df["home_runs_total"], 0) - _safe_numeric(df["home_runs_f5"], 0), 0),
        }
    )

    b = pd.concat([home_tbl, away_tbl], ignore_index=True).copy()
    b["date_dt"] = pd.to_datetime(b["date"])
    b = b.sort_values(["team", "date_dt", "game_id"]).reset_index(drop=True)
    b["appearance_flag"] = 1

    by_team = b.groupby("team")
    b["bullpen_runs_allowed_L5"] = _rolling_shifted_mean(by_team["bullpen_runs_allowed"], 5)
    b["bullpen_runs_allowed_L10"] = _rolling_shifted_mean(by_team["bullpen_runs_allowed"], 10)
    b["bullpen_load_L3"] = _rolling_shifted_sum(by_team["appearance_flag"], 3)
    b["bullpen_load_L5"] = _rolling_shifted_sum(by_team["appearance_flag"], 5)

    out = b[
        [
            "game_id",
            "team",
            "bullpen_runs_allowed_L5",
            "bullpen_runs_allowed_L10",
            "bullpen_load_L3",
            "bullpen_load_L5",
        ]
    ].copy()

    for c in [
        "bullpen_runs_allowed_L5",
        "bullpen_runs_allowed_L10",
        "bullpen_load_L3",
        "bullpen_load_L5",
    ]:
        out[c] = out[c].fillna(0.0)

    return out


# =========================
# Main builder
# =========================
def build_features() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA)

    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date_dt", "game_id"]).reset_index(drop=True)

    df = calculate_elo_ratings(df)

    df["TARGET_home_win"] = (df["home_runs_total"] > df["away_runs_total"]).astype(int)
    df["TARGET_yrfi"] = ((df["home_r1"] + df["away_r1"]) > 0).astype(int)
    df["TARGET_home_win_f5"] = (df["home_runs_f5"] > df["away_runs_f5"]).astype(int)

    rolling_features = calculate_team_rolling_features(df)
    home_surface_features, away_surface_features = calculate_surface_split_features(df)
    pitcher_features = build_pitcher_game_table(df)
    bullpen_features = build_bullpen_proxy_features(df)

    # Home team rolling
    df = pd.merge(
        df,
        rolling_features,
        left_on=["game_id", "home_team"],
        right_on=["game_id", "team"],
        how="left",
    )
    df = df.rename(columns={c: f"home_{c}" for c in rolling_features.columns if c not in ["game_id", "team"]}).drop(columns=["team"])

    # Away team rolling
    df = pd.merge(
        df,
        rolling_features,
        left_on=["game_id", "away_team"],
        right_on=["game_id", "team"],
        how="left",
    )
    df = df.rename(columns={c: f"away_{c}" for c in rolling_features.columns if c not in ["game_id", "team"]}).drop(columns=["team"])

    # Home surface
    df = pd.merge(
        df,
        home_surface_features,
        left_on=["game_id", "home_team"],
        right_on=["game_id", "team"],
        how="left",
    )
    df = df.rename(columns={c: f"home_{c}" for c in home_surface_features.columns if c not in ["game_id", "team"]}).drop(columns=["team"])

    # Away surface
    df = pd.merge(
        df,
        away_surface_features,
        left_on=["game_id", "away_team"],
        right_on=["game_id", "team"],
        how="left",
    )
    df = df.rename(columns={c: f"away_{c}" for c in away_surface_features.columns if c not in ["game_id", "team"]}).drop(columns=["team"])

    # Home pitcher
    df = pd.merge(
        df,
        pitcher_features,
        left_on=["game_id", "home_team"],
        right_on=["game_id", "team"],
        how="left",
    )
    df = df.rename(
        columns={
            c: f"home_{c}"
            for c in pitcher_features.columns
            if c not in ["game_id", "team"]
        }
    ).drop(columns=["team"])

    # Away pitcher
    df = pd.merge(
        df,
        pitcher_features,
        left_on=["game_id", "away_team"],
        right_on=["game_id", "team"],
        how="left",
    )
    df = df.rename(
        columns={
            c: f"away_{c}"
            for c in pitcher_features.columns
            if c not in ["game_id", "team"]
        }
    ).drop(columns=["team"])

    # Home bullpen
    df = pd.merge(
        df,
        bullpen_features,
        left_on=["game_id", "home_team"],
        right_on=["game_id", "team"],
        how="left",
    )
    df = df.rename(
        columns={c: f"home_{c}" for c in bullpen_features.columns if c not in ["game_id", "team"]}
    ).drop(columns=["team"])

    # Away bullpen
    df = pd.merge(
        df,
        bullpen_features,
        left_on=["game_id", "away_team"],
        right_on=["game_id", "team"],
        how="left",
    )
    df = df.rename(
        columns={c: f"away_{c}" for c in bullpen_features.columns if c not in ["game_id", "team"]}
    ).drop(columns=["team"])

    # Necesitamos historia mínima real
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

    # Merge advanced cached features if available
    try:
        if UMPIRE_STATS_FILE.exists():
            ump = pd.read_csv(UMPIRE_STATS_FILE)
            # expected columns: umpire, zone_rate
            if "umpire" in ump.columns:
                ump = ump.rename(columns={"umpire": "umpire_name"})
                df = df.merge(ump, left_on="home_plate_umpire", right_on="umpire_name", how="left")
                df = df.rename(columns={"zone_rate": "home_umpire_zone_rate"})
                df = df.drop(columns=[c for c in ["umpire_name"] if c in df.columns])
    except Exception:
        pass

    try:
        if LINEUP_STRENGTH_FILE.exists():
            ls = pd.read_csv(LINEUP_STRENGTH_FILE)
            # expected columns: date, team, lineup_strength
            if {"date", "team", "lineup_strength"}.issubset(ls.columns):
                ls["date"] = ls["date"].astype(str)
                df = df.merge(ls, left_on=["date", "home_team"], right_on=["date", "team"], how="left")
                df = df.rename(columns={"lineup_strength": "home_lineup_strength"}).drop(columns=["team"])
                df = df.merge(ls, left_on=["date", "away_team"], right_on=["date", "team"], how="left")
                df = df.rename(columns={"lineup_strength": "away_lineup_strength"}).drop(columns=["team"])
    except Exception:
        pass

    try:
        if LINE_MOVEMENT_FILE.exists():
            lm = pd.read_csv(LINE_MOVEMENT_FILE)
            # expected columns: game_id, open_line, current_line
            if {"game_id", "open_line", "current_line"}.issubset(lm.columns):
                lm["game_id"] = lm["game_id"].astype(str)
                df = df.merge(lm, left_on="game_id", right_on="game_id", how="left")
                df["line_movement"] = df["current_line"] - df["open_line"].fillna(df["current_line"])
    except Exception:
        pass

    # =========================
    # Core differentials
    # =========================
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

    # =========================
    # Momentum / form / fatigue
    # =========================
    df["home_momentum_win"] = df["home_win_pct_L5"] - df["home_win_pct_L10"]
    df["away_momentum_win"] = df["away_win_pct_L5"] - df["away_win_pct_L10"]
    df["diff_momentum_win"] = df["home_momentum_win"] - df["away_momentum_win"]

    df["home_momentum_run_diff"] = df["home_run_diff_L5"] - df["home_run_diff_L10"]
    df["away_momentum_run_diff"] = df["away_run_diff_L5"] - df["away_run_diff_L10"]
    df["diff_momentum_run_diff"] = df["home_momentum_run_diff"] - df["away_momentum_run_diff"]

    df["home_surface_edge"] = df["home_home_only_win_pct_L5"] - df["home_win_pct_L10"]
    df["away_surface_edge"] = df["away_away_only_win_pct_L5"] - df["away_win_pct_L10"]
    df["diff_surface_edge"] = df["home_surface_edge"] - df["away_surface_edge"]

    df["home_fatigue_index"] = df["home_games_last_5_days"] - df["home_rest_days"]
    df["away_fatigue_index"] = df["away_games_last_5_days"] - df["away_rest_days"]
    df["diff_fatigue_index"] = df["home_fatigue_index"] - df["away_fatigue_index"]

    df["home_form_power"] = df["home_win_pct_L10"] * df["home_run_diff_L10"]
    df["away_form_power"] = df["away_win_pct_L10"] * df["away_run_diff_L10"]
    df["diff_form_power"] = df["home_form_power"] - df["away_form_power"]
    df["diff_signed_win_streak"] = df["home_signed_win_streak"] - df["away_signed_win_streak"]

    # =========================
    # Mean reversion / upset-risk
    # =========================
    home_quality_edge = (
        0.65 * df["home_win_pct_L10_vs_league"]
        + 0.35 * _signed_squash(df["home_run_diff_L10_vs_league"])
    )
    away_quality_edge = (
        0.65 * df["away_win_pct_L10_vs_league"]
        + 0.35 * _signed_squash(df["away_run_diff_L10_vs_league"])
    )

    home_variance_pressure = 1.0 + df["home_runs_scored_std_L10"] + df["home_runs_allowed_std_L10"]
    away_variance_pressure = 1.0 + df["away_runs_scored_std_L10"] + df["away_runs_allowed_std_L10"]

    df["home_regression_risk"] = (
        df["home_momentum_win"].clip(lower=0.0)
        * home_variance_pressure
        * (1.0 + home_quality_edge.clip(lower=0.0))
    ) + (0.35 * df["home_momentum_run_diff"].clip(lower=0.0))
    df["away_regression_risk"] = (
        df["away_momentum_win"].clip(lower=0.0)
        * away_variance_pressure
        * (1.0 + away_quality_edge.clip(lower=0.0))
    ) + (0.35 * df["away_momentum_run_diff"].clip(lower=0.0))
    df["diff_regression_risk"] = df["home_regression_risk"] - df["away_regression_risk"]

    df["home_bounce_back_signal"] = (
        (-df["home_momentum_win"]).clip(lower=0.0)
        * (1.0 + home_quality_edge.clip(lower=0.0))
        * (1.0 + df["home_rest_days"].clip(lower=0.0))
    )
    df["away_bounce_back_signal"] = (
        (-df["away_momentum_win"]).clip(lower=0.0)
        * (1.0 + away_quality_edge.clip(lower=0.0))
        * (1.0 + df["away_rest_days"].clip(lower=0.0))
    )
    df["diff_bounce_back_signal"] = df["home_bounce_back_signal"] - df["away_bounce_back_signal"]

    home_favorite_flag = pd.to_numeric(df["home_is_favorite"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    away_favorite_flag = 1.0 - home_favorite_flag
    df["favorite_trap_signal"] = (
        home_favorite_flag * (df["home_regression_risk"] - df["away_bounce_back_signal"])
        + away_favorite_flag * (df["away_regression_risk"] - df["home_bounce_back_signal"])
    )
    df["underdog_upset_signal"] = (
        home_favorite_flag * (df["away_bounce_back_signal"] - df["home_regression_risk"])
        + away_favorite_flag * (df["home_bounce_back_signal"] - df["away_regression_risk"])
    )

    home_dog_flag = 1.0 - home_favorite_flag
    away_dog_flag = home_favorite_flag
    total_line = pd.to_numeric(df["odds_over_under"], errors="coerce").fillna(8.5)
    low_total_factor = ((8.5 - total_line).clip(lower=0.0)) / 2.5

    df["home_public_pressure"] = home_favorite_flag * (
        df["home_signed_win_streak"].clip(lower=0.0)
        + df["home_momentum_win"].clip(lower=0.0)
        + home_quality_edge.clip(lower=0.0)
    )
    df["away_public_pressure"] = away_favorite_flag * (
        df["away_signed_win_streak"].clip(lower=0.0)
        + df["away_momentum_win"].clip(lower=0.0)
        + away_quality_edge.clip(lower=0.0)
    )
    df["diff_public_pressure"] = df["home_public_pressure"] - df["away_public_pressure"]

    df["home_contrarian_value"] = home_dog_flag * (
        df["home_bounce_back_signal"]
        + home_quality_edge.clip(lower=0.0)
        - df["home_regression_risk"]
    )
    df["away_contrarian_value"] = away_dog_flag * (
        df["away_bounce_back_signal"]
        + away_quality_edge.clip(lower=0.0)
        - df["away_regression_risk"]
    )
    df["diff_contrarian_value"] = df["home_contrarian_value"] - df["away_contrarian_value"]

    df["home_dog_live_low_total"] = home_dog_flag * low_total_factor * (1.0 + home_quality_edge.clip(lower=0.0))
    df["away_dog_live_low_total"] = away_dog_flag * low_total_factor * (1.0 + away_quality_edge.clip(lower=0.0))
    df["diff_dog_live_low_total"] = df["home_dog_live_low_total"] - df["away_dog_live_low_total"]

    # =========================
    # New: pitcher differentials
    # =========================
    df["diff_pitcher_rest_days"] = df["home_pitcher_rest_days"] - df["away_pitcher_rest_days"]
    df["diff_pitcher_era_L5"] = df["home_pitcher_era_L5"] - df["away_pitcher_era_L5"]
    df["diff_pitcher_whip_L5"] = df["home_pitcher_whip_L5"] - df["away_pitcher_whip_L5"]
    df["diff_pitcher_k_bb_L5"] = df["home_pitcher_k_bb_L5"] - df["away_pitcher_k_bb_L5"]
    df["diff_pitcher_hr9_L5"] = df["home_pitcher_hr9_L5"] - df["away_pitcher_hr9_L5"]
    df["diff_pitcher_ip_L5"] = df["home_pitcher_ip_L5"] - df["away_pitcher_ip_L5"]
    df["diff_pitcher_r1_allowed_rate_L10"] = (
        df["home_pitcher_r1_allowed_rate_L10"] - df["away_pitcher_r1_allowed_rate_L10"]
    )
    df["diff_pitcher_f5_runs_allowed_L5"] = (
        df["home_pitcher_f5_runs_allowed_L5"] - df["away_pitcher_f5_runs_allowed_L5"]
    )
    df["diff_pitcher_quality_start_rate_L10"] = (
        df["home_pitcher_quality_start_rate_L10"] - df["away_pitcher_quality_start_rate_L10"]
    )
    df["diff_pitcher_blowup_rate_L10"] = (
        df["home_pitcher_blowup_rate_L10"] - df["away_pitcher_blowup_rate_L10"]
    )
    df["diff_pitcher_era_trend"] = df["home_pitcher_era_trend"] - df["away_pitcher_era_trend"]
    df["diff_pitcher_whip_trend"] = df["home_pitcher_whip_trend"] - df["away_pitcher_whip_trend"]
    df["diff_pitcher_recent_quality_score"] = (
        df["home_pitcher_recent_quality_score"] - df["away_pitcher_recent_quality_score"]
    )

    # =========================
    # New: bullpen differentials
    # =========================
    df["diff_bullpen_runs_allowed_L5"] = df["home_bullpen_runs_allowed_L5"] - df["away_bullpen_runs_allowed_L5"]
    df["diff_bullpen_runs_allowed_L10"] = df["home_bullpen_runs_allowed_L10"] - df["away_bullpen_runs_allowed_L10"]
    df["diff_bullpen_load_L3"] = df["home_bullpen_load_L3"] - df["away_bullpen_load_L3"]
    df["diff_bullpen_load_L5"] = df["home_bullpen_load_L5"] - df["away_bullpen_load_L5"]

    # =========================
    # New: matchup features
    # =========================
    df["home_offense_vs_away_pitcher"] = df["home_runs_scored_L5"] - df["away_pitcher_era_L5"]
    df["away_offense_vs_home_pitcher"] = df["away_runs_scored_L5"] - df["home_pitcher_era_L5"]
    df["diff_offense_vs_pitcher"] = df["home_offense_vs_away_pitcher"] - df["away_offense_vs_home_pitcher"]

    df["home_hits_vs_away_whip"] = df["home_hits_L5"] - df["away_pitcher_whip_L5"]
    df["away_hits_vs_home_whip"] = df["away_hits_L5"] - df["home_pitcher_whip_L5"]
    df["diff_hits_vs_whip"] = df["home_hits_vs_away_whip"] - df["away_hits_vs_home_whip"]

    df["home_r1_vs_away_pitcher"] = df["home_r1_scored_rate_L10"] - df["away_pitcher_r1_allowed_rate_L10"]
    df["away_r1_vs_home_pitcher"] = df["away_r1_scored_rate_L10"] - df["home_pitcher_r1_allowed_rate_L10"]
    df["diff_r1_vs_pitcher"] = df["home_r1_vs_away_pitcher"] - df["away_r1_vs_home_pitcher"]

    df["home_f5_vs_away_pitcher"] = df["home_f5_diff_L5"] - df["away_pitcher_f5_runs_allowed_L5"]
    df["away_f5_vs_home_pitcher"] = df["away_f5_diff_L5"] - df["home_pitcher_f5_runs_allowed_L5"]
    df["diff_f5_vs_pitcher"] = df["home_f5_vs_away_pitcher"] - df["away_f5_vs_home_pitcher"]

    # =========================
    # New: variance / stability
    # =========================
    df["diff_runs_scored_std_L10"] = df["home_runs_scored_std_L10"] - df["away_runs_scored_std_L10"]
    df["diff_runs_allowed_std_L10"] = df["home_runs_allowed_std_L10"] - df["away_runs_allowed_std_L10"]

    # =========================
    # FEATURES EXTRA YRFI
    # =========================

# Aseguramos columna de ambos pitchers disponible y limpia
    if "pitcher_data_available" in df.columns:
        df["both_pitchers_available"] = df["pitcher_data_available"].fillna(0).astype(int)
    elif "both_pitchers_available" in df.columns:
        df["both_pitchers_available"] = df["both_pitchers_available"].fillna(0).astype(int)
    else:
        df["both_pitchers_available"] = 0

    for side in ["home", "away"]:
        opp = "away" if side == "home" else "home"

        # Proxies matchup L5: mientras no exista un matchup rolling real,
        # usamos la señal actual ya calculada como proxy consistente.
        if f"{side}_r1_vs_{opp}_pitcher" in df.columns:
            df[f"{side}_r1_vs_{opp}_pitcher_L5_proxy"] = df[f"{side}_r1_vs_{opp}_pitcher"]

        # Fallback defensivo si por alguna razón faltan stds.
        if f"{side}_r1_scored_std_L10" not in df.columns and f"{side}_r1_scored_rate_L10" in df.columns:
            df[f"{side}_r1_scored_std_L10"] = 0.0

        if f"{side}_r1_allowed_std_L10" not in df.columns and f"{side}_r1_allowed_rate_L10" in df.columns:
            df[f"{side}_r1_allowed_std_L10"] = 0.0

    # Diferenciales L5
    if {"home_r1_scored_rate_L5", "away_r1_scored_rate_L5"}.issubset(df.columns):
        df["diff_r1_scored_rate_L5"] = df["home_r1_scored_rate_L5"] - df["away_r1_scored_rate_L5"]

    if {"home_r1_allowed_rate_L5", "away_r1_allowed_rate_L5"}.issubset(df.columns):
        df["diff_r1_allowed_rate_L5"] = df["home_r1_allowed_rate_L5"] - df["away_r1_allowed_rate_L5"]

    if {"home_pitcher_r1_allowed_rate_L5", "away_pitcher_r1_allowed_rate_L5"}.issubset(df.columns):
        df["diff_pitcher_r1_allowed_rate_L5"] = (
            df["home_pitcher_r1_allowed_rate_L5"] - df["away_pitcher_r1_allowed_rate_L5"]
        )

    if {"home_r1_vs_away_pitcher_L5_proxy", "away_r1_vs_home_pitcher_L5_proxy"}.issubset(df.columns):
        df["diff_r1_vs_pitcher_L5"] = (
            df["home_r1_vs_away_pitcher_L5_proxy"] - df["away_r1_vs_home_pitcher_L5_proxy"]
        )

    # Diferenciales de volatilidad
    if {"home_r1_scored_std_L10", "away_r1_scored_std_L10"}.issubset(df.columns):
        df["diff_r1_scored_std_L10"] = df["home_r1_scored_std_L10"] - df["away_r1_scored_std_L10"]

    if {"home_r1_allowed_std_L10", "away_r1_allowed_std_L10"}.issubset(df.columns):
        df["diff_r1_allowed_std_L10"] = df["home_r1_allowed_std_L10"] - df["away_r1_allowed_std_L10"]

    # Presión YRFI
    if {"home_r1_scored_rate_L10", "away_pitcher_r1_allowed_rate_L10"}.issubset(df.columns):
        df["yrfi_pressure_home"] = df["home_r1_scored_rate_L10"] + df["away_pitcher_r1_allowed_rate_L10"]

    if {"away_r1_scored_rate_L10", "home_pitcher_r1_allowed_rate_L10"}.issubset(df.columns):
        df["yrfi_pressure_away"] = df["away_r1_scored_rate_L10"] + df["home_pitcher_r1_allowed_rate_L10"]

    if {"yrfi_pressure_home", "yrfi_pressure_away"}.issubset(df.columns):
        df["diff_yrfi_pressure"] = df["yrfi_pressure_home"] - df["yrfi_pressure_away"]
        df["total_yrfi_pressure"] = df["yrfi_pressure_home"] + df["yrfi_pressure_away"]

    if {"home_r1_scored_rate_L5", "away_pitcher_r1_allowed_rate_L5"}.issubset(df.columns):
        df["yrfi_pressure_home_L5"] = df["home_r1_scored_rate_L5"] + df["away_pitcher_r1_allowed_rate_L5"]

    if {"away_r1_scored_rate_L5", "home_pitcher_r1_allowed_rate_L5"}.issubset(df.columns):
        df["yrfi_pressure_away_L5"] = df["away_r1_scored_rate_L5"] + df["home_pitcher_r1_allowed_rate_L5"]

    if {"yrfi_pressure_home_L5", "yrfi_pressure_away_L5"}.issubset(df.columns):
        df["diff_yrfi_pressure_L5"] = df["yrfi_pressure_home_L5"] - df["yrfi_pressure_away_L5"]
        df["total_yrfi_pressure_L5"] = df["yrfi_pressure_home_L5"] + df["yrfi_pressure_away_L5"]

    # Consistencia
    if "home_r1_scored_std_L10" in df.columns:
        df["home_yrfi_consistency_L10"] = 1.0 - df["home_r1_scored_std_L10"]

    if "away_r1_scored_std_L10" in df.columns:
        df["away_yrfi_consistency_L10"] = 1.0 - df["away_r1_scored_std_L10"]

    if {"home_yrfi_consistency_L10", "away_yrfi_consistency_L10"}.issubset(df.columns):
        df["diff_yrfi_consistency_L10"] = (
            df["home_yrfi_consistency_L10"] - df["away_yrfi_consistency_L10"]
        )

    # Defragmentación ligera tras muchas columnas nuevas
    df = df.copy()

    # Cleanup
    numeric_cols = [c for c in df.columns if c not in ["game_id", "date", "time", "season", "home_team", "away_team"]]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    # Add simple park factors and weather features (from cache)
    df = ensure_park_weather_columns(df)

    model_columns = [
        "game_id",
        "date",
        "time",
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

        "home_signed_win_streak",
        "away_signed_win_streak",
        "diff_signed_win_streak",

        "home_runs_scored_L5",
        "away_runs_scored_L5",
        "diff_runs_scored_L5",

        "home_runs_allowed_L5",
        "away_runs_allowed_L5",
        "diff_runs_allowed_L5",

        "home_runs_scored_std_L10",
        "away_runs_scored_std_L10",
        "diff_runs_scored_std_L10",

        "home_runs_allowed_std_L10",
        "away_runs_allowed_std_L10",
        "diff_runs_allowed_std_L10",

        "home_yrfi_rate_L10",
        "away_yrfi_rate_L10",
        "diff_yrfi_rate_L10",

        "home_r1_scored_rate_L10",
        "away_r1_scored_rate_L10",
        "diff_r1_scored_rate_L10",

        "home_r1_scored_rate_L5",
        "away_r1_scored_rate_L5",
        "diff_r1_scored_rate_L5",

        "home_r1_allowed_rate_L10",
        "away_r1_allowed_rate_L10",
        "diff_r1_allowed_rate_L10",

        "home_r1_allowed_rate_L5",
        "away_r1_allowed_rate_L5",
        "diff_r1_allowed_rate_L5",

        "home_r1_scored_std_L10",
        "away_r1_scored_std_L10",
        "diff_r1_scored_std_L10",

        "home_r1_allowed_std_L10",
        "away_r1_allowed_std_L10",
        "diff_r1_allowed_std_L10",

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

        "home_regression_risk",
        "away_regression_risk",
        "diff_regression_risk",

        "home_bounce_back_signal",
        "away_bounce_back_signal",
        "diff_bounce_back_signal",

        "favorite_trap_signal",
        "underdog_upset_signal",

        "home_public_pressure",
        "away_public_pressure",
        "diff_public_pressure",

        "home_contrarian_value",
        "away_contrarian_value",
        "diff_contrarian_value",

        "home_dog_live_low_total",
        "away_dog_live_low_total",
        "diff_dog_live_low_total",

        # Pitchers
        "home_pitcher_rest_days",
        "away_pitcher_rest_days",
        "diff_pitcher_rest_days",

        "home_pitcher_era_L5",
        "away_pitcher_era_L5",
        "diff_pitcher_era_L5",

        "home_pitcher_whip_L5",
        "away_pitcher_whip_L5",
        "diff_pitcher_whip_L5",

        "home_pitcher_k_bb_L5",
        "away_pitcher_k_bb_L5",
        "diff_pitcher_k_bb_L5",

        "home_pitcher_hr9_L5",
        "away_pitcher_hr9_L5",
        "diff_pitcher_hr9_L5",

        "home_pitcher_ip_L5",
        "away_pitcher_ip_L5",
        "diff_pitcher_ip_L5",

        "home_pitcher_r1_allowed_rate_L10",
        "away_pitcher_r1_allowed_rate_L10",
        "diff_pitcher_r1_allowed_rate_L10",

        "home_pitcher_r1_allowed_rate_L5",
        "away_pitcher_r1_allowed_rate_L5",
        "diff_pitcher_r1_allowed_rate_L5",

        "home_pitcher_f5_runs_allowed_L5",
        "away_pitcher_f5_runs_allowed_L5",
        "diff_pitcher_f5_runs_allowed_L5",

        "home_pitcher_quality_start_rate_L10",
        "away_pitcher_quality_start_rate_L10",
        "diff_pitcher_quality_start_rate_L10",

        "home_pitcher_blowup_rate_L10",
        "away_pitcher_blowup_rate_L10",
        "diff_pitcher_blowup_rate_L10",

        "home_pitcher_era_trend",
        "away_pitcher_era_trend",
        "diff_pitcher_era_trend",

        "home_pitcher_whip_trend",
        "away_pitcher_whip_trend",
        "diff_pitcher_whip_trend",

        "home_pitcher_recent_quality_score",
        "away_pitcher_recent_quality_score",
        "diff_pitcher_recent_quality_score",

        # Bullpen
        "home_bullpen_runs_allowed_L5",
        "away_bullpen_runs_allowed_L5",
        "diff_bullpen_runs_allowed_L5",

        "home_bullpen_runs_allowed_L10",
        "away_bullpen_runs_allowed_L10",
        "diff_bullpen_runs_allowed_L10",

        "home_bullpen_load_L3",
        "away_bullpen_load_L3",
        "diff_bullpen_load_L3",

        "home_bullpen_load_L5",
        "away_bullpen_load_L5",
        "diff_bullpen_load_L5",

        # Matchups
        "home_offense_vs_away_pitcher",
        "away_offense_vs_home_pitcher",
        "diff_offense_vs_pitcher",

        "home_hits_vs_away_whip",
        "away_hits_vs_home_whip",
        "diff_hits_vs_whip",

        "home_r1_vs_away_pitcher",
        "away_r1_vs_home_pitcher",
        "diff_r1_vs_pitcher",

        "home_r1_vs_away_pitcher_L5_proxy",
        "away_r1_vs_home_pitcher_L5_proxy",
        "diff_r1_vs_pitcher_L5",

        "yrfi_pressure_home",
        "yrfi_pressure_away",
        "diff_yrfi_pressure",
        "total_yrfi_pressure",

        "yrfi_pressure_home_L5",
        "yrfi_pressure_away_L5",
        "diff_yrfi_pressure_L5",
        "total_yrfi_pressure_L5",

        "home_f5_vs_away_pitcher",
        "away_f5_vs_home_pitcher",
        "diff_f5_vs_pitcher",

        "home_yrfi_consistency_L10",
        "away_yrfi_consistency_L10",
        "diff_yrfi_consistency_L10",

        "both_pitchers_available",

        "home_park_factor",
        "away_park_factor",
        "avg_park_factor",
        "weather_temp",
        "weather_wind",
        "weather_humidity",

        "home_is_favorite",
        "odds_over_under",
        "market_missing",

        "TARGET_home_win",
        "TARGET_yrfi",
        "TARGET_home_win_f5",
    ]

    final_df = df[model_columns].copy()
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Features MLB generadas. Partidos listos: {len(final_df)}")
    print(f"💾 Archivo guardado en: {OUTPUT_FILE}")
    return final_df


if __name__ == "__main__":
    build_features()
