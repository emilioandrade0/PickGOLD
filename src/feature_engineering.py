import numpy as np
import pandas as pd
from pathlib import Path

# --- RUTAS ---
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA = BASE_DIR / "data" / "raw" / "nba_advanced_history.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = PROCESSED_DATA_DIR / "model_ready_features.csv"


TEAM_CONF_DIV = {
    "ATL": ("EAST", "SOUTHEAST"),
    "BOS": ("EAST", "ATLANTIC"),
    "BKN": ("EAST", "ATLANTIC"),
    "CHA": ("EAST", "SOUTHEAST"),
    "CHI": ("EAST", "CENTRAL"),
    "CLE": ("EAST", "CENTRAL"),
    "DAL": ("WEST", "SOUTHWEST"),
    "DEN": ("WEST", "NORTHWEST"),
    "DET": ("EAST", "CENTRAL"),
    "GSW": ("WEST", "PACIFIC"),
    "HOU": ("WEST", "SOUTHWEST"),
    "IND": ("EAST", "CENTRAL"),
    "LAC": ("WEST", "PACIFIC"),
    "LAL": ("WEST", "PACIFIC"),
    "MEM": ("WEST", "SOUTHWEST"),
    "MIA": ("EAST", "SOUTHEAST"),
    "MIL": ("EAST", "CENTRAL"),
    "MIN": ("WEST", "NORTHWEST"),
    "NOP": ("WEST", "SOUTHWEST"),
    "NYK": ("EAST", "ATLANTIC"),
    "OKC": ("WEST", "NORTHWEST"),
    "ORL": ("EAST", "SOUTHEAST"),
    "PHI": ("EAST", "ATLANTIC"),
    "PHX": ("WEST", "PACIFIC"),
    "POR": ("WEST", "NORTHWEST"),
    "SAC": ("WEST", "PACIFIC"),
    "SAS": ("WEST", "SOUTHWEST"),
    "TOR": ("EAST", "ATLANTIC"),
    "UTA": ("WEST", "NORTHWEST"),
    "WAS": ("EAST", "SOUTHEAST"),
}

TEAM_TIMEZONE = {
    "ATL": "ET", "BOS": "ET", "BKN": "ET", "CHA": "ET", "CHI": "CT", "CLE": "ET",
    "DAL": "CT", "DEN": "MT", "DET": "ET", "GSW": "PT", "HOU": "CT", "IND": "ET",
    "LAC": "PT", "LAL": "PT", "MEM": "CT", "MIA": "ET", "MIL": "CT", "MIN": "CT",
    "NOP": "CT", "NYK": "ET", "OKC": "CT", "ORL": "ET", "PHI": "ET", "PHX": "MT",
    "POR": "PT", "SAC": "PT", "SAS": "CT", "TOR": "ET", "UTA": "MT", "WAS": "ET",
}

TZ_OFFSET = {"ET": 0, "CT": -1, "MT": -2, "PT": -3}


def days_to_playoffs(game_date: pd.Timestamp) -> int:
    playoff_year = game_date.year + 1 if game_date.month >= 7 else game_date.year
    playoff_start = pd.Timestamp(playoff_year, 4, 15)
    return int(max((playoff_start - game_date).days, 0))


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    date_dt = pd.to_datetime(df["date"])

    def _conf(team: str) -> str:
        return TEAM_CONF_DIV.get(str(team), ("UNK", "UNK"))[0]

    def _div(team: str) -> str:
        return TEAM_CONF_DIV.get(str(team), ("UNK", "UNK"))[1]

    home_conf = df["home_team"].map(_conf)
    away_conf = df["away_team"].map(_conf)
    home_div = df["home_team"].map(_div)
    away_div = df["away_team"].map(_div)

    home_tz = df["home_team"].map(lambda t: TEAM_TIMEZONE.get(str(t), "ET"))
    away_tz = df["away_team"].map(lambda t: TEAM_TIMEZONE.get(str(t), "ET"))

    df["same_conference"] = (home_conf == away_conf).astype(int)
    df["same_division"] = (home_div == away_div).astype(int)
    df["away_tz_diff"] = (home_tz.map(TZ_OFFSET) - away_tz.map(TZ_OFFSET)).abs().astype(int)
    df["interconference_travel"] = ((df["same_conference"] == 0) & (df["away_tz_diff"] >= 2)).astype(int)

    df["days_to_playoffs"] = date_dt.map(days_to_playoffs).clip(lower=0, upper=240)
    df["playoff_pressure"] = (1 - (df["days_to_playoffs"] / 120.0)).clip(lower=0, upper=1)

    return df


def calculate_elo_ratings(df: pd.DataFrame, k: float = 20, home_advantage: float = 100) -> pd.DataFrame:
    print("📈 Calculando Sistema ELO de Fuerza de Equipos...")

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

        actual_home = 1 if row["home_pts_total"] > row["away_pts_total"] else 0
        actual_away = 1 - actual_home

        elo_dict[home] = home_elo_pre + k * (actual_home - expected_home)
        elo_dict[away] = away_elo_pre + k * (actual_away - expected_away)

    df["home_elo_pre"] = elo_home_before
    df["away_elo_pre"] = elo_away_before
    return df


def ensure_market_columns(df: pd.DataFrame) -> pd.DataFrame:
    market_cols = ["home_spread", "spread_abs", "home_is_favorite", "odds_over_under"]

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
    """
    Cuenta cuántos partidos previos tuvo un equipo en los últimos N días.
    No usa groupby.apply, así que evita el FutureWarning y suele ser más rápido.
    """
    dates = group["date_dt"].to_numpy(dtype="datetime64[D]")
    counts = np.zeros(len(dates), dtype=int)

    for i in range(len(dates)):
        current_date = dates[i]
        start_date = current_date - np.timedelta64(days, "D")
        # contar juegos previos entre (current_date - days) y current_date, excluyendo el actual
        counts[i] = np.sum((dates[:i] >= start_date) & (dates[:i] < current_date))

    return pd.Series(counts, index=group.index)


def calculate_team_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    print("⚙️ Generando variables generales por equipo...")

    home_df = df[
        ["date", "game_id", "home_team", "home_pts_total", "away_pts_total", "home_q1", "away_q1"]
    ].copy()
    home_df.columns = [
        "date", "game_id", "team", "pts_scored", "pts_conceded", "q1_scored", "q1_conceded"
    ]
    home_df["is_home"] = 1

    away_df = df[
        ["date", "game_id", "away_team", "away_pts_total", "home_pts_total", "away_q1", "home_q1"]
    ].copy()
    away_df.columns = [
        "date", "game_id", "team", "pts_scored", "pts_conceded", "q1_scored", "q1_conceded"
    ]
    away_df["is_home"] = 0

    all_stats = pd.concat([home_df, away_df], ignore_index=True).copy()
    all_stats["date_dt"] = pd.to_datetime(all_stats["date"])
    all_stats = all_stats.sort_values(["team", "date_dt", "game_id"]).reset_index(drop=True)

    all_stats["won_game"] = (all_stats["pts_scored"] > all_stats["pts_conceded"]).astype(int)
    all_stats["pts_diff"] = all_stats["pts_scored"] - all_stats["pts_conceded"]
    all_stats["won_q1"] = (all_stats["q1_scored"] > all_stats["q1_conceded"]).astype(int)
    all_stats["q1_diff"] = all_stats["q1_scored"] - all_stats["q1_conceded"]

    all_stats["last_game_date"] = all_stats.groupby("team")["date_dt"].shift(1)
    all_stats["rest_days"] = (
        (all_stats["date_dt"] - all_stats["last_game_date"]).dt.days.fillna(5).clip(lower=0, upper=7)
    )
    all_stats["is_b2b"] = (all_stats["rest_days"] == 1).astype(int)

    # Fatiga avanzada sin groupby.apply
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

    def roll_decay_mean(col: str, window: int, decay: float = 0.85) -> pd.Series:
        def _weighted(values):
            n = len(values)
            if n <= 0:
                return np.nan
            weights = np.power(decay, np.arange(n - 1, -1, -1, dtype=float))
            return float(np.dot(values, weights) / weights.sum())

        return all_stats.groupby("team")[col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).apply(_weighted, raw=True)
        )

    all_stats["win_pct_L5"] = roll_mean("won_game", 5)
    all_stats["win_pct_L10"] = roll_mean("won_game", 10)
    all_stats["pts_diff_L5"] = roll_mean("pts_diff", 5)
    all_stats["pts_diff_L10"] = roll_mean("pts_diff", 10)
    all_stats["q1_win_pct_L5"] = roll_mean("won_q1", 5)
    all_stats["q1_diff_L5"] = roll_mean("q1_diff", 5)
    all_stats["pts_scored_L5"] = roll_mean("pts_scored", 5)
    all_stats["pts_conceded_L5"] = roll_mean("pts_conceded", 5)

    # Forma reciente ponderada: prioriza partidos más cercanos.
    all_stats["win_pct_D10"] = roll_decay_mean("won_game", 10, decay=0.85)
    all_stats["pts_diff_D10"] = roll_decay_mean("pts_diff", 10, decay=0.85)
    all_stats["q1_diff_D10"] = roll_decay_mean("q1_diff", 10, decay=0.85)

    # Momentum vs regresión: cuánto se separa el corto plazo del baseline reciente ponderado.
    all_stats["momentum_win"] = all_stats["win_pct_L5"] - all_stats["win_pct_D10"]
    all_stats["momentum_pts"] = all_stats["pts_diff_L5"] - all_stats["pts_diff_D10"]
    all_stats["momentum_q1"] = all_stats["q1_diff_L5"] - all_stats["q1_diff_D10"]
    all_stats["regression_alert"] = all_stats["momentum_win"].abs() + (all_stats["momentum_pts"].abs() / 20.0)

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
            "pts_diff_L5",
            "pts_diff_L10",
            "q1_win_pct_L5",
            "q1_diff_L5",
            "pts_scored_L5",
            "pts_conceded_L5",
            "win_pct_D10",
            "pts_diff_D10",
            "q1_diff_D10",
            "momentum_win",
            "momentum_pts",
            "momentum_q1",
            "regression_alert",
        ]
    ].copy()


def calculate_surface_split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("🏠✈️ Generando splits Home/Away...")

    # HOME ONLY
    home_only = df[
        ["date", "game_id", "home_team", "home_pts_total", "away_pts_total", "home_q1", "away_q1"]
    ].copy()
    home_only.columns = [
        "date", "game_id", "team", "pts_scored", "pts_conceded", "q1_scored", "q1_conceded"
    ]
    home_only["date_dt"] = pd.to_datetime(home_only["date"])
    home_only = home_only.sort_values(["team", "date_dt", "game_id"]).reset_index(drop=True)

    home_only["won_game"] = (home_only["pts_scored"] > home_only["pts_conceded"]).astype(int)
    home_only["pts_diff"] = home_only["pts_scored"] - home_only["pts_conceded"]
    home_only["won_q1"] = (home_only["q1_scored"] > home_only["q1_conceded"]).astype(int)

    def roll_home(col: str, window: int) -> pd.Series:
        return home_only.groupby("team")[col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

    home_only["home_only_win_pct_L5"] = roll_home("won_game", 5)
    home_only["home_only_pts_diff_L5"] = roll_home("pts_diff", 5)
    home_only["home_only_q1_win_pct_L5"] = roll_home("won_q1", 5)

    home_features = home_only[
        ["game_id", "team", "home_only_win_pct_L5", "home_only_pts_diff_L5", "home_only_q1_win_pct_L5"]
    ].copy()

    # AWAY ONLY
    away_only = df[
        ["date", "game_id", "away_team", "away_pts_total", "home_pts_total", "away_q1", "home_q1"]
    ].copy()
    away_only.columns = [
        "date", "game_id", "team", "pts_scored", "pts_conceded", "q1_scored", "q1_conceded"
    ]
    away_only["date_dt"] = pd.to_datetime(away_only["date"])
    away_only = away_only.sort_values(["team", "date_dt", "game_id"]).reset_index(drop=True)

    away_only["won_game"] = (away_only["pts_scored"] > away_only["pts_conceded"]).astype(int)
    away_only["pts_diff"] = away_only["pts_scored"] - away_only["pts_conceded"]
    away_only["won_q1"] = (away_only["q1_scored"] > away_only["q1_conceded"]).astype(int)

    def roll_away(col: str, window: int) -> pd.Series:
        return away_only.groupby("team")[col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

    away_only["away_only_win_pct_L5"] = roll_away("won_game", 5)
    away_only["away_only_pts_diff_L5"] = roll_away("pts_diff", 5)
    away_only["away_only_q1_win_pct_L5"] = roll_away("won_q1", 5)

    away_features = away_only[
        ["game_id", "team", "away_only_win_pct_L5", "away_only_pts_diff_L5", "away_only_q1_win_pct_L5"]
    ].copy()

    return home_features, away_features


def calculate_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features de enfrentamiento directo (H2H) usando solo historial previo al juego actual.
    """
    print("🤝 Generando features de enfrentamientos directos...")

    work = df[[
        "game_id",
        "date",
        "home_team",
        "away_team",
        "home_pts_total",
        "away_pts_total",
        "home_q1",
        "away_q1",
    ]].copy()

    work["date_dt"] = pd.to_datetime(work["date"])
    work = work.sort_values(["date_dt", "game_id"]).reset_index(drop=True)

    h2h_history = {}
    out_rows = []

    for _, row in work.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        key = tuple(sorted([home, away]))

        previous = h2h_history.get(key, [])
        last5 = previous[-5:]
        last10 = previous[-10:]

        def _home_win_pct(records):
            if not records:
                return 0.5
            wins = sum(1 for r in records if r["full_game_winner"] == home)
            return wins / len(records)

        def _home_q1_win_pct(records):
            if not records:
                return 0.5
            wins = sum(1 for r in records if r["q1_winner"] == home)
            return wins / len(records)

        def _home_pts_diff(records):
            if not records:
                return 0.0
            diffs = [r["pts_diff_from_a"][home] for r in records if home in r["pts_diff_from_a"]]
            return float(np.mean(diffs)) if diffs else 0.0

        out_rows.append(
            {
                "game_id": row["game_id"],
                "matchup_home_win_pct_L5": _home_win_pct(last5),
                "matchup_home_win_pct_L10": _home_win_pct(last10),
                "matchup_home_q1_win_pct_L5": _home_q1_win_pct(last5),
                "matchup_home_pts_diff_L5": _home_pts_diff(last5),
            }
        )

        home_score = int(row["home_pts_total"])
        away_score = int(row["away_pts_total"])
        home_q1 = int(row["home_q1"])
        away_q1 = int(row["away_q1"])

        if home_score > away_score:
            full_winner = home
        elif away_score > home_score:
            full_winner = away
        else:
            full_winner = "TIE"

        if home_q1 > away_q1:
            q1_winner = home
        elif away_q1 > home_q1:
            q1_winner = away
        else:
            q1_winner = "TIE"

        record = {
            "full_game_winner": full_winner,
            "q1_winner": q1_winner,
            "pts_diff_from_a": {
                home: home_score - away_score,
                away: away_score - home_score,
            },
        }

        h2h_history.setdefault(key, []).append(record)

    return pd.DataFrame(out_rows)


def add_league_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compara rendimiento reciente de cada equipo vs promedio de liga de ese mismo día.
    """
    print("🌐 Generando features relativas a promedio de liga...")

    home_snapshot = df[["date", "home_win_pct_L10", "home_pts_diff_L10", "home_q1_win_pct_L5"]].rename(
        columns={
            "home_win_pct_L10": "win_pct_L10",
            "home_pts_diff_L10": "pts_diff_L10",
            "home_q1_win_pct_L5": "q1_win_pct_L5",
        }
    )
    away_snapshot = df[["date", "away_win_pct_L10", "away_pts_diff_L10", "away_q1_win_pct_L5"]].rename(
        columns={
            "away_win_pct_L10": "win_pct_L10",
            "away_pts_diff_L10": "pts_diff_L10",
            "away_q1_win_pct_L5": "q1_win_pct_L5",
        }
    )

    league_snapshot = pd.concat([home_snapshot, away_snapshot], ignore_index=True)
    league_means = (
        league_snapshot.groupby("date", as_index=False)
        .agg(
            league_win_pct_L10=("win_pct_L10", "mean"),
            league_pts_diff_L10=("pts_diff_L10", "mean"),
            league_q1_win_pct_L5=("q1_win_pct_L5", "mean"),
        )
    )

    df = df.merge(league_means, on="date", how="left")

    df["home_win_pct_L10_vs_league"] = df["home_win_pct_L10"] - df["league_win_pct_L10"]
    df["away_win_pct_L10_vs_league"] = df["away_win_pct_L10"] - df["league_win_pct_L10"]

    df["home_pts_diff_L10_vs_league"] = df["home_pts_diff_L10"] - df["league_pts_diff_L10"]
    df["away_pts_diff_L10_vs_league"] = df["away_pts_diff_L10"] - df["league_pts_diff_L10"]

    df["home_q1_win_pct_L5_vs_league"] = df["home_q1_win_pct_L5"] - df["league_q1_win_pct_L5"]
    df["away_q1_win_pct_L5_vs_league"] = df["away_q1_win_pct_L5"] - df["league_q1_win_pct_L5"]

    return df


def build_features() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA)

    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date_dt", "game_id"]).reset_index(drop=True)
    df = df.drop(columns=["date_dt"])

    df = calculate_elo_ratings(df)

    df["TARGET_home_win"] = (df["home_pts_total"] > df["away_pts_total"]).astype(int)
    df["TARGET_home_win_q1"] = (df["home_q1"] > df["away_q1"]).astype(int)

    rolling_features = calculate_team_rolling_features(df)
    home_surface_features, away_surface_features = calculate_surface_split_features(df)
    matchup_features = calculate_matchup_features(df)

    # Join general home
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

    # Join general away
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

    # Join home-only for home team
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

    # Join away-only for away team
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

    # Join matchup features
    df = pd.merge(df, matchup_features, on="game_id", how="left")

    df = df.dropna(subset=["home_win_pct_L10", "away_win_pct_L10"]).copy()

    split_cols = [
        "home_home_only_win_pct_L5",
        "home_home_only_pts_diff_L5",
        "home_home_only_q1_win_pct_L5",
        "away_away_only_win_pct_L5",
        "away_away_only_pts_diff_L5",
        "away_away_only_q1_win_pct_L5",
    ]
    for col in split_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df = ensure_market_columns(df)
    df = add_context_features(df)
    df = add_league_relative_features(df)

    # Diferenciales base
    df["diff_elo"] = df["home_elo_pre"] - df["away_elo_pre"]
    df["diff_rest_days"] = df["home_rest_days"] - df["away_rest_days"]
    df["diff_is_b2b"] = df["home_is_b2b"] - df["away_is_b2b"]
    df["diff_games_last_3_days"] = df["home_games_last_3_days"] - df["away_games_last_3_days"]
    df["diff_games_last_5_days"] = df["home_games_last_5_days"] - df["away_games_last_5_days"]
    df["diff_games_last_7_days"] = df["home_games_last_7_days"] - df["away_games_last_7_days"]

    df["diff_win_pct_L5"] = df["home_win_pct_L5"] - df["away_win_pct_L5"]
    df["diff_win_pct_L10"] = df["home_win_pct_L10"] - df["away_win_pct_L10"]
    df["diff_pts_diff_L5"] = df["home_pts_diff_L5"] - df["away_pts_diff_L5"]
    df["diff_pts_diff_L10"] = df["home_pts_diff_L10"] - df["away_pts_diff_L10"]
    df["diff_q1_win_pct_L5"] = df["home_q1_win_pct_L5"] - df["away_q1_win_pct_L5"]
    df["diff_q1_diff_L5"] = df["home_q1_diff_L5"] - df["away_q1_diff_L5"]
    df["diff_pts_scored_L5"] = df["home_pts_scored_L5"] - df["away_pts_scored_L5"]
    df["diff_pts_conceded_L5"] = df["home_pts_conceded_L5"] - df["away_pts_conceded_L5"]
    df["diff_win_pct_D10"] = df["home_win_pct_D10"] - df["away_win_pct_D10"]
    df["diff_pts_diff_D10"] = df["home_pts_diff_D10"] - df["away_pts_diff_D10"]
    df["diff_q1_diff_D10"] = df["home_q1_diff_D10"] - df["away_q1_diff_D10"]
    df["diff_momentum_win"] = df["home_momentum_win"] - df["away_momentum_win"]
    df["diff_momentum_pts"] = df["home_momentum_pts"] - df["away_momentum_pts"]
    df["diff_momentum_q1"] = df["home_momentum_q1"] - df["away_momentum_q1"]
    df["diff_regression_alert"] = df["home_regression_alert"] - df["away_regression_alert"]

    # Diferenciales de home/away split
    df["diff_surface_win_pct_L5"] = df["home_home_only_win_pct_L5"] - df["away_away_only_win_pct_L5"]
    df["diff_surface_pts_diff_L5"] = df["home_home_only_pts_diff_L5"] - df["away_away_only_pts_diff_L5"]
    df["diff_surface_q1_win_pct_L5"] = df["home_home_only_q1_win_pct_L5"] - df["away_away_only_q1_win_pct_L5"]

    # Diferenciales de matchup
    df["diff_matchup_home_edge_L5"] = (df["matchup_home_win_pct_L5"] - 0.5) * 2
    df["diff_matchup_home_edge_L10"] = (df["matchup_home_win_pct_L10"] - 0.5) * 2

    # Diferenciales relativos a liga
    df["diff_win_pct_L10_vs_league"] = df["home_win_pct_L10_vs_league"] - df["away_win_pct_L10_vs_league"]
    df["diff_pts_diff_L10_vs_league"] = df["home_pts_diff_L10_vs_league"] - df["away_pts_diff_L10_vs_league"]
    df["diff_q1_win_pct_L5_vs_league"] = df["home_q1_win_pct_L5_vs_league"] - df["away_q1_win_pct_L5_vs_league"]

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

        "home_pts_diff_L5",
        "away_pts_diff_L5",
        "diff_pts_diff_L5",

        "home_pts_diff_L10",
        "away_pts_diff_L10",
        "diff_pts_diff_L10",

        "home_q1_win_pct_L5",
        "away_q1_win_pct_L5",
        "diff_q1_win_pct_L5",

        "home_q1_diff_L5",
        "away_q1_diff_L5",
        "diff_q1_diff_L5",

        "home_pts_scored_L5",
        "away_pts_scored_L5",
        "diff_pts_scored_L5",

        "home_pts_conceded_L5",
        "away_pts_conceded_L5",
        "diff_pts_conceded_L5",

        "home_win_pct_D10",
        "away_win_pct_D10",
        "diff_win_pct_D10",

        "home_pts_diff_D10",
        "away_pts_diff_D10",
        "diff_pts_diff_D10",

        "home_q1_diff_D10",
        "away_q1_diff_D10",
        "diff_q1_diff_D10",

        "home_momentum_win",
        "away_momentum_win",
        "diff_momentum_win",

        "home_momentum_pts",
        "away_momentum_pts",
        "diff_momentum_pts",

        "home_momentum_q1",
        "away_momentum_q1",
        "diff_momentum_q1",

        "home_regression_alert",
        "away_regression_alert",
        "diff_regression_alert",

        "home_home_only_win_pct_L5",
        "away_away_only_win_pct_L5",
        "diff_surface_win_pct_L5",

        "home_home_only_pts_diff_L5",
        "away_away_only_pts_diff_L5",
        "diff_surface_pts_diff_L5",

        "home_home_only_q1_win_pct_L5",
        "away_away_only_q1_win_pct_L5",
        "diff_surface_q1_win_pct_L5",

        "home_win_pct_L10_vs_league",
        "away_win_pct_L10_vs_league",
        "diff_win_pct_L10_vs_league",

        "home_pts_diff_L10_vs_league",
        "away_pts_diff_L10_vs_league",
        "diff_pts_diff_L10_vs_league",

        "home_q1_win_pct_L5_vs_league",
        "away_q1_win_pct_L5_vs_league",
        "diff_q1_win_pct_L5_vs_league",

        "home_spread",
        "spread_abs",
        "home_is_favorite",
        "odds_over_under",
        "market_missing",

        "TARGET_home_win",
        "TARGET_home_win_q1",
    ]

    final_df = df[model_columns].copy()
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Features avanzadas generadas. Partidos listos: {len(final_df)}")
    print(f"💾 Archivo guardado en: {OUTPUT_FILE}")
    return final_df


if __name__ == "__main__":
    build_features()