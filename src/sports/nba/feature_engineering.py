import numpy as np
import pandas as pd
from pathlib import Path

# --- RUTAS ---
import sys
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
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


def compute_intensity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add compact pregame-valid intensity features.

    Produces:
      - timezone_difference
      - playoff_multiplier
      - travel_penalty
      - intensity_score
      - is_high_intensity_game

    Uses existing context columns produced by `add_context_features`.
    """
    # timezone difference: reuse existing column if available
    if "away_tz_diff" in df.columns:
        df["timezone_difference"] = df["away_tz_diff"].fillna(0).astype(int)
    else:
        home_tz = df["home_team"].map(lambda t: TEAM_TIMEZONE.get(str(t), "ET"))
        away_tz = df["away_team"].map(lambda t: TEAM_TIMEZONE.get(str(t), "ET"))
        df["timezone_difference"] = (home_tz.map(TZ_OFFSET) - away_tz.map(TZ_OFFSET)).abs().fillna(0).astype(int)

    def _playoff_multiplier(days):
        try:
            d = int(days)
        except Exception:
            return 1.0
        if d <= 0:
            return 1.5
        factor = max(0, 30 - min(d, 30))
        return 1.0 + 0.5 * (factor / 30.0)

    df["playoff_multiplier"] = df.get("days_to_playoffs", 30).map(_playoff_multiplier).astype(float)

    df["travel_penalty"] = 0.0
    if "interconference_travel" in df.columns:
        df.loc[df["interconference_travel"] == 1, "travel_penalty"] = -0.15
    else:
        df.loc[(df["timezone_difference"] >= 3) & (df.get("same_conference", 0) == 0), "travel_penalty"] = -0.15

    div_bonus = (df.get("same_division", 0) == 1).astype(int) * 0.40
    conf_bonus = ((df.get("same_conference", 0) == 1) & (df.get("same_division", 0) == 0)).astype(int) * 0.25

    base_intensity = 1.0 + div_bonus + conf_bonus + df["travel_penalty"]
    df["intensity_score"] = (base_intensity * df["playoff_multiplier"]).round(6)
    df["is_high_intensity_game"] = (df["intensity_score"] >= 1.3).astype(int)

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


def add_trueskill_from_elo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas de TrueSkill aproximadas a partir del ELO pre-game calculado.
    - Implementación segura y ligera: si existe una migración a trueskill más adelante, se puede reemplazar.
    - No introduce dependencias nuevas ni hace forward-looking leaks: usa `home_elo_pre`/`away_elo_pre`.
    """
    print("🔔 Generando TrueSkill (aprox. desde ELO) ...")
    # Defaults compatibles con TrueSkill: mu ~25, sigma ~8.333
    base_mu = 25.0
    base_sigma = 8.333

    # Map ELO -> pseudo-TS mu: escala lineal (segura, histórica porque usa elo_pre)
    # 1500 ELO -> mu 25.0. Cada 40 puntos de ELO ~ 1 punto de mu.
    if "home_elo_pre" in df.columns and "away_elo_pre" in df.columns:
        df["home_ts_mu"] = base_mu + (df["home_elo_pre"] - 1500.0) / 40.0
        df["away_ts_mu"] = base_mu + (df["away_elo_pre"] - 1500.0) / 40.0
    else:
        df["home_ts_mu"] = base_mu
        df["away_ts_mu"] = base_mu

    # Simple constant sigma (conservador). Si queremos refinamiento, se puede ajustar
    df["home_ts_sigma"] = base_sigma
    df["away_ts_sigma"] = base_sigma

    df["diff_ts_mu"] = df["home_ts_mu"] - df["away_ts_mu"]
    df["diff_ts_sigma"] = df["home_ts_sigma"] - df["away_ts_sigma"]

    return df


def ensure_market_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Añadimos 'home_moneyline_odds' a la lista para limpiarla también
    market_cols = ["home_spread", "spread_abs", "home_is_favorite", "odds_over_under", "home_moneyline_odds"]

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
    # base columns for home/away (safe selection to avoid KeyError if optional boxscore columns missing)
    home_base = ["date", "game_id", "home_team", "home_pts_total", "away_pts_total", "home_q1", "away_q1"]
    away_base = ["date", "game_id", "away_team", "away_pts_total", "home_pts_total", "away_q1", "home_q1"]

    home_df = df[home_base].copy()
    home_df.columns = ["date", "game_id", "team", "pts_scored", "pts_conceded", "q1_scored", "q1_conceded"]
    home_df["is_home"] = 1

    away_df = df[away_base].copy()
    away_df.columns = ["date", "game_id", "team", "pts_scored", "pts_conceded", "q1_scored", "q1_conceded"]
    away_df["is_home"] = 0

    # map optional boxscore columns if present in original dataframe; align using original df index
    for src, dst in [("home_fgm", "fgm"), ("home_fga", "fga"), ("home_3pm", "three_pm"), ("home_tov", "tov"), ("home_orb", "orb"), ("home_fta", "fta")]:
        if src in df.columns:
            home_df[dst] = df.loc[home_df.index, src].astype(float)
        else:
            home_df[dst] = np.nan

    for src, dst in [("away_fgm", "fgm"), ("away_fga", "fga"), ("away_3pm", "three_pm"), ("away_tov", "tov"), ("away_orb", "orb"), ("away_fta", "fta")]:
        if src in df.columns:
            away_df[dst] = df.loc[away_df.index, src].astype(float)
        else:
            away_df[dst] = np.nan

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

    # Conteo para 6 días (útil para 4-in-6)
    all_stats["games_last_6_days"] = (
        all_stats.groupby("team", group_keys=False)
        .apply(lambda g: count_games_in_last_days(g, 6), include_groups=False)
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
    # Last-3 win% (históricamente válido, shift para evitar leakage)
    all_stats["last_3_win_pct"] = all_stats.groupby("team")["won_game"].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    all_stats["win_pct_L10"] = roll_mean("won_game", 10)
    all_stats["pts_diff_L5"] = roll_mean("pts_diff", 5)
    all_stats["pts_diff_L10"] = roll_mean("pts_diff", 10)
    all_stats["q1_win_pct_L5"] = roll_mean("won_q1", 5)
    all_stats["q1_diff_L5"] = roll_mean("q1_diff", 5)
    all_stats["pts_scored_L5"] = roll_mean("pts_scored", 5)
    all_stats["pts_conceded_L5"] = roll_mean("pts_conceded", 5)

    # Keep a last-5 alias (requested naming)
    all_stats["last_5_win_pct"] = all_stats["win_pct_L5"]

    # Forma reciente ponderada: prioriza partidos más cercanos.
    all_stats["win_pct_D10"] = roll_decay_mean("won_game", 10, decay=0.85)
    all_stats["pts_diff_D10"] = roll_decay_mean("pts_diff", 10, decay=0.85)
    all_stats["q1_diff_D10"] = roll_decay_mean("q1_diff", 10, decay=0.85)

    # Momentum vs regresión: cuánto se separa el corto plazo del baseline reciente ponderado.
    all_stats["momentum_win"] = all_stats["win_pct_L5"] - all_stats["win_pct_D10"]
    all_stats["momentum_pts"] = all_stats["pts_diff_L5"] - all_stats["pts_diff_D10"]
    all_stats["momentum_q1"] = all_stats["q1_diff_L5"] - all_stats["q1_diff_D10"]
    all_stats["regression_alert"] = all_stats["momentum_win"].abs() + (all_stats["momentum_pts"].abs() / 20.0)

    # --- Nuevas señales solicitadas: recent form, streak, volatility, momentum (compact)
    # days_since_last_game: alias a rest_days (ya es histórico, shift aplicado)
    all_stats["days_since_last_game"] = all_stats["rest_days"].fillna(5).astype(float)

    # game_frequency: aproximación simple = juegos en las últimas 7 días / 7
    all_stats["game_frequency"] = all_stats["games_last_7_days"] / 7.0

    # recent_form_score: mezcla simple de short-term (L3) y L5
    all_stats["recent_form_score"] = (0.6 * all_stats["last_3_win_pct"]) + (0.4 * all_stats["last_5_win_pct"])

    # Volatility: std de outcomes (won_game) en ventana 10 (históricamente válida)
    all_stats["volatility"] = all_stats.groupby("team")["won_game"].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=1).std().fillna(0)
    )

    # Momentum: alias compacto (ya se calcula como momentum_win), pero dejamos campo 'momentum'
    all_stats["momentum"] = all_stats["momentum_win"]

    # Streak: racha vigente antes del partido (positivo = wins, negativo = losses)
    def _compute_streak_group(g: pd.Series) -> pd.Series:
        # g is the group's won_game series with original index
        vals = g.to_numpy()
        out = np.zeros(len(vals), dtype=int)
        for i in range(len(vals)):
            if i == 0:
                out[i] = 0
                continue
            prev = vals[i - 1]
            direction = 1 if prev == 1 else -1
            s = direction
            j = i - 2
            while j >= 0 and vals[j] == prev:
                s += direction
                j -= 1
            out[i] = s
        return pd.Series(out, index=g.index)

    # compute streak per group and assign into an array to guarantee alignment
    streak_arr = np.zeros(len(all_stats), dtype=int)
    for team, g in all_stats.groupby("team")["won_game"]:
        vals = g.to_numpy()
        out = np.zeros(len(vals), dtype=int)
        for i in range(len(vals)):
            if i == 0:
                out[i] = 0
                continue
            prev = vals[i - 1]
            direction = 1 if prev == 1 else -1
            s = direction
            j = i - 2
            while j >= 0 and vals[j] == prev:
                s += direction
                j -= 1
            out[i] = s
        streak_arr[g.index] = out

    all_stats["streak"] = streak_arr

    # --- FOUR FACTORS (L10) ---
    # Compute per-game four-factor pieces when boxscore pieces exist, otherwise produce NaNs
    # possessions estimate: FGA + 0.4*FTA - ORB + TOV (classic approximation)
    def _safe_div(a, b):
        return np.where((b == 0) | np.isnan(b), np.nan, a / b)

    poss = all_stats.get("fga", pd.Series(np.nan, index=all_stats.index)).fillna(np.nan) + 0.4 * all_stats.get("fta", pd.Series(np.nan, index=all_stats.index)).fillna(np.nan) - all_stats.get("orb", pd.Series(np.nan, index=all_stats.index)).fillna(np.nan) + all_stats.get("tov", pd.Series(np.nan, index=all_stats.index)).fillna(np.nan)

    # eFG% = (FGM + 0.5*3PM) / FGA
    all_stats["efg_pct"] = _safe_div(all_stats.get("fgm", 0).fillna(0) + 0.5 * all_stats.get("three_pm", 0).fillna(0), all_stats.get("fga", np.nan))
    # TOV rate = TOV / possessions
    all_stats["tov_rate"] = _safe_div(all_stats.get("tov", np.nan).fillna(np.nan), poss)
    # ORB rate: proxy = ORB / (ORB + 1) if opponent ORB not available
    all_stats["orb_rate"] = _safe_div(all_stats.get("orb", np.nan).fillna(np.nan), (all_stats.get("orb", 0).fillna(0) + 1))
    # FT rate = FTA / possessions
    all_stats["ft_rate"] = _safe_div(all_stats.get("fta", np.nan).fillna(np.nan), poss)

    # Rolling L10 for four factors (historic: shift to avoid leakage)
    all_stats["efg_pct_L10"] = all_stats.groupby("team")["efg_pct"].transform(lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())
    all_stats["tov_rate_L10"] = all_stats.groupby("team")["tov_rate"].transform(lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())
    all_stats["orb_rate_L10"] = all_stats.groupby("team")["orb_rate"].transform(lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())
    all_stats["ft_rate_L10"] = all_stats.groupby("team")["ft_rate"].transform(lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())

    # --- NET RATING and EWMA ---
    # estimate possessions per game if possible; else use simple proxy based on scoring
    # fallback possessions proxy to avoid NaNs: (pts_scored + pts_conceded) / 1.02
    poss_fallback = (all_stats["pts_scored"] + all_stats["pts_conceded"]) / 1.02
    possessions = poss.copy().fillna(poss_fallback)

    all_stats["net_rating_game"] = ((all_stats["pts_scored"] - all_stats["pts_conceded"]) / possessions) * 100.0
    # rolling L10 net rating
    all_stats["net_rating_L10"] = all_stats.groupby("team")["net_rating_game"].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
    )
    # EWMA net rating (span ~10 -> alpha ~ 2/(n+1))
    def _ewma_shifted(x, span=10):
        return x.shift(1).ewm(span=span, adjust=False).mean()

    all_stats["ewma_net_rating"] = all_stats.groupby("team")["net_rating_game"].transform(lambda x: _ewma_shifted(x, span=10))

    # --- Q1-specific rolling metrics (L10) ---
    all_stats["q1_pts_scored_L10"] = all_stats.groupby("team")["q1_scored"].transform(lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())
    all_stats["q1_pts_allowed_L10"] = all_stats.groupby("team")["q1_conceded"].transform(lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())
    all_stats["q1_net_rating_L10"] = all_stats["q1_pts_scored_L10"] - all_stats["q1_pts_allowed_L10"]
    all_stats["q1_ewma_scoring"] = all_stats.groupby("team")["q1_scored"].transform(lambda x: _ewma_shifted(x, span=8))

    # --- FATIGUE / SCHEDULE LOAD refinements ---
    # games in last 4 days
    all_stats["games_last_4_days"] = all_stats.groupby("team", group_keys=False).apply(lambda g: count_games_in_last_days(g, 4), include_groups=False).sort_index()
    all_stats["4in6_flag"] = (all_stats["games_last_6_days"] >= 4).astype(int)
    all_stats["3in4_flag"] = (all_stats["games_last_4_days"] >= 3).astype(int)
    # fatigue_load: weighted sum (short-term emphasis)
    all_stats["fatigue_load"] = (
        all_stats["games_last_3_days"].fillna(0) * 1.0
        + all_stats["games_last_5_days"].fillna(0) * 0.5
        + all_stats["is_b2b"].fillna(0) * 0.8
        - all_stats["rest_days"].fillna(3) * 0.1
    )

    # --- BAYESIAN SHRINKAGE (simple) ---
    # shrink observed proportions toward league mean using effective sample size k
    def _shrink_mean(obs_series, count_series, league_mean, k=5.0):
        # obs_series and count_series are aligned series
        w = count_series / (count_series + k)
        return (w * obs_series) + ((1 - w) * league_mean)

    # compute rolling counts for windows
    all_stats["count_L10"] = all_stats.groupby("team")["won_game"].transform(lambda x: x.shift(1).rolling(window=10, min_periods=1).count())
    # league mean for win_pct across the dataset (safe prior)
    global_win_mean = all_stats["won_game"].mean() if len(all_stats) > 0 else 0.5
    all_stats["win_pct_L10_shrunk"] = _shrink_mean(all_stats["win_pct_L10"].fillna(global_win_mean), all_stats["count_L10"].fillna(0), global_win_mean, k=5.0)

    # Q1 shrink (window 5)
    all_stats["count_q1_L5"] = all_stats.groupby("team")["won_q1"].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).count())
    global_q1_mean = all_stats["won_q1"].mean() if len(all_stats) > 0 else 0.5
    all_stats["q1_win_pct_L5_shrunk"] = _shrink_mean(all_stats["q1_win_pct_L5"].fillna(global_q1_mean), all_stats["count_q1_L5"].fillna(0), global_q1_mean, k=3.0)

    # net rating shrink toward zero (league neutral) using effective games
    all_stats["net_count_L10"] = all_stats.groupby("team")["net_rating_game"].transform(lambda x: x.shift(1).rolling(window=10, min_periods=1).count())
    all_stats["net_rating_L10_shrunk"] = _shrink_mean(all_stats["net_rating_L10"].fillna(0.0), all_stats["net_count_L10"].fillna(0), 0.0, k=8.0)

    return all_stats[
        [
            "game_id",
            "team",
            "rest_days",
            "is_b2b",
            "games_last_3_days",
            "games_last_5_days",
            "games_last_6_days",
            "games_last_7_days",
            "game_frequency",
            "days_since_last_game",
            "last_3_win_pct",
            "last_5_win_pct",
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
            "momentum",
            "volatility",
            "recent_form_score",
            "streak",
            "regression_alert",
            # Four factors L10
            "efg_pct_L10",
            "tov_rate_L10",
            "orb_rate_L10",
            "ft_rate_L10",
            # Net rating / EWMA
            "net_rating_L10",
            "ewma_net_rating",
            # Q1 specific
            "q1_pts_scored_L10",
            "q1_pts_allowed_L10",
            "q1_net_rating_L10",
            "q1_ewma_scoring",
            # Fatigue / schedule
            "games_last_4_days",
            "4in6_flag",
            "3in4_flag",
            "fatigue_load",
            # Bayesian shrinkage
            "win_pct_L10_shrunk",
            "q1_win_pct_L5_shrunk",
            "net_rating_L10_shrunk",
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


def add_full_game_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    # Leakage-safe engineered signals built only from pregame columns.
    def _num(col: str, default: float = 0.0) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
        return pd.Series(default, index=df.index, dtype=float)

    home_elo = _num("home_elo_pre")
    away_elo = _num("away_elo_pre")
    home_spread = _num("home_spread")

    home_rest = _num("home_rest_days")
    away_rest = _num("away_rest_days")
    home_b2b = _num("home_is_b2b")
    away_b2b = _num("away_is_b2b")
    home_g5 = _num("home_games_last_5_days")
    away_g5 = _num("away_games_last_5_days")

    diff_win_l10 = _num("diff_win_pct_L10")
    diff_pts_l10 = _num("diff_pts_diff_L10")
    diff_surface_pts = _num("diff_surface_pts_diff_L5")
    diff_matchup_l5 = _num("diff_matchup_home_edge_L5")
    diff_pts_vs_league = _num("diff_pts_diff_L10_vs_league")
    diff_rest = _num("diff_rest_days")
    diff_is_b2b = _num("diff_is_b2b")
    diff_g5 = _num("diff_games_last_5_days")

    # Home spread expected from ELO (same convention as home_spread).
    elo_expected_spread = (away_elo - (home_elo + 100.0)) / 28.0
    df["elo_spread_gap"] = home_spread - elo_expected_spread
    df["elo_spread_gap_abs"] = df["elo_spread_gap"].abs()

    # Composite home edge from stable macro signals.
    df["fullgame_form_strength_edge"] = (
        (diff_win_l10 * 0.45)
        + (diff_pts_l10 * 0.20)
        + (diff_surface_pts * 0.20)
        + (diff_matchup_l5 * 0.15)
    )

    # Schedule pressure differential (positive => better for home).
    home_stress = (home_g5 * 0.70) + (home_b2b * 1.40) - (home_rest * 0.50)
    away_stress = (away_g5 * 0.70) + (away_b2b * 1.40) - (away_rest * 0.50)
    df["schedule_stress_edge"] = away_stress - home_stress

    # Interactions to capture context non-linearity.
    df["fullgame_context_interaction"] = diff_win_l10 * diff_matchup_l5
    df["rest_form_interaction"] = diff_rest * diff_pts_vs_league
    df["fatigue_penalty_edge"] = (diff_is_b2b * 0.8) + (diff_g5 * 0.25)

    return df


def build_features() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA)

# --- NUEVO: Limpieza de columnas de lesiones ---
    if "home_injuries_count" in df.columns:
        df["home_injuries_count"] = pd.to_numeric(df["home_injuries_count"], errors="coerce").fillna(0)
    else:
        df["home_injuries_count"] = 0
    if "away_injuries_count" in df.columns:
        df["away_injuries_count"] = pd.to_numeric(df["away_injuries_count"], errors="coerce").fillna(0)
    else:
        df["away_injuries_count"] = 0
    # -----------------------------------------------

    df["date_dt"] = pd.to_datetime(df["date"])

    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date_dt", "game_id"]).reset_index(drop=True)
    df = df.drop(columns=["date_dt"])

    df = calculate_elo_ratings(df)
    # Añadimos TrueSkill aproximado (a partir del ELO pre-game) - seguro y sin leakage
    df = add_trueskill_from_elo(df)

    df["TARGET_home_win"] = (df["home_pts_total"] > df["away_pts_total"]).astype(int)
    df["TARGET_home_win_q1"] = (df["home_q1"] > df["away_q1"]).astype(int)
    first_half_home = pd.to_numeric(df.get("home_q1", 0), errors="coerce").fillna(0) + pd.to_numeric(df.get("home_q2", 0), errors="coerce").fillna(0)
    first_half_away = pd.to_numeric(df.get("away_q1", 0), errors="coerce").fillna(0) + pd.to_numeric(df.get("away_q2", 0), errors="coerce").fillna(0)
    df["TARGET_home_win_h1"] = np.where(first_half_home > first_half_away, 1, np.where(first_half_home < first_half_away, 0, np.nan))
    # Nuevos mercados (sin leakage): spread y total usan línea pregame
    home_margin = pd.to_numeric(df["home_pts_total"], errors="coerce") - pd.to_numeric(df["away_pts_total"], errors="coerce")
    home_spread_num = pd.to_numeric(df.get("home_spread", 0), errors="coerce")
    total_line_num = pd.to_numeric(df.get("odds_over_under", 0), errors="coerce")
    game_total_points = pd.to_numeric(df["home_pts_total"], errors="coerce") + pd.to_numeric(df["away_pts_total"], errors="coerce")

    spread_result = home_margin + home_spread_num
    df["TARGET_home_cover_spread"] = np.where(
        home_spread_num.notna() & (home_spread_num != 0),
        np.where(spread_result > 0, 1, np.where(spread_result < 0, 0, np.nan)),
        np.nan,
    )

    df["TARGET_over_total"] = np.where(
        total_line_num.notna() & (total_line_num > 0),
        np.where(game_total_points > total_line_num, 1, np.where(game_total_points < total_line_num, 0, np.nan)),
        np.nan,
    )

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
    # Intensity features (compact, pregame-valid, non-duplicative)
    df = compute_intensity_features(df)
    df = add_league_relative_features(df)

    # Diferenciales base
    df["diff_elo"] = df["home_elo_pre"] - df["away_elo_pre"]
    df["diff_rest_days"] = df["home_rest_days"] - df["away_rest_days"]
    df["diff_is_b2b"] = df["home_is_b2b"] - df["away_is_b2b"]
    df["diff_games_last_3_days"] = df["home_games_last_3_days"] - df["away_games_last_3_days"]
    df["diff_games_last_5_days"] = df["home_games_last_5_days"] - df["away_games_last_5_days"]
    df["diff_games_last_6_days"] = df["home_games_last_6_days"] - df["away_games_last_6_days"]
    df["diff_games_last_7_days"] = df["home_games_last_7_days"] - df["away_games_last_7_days"]
    # diff for games_last_4_days (added in rolling features)
    if "home_games_last_4_days" in df.columns and "away_games_last_4_days" in df.columns:
        df["diff_games_last_4_days"] = df["home_games_last_4_days"] - df["away_games_last_4_days"]

    # Four-factors diffs (L10)
    for base in ["efg_pct_L10", "tov_rate_L10", "orb_rate_L10", "ft_rate_L10"]:
        h = f"home_{base}"
        a = f"away_{base}"
        d = f"diff_{base}"
        if h in df.columns and a in df.columns:
            df[d] = df[h] - df[a]

    # Net rating diffs
    if "home_net_rating_L10" in df.columns and "away_net_rating_L10" in df.columns:
        df["diff_net_rating_L10"] = df["home_net_rating_L10"] - df["away_net_rating_L10"]
    if "home_ewma_net_rating" in df.columns and "away_ewma_net_rating" in df.columns:
        df["diff_ewma_net_rating"] = df["home_ewma_net_rating"] - df["away_ewma_net_rating"]

    # Q1-specific diffs
    for base in ["q1_pts_scored_L10", "q1_pts_allowed_L10", "q1_net_rating_L10", "q1_ewma_scoring"]:
        h = f"home_{base}"
        a = f"away_{base}"
        d = f"diff_{base}"
        if h in df.columns and a in df.columns:
            df[d] = df[h] - df[a]

    # Fatigue diffs / flags
    if "home_4in6_flag" in df.columns and "away_4in6_flag" in df.columns:
        df["diff_4in6_flag"] = df["home_4in6_flag"] - df["away_4in6_flag"]
    if "home_3in4_flag" in df.columns and "away_3in4_flag" in df.columns:
        df["diff_3in4_flag"] = df["home_3in4_flag"] - df["away_3in4_flag"]
    if "home_fatigue_load" in df.columns and "away_fatigue_load" in df.columns:
        df["diff_fatigue_load"] = df["home_fatigue_load"] - df["away_fatigue_load"]

    # Shrinkage diffs
    if "home_win_pct_L10_shrunk" in df.columns and "away_win_pct_L10_shrunk" in df.columns:
        df["diff_win_pct_L10_shrunk"] = df["home_win_pct_L10_shrunk"] - df["away_win_pct_L10_shrunk"]
    if "home_q1_win_pct_L5_shrunk" in df.columns and "away_q1_win_pct_L5_shrunk" in df.columns:
        df["diff_q1_win_pct_L5_shrunk"] = df["home_q1_win_pct_L5_shrunk"] - df["away_q1_win_pct_L5_shrunk"]
    if "home_net_rating_L10_shrunk" in df.columns and "away_net_rating_L10_shrunk" in df.columns:
        df["diff_net_rating_L10_shrunk"] = df["home_net_rating_L10_shrunk"] - df["away_net_rating_L10_shrunk"]
    
    # --- FASE 2: FEATURES DE LESIONES ---
    # Diferencia neta: Un número negativo significa que el local tiene MENOS lesionados que el visitante (Ventaja Local)
    df["diff_injuries"] = df["home_injuries_count"] - df["away_injuries_count"]
    
    # Alerta de ventaja física: ¿Hay una diferencia de 3 o más jugadores?
    # Esto ayuda a la IA a detectar partidos donde un equipo viene "parchado"
    df["injury_advantage_home"] = (df["diff_injuries"] <= -3).astype(int)
    df["injury_advantage_away"] = (df["diff_injuries"] >= 3).astype(int)
    
    # Interacción Lesiones + Odds (Para detectar lo que decías del dinero raro)
    # Si un equipo tiene ventaja de lesiones pero NO es el favorito, hay algo raro
    df["injury_surprise_upset"] = ((df["injury_advantage_home"] == 1) & (df["home_is_favorite"] == 0)).astype(int)
    # -------------------------------------

    # Mejora de impacto de lesiones
    # Flag si falta el top scorer detectado (posible data-missing o baja importante)
    if "home_top_scorer_pts" in df.columns and "away_top_scorer_pts" in df.columns:
        df["home_missing_top_scorer"] = (df["home_top_scorer_pts"] == 0).astype(int)
        df["away_missing_top_scorer"] = (df["away_top_scorer_pts"] == 0).astype(int)
    else:
        df["home_missing_top_scorer"] = 0
        df["away_missing_top_scorer"] = 0

    # Puntuación simple de impacto por lesiones (para que el modelo aprenda magnitude)
    df["injury_impact_score"] = (df["away_injuries_count"] - df["home_injuries_count"]).abs() * 0.5
    # Si falta top scorer, aumentamos la señal
    df["injury_impact_score"] += df["home_missing_top_scorer"] * 1.5 + df["away_missing_top_scorer"] * 1.5

# --- FASE 3: FEATURES DE VALOR DE MERCADO (ACCURACY BOOST) ---
    # 1. Spread esperado según nuestro ELO (Regla: ~28 pts ELO = 1 pt spread)
    df["elo_expected_spread"] = (df["away_elo_pre"] - (df["home_elo_pre"] + 100)) / 28
    
    # 2. Market Edge: Diferencia entre nuestro cálculo y el del casino
    df["market_diff_spread"] = df["elo_expected_spread"] - df["home_spread"]

    # --- MARKET EDGE (nombre alternativo para entrenamiento/interpretabilidad) ---
    df["model_edge"] = df["elo_expected_spread"] - df["home_spread"]
    df["is_huge_edge"] = (df["model_edge"].abs() > 5).astype(int)

    # 3. Probabilidad implícita del mercado (robusta para distintos formatos)
    def american_to_implied(american):
        try:
            american = float(american)
        except Exception:
            return np.nan
        if american == 0 or pd.isna(american):
            return 0.5
        if 1.0 < american < 25.0:
            return 1.0 / american
        if american > 0:
            return 100.0 / (american + 100.0)
        return abs(american) / (abs(american) + 100.0)

    # Usamos momios si están; si no, aproximamos desde spread
    if "home_moneyline_odds" in df.columns and df["home_moneyline_odds"].notna().any():
        df["market_prob_home"] = df["home_moneyline_odds"].apply(american_to_implied)
    else:
        df["market_prob_home"] = 1 / (1 + 10 ** (df["home_spread"] * 25.0 / 400.0))

    # Implied prob exacta desde moneyline (si existe) para comparar con el modelo
    if "home_moneyline_odds" in df.columns:
        df["implied_prob_home_ml"] = df["home_moneyline_odds"].apply(american_to_implied)
    else:
        df["implied_prob_home_ml"] = df["market_prob_home"]

    # --- MARKET vs MODEL disagreement (usar columnas meta si disponibles, sino campos model_prob_home)
    df["market_model_prob_signed_diff"] = np.nan
    df["market_model_prob_abs_diff"] = np.nan
    model_prob_col = None
    if "full_game_meta_prob_home" in df.columns:
        model_prob_col = "full_game_meta_prob_home"
    elif "model_prob_home" in df.columns:
        model_prob_col = "model_prob_home"

    if model_prob_col is not None:
        df["market_model_prob_signed_diff"] = df[model_prob_col] - df["implied_prob_home_ml"]
        df["market_model_prob_abs_diff"] = df["market_model_prob_signed_diff"].abs().fillna(0)

    # Favorito según mercado
    df["market_favorite_is_home"] = (df["home_spread"] < 0).astype(int)
    # Favorito según modelo (si existe)
    df["model_favorite_is_home"] = np.nan
    if "full_game_meta_prob_home" in df.columns:
        df["model_favorite_is_home"] = (df["full_game_meta_prob_home"] >= 0.5).astype(int)

    # Si el modelo favorece al underdog y la diferencia es grande, marcarlo
    df["is_model_underdog_with_edge"] = 0
    if "market_model_prob_abs_diff" in df.columns:
        df.loc[
            (df["model_favorite_is_home"] != df["market_favorite_is_home"]) & (df["market_model_prob_abs_diff"] > 0.07),
            "is_model_underdog_with_edge",
        ] = 1
    # -------------------------------------------------------------

    # --- FATIGA EXTREMA / SCHEDULE STRESS ---
    # 3 juegos en 4 noches (3-in-4)
    df["is_3_in_4"] = ((df["home_games_last_5_days"] >= 3) & (df["home_rest_days"] <= 1)).astype(int)
    # 5 juegos en 7 noches (5-in-7)
    df["is_5_in_7"] = (df["home_games_last_7_days"] >= 5).astype(int)
    # 4 juegos en 6 noches (4-in-6)
    df["is_4_in_6"] = (df["home_games_last_6_days"] >= 4).astype(int)

    # Return from road trip (heurística): si los últimos 2 juegos del equipo fueron away
    # y ahora juega en casa con descanso corto
    if "home_games_last_3_days" in df.columns and "home_rest_days" in df.columns:
        df["home_return_from_roadtrip"] = ((df["home_games_last_3_days"] >= 2) & (df["home_rest_days"] <= 2)).astype(int)
    else:
        df["home_return_from_roadtrip"] = 0

    # --- ALTITUD ---
    df["altitude_advantage"] = df["home_team"].isin(["DEN", "UTA"]).astype(int)

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

    # Diffs para nuevas señales (recent form / streak / volatility / momentum)
    df["diff_last_3_win_pct"] = df["home_last_3_win_pct"] - df["away_last_3_win_pct"]
    df["diff_last_5_win_pct"] = df["home_last_5_win_pct"] - df["away_last_5_win_pct"]

    df["diff_game_frequency"] = df["home_game_frequency"] - df["away_game_frequency"]

    df["diff_days_since_last_game"] = df["home_days_since_last_game"] - df["away_days_since_last_game"]

    df["diff_recent_form_score"] = df["home_recent_form_score"] - df["away_recent_form_score"]

    df["diff_streak"] = df["home_streak"] - df["away_streak"]

    df["diff_volatility"] = df["home_volatility"] - df["away_volatility"]

    df["diff_momentum"] = df["home_momentum"] - df["away_momentum"]

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
    df = add_full_game_signal_features(df)

    model_columns = [
        "game_id",
        "date",
        "season",
        "home_team",
        "away_team",

        "home_elo_pre",
        "away_elo_pre",
        "diff_elo",

        # TrueSkill (aprox. desde ELO)
        "home_ts_mu",
        "away_ts_mu",
        "diff_ts_mu",
        "home_ts_sigma",
        "away_ts_sigma",
        "diff_ts_sigma",

        "home_rest_days",
        "away_rest_days",
        "diff_rest_days",

        # Recent form / streak / volatility / momentum
        "home_last_3_win_pct",
        "away_last_3_win_pct",
        "diff_last_3_win_pct",

        "home_last_5_win_pct",
        "away_last_5_win_pct",
        "diff_last_5_win_pct",

        "home_game_frequency",
        "away_game_frequency",
        "diff_game_frequency",

        "home_days_since_last_game",
        "away_days_since_last_game",
        "diff_days_since_last_game",

        "home_recent_form_score",
        "away_recent_form_score",
        "diff_recent_form_score",

        "home_streak",
        "away_streak",
        "diff_streak",

        "home_volatility",
        "away_volatility",
        "diff_volatility",

        "home_momentum",
        "away_momentum",
        "diff_momentum",

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

        # --- FATIGUE / SCHEDULE LOAD ---
        "home_games_last_4_days",
        "away_games_last_4_days",
        "diff_games_last_4_days",
        "home_4in6_flag",
        "away_4in6_flag",
        "diff_4in6_flag",
        "home_3in4_flag",
        "away_3in4_flag",
        "diff_3in4_flag",
        "home_fatigue_load",
        "away_fatigue_load",
        "diff_fatigue_load",

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

        # --- Q1-specific L10 and EWMA ---
        "home_q1_pts_scored_L10",
        "away_q1_pts_scored_L10",
        "diff_q1_pts_scored_L10",
        "home_q1_pts_allowed_L10",
        "away_q1_pts_allowed_L10",
        "diff_q1_pts_allowed_L10",
        "home_q1_net_rating_L10",
        "away_q1_net_rating_L10",
        "diff_q1_net_rating_L10",
        "home_q1_ewma_scoring",
        "away_q1_ewma_scoring",
        "diff_q1_ewma_scoring",

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

        # --- FOUR FACTORS L10 ---
        "home_efg_pct_L10",
        "away_efg_pct_L10",
        "diff_efg_pct_L10",
        "home_tov_rate_L10",
        "away_tov_rate_L10",
        "diff_tov_rate_L10",
        "home_orb_rate_L10",
        "away_orb_rate_L10",
        "diff_orb_rate_L10",
        "home_ft_rate_L10",
        "away_ft_rate_L10",
        "diff_ft_rate_L10",

        # --- NET RATING / EWMA ---
        "home_net_rating_L10",
        "away_net_rating_L10",
        "diff_net_rating_L10",
        "home_ewma_net_rating",
        "away_ewma_net_rating",
        "diff_ewma_net_rating",

        # Intensity/context features
        "timezone_difference",
        "playoff_multiplier",
        "travel_penalty",
        "intensity_score",
        "is_high_intensity_game",

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

        # --- BAYESIAN SHRINKAGE ---
        "home_win_pct_L10_shrunk",
        "away_win_pct_L10_shrunk",
        "diff_win_pct_L10_shrunk",
        "home_q1_win_pct_L5_shrunk",
        "away_q1_win_pct_L5_shrunk",
        "diff_q1_win_pct_L5_shrunk",
        "home_net_rating_L10_shrunk",
        "away_net_rating_L10_shrunk",
        "diff_net_rating_L10_shrunk",

        "home_pts_diff_L10_vs_league",
        "away_pts_diff_L10_vs_league",
        "diff_pts_diff_L10_vs_league",

        "home_q1_win_pct_L5_vs_league",
        "away_q1_win_pct_L5_vs_league",
        "diff_q1_win_pct_L5_vs_league",

        # Full-game signal engineering
        "elo_spread_gap",
        "elo_spread_gap_abs",
        "fullgame_form_strength_edge",
        "schedule_stress_edge",
        "fullgame_context_interaction",
        "rest_form_interaction",
        "fatigue_penalty_edge",

        "home_spread",
        "spread_abs",
        "home_is_favorite",
        "odds_over_under",
        "market_missing",

        # --- MARKET VALUE FEATURES ---
        "elo_expected_spread",
        "market_diff_spread",
        "market_prob_home",

        # --- LESIONES ---
        "home_injuries_count",
        "away_injuries_count",
        "diff_injuries",
        "injury_advantage_home",
        "injury_advantage_away",
        "injury_surprise_upset",

        # --- AUDIT DATA (SÓLO PARA UI/CSV, NO PARA ENTRENAR) ---
        "home_pts_total",
        "away_pts_total",
        "home_q1",
        "away_q1",

        # --- TARGETS ---
        "TARGET_home_win",
        "TARGET_home_win_q1",
        "TARGET_home_win_h1",
        "TARGET_home_cover_spread",
        "TARGET_over_total",
    ]

    available_cols = [c for c in model_columns if c in df.columns]
    missing_cols = [c for c in model_columns if c not in df.columns]
    if missing_cols:
        print(f"⚠️ Columnas no disponibles y omitidas: {len(missing_cols)}")
        print("   " + ", ".join(missing_cols[:12]) + (" ..." if len(missing_cols) > 12 else ""))

    final_df = df[available_cols].copy()
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Features avanzadas generadas. Partidos listos: {len(final_df)}")
    print(f"💾 Archivo guardado en: {OUTPUT_FILE}")
    return final_df


if __name__ == "__main__":
    build_features()
