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
EXTERNAL_PRIOR_ARCHIVE_DIR = Path(
    os.getenv(
        "NBA_MLB_EXTERNAL_PRIOR_ARCHIVE_DIR",
        r"C:\Users\andra\Desktop\mlb-predictions-main\mlb-predictions-main\predictions\archive",
    )
)
USE_EXTERNAL_PRIOR = int(np.clip(float(os.getenv("NBA_MLB_USE_EXTERNAL_PRIOR", "0") or 0.0), 0.0, 1.0))
DISABLE_EXTERNAL_PRIOR_BLOCK = int(
    np.clip(float(os.getenv("NBA_MLB_DISABLE_EXTERNAL_PRIOR", "0") or 0.0), 0.0, 1.0)
)
DISABLE_PREV_SEASON_BLEND_BLOCK = int(
    np.clip(float(os.getenv("NBA_MLB_DISABLE_PREV_SEASON_BLEND", "1") or 0.0), 0.0, 1.0)
)
DISABLE_UMPIRE_EXPANDED_BLOCK = int(
    np.clip(float(os.getenv("NBA_MLB_DISABLE_UMPIRE_EXPANDED", "1") or 0.0), 0.0, 1.0)
)

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

    # Preserve existing ingest weather and only fill missing values.
    if "weather_temp" not in df.columns:
        df["weather_temp"] = np.nan
    if "weather_wind" not in df.columns:
        df["weather_wind"] = np.nan
    if "weather_humidity" not in df.columns:
        df["weather_humidity"] = np.nan

    # Try to fill from cache
    if "date" in df.columns and "home_team" in df.columns:
        for idx, row in df[["date", "home_team"]].iterrows():
            d = str(row["date"]) if row["date"] is not None else ""
            t = str(row["home_team"]) if row["home_team"] is not None else ""
            w = fetch_weather_for_game(d, t)
            if w:
                try:
                    if pd.isna(df.at[idx, "weather_temp"]):
                        df.at[idx, "weather_temp"] = float(w.get("temp"))
                    if pd.isna(df.at[idx, "weather_wind"]):
                        df.at[idx, "weather_wind"] = float(w.get("wind"))
                    if pd.isna(df.at[idx, "weather_humidity"]):
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


def _normalize_person_name(value) -> str:
    raw = str(value or "").strip().lower()
    raw = re.sub(r"[^a-z\s]", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def _safe_divide(a, b, default=0.0):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    out = np.where((b.notna()) & (b != 0), a / b, default)
    return pd.Series(out, index=a.index if hasattr(a, "index") else None)


def _american_to_implied_prob(series: pd.Series) -> pd.Series:
    odds = pd.to_numeric(series, errors="coerce")
    pos = odds > 0
    neg = odds < 0
    out = pd.Series(np.nan, index=odds.index, dtype=float)
    out.loc[pos] = 100.0 / (odds.loc[pos] + 100.0)
    out.loc[neg] = np.abs(odds.loc[neg]) / (np.abs(odds.loc[neg]) + 100.0)
    out = out.clip(lower=0.0, upper=1.0)
    return out


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


def _baseball_ip_to_outs(series, default=0.0):
    values = pd.to_numeric(series, errors="coerce").fillna(default)
    whole = np.floor(values).astype(int)
    frac_digit = np.round((values - whole) * 10).astype(int)
    frac_digit = np.clip(frac_digit, 0, 2)
    return pd.Series((whole * 3) + frac_digit, index=values.index)


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


def _normalize_team_abbr(team_value: str) -> str:
    t = str(team_value or "").strip().upper()
    team_map = {
        "OAK": "ATH",
        "SDP": "SD",
        "TBD": "TB",
        "KCR": "KC",
        "CHW": "CWS",
        "ANA": "LAA",
        "SFG": "SF",
        "WSN": "WSH",
    }
    return team_map.get(t, t)


def load_external_home_prob_prior() -> pd.DataFrame:
    """
    Carga probabilidades históricas pre-game externas (si existen) y
    devuelve columnas listas para merge por date/home_team/away_team.
    """
    cols = ["date", "home_team", "away_team", "external_home_prob", "external_away_prob"]
    if not EXTERNAL_PRIOR_ARCHIVE_DIR.exists():
        return pd.DataFrame(columns=cols)

    frames = []
    for csv_path in EXTERNAL_PRIOR_ARCHIVE_DIR.rglob("*.csv"):
        name = csv_path.name.lower()
        if "validation" in name:
            continue
        try:
            tmp = pd.read_csv(
                csv_path,
                usecols=["date", "hometeam", "awayteam", "hometeamodds", "awayteamodds"],
            )
        except Exception:
            continue
        if tmp.empty:
            continue
        tmp = tmp.rename(
            columns={
                "hometeam": "home_team",
                "awayteam": "away_team",
                "hometeamodds": "external_home_prob",
                "awayteamodds": "external_away_prob",
            }
        )
        tmp["date"] = tmp["date"].astype(str).str.slice(0, 10)
        tmp["home_team"] = tmp["home_team"].map(_normalize_team_abbr)
        tmp["away_team"] = tmp["away_team"].map(_normalize_team_abbr)
        for pcol in ["external_home_prob", "external_away_prob"]:
            tmp[pcol] = pd.to_numeric(tmp[pcol], errors="coerce")
        tmp = tmp.dropna(subset=["date", "home_team", "away_team", "external_home_prob", "external_away_prob"])
        if tmp.empty:
            continue
        tmp["external_home_prob"] = tmp["external_home_prob"].clip(0.0, 1.0)
        tmp["external_away_prob"] = tmp["external_away_prob"].clip(0.0, 1.0)
        frames.append(tmp[cols])

    if not frames:
        return pd.DataFrame(columns=cols)

    ext = pd.concat(frames, ignore_index=True)
    ext = (
        ext.groupby(["date", "home_team", "away_team"], as_index=False)[
            ["external_home_prob", "external_away_prob"]
        ]
        .mean()
        .sort_values(["date", "home_team", "away_team"])
    )
    return ext


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


def add_prev_season_blend_features(df: pd.DataFrame) -> pd.DataFrame:
    print("🧬 Generando blend conservador de temporada previa...")

    def _merge_with_guard(base_df: pd.DataFrame, right_df: pd.DataFrame, merge_name: str, **kwargs) -> pd.DataFrame:
        before_shape = base_df.shape
        print(f"   [SHAPE] {merge_name} before={before_shape}")
        out_df = base_df.merge(right_df, **kwargs)
        print(f"   [SHAPE] {merge_name} after={out_df.shape}")
        if out_df.shape[0] != before_shape[0]:
            raise RuntimeError(
                f"Row count changed in {merge_name}: {before_shape[0]} -> {out_df.shape[0]}"
            )
        return out_df

    def _asof_with_guard(left_df: pd.DataFrame, right_df: pd.DataFrame, merge_name: str) -> pd.DataFrame:
        before_shape = left_df.shape
        print(f"   [SHAPE] {merge_name} before={before_shape}")
        left_sorted = left_df.sort_values(["date_dt", "team", "season_target", "_order"]).reset_index(drop=True)
        right_sorted = right_df.sort_values(["date_dt", "team", "season_target"]).reset_index(drop=True)
        out_df = pd.merge_asof(
            left_sorted,
            right_sorted,
            on="date_dt",
            by=["team", "season_target"],
            direction="backward",
            allow_exact_matches=False,
        )
        out_df = out_df.sort_values("_order").reset_index(drop=True)
        print(f"   [SHAPE] {merge_name} after={out_df.shape}")
        if out_df.shape[0] != before_shape[0]:
            raise RuntimeError(
                f"Row count changed in {merge_name}: {before_shape[0]} -> {out_df.shape[0]}"
            )
        return out_df

    if DISABLE_PREV_SEASON_BLEND_BLOCK > 0:
        neutral_defaults = {
            "home_games_in_season_before": 0.0,
            "away_games_in_season_before": 0.0,
            "diff_games_in_season_before": 0.0,
            "home_prev_win_pct": 0.50,
            "away_prev_win_pct": 0.50,
            "diff_prev_win_pct": 0.0,
            "home_prev_run_diff_pg": 0.0,
            "away_prev_run_diff_pg": 0.0,
            "diff_prev_run_diff_pg": 0.0,
            "home_prev_runs_scored_pg": 4.50,
            "away_prev_runs_scored_pg": 4.50,
            "diff_prev_runs_scored_pg": 0.0,
            "home_prev_runs_allowed_pg": 4.50,
            "away_prev_runs_allowed_pg": 4.50,
            "diff_prev_runs_allowed_pg": 0.0,
            "home_season_blend_weight": 0.0,
            "away_season_blend_weight": 0.0,
            "diff_season_blend_weight": 0.0,
            "prev_season_data_available": 0,
        }
        for col, default in neutral_defaults.items():
            df[col] = default

        passthrough_pairs = [
            ("home_win_pct_L10_blend", "home_win_pct_L10"),
            ("away_win_pct_L10_blend", "away_win_pct_L10"),
            ("home_run_diff_L10_blend", "home_run_diff_L10"),
            ("away_run_diff_L10_blend", "away_run_diff_L10"),
            ("home_runs_scored_L5_blend", "home_runs_scored_L5"),
            ("away_runs_scored_L5_blend", "away_runs_scored_L5"),
            ("home_runs_allowed_L5_blend", "home_runs_allowed_L5"),
            ("away_runs_allowed_L5_blend", "away_runs_allowed_L5"),
        ]
        for out_col, src_col in passthrough_pairs:
            df[out_col] = pd.to_numeric(df.get(src_col, 0.0), errors="coerce").fillna(0.0)

        df["diff_win_pct_L10_blend"] = df["home_win_pct_L10_blend"] - df["away_win_pct_L10_blend"]
        df["diff_run_diff_L10_blend"] = df["home_run_diff_L10_blend"] - df["away_run_diff_L10_blend"]
        df["diff_runs_scored_L5_blend"] = df["home_runs_scored_L5_blend"] - df["away_runs_scored_L5_blend"]
        df["diff_runs_allowed_L5_blend"] = df["home_runs_allowed_L5_blend"] - df["away_runs_allowed_L5_blend"]
        return df

    required_cols = [
        "game_id",
        "season",
        "date_dt",
        "home_team",
        "away_team",
        "home_runs_total",
        "away_runs_total",
        "home_win_pct_L10",
        "away_win_pct_L10",
        "home_run_diff_L10",
        "away_run_diff_L10",
        "home_runs_scored_L5",
        "away_runs_scored_L5",
        "home_runs_allowed_L5",
        "away_runs_allowed_L5",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"   ⚠️ Priors omitidos: faltan columnas ({missing[:6]})")
        return df

    season_col = pd.to_numeric(df["season"], errors="coerce").fillna(0).astype(int)
    game_dt = pd.to_datetime(df["date_dt"], errors="coerce")
    home_runs_total = _safe_numeric(df["home_runs_total"], default=0.0)
    away_runs_total = _safe_numeric(df["away_runs_total"], default=0.0)

    home_tbl = pd.DataFrame(
        {
            "game_id": df["game_id"],
            "season": season_col,
            "date_dt": game_dt,
            "team": df["home_team"].astype(str),
            "win_flag": (home_runs_total > away_runs_total).astype(float),
            "run_diff": (home_runs_total - away_runs_total).astype(float),
            "runs_scored": home_runs_total.astype(float),
            "runs_allowed": away_runs_total.astype(float),
        }
    )

    away_tbl = pd.DataFrame(
        {
            "game_id": df["game_id"],
            "season": season_col,
            "date_dt": game_dt,
            "team": df["away_team"].astype(str),
            "win_flag": (away_runs_total > home_runs_total).astype(float),
            "run_diff": (away_runs_total - home_runs_total).astype(float),
            "runs_scored": away_runs_total.astype(float),
            "runs_allowed": home_runs_total.astype(float),
        }
    )

    team_games = pd.concat([home_tbl, away_tbl], ignore_index=True)
    team_games = team_games.dropna(subset=["date_dt"]).copy()
    team_games = team_games.sort_values(["team", "season", "date_dt", "game_id"]).reset_index(drop=True)
    team_games["games_in_season_before"] = team_games.groupby(["team", "season"]).cumcount().astype(float)

    dataset_mean_win_pct = float(pd.to_numeric(team_games["win_flag"], errors="coerce").mean())
    dataset_mean_run_diff = float(pd.to_numeric(team_games["run_diff"], errors="coerce").mean())
    dataset_mean_runs_scored = float(pd.to_numeric(team_games["runs_scored"], errors="coerce").mean())
    dataset_mean_runs_allowed = float(pd.to_numeric(team_games["runs_allowed"], errors="coerce").mean())

    if not np.isfinite(dataset_mean_win_pct):
        dataset_mean_win_pct = 0.50
    if not np.isfinite(dataset_mean_run_diff):
        dataset_mean_run_diff = 0.00
    if not np.isfinite(dataset_mean_runs_scored):
        dataset_mean_runs_scored = 4.50
    if not np.isfinite(dataset_mean_runs_allowed):
        dataset_mean_runs_allowed = 4.50

    home_games = team_games[["game_id", "team", "games_in_season_before"]].rename(
        columns={"team": "home_team", "games_in_season_before": "home_games_in_season_before"}
    )
    away_games = team_games[["game_id", "team", "games_in_season_before"]].rename(
        columns={"team": "away_team", "games_in_season_before": "away_games_in_season_before"}
    )

    df = _merge_with_guard(
        df,
        home_games,
        "prev_season_home_games",
        on=["game_id", "home_team"],
        how="left",
    )
    df = _merge_with_guard(
        df,
        away_games,
        "prev_season_away_games",
        on=["game_id", "away_team"],
        how="left",
    )

    prev_source = team_games[["team", "season", "date_dt", "win_flag", "run_diff", "runs_scored"]].copy()
    prev_source["season_target"] = prev_source["season"] + 1
    prev_source = prev_source.sort_values(["team", "season_target", "date_dt"]).reset_index(drop=True)
    prev_source["prev_games_count"] = prev_source.groupby(["team", "season_target"]).cumcount() + 1

    grp = prev_source.groupby(["team", "season_target"])
    prev_source["prev_win_pct"] = grp["win_flag"].cumsum() / prev_source["prev_games_count"]
    prev_source["prev_run_diff_pg"] = grp["run_diff"].cumsum() / prev_source["prev_games_count"]
    prev_source["prev_runs_scored_pg"] = grp["runs_scored"].cumsum() / prev_source["prev_games_count"]

    prev_source = prev_source[
        [
            "team",
            "season_target",
            "date_dt",
            "prev_games_count",
            "prev_win_pct",
            "prev_run_diff_pg",
            "prev_runs_scored_pg",
        ]
    ].copy()

    home_lookup = df[["game_id", "home_team", "date_dt", "season"]].copy()
    home_lookup["season_target"] = pd.to_numeric(home_lookup["season"], errors="coerce").fillna(0).astype(int)
    home_lookup["date_dt"] = pd.to_datetime(home_lookup["date_dt"], errors="coerce")
    home_lookup = home_lookup.rename(columns={"home_team": "team"})
    home_lookup["_order"] = np.arange(len(home_lookup))
    home_lookup = home_lookup.sort_values(["team", "season_target", "date_dt", "_order"]).reset_index(drop=True)
    home_lookup = _asof_with_guard(home_lookup, prev_source, "prev_season_home_asof")
    home_lookup = home_lookup.sort_values("_order").reset_index(drop=True)
    home_lookup["home_prev_data_available"] = home_lookup["prev_games_count"].notna().astype(int)
    home_lookup = home_lookup.rename(
        columns={
            "team": "home_team",
            "prev_win_pct": "home_prev_win_pct",
            "prev_run_diff_pg": "home_prev_run_diff_pg",
            "prev_runs_scored_pg": "home_prev_runs_scored_pg",
        }
    )
    home_lookup = home_lookup[
        [
            "game_id",
            "home_team",
            "home_prev_data_available",
            "home_prev_win_pct",
            "home_prev_run_diff_pg",
            "home_prev_runs_scored_pg",
        ]
    ].copy()

    away_lookup = df[["game_id", "away_team", "date_dt", "season"]].copy()
    away_lookup["season_target"] = pd.to_numeric(away_lookup["season"], errors="coerce").fillna(0).astype(int)
    away_lookup["date_dt"] = pd.to_datetime(away_lookup["date_dt"], errors="coerce")
    away_lookup = away_lookup.rename(columns={"away_team": "team"})
    away_lookup["_order"] = np.arange(len(away_lookup))
    away_lookup = away_lookup.sort_values(["team", "season_target", "date_dt", "_order"]).reset_index(drop=True)
    away_lookup = _asof_with_guard(away_lookup, prev_source, "prev_season_away_asof")
    away_lookup = away_lookup.sort_values("_order").reset_index(drop=True)
    away_lookup["away_prev_data_available"] = away_lookup["prev_games_count"].notna().astype(int)
    away_lookup = away_lookup.rename(
        columns={
            "team": "away_team",
            "prev_win_pct": "away_prev_win_pct",
            "prev_run_diff_pg": "away_prev_run_diff_pg",
            "prev_runs_scored_pg": "away_prev_runs_scored_pg",
        }
    )
    away_lookup = away_lookup[
        [
            "game_id",
            "away_team",
            "away_prev_data_available",
            "away_prev_win_pct",
            "away_prev_run_diff_pg",
            "away_prev_runs_scored_pg",
        ]
    ].copy()

    df = _merge_with_guard(
        df,
        home_lookup,
        "prev_season_home_stats",
        on=["game_id", "home_team"],
        how="left",
    )
    df = _merge_with_guard(
        df,
        away_lookup,
        "prev_season_away_stats",
        on=["game_id", "away_team"],
        how="left",
    )

    df["home_prev_data_available"] = pd.to_numeric(df.get("home_prev_data_available", 0), errors="coerce").fillna(0).astype(int)
    df["away_prev_data_available"] = pd.to_numeric(df.get("away_prev_data_available", 0), errors="coerce").fillna(0).astype(int)
    df["prev_season_data_available"] = ((df["home_prev_data_available"] > 0) & (df["away_prev_data_available"] > 0)).astype(int)

    df["home_prev_win_pct"] = pd.to_numeric(df.get("home_prev_win_pct", np.nan), errors="coerce").fillna(dataset_mean_win_pct)
    df["away_prev_win_pct"] = pd.to_numeric(df.get("away_prev_win_pct", np.nan), errors="coerce").fillna(dataset_mean_win_pct)
    df["home_prev_run_diff_pg"] = pd.to_numeric(df.get("home_prev_run_diff_pg", np.nan), errors="coerce").fillna(dataset_mean_run_diff)
    df["away_prev_run_diff_pg"] = pd.to_numeric(df.get("away_prev_run_diff_pg", np.nan), errors="coerce").fillna(dataset_mean_run_diff)
    df["home_prev_runs_scored_pg"] = pd.to_numeric(df.get("home_prev_runs_scored_pg", np.nan), errors="coerce").fillna(dataset_mean_runs_scored)
    df["away_prev_runs_scored_pg"] = pd.to_numeric(df.get("away_prev_runs_scored_pg", np.nan), errors="coerce").fillna(dataset_mean_runs_scored)

    df["home_prev_runs_allowed_pg"] = dataset_mean_runs_allowed
    df["away_prev_runs_allowed_pg"] = dataset_mean_runs_allowed

    df["home_games_in_season_before"] = pd.to_numeric(
        df.get("home_games_in_season_before", 0), errors="coerce"
    ).fillna(0.0)
    df["away_games_in_season_before"] = pd.to_numeric(
        df.get("away_games_in_season_before", 0), errors="coerce"
    ).fillna(0.0)
    df["diff_games_in_season_before"] = df["home_games_in_season_before"] - df["away_games_in_season_before"]

    # Blend conservador: el prior nunca pesa mas de 0.2.
    home_prior_weight = np.minimum(0.2, 5.0 / (df["home_games_in_season_before"] + 5.0))
    away_prior_weight = np.minimum(0.2, 5.0 / (df["away_games_in_season_before"] + 5.0))

    home_prior_weight = np.where(df["home_prev_data_available"] > 0, home_prior_weight, 0.0)
    away_prior_weight = np.where(df["away_prev_data_available"] > 0, away_prior_weight, 0.0)

    df["home_season_blend_weight"] = np.clip(home_prior_weight, 0.0, 0.2)
    df["away_season_blend_weight"] = np.clip(away_prior_weight, 0.0, 0.2)
    df["diff_season_blend_weight"] = df["home_season_blend_weight"] - df["away_season_blend_weight"]

    df["home_win_pct_L10_blend"] = (
        (1.0 - df["home_season_blend_weight"]) * pd.to_numeric(df["home_win_pct_L10"], errors="coerce").fillna(dataset_mean_win_pct)
        + df["home_season_blend_weight"] * df["home_prev_win_pct"]
    )
    df["away_win_pct_L10_blend"] = (
        (1.0 - df["away_season_blend_weight"]) * pd.to_numeric(df["away_win_pct_L10"], errors="coerce").fillna(dataset_mean_win_pct)
        + df["away_season_blend_weight"] * df["away_prev_win_pct"]
    )
    df["diff_win_pct_L10_blend"] = df["home_win_pct_L10_blend"] - df["away_win_pct_L10_blend"]

    df["home_run_diff_L10_blend"] = (
        (1.0 - df["home_season_blend_weight"]) * pd.to_numeric(df["home_run_diff_L10"], errors="coerce").fillna(dataset_mean_run_diff)
        + df["home_season_blend_weight"] * df["home_prev_run_diff_pg"]
    )
    df["away_run_diff_L10_blend"] = (
        (1.0 - df["away_season_blend_weight"]) * pd.to_numeric(df["away_run_diff_L10"], errors="coerce").fillna(dataset_mean_run_diff)
        + df["away_season_blend_weight"] * df["away_prev_run_diff_pg"]
    )
    df["diff_run_diff_L10_blend"] = df["home_run_diff_L10_blend"] - df["away_run_diff_L10_blend"]

    df["home_runs_scored_L5_blend"] = (
        (1.0 - df["home_season_blend_weight"]) * pd.to_numeric(df["home_runs_scored_L5"], errors="coerce").fillna(dataset_mean_runs_scored)
        + df["home_season_blend_weight"] * df["home_prev_runs_scored_pg"]
    )
    df["away_runs_scored_L5_blend"] = (
        (1.0 - df["away_season_blend_weight"]) * pd.to_numeric(df["away_runs_scored_L5"], errors="coerce").fillna(dataset_mean_runs_scored)
        + df["away_season_blend_weight"] * df["away_prev_runs_scored_pg"]
    )
    df["diff_runs_scored_L5_blend"] = df["home_runs_scored_L5_blend"] - df["away_runs_scored_L5_blend"]

    # Conservador: sin prior adicional de runs_allowed para evitar ruido.
    df["home_runs_allowed_L5_blend"] = pd.to_numeric(df["home_runs_allowed_L5"], errors="coerce").fillna(dataset_mean_runs_allowed)
    df["away_runs_allowed_L5_blend"] = pd.to_numeric(df["away_runs_allowed_L5"], errors="coerce").fillna(dataset_mean_runs_allowed)
    df["diff_runs_allowed_L5_blend"] = df["home_runs_allowed_L5_blend"] - df["away_runs_allowed_L5_blend"]

    df["diff_prev_win_pct"] = df["home_prev_win_pct"] - df["away_prev_win_pct"]
    df["diff_prev_run_diff_pg"] = df["home_prev_run_diff_pg"] - df["away_prev_run_diff_pg"]
    df["diff_prev_runs_scored_pg"] = df["home_prev_runs_scored_pg"] - df["away_prev_runs_scored_pg"]
    df["diff_prev_runs_allowed_pg"] = df["home_prev_runs_allowed_pg"] - df["away_prev_runs_allowed_pg"]

    df = df.drop(columns=[c for c in ["home_prev_data_available", "away_prev_data_available"] if c in df.columns])
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
    print("Generando proxies de bullpen MLB...")

    if "home_starting_pitcher_ip" in df.columns:
        home_starter_ip = _safe_numeric(df["home_starting_pitcher_ip"], 0.0)
    else:
        home_starter_ip = pd.Series(0.0, index=df.index)

    if "away_starting_pitcher_ip" in df.columns:
        away_starter_ip = _safe_numeric(df["away_starting_pitcher_ip"], 0.0)
    else:
        away_starter_ip = pd.Series(0.0, index=df.index)

    home_tbl = pd.DataFrame(
        {
            "date": df["date"],
            "game_id": df["game_id"],
            "team": df["home_team"],
            "side": "home",
            "bullpen_runs_allowed": np.maximum(_safe_numeric(df["away_runs_total"], 0) - _safe_numeric(df["away_runs_f5"], 0), 0),
            "starter_ip": home_starter_ip,
        }
    )

    away_tbl = pd.DataFrame(
        {
            "date": df["date"],
            "game_id": df["game_id"],
            "team": df["away_team"],
            "side": "away",
            "bullpen_runs_allowed": np.maximum(_safe_numeric(df["home_runs_total"], 0) - _safe_numeric(df["home_runs_f5"], 0), 0),
            "starter_ip": away_starter_ip,
        }
    )

    b = pd.concat([home_tbl, away_tbl], ignore_index=True).copy()
    b["date_dt"] = pd.to_datetime(b["date"])
    b = b.sort_values(["team", "date_dt", "game_id"]).reset_index(drop=True)
    b["appearance_flag"] = 1
    b["starter_outs"] = _baseball_ip_to_outs(b["starter_ip"], default=0.0)
    # Approximation: bullpen outs are remaining outs from a 9-inning game.
    b["bullpen_outs"] = np.clip(27.0 - b["starter_outs"], 0.0, 27.0)
    b["bullpen_ip"] = b["bullpen_outs"] / 3.0

    by_team = b.groupby("team")
    b["bullpen_runs_allowed_L5"] = _rolling_shifted_mean(by_team["bullpen_runs_allowed"], 5)
    b["bullpen_runs_allowed_L10"] = _rolling_shifted_mean(by_team["bullpen_runs_allowed"], 10)
    b["bullpen_load_L3"] = _rolling_shifted_sum(by_team["appearance_flag"], 3)
    b["bullpen_load_L5"] = _rolling_shifted_sum(by_team["appearance_flag"], 5)
    b["bullpen_outs_yesterday"] = by_team["bullpen_outs"].transform(lambda x: x.shift(1))
    b["bullpen_outs_3d"] = _rolling_shifted_sum(by_team["bullpen_outs"], 3)
    b["bullpen_ip_yesterday"] = b["bullpen_outs_yesterday"] / 3.0
    b["bullpen_ip_3d"] = b["bullpen_outs_3d"] / 3.0

    metric_cols = [
        "bullpen_runs_allowed_L5",
        "bullpen_runs_allowed_L10",
        "bullpen_load_L3",
        "bullpen_load_L5",
        "bullpen_outs_yesterday",
        "bullpen_outs_3d",
        "bullpen_ip_yesterday",
        "bullpen_ip_3d",
    ]
    out = b[[
        "game_id",
        "team",
        "side",
        *metric_cols,
    ]].copy()

    for c in metric_cols:
        out[c] = out[c].fillna(0.0)

    home_out = out[out["side"] == "home"][["game_id", *metric_cols]].copy()
    home_out = home_out.drop_duplicates(subset=["game_id"], keep="last")
    home_out = home_out.rename(columns={c: f"home_{c}" for c in metric_cols})

    away_out = out[out["side"] == "away"][["game_id", *metric_cols]].copy()
    away_out = away_out.drop_duplicates(subset=["game_id"], keep="last")
    away_out = away_out.rename(columns={c: f"away_{c}" for c in metric_cols})

    wide = pd.merge(home_out, away_out, on="game_id", how="outer")
    for c in wide.columns:
        if c != "game_id":
            wide[c] = pd.to_numeric(wide[c], errors="coerce").fillna(0.0)

    return wide


def add_series_context_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df[["game_id", "date_dt", "season", "home_team", "away_team", "home_runs_total", "away_runs_total"]].copy()
    home_team_str = work["home_team"].astype(str)
    away_team_str = work["away_team"].astype(str)
    team_low = home_team_str.where(home_team_str <= away_team_str, away_team_str)
    team_high = home_team_str.where(home_team_str > away_team_str, away_team_str)
    work["series_pair_key"] = work["season"].astype(str) + "|" + team_low + "|" + team_high
    work = work.sort_values(["series_pair_key", "date_dt", "game_id"]).reset_index(drop=True)
    date_gap = work.groupby("series_pair_key")["date_dt"].diff().dt.days.fillna(99)
    new_series = (date_gap > 3).astype(int)
    work["series_id"] = work["series_pair_key"] + "|" + new_series.groupby(work["series_pair_key"]).cumsum().astype(str)
    work["home_margin"] = _safe_numeric(work["home_runs_total"], 0.0) - _safe_numeric(work["away_runs_total"], 0.0)
    work["home_win_flag"] = (work["home_margin"] > 0).astype(int)
    work["series_game_number"] = (work.groupby("series_id").cumcount() + 1).astype(float)
    work["series_run_diff_so_far"] = (work.groupby("series_id")["home_margin"].cumsum() - work["home_margin"]).astype(float)
    work["did_home_win_previous_game_in_series"] = work.groupby("series_id")["home_win_flag"].shift(1).fillna(0).astype(float)
    return df.merge(
        work[["game_id", "series_game_number", "series_run_diff_so_far", "did_home_win_previous_game_in_series"]],
        on="game_id",
        how="left",
    )


def add_matchup_history_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df[[
        "game_id", "date_dt", "season", "home_team", "away_team",
        "home_runs_total", "away_runs_total",
    ]].copy()
    work["home_team"] = work["home_team"].astype(str)
    work["away_team"] = work["away_team"].astype(str)
    team_low = work["home_team"].where(work["home_team"] <= work["away_team"], work["away_team"])
    team_high = work["home_team"].where(work["home_team"] > work["away_team"], work["away_team"])
    work["matchup_pair_key"] = team_low + "|" + team_high
    work = work.sort_values(["matchup_pair_key", "date_dt", "game_id"]).reset_index(drop=True)

    hist_win_pct = []
    hist_run_diff = []
    season_win_pct = []

    overall_state: dict[str, dict[str, float]] = {}
    season_state: dict[tuple[int, str], dict[str, float]] = {}

    for _, row in work.iterrows():
        pair_key = str(row["matchup_pair_key"])
        season_key = (int(row["season"]), pair_key)
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        home_margin = float(_safe_numeric(pd.Series([row["home_runs_total"]]), 0.0).iloc[0]) - float(
            _safe_numeric(pd.Series([row["away_runs_total"]]), 0.0).iloc[0]
        )

        pair_overall = overall_state.get(pair_key, {"games": 0.0, "home_team_wins": {}, "home_team_run_diff": {}})
        pair_season = season_state.get(season_key, {"games": 0.0, "home_team_wins": {}})

        overall_games = float(pair_overall["games"])
        home_prev_wins = float(pair_overall["home_team_wins"].get(home_team, 0.0))
        home_prev_run_diff = float(pair_overall["home_team_run_diff"].get(home_team, 0.0))
        season_games = float(pair_season["games"])
        season_prev_wins = float(pair_season["home_team_wins"].get(home_team, 0.0))

        hist_win_pct.append(home_prev_wins / overall_games if overall_games > 0 else 0.5)
        hist_run_diff.append(home_prev_run_diff / overall_games if overall_games > 0 else 0.0)
        season_win_pct.append(season_prev_wins / season_games if season_games > 0 else 0.5)

        home_win = 1.0 if home_margin > 0 else 0.0
        away_win = 1.0 - home_win

        pair_overall["games"] = overall_games + 1.0
        pair_overall["home_team_wins"][home_team] = home_prev_wins + home_win
        pair_overall["home_team_wins"][away_team] = float(pair_overall["home_team_wins"].get(away_team, 0.0)) + away_win
        pair_overall["home_team_run_diff"][home_team] = home_prev_run_diff + home_margin
        pair_overall["home_team_run_diff"][away_team] = float(pair_overall["home_team_run_diff"].get(away_team, 0.0)) - home_margin
        overall_state[pair_key] = pair_overall

        pair_season["games"] = season_games + 1.0
        pair_season["home_team_wins"][home_team] = season_prev_wins + home_win
        pair_season["home_team_wins"][away_team] = float(pair_season["home_team_wins"].get(away_team, 0.0)) + away_win
        season_state[season_key] = pair_season

    work["home_vs_away_matchup_win_pct_pre"] = pd.Series(hist_win_pct, index=work.index, dtype=float)
    work["home_vs_away_matchup_run_diff_pre"] = pd.Series(hist_run_diff, index=work.index, dtype=float)
    work["home_vs_away_in_season_record_pre"] = pd.Series(season_win_pct, index=work.index, dtype=float)

    return df.merge(
        work[[
            "game_id",
            "home_vs_away_matchup_win_pct_pre",
            "home_vs_away_matchup_run_diff_pre",
            "home_vs_away_in_season_record_pre",
        ]],
        on="game_id",
        how="left",
    )


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
    df = add_series_context_features(df)
    df = add_matchup_history_features(df)

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

    # Bullpen proxy already comes pivoted as home_/away_ columns.
    df = pd.merge(df, bullpen_features, on="game_id", how="left")

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
    df = add_prev_season_blend_features(df)

    # Optional external prior from archived third-party pregame probabilities.
    if DISABLE_EXTERNAL_PRIOR_BLOCK > 0:
        df["external_home_prob"] = 0.5
        df["external_away_prob"] = 0.5
        df["external_prior_available"] = 0
        df["external_prob_gap"] = 0.0
        df["external_home_edge"] = 0.0
        df["external_confidence"] = 0.0
        df["external_market_agreement"] = 0
    else:
        if USE_EXTERNAL_PRIOR > 0:
            ext_prior = load_external_home_prob_prior()
        else:
            ext_prior = pd.DataFrame(columns=["date", "home_team", "away_team", "external_home_prob", "external_away_prob"])
        if not ext_prior.empty and {"date", "home_team", "away_team"}.issubset(df.columns):
            before_merge_rows = len(df)
            print(f"   external prior merge rows before={before_merge_rows}")
            df["date"] = df["date"].astype(str).str.slice(0, 10)
            df["home_team"] = df["home_team"].map(_normalize_team_abbr)
            df["away_team"] = df["away_team"].map(_normalize_team_abbr)
            df = df.merge(ext_prior, on=["date", "home_team", "away_team"], how="left")
            print(f"   external prior merge rows after={len(df)}")
        else:
            df["external_home_prob"] = np.nan
            df["external_away_prob"] = np.nan

        df["external_prior_available"] = (
            pd.to_numeric(df["external_home_prob"], errors="coerce").notna()
            & pd.to_numeric(df["external_away_prob"], errors="coerce").notna()
        ).astype(int)
        df["external_home_prob"] = pd.to_numeric(df["external_home_prob"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
        df["external_away_prob"] = pd.to_numeric(df["external_away_prob"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
        df["external_prob_gap"] = df["external_home_prob"] - df["external_away_prob"]
        df["external_home_edge"] = df["external_home_prob"] - 0.5
        df["external_confidence"] = np.abs(df["external_home_edge"])
        home_fav_flag_ext = pd.to_numeric(df["home_is_favorite"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        home_pick_ext = (df["external_home_prob"] >= 0.5).astype(int)
        df["external_market_agreement"] = (home_pick_ext == (home_fav_flag_ext >= 0.5).astype(int)).astype(int)

    # Merge advanced cached features if available
    try:
        if UMPIRE_STATS_FILE.exists():
            ump = pd.read_csv(UMPIRE_STATS_FILE)
            # expected columns: umpire, zone_rate, games_sample, consistency_rate, favor_abs_mean, weighted_score
            if "umpire" in ump.columns:
                ump = ump.rename(columns={"umpire": "umpire_name"})
                ump["umpire_name_norm"] = ump["umpire_name"].map(_normalize_person_name)
                df["home_plate_umpire_norm"] = df.get("home_plate_umpire", pd.Series(index=df.index, dtype=object)).map(
                    _normalize_person_name
                )
                before_merge_rows = len(df)
                print(f"   [SHAPE] umpire merge before={df.shape}")
                df = df.merge(ump, left_on="home_plate_umpire_norm", right_on="umpire_name_norm", how="left")
                print(f"   [SHAPE] umpire merge after={df.shape}")
                if len(df) != before_merge_rows:
                    raise RuntimeError(f"Row count changed in umpire merge: {before_merge_rows} -> {len(df)}")

                rename_map = {
                    "zone_rate": "home_umpire_zone_rate",
                    "games_sample": "home_umpire_games_sample",
                }
                if DISABLE_UMPIRE_EXPANDED_BLOCK <= 0:
                    rename_map.update(
                        {
                            "consistency_rate": "home_umpire_consistency_rate",
                            "favor_abs_mean": "home_umpire_favor_abs_mean",
                            "weighted_score": "home_umpire_weighted_score",
                        }
                    )
                df = df.rename(columns=rename_map)
                df = df.drop(columns=[c for c in ["umpire_name", "umpire_name_norm", "home_plate_umpire_norm"] if c in df.columns])
    except Exception:
        pass

    if "home_umpire_zone_rate" not in df.columns:
        df["home_umpire_zone_rate"] = 0.5
    if "home_umpire_games_sample" not in df.columns:
        df["home_umpire_games_sample"] = 0.0
    if "home_umpire_consistency_rate" not in df.columns:
        df["home_umpire_consistency_rate"] = 0.5
    if "home_umpire_favor_abs_mean" not in df.columns:
        df["home_umpire_favor_abs_mean"] = 0.0
    if "home_umpire_weighted_score" not in df.columns:
        df["home_umpire_weighted_score"] = 0.0

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

    market_micro_cols = [
        "open_line",
        "current_line",
        "line_movement",
        "open_total",
        "current_total",
        "total_movement",
        "current_home_moneyline",
        "current_away_moneyline",
        "current_total_line",
        "bookmakers_count",
        "snapshot_count",
    ]
    if DISABLE_EXTERNAL_PRIOR_BLOCK > 0:
        for col in market_micro_cols + [
            "home_moneyline_odds",
            "away_moneyline_odds",
            "home_implied_prob_ml",
            "away_implied_prob_ml",
            "market_overround_ml",
            "home_prob_no_vig_ml",
            "away_prob_no_vig_ml",
            "market_home_edge_no_vig_ml",
            "market_favorite_strength_ml",
            "market_favorite_alignment_ml",
        ]:
            if col not in df.columns:
                df[col] = 0.0
        df["market_moneyline_gap"] = 0.0
        df["market_total_delta"] = 0.0
        df["market_line_velocity"] = 0.0
        df["market_micro_missing"] = 1
        df["home_implied_prob_ml"] = 0.5
        df["away_implied_prob_ml"] = 0.5
        df["market_overround_ml"] = 1.0
        df["home_prob_no_vig_ml"] = 0.5
        df["away_prob_no_vig_ml"] = 0.5
        df["market_home_edge_no_vig_ml"] = 0.0
        df["market_favorite_strength_ml"] = 0.0
        df["market_favorite_alignment_ml"] = 0
    else:
        for col in market_micro_cols:
            if col not in df.columns:
                df[col] = np.nan

        df["market_moneyline_gap"] = pd.to_numeric(df["current_home_moneyline"], errors="coerce") - pd.to_numeric(
            df["current_away_moneyline"], errors="coerce"
        )
        df["market_total_delta"] = pd.to_numeric(df["current_total_line"], errors="coerce") - pd.to_numeric(
            df["odds_over_under"], errors="coerce"
        )
        velocity_denom = pd.to_numeric(df["snapshot_count"], errors="coerce").replace(0, np.nan)
        df["market_line_velocity"] = _safe_divide(df["line_movement"], velocity_denom, default=0.0)
        df["market_micro_missing"] = df[market_micro_cols].isna().any(axis=1).astype(int)

        # Moneyline strength / no-vig features (inspired by sportsbook-style modeling).
        if "home_moneyline_odds" not in df.columns:
            df["home_moneyline_odds"] = np.nan
        if "away_moneyline_odds" not in df.columns:
            df["away_moneyline_odds"] = np.nan

        df["home_implied_prob_ml"] = _american_to_implied_prob(df["home_moneyline_odds"])
        df["away_implied_prob_ml"] = _american_to_implied_prob(df["away_moneyline_odds"])
        df["market_overround_ml"] = (
            pd.to_numeric(df["home_implied_prob_ml"], errors="coerce")
            + pd.to_numeric(df["away_implied_prob_ml"], errors="coerce")
        )
        overround_denom = pd.to_numeric(df["market_overround_ml"], errors="coerce").replace(0, np.nan)
        df["home_prob_no_vig_ml"] = _safe_divide(df["home_implied_prob_ml"], overround_denom, default=0.5).clip(0.0, 1.0)
        df["away_prob_no_vig_ml"] = _safe_divide(df["away_implied_prob_ml"], overround_denom, default=0.5).clip(0.0, 1.0)
        df["market_home_edge_no_vig_ml"] = df["home_prob_no_vig_ml"] - 0.5
        df["market_favorite_strength_ml"] = np.abs(df["market_home_edge_no_vig_ml"])
        fav_flag = pd.to_numeric(df["home_is_favorite"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        market_home_pick = (df["home_prob_no_vig_ml"] >= 0.5).astype(int)
        df["market_favorite_alignment_ml"] = (market_home_pick == (fav_flag >= 0.5).astype(int)).astype(int)

    # =========================
    # Core differentials
    # =========================
    df["diff_elo"] = df["home_elo_pre"] - df["away_elo_pre"]
    df["elo_sum"] = df["home_elo_pre"] + df["away_elo_pre"]
    df["elo_gap_abs"] = df["diff_elo"].abs()
    home_fav_flag = pd.to_numeric(df.get("home_is_favorite", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    df["favorite_elo_gap_signed"] = np.where(home_fav_flag >= 0.5, df["diff_elo"], -df["diff_elo"])

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
    df["elo_form_interaction"] = df["diff_elo"] * df["diff_win_pct_L10"]
    df["elo_rest_interaction"] = df["diff_elo"] * df["diff_rest_days"]

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
    df["elo_pitcher_alignment"] = df["diff_elo"] * df["diff_pitcher_recent_quality_score"]

    # =========================
    # New: bullpen differentials
    # =========================
    df["diff_bullpen_runs_allowed_L5"] = df["home_bullpen_runs_allowed_L5"] - df["away_bullpen_runs_allowed_L5"]
    df["diff_bullpen_runs_allowed_L10"] = df["home_bullpen_runs_allowed_L10"] - df["away_bullpen_runs_allowed_L10"]
    df["diff_bullpen_load_L3"] = df["home_bullpen_load_L3"] - df["away_bullpen_load_L3"]
    df["diff_bullpen_load_L5"] = df["home_bullpen_load_L5"] - df["away_bullpen_load_L5"]
    for col in [
        "home_bullpen_outs_3d",
        "away_bullpen_outs_3d",
        "home_bullpen_outs_5d",
        "away_bullpen_outs_5d",
        "home_bullpen_ip_3d",
        "away_bullpen_ip_3d",
        "home_bullpen_ip_5d",
        "away_bullpen_ip_5d",
        "home_bullpen_games_3d",
        "away_bullpen_games_3d",
        "home_bullpen_games_5d",
        "away_bullpen_games_5d",
        "home_bullpen_heavy_use_3d",
        "away_bullpen_heavy_use_3d",
        "home_bullpen_heavy_use_5d",
        "away_bullpen_heavy_use_5d",
        "home_bullpen_extreme_use_3d",
        "away_bullpen_extreme_use_3d",
        "home_bullpen_ip_yesterday",
        "away_bullpen_ip_yesterday",
        "home_bullpen_outs_yesterday",
        "away_bullpen_outs_yesterday",
    ]:
        if col not in df.columns:
            df[col] = 0.0
    df["diff_bullpen_outs_3d"] = df["home_bullpen_outs_3d"] - df["away_bullpen_outs_3d"]
    df["diff_bullpen_outs_5d"] = df["home_bullpen_outs_5d"] - df["away_bullpen_outs_5d"]
    df["diff_bullpen_ip_3d"] = df["home_bullpen_ip_3d"] - df["away_bullpen_ip_3d"]
    df["diff_bullpen_ip_5d"] = df["home_bullpen_ip_5d"] - df["away_bullpen_ip_5d"]
    df["diff_bullpen_games_3d"] = df["home_bullpen_games_3d"] - df["away_bullpen_games_3d"]
    df["diff_bullpen_games_5d"] = df["home_bullpen_games_5d"] - df["away_bullpen_games_5d"]
    df["diff_bullpen_heavy_use_3d"] = df["home_bullpen_heavy_use_3d"] - df["away_bullpen_heavy_use_3d"]
    df["diff_bullpen_heavy_use_5d"] = df["home_bullpen_heavy_use_5d"] - df["away_bullpen_heavy_use_5d"]
    df["diff_bullpen_extreme_use_3d"] = df["home_bullpen_extreme_use_3d"] - df["away_bullpen_extreme_use_3d"]
    df["diff_bullpen_ip_yesterday"] = df["home_bullpen_ip_yesterday"] - df["away_bullpen_ip_yesterday"]
    df["diff_bullpen_outs_yesterday"] = df["home_bullpen_outs_yesterday"] - df["away_bullpen_outs_yesterday"]

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

    # Park features are created late in the pipeline, so derive context interactions here.
    df["park_factor_delta"] = df["avg_park_factor"] - 1.0
    df["umpire_zone_delta"] = df["home_umpire_zone_rate"] - 0.5
    df["umpire_sample_log"] = np.log1p(df["home_umpire_games_sample"].clip(lower=0.0))

    df["park_offense_pressure"] = df["park_factor_delta"] * df["diff_offense_vs_pitcher"]
    df["park_bullpen_pressure"] = df["park_factor_delta"] * df["diff_bullpen_runs_allowed_L5"]
    df["park_form_pressure"] = df["park_factor_delta"] * df["diff_form_power"]

    if DISABLE_UMPIRE_EXPANDED_BLOCK <= 0:
        df["umpire_consistency_delta"] = df["home_umpire_consistency_rate"] - 0.5
        df["umpire_favor_abs"] = df["home_umpire_favor_abs_mean"].abs()
        df["umpire_weighted_score_log"] = np.log1p(df["home_umpire_weighted_score"].clip(lower=0.0))
        df["umpire_pitcher_command_edge"] = df["umpire_zone_delta"] * df["diff_pitcher_k_bb_L5"]
        df["umpire_pitcher_whip_edge"] = df["umpire_zone_delta"] * (-df["diff_pitcher_whip_L5"])
        df["umpire_recent_quality_edge"] = df["umpire_consistency_delta"] * df["diff_pitcher_recent_quality_score"]
        df["umpire_volatility_risk"] = df["umpire_favor_abs"] * df["diff_runs_scored_std_L10"].abs()
        df["umpire_confidence_signal"] = df["umpire_sample_log"] * df["umpire_consistency_delta"]
    else:
        # Modo quirurgico: solo dos variables de umpire activas.
        df["umpire_consistency_delta"] = 0.0
        df["umpire_favor_abs"] = 0.0
        df["umpire_weighted_score_log"] = 0.0
        df["umpire_pitcher_command_edge"] = 0.0
        df["umpire_pitcher_whip_edge"] = 0.0
        df["umpire_recent_quality_edge"] = 0.0
        df["umpire_volatility_risk"] = 0.0
        df["umpire_confidence_signal"] = 0.0

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
        "elo_sum",
        "elo_gap_abs",
        "favorite_elo_gap_signed",
        "elo_form_interaction",
        "elo_rest_interaction",
        "elo_pitcher_alignment",
        "series_game_number",
        "series_run_diff_so_far",
        "did_home_win_previous_game_in_series",

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

        "home_games_in_season_before",
        "away_games_in_season_before",
        "diff_games_in_season_before",

        "home_prev_win_pct",
        "away_prev_win_pct",
        "diff_prev_win_pct",

        "home_prev_run_diff_pg",
        "away_prev_run_diff_pg",
        "diff_prev_run_diff_pg",

        "home_prev_runs_scored_pg",
        "away_prev_runs_scored_pg",
        "diff_prev_runs_scored_pg",

        "home_prev_runs_allowed_pg",
        "away_prev_runs_allowed_pg",
        "diff_prev_runs_allowed_pg",

        "home_season_blend_weight",
        "away_season_blend_weight",
        "diff_season_blend_weight",
        "prev_season_data_available",

        "home_win_pct_L10_blend",
        "away_win_pct_L10_blend",
        "diff_win_pct_L10_blend",

        "home_run_diff_L10_blend",
        "away_run_diff_L10_blend",
        "diff_run_diff_L10_blend",

        "home_runs_scored_L5_blend",
        "away_runs_scored_L5_blend",
        "diff_runs_scored_L5_blend",

        "home_runs_allowed_L5_blend",
        "away_runs_allowed_L5_blend",
        "diff_runs_allowed_L5_blend",

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

        "home_bullpen_outs_3d",
        "away_bullpen_outs_3d",
        "diff_bullpen_outs_3d",

        "home_bullpen_outs_5d",
        "away_bullpen_outs_5d",
        "diff_bullpen_outs_5d",

        "home_bullpen_ip_3d",
        "away_bullpen_ip_3d",
        "diff_bullpen_ip_3d",

        "home_bullpen_ip_5d",
        "away_bullpen_ip_5d",
        "diff_bullpen_ip_5d",

        "home_bullpen_games_3d",
        "away_bullpen_games_3d",
        "diff_bullpen_games_3d",

        "home_bullpen_games_5d",
        "away_bullpen_games_5d",
        "diff_bullpen_games_5d",

        "home_bullpen_heavy_use_3d",
        "away_bullpen_heavy_use_3d",
        "diff_bullpen_heavy_use_3d",

        "home_bullpen_heavy_use_5d",
        "away_bullpen_heavy_use_5d",
        "diff_bullpen_heavy_use_5d",

        "home_bullpen_extreme_use_3d",
        "away_bullpen_extreme_use_3d",
        "diff_bullpen_extreme_use_3d",

        "home_bullpen_ip_yesterday",
        "away_bullpen_ip_yesterday",
        "diff_bullpen_ip_yesterday",

        "home_bullpen_outs_yesterday",
        "away_bullpen_outs_yesterday",
        "diff_bullpen_outs_yesterday",

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
        "home_umpire_zone_rate",
        "home_umpire_games_sample",
        "home_umpire_consistency_rate",
        "home_umpire_favor_abs_mean",
        "home_umpire_weighted_score",
        "park_factor_delta",
        "umpire_zone_delta",
        "umpire_consistency_delta",
        "umpire_sample_log",
        "umpire_favor_abs",
        "umpire_weighted_score_log",
        "park_offense_pressure",
        "park_bullpen_pressure",
        "park_form_pressure",
        "umpire_pitcher_command_edge",
        "umpire_pitcher_whip_edge",
        "umpire_recent_quality_edge",
        "umpire_volatility_risk",
        "umpire_confidence_signal",

        "home_is_favorite",
        "odds_over_under",
        "open_line",
        "current_line",
        "line_movement",
        "open_total",
        "current_total",
        "total_movement",
        "current_home_moneyline",
        "current_away_moneyline",
        "current_total_line",
        "bookmakers_count",
        "snapshot_count",
        "market_moneyline_gap",
        "market_total_delta",
        "market_line_velocity",
        "market_missing",
        "market_micro_missing",
        "home_implied_prob_ml",
        "away_implied_prob_ml",
        "market_overround_ml",
        "home_prob_no_vig_ml",
        "away_prob_no_vig_ml",
        "market_home_edge_no_vig_ml",
        "market_favorite_strength_ml",
        "market_favorite_alignment_ml",

        "external_prior_available",
        "external_home_prob",
        "external_away_prob",
        "external_prob_gap",
        "external_home_edge",
        "external_confidence",
        "external_market_agreement",

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
