from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "player_a_rank",
    "player_b_rank",
    "diff_rank",
    "player_a_odds",
    "player_b_odds",
    "player_a_implied_prob",
    "player_b_implied_prob",
    "diff_implied_prob",
    "player_a_matches_before",
    "player_b_matches_before",
    "player_a_win_rate_l5",
    "player_b_win_rate_l5",
    "diff_win_rate_l5",
    "player_a_win_rate_all",
    "player_b_win_rate_all",
    "diff_win_rate_all",
    "player_a_streak",
    "player_b_streak",
    "diff_streak",
]

OUTPUT_COLUMNS = [
    "match_id",
    "date",
    "time",
    "tour",
    "tournament",
    "surface",
    "round",
    "player_a",
    "player_b",
    *FEATURE_COLUMNS,
    "TARGET_player_a_win",
]


@dataclass
class PlayerState:
    matches: int = 0
    wins: int = 0
    last5: deque = field(default_factory=lambda: deque(maxlen=5))
    streak: int = 0

    def snapshot(self) -> dict[str, float]:
        win_rate_all = self.wins / self.matches if self.matches else np.nan
        win_rate_l5 = (sum(self.last5) / len(self.last5)) if self.last5 else np.nan
        return {
            "matches": float(self.matches),
            "wins": float(self.wins),
            "win_rate_all": float(win_rate_all) if pd.notna(win_rate_all) else np.nan,
            "win_rate_l5": float(win_rate_l5) if pd.notna(win_rate_l5) else np.nan,
            "streak": float(self.streak),
        }

    def update(self, won: bool) -> None:
        self.matches += 1
        if won:
            self.wins += 1
            self.streak = self.streak + 1 if self.streak >= 0 else 1
            self.last5.append(1)
        else:
            self.streak = self.streak - 1 if self.streak <= 0 else -1
            self.last5.append(0)


def implied_prob_from_decimal(value) -> float:
    try:
        dec = float(value)
    except Exception:
        return np.nan
    if dec <= 1.0:
        return np.nan
    return 1.0 / dec


def _safe_num(series_or_value):
    return pd.to_numeric(series_or_value, errors="coerce")


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    out["match_id"] = out.get("match_id", "").astype(str)
    out["player_a"] = out.get("player_a", "").astype(str).str.strip()
    out["player_b"] = out.get("player_b", "").astype(str).str.strip()
    out["date"] = out.get("date", "").astype(str)
    out["time"] = out.get("time", "").astype(str)
    out["player_a_rank"] = _safe_num(out.get("player_a_rank"))
    out["player_b_rank"] = _safe_num(out.get("player_b_rank"))
    out["player_a_odds"] = _safe_num(out.get("player_a_odds"))
    out["player_b_odds"] = _safe_num(out.get("player_b_odds"))
    out["_sort_ts"] = pd.to_datetime(out["date"] + " " + out["time"], errors="coerce")
    out = out.sort_values(["_sort_ts", "match_id"], kind="stable").reset_index(drop=True)
    return out


def _feature_row(row: pd.Series, state_a: PlayerState, state_b: PlayerState) -> dict:
    snap_a = state_a.snapshot()
    snap_b = state_b.snapshot()
    rank_a = _safe_num(row.get("player_a_rank"))
    rank_b = _safe_num(row.get("player_b_rank"))
    odds_a = _safe_num(row.get("player_a_odds"))
    odds_b = _safe_num(row.get("player_b_odds"))
    prob_a = implied_prob_from_decimal(odds_a)
    prob_b = implied_prob_from_decimal(odds_b)
    return {
        "match_id": str(row.get("match_id") or ""),
        "date": str(row.get("date") or ""),
        "time": str(row.get("time") or ""),
        "tour": str(row.get("tour") or ""),
        "tournament": str(row.get("tournament") or ""),
        "surface": str(row.get("surface") or ""),
        "round": str(row.get("round") or ""),
        "player_a": str(row.get("player_a") or ""),
        "player_b": str(row.get("player_b") or ""),
        "player_a_rank": rank_a,
        "player_b_rank": rank_b,
        "diff_rank": rank_b - rank_a if pd.notna(rank_a) and pd.notna(rank_b) else np.nan,
        "player_a_odds": odds_a,
        "player_b_odds": odds_b,
        "player_a_implied_prob": prob_a,
        "player_b_implied_prob": prob_b,
        "diff_implied_prob": prob_a - prob_b if pd.notna(prob_a) and pd.notna(prob_b) else np.nan,
        "player_a_matches_before": snap_a["matches"],
        "player_b_matches_before": snap_b["matches"],
        "player_a_win_rate_l5": snap_a["win_rate_l5"],
        "player_b_win_rate_l5": snap_b["win_rate_l5"],
        "diff_win_rate_l5": snap_a["win_rate_l5"] - snap_b["win_rate_l5"] if pd.notna(snap_a["win_rate_l5"]) and pd.notna(snap_b["win_rate_l5"]) else np.nan,
        "player_a_win_rate_all": snap_a["win_rate_all"],
        "player_b_win_rate_all": snap_b["win_rate_all"],
        "diff_win_rate_all": snap_a["win_rate_all"] - snap_b["win_rate_all"] if pd.notna(snap_a["win_rate_all"]) and pd.notna(snap_b["win_rate_all"]) else np.nan,
        "player_a_streak": snap_a["streak"],
        "player_b_streak": snap_b["streak"],
        "diff_streak": snap_a["streak"] - snap_b["streak"],
    }


def build_history_features(raw_history_df: pd.DataFrame) -> pd.DataFrame:
    prepared = _prepare_frame(raw_history_df)
    if prepared.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    states: dict[str, PlayerState] = {}
    rows: list[dict] = []

    for _, row in prepared.iterrows():
        player_a = str(row.get("player_a") or "").strip()
        player_b = str(row.get("player_b") or "").strip()
        winner = str(row.get("winner") or "").strip()
        if not player_a or not player_b or winner not in {player_a, player_b}:
            continue

        state_a = states.setdefault(player_a, PlayerState())
        state_b = states.setdefault(player_b, PlayerState())
        features = _feature_row(row, state_a, state_b)
        features["TARGET_player_a_win"] = 1 if winner == player_a else 0
        rows.append(features)

        state_a.update(winner == player_a)
        state_b.update(winner == player_b)

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def build_upcoming_features(raw_history_df: pd.DataFrame, upcoming_df: pd.DataFrame) -> pd.DataFrame:
    history_prepared = _prepare_frame(raw_history_df)
    upcoming_prepared = _prepare_frame(upcoming_df)
    if upcoming_prepared.empty:
        return pd.DataFrame(columns=[c for c in OUTPUT_COLUMNS if c != "TARGET_player_a_win"])

    states: dict[str, PlayerState] = {}
    for _, row in history_prepared.iterrows():
        player_a = str(row.get("player_a") or "").strip()
        player_b = str(row.get("player_b") or "").strip()
        winner = str(row.get("winner") or "").strip()
        if not player_a or not player_b or winner not in {player_a, player_b}:
            continue
        states.setdefault(player_a, PlayerState()).update(winner == player_a)
        states.setdefault(player_b, PlayerState()).update(winner == player_b)

    rows: list[dict] = []
    for _, row in upcoming_prepared.iterrows():
        player_a = str(row.get("player_a") or "").strip()
        player_b = str(row.get("player_b") or "").strip()
        state_a = states.setdefault(player_a, PlayerState())
        state_b = states.setdefault(player_b, PlayerState())
        rows.append(_feature_row(row, state_a, state_b))

    cols = [c for c in OUTPUT_COLUMNS if c != "TARGET_player_a_win"]
    return pd.DataFrame(rows, columns=cols)
