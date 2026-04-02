from pathlib import Path
import requests
import pandas as pd
import os
import io
import sys

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
CACHE_DIR = BASE_DIR / "data" / "mlb" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LINEUP_CACHE = CACHE_DIR / "lineup_strength.csv"


def fetch_baseballmonster_lineups(save_path=LINEUP_CACHE, timeout=30):
    """Fetch CSV lineup from BaseballMonster if available.

    URL observed: https://baseballmonster.com/Lineups.aspx?csv=1
    This is best-effort and may break if the provider changes.
    """
    url = "https://baseballmonster.com/Lineups.aspx?csv=1"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    txt = resp.content
    df = pd.read_csv(io.BytesIO(txt))
    # save raw
    df.to_csv(save_path, index=False)
    print(f"Lineups saved to: {save_path}")
    return save_path


def load_lineup_cache(path=LINEUP_CACHE):
    if not Path(path).exists():
        raise FileNotFoundError(f"Lineup cache not found: {path}")
    return pd.read_csv(path)


def compute_lineup_strength(lineup_df: pd.DataFrame, player_stats_df: pd.DataFrame = None, save_path=LINEUP_CACHE):
    """Compute a simple lineup_strength per team/date.

    If `player_stats_df` is provided (wOBA/wRC+), we sum the batter metric; otherwise
    fall back to counting starters as proxy strength.
    """
    out_rows = []
    # expected columns: Date, Team, Player, Slot (varies by provider)
    for _, row in lineup_df.iterrows():
        try:
            date = row.get("Date") or row.get("date")
            team = row.get("Team") or row.get("team") or row.get("Home")
            players = []
            # many CSVs pack lineup as comma-separated players in one cell
            for col in row.index:
                if isinstance(row[col], str) and row[col].count(" ") >= 1 and row[col] == row[col].strip():
                    players.append(row[col])
            strength = None
            if player_stats_df is not None:
                # try to sum wOBA or wRC+
                s = 0.0
                found = 0
                for p in players:
                    r = player_stats_df[player_stats_df["player_name"].str.contains(p, na=False, case=False)]
                    if not r.empty:
                        val = r.iloc[0].get("wOBA") or r.iloc[0].get("wrc_plus") or 0.0
                        s += float(val)
                        found += 1
                if found:
                    strength = s / found
            if strength is None:
                strength = len(players)
            out_rows.append({"date": date, "team": team, "lineup_strength": strength})
        except Exception:
            continue

    out = pd.DataFrame(out_rows)
    out.to_csv(save_path, index=False)
    print(f"Lineup strength saved to: {save_path}")
    return save_path


def main():
    # attempt to fetch BaseballMonster CSV, else instruct user to place CSV
    try:
        fetch_baseballmonster_lineups()
    except Exception as e:
        print("Failed to download lineups:", e)
        print("Place a lineup CSV at:", LINEUP_CACHE)


if __name__ == "__main__":
    main()
