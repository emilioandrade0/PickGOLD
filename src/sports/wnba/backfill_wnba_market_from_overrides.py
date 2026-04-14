from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_FILE = BASE_DIR / "data" / "wnba" / "raw" / "wnba_advanced_history.csv"
OVERRIDES_FILE = BASE_DIR / "data" / "odds_provider" / "closing_odds_overrides.csv"


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _fill_if_missing(dst: pd.Series, src: pd.Series) -> pd.Series:
    dst_num = _to_num(dst)
    src_num = _to_num(src)
    missing = dst_num.isna() | (dst_num == 0)
    out = dst_num.copy()
    out.loc[missing] = src_num.loc[missing]
    return out


def _american_to_prob(v):
    try:
        x = float(v)
    except Exception:
        return np.nan
    if x == 0 or np.isnan(x):
        return np.nan
    if x > 0:
        return 100.0 / (x + 100.0)
    return abs(x) / (abs(x) + 100.0)


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"No existe RAW: {RAW_FILE}")
    if not OVERRIDES_FILE.exists():
        raise FileNotFoundError(f"No existe overrides: {OVERRIDES_FILE}")

    raw = pd.read_csv(RAW_FILE, dtype={"game_id": str})
    ovr = pd.read_csv(OVERRIDES_FILE, dtype={"game_id": str})

    if "sport" in ovr.columns:
        ovr = ovr[ovr["sport"].astype(str).str.lower() == "wnba"].copy()
    if ovr.empty:
        print("No hay filas WNBA en closing_odds_overrides.csv")
        return

    keep_cols = [
        "game_id",
        "home_moneyline_odds",
        "away_moneyline_odds",
        "closing_moneyline_odds",
        "closing_spread_odds",
        "closing_total_odds",
        "closing_spread_line",
        "closing_total_line",
        "odds_over_under",
        "odds_source_provider",
    ]
    for c in keep_cols:
        if c not in ovr.columns:
            ovr[c] = np.nan
    ovr = ovr[keep_cols].drop_duplicates(subset=["game_id"], keep="last")

    merged = raw.merge(ovr, on="game_id", how="left", suffixes=("", "__ovr"))
    before_quality_real = (merged["odds_data_quality"].astype(str).str.lower() == "real").sum() if "odds_data_quality" in merged.columns else 0

    for col in ["home_moneyline_odds", "away_moneyline_odds", "closing_moneyline_odds", "closing_spread_odds", "closing_total_odds", "odds_over_under"]:
        src = f"{col}__ovr"
        if src in merged.columns and col in merged.columns:
            merged[col] = _fill_if_missing(merged[col], merged[src])

    # Fill total line fallback.
    if "closing_total_line__ovr" in merged.columns and "odds_over_under" in merged.columns:
        merged["odds_over_under"] = _fill_if_missing(merged["odds_over_under"], merged["closing_total_line__ovr"])

    # Rebuild spread-related columns when missing and line is available.
    line = _to_num(merged.get("closing_spread_line__ovr", pd.Series(np.nan, index=merged.index)))
    home_ml = _to_num(merged.get("home_moneyline_odds", pd.Series(np.nan, index=merged.index)))
    away_ml = _to_num(merged.get("away_moneyline_odds", pd.Series(np.nan, index=merged.index)))

    home_prob = home_ml.apply(_american_to_prob)
    away_prob = away_ml.apply(_american_to_prob)

    existing_home_spread = _to_num(merged.get("home_spread", pd.Series(np.nan, index=merged.index)))
    need_spread = existing_home_spread.isna() | (existing_home_spread == 0)

    home_fav = home_prob > away_prob
    away_fav = away_prob > home_prob
    signed_home_spread = pd.Series(np.nan, index=merged.index, dtype=float)
    signed_home_spread.loc[home_fav] = -line.loc[home_fav].abs()
    signed_home_spread.loc[away_fav] = line.loc[away_fav].abs()
    signed_home_spread = signed_home_spread.where(line.notna(), np.nan)

    merged.loc[need_spread, "home_spread"] = signed_home_spread.loc[need_spread]
    if "spread_abs" in merged.columns:
        merged["spread_abs"] = _to_num(merged["home_spread"]).abs().fillna(_to_num(merged["spread_abs"]))
    if "home_is_favorite" in merged.columns:
        hs = _to_num(merged["home_spread"])
        merged["home_is_favorite"] = np.where(hs < 0, 1, np.where(hs > 0, 0, merged["home_is_favorite"]))
    if "odds_spread" in merged.columns:
        hs = _to_num(merged["home_spread"])
        merged["odds_spread"] = np.where(
            hs < 0,
            merged["home_team"].astype(str) + " " + hs.round(1).astype(str),
            np.where(
                hs > 0,
                merged["away_team"].astype(str) + " -" + hs.round(1).astype(str),
                merged["odds_spread"],
            ),
        )

    # Update quality/source flags when we now have usable market fields.
    has_market_now = (_to_num(merged.get("odds_over_under", pd.Series(np.nan, index=merged.index))).fillna(0) > 0) | (_to_num(merged.get("home_spread", pd.Series(np.nan, index=merged.index))).fillna(0) != 0)
    if "odds_data_quality" in merged.columns:
        merged.loc[has_market_now, "odds_data_quality"] = merged.loc[has_market_now, "odds_data_quality"].replace({np.nan: "real"}).fillna("real")
        merged.loc[has_market_now & merged["odds_data_quality"].astype(str).str.lower().eq("fallback"), "odds_data_quality"] = "real"
    if "market_source" in merged.columns:
        merged.loc[has_market_now & merged["market_source"].isna(), "market_source"] = "odds_overrides"

    drop_cols = [c for c in merged.columns if c.endswith("__ovr")]
    merged = merged.drop(columns=drop_cols)

    after_quality_real = (merged["odds_data_quality"].astype(str).str.lower() == "real").sum() if "odds_data_quality" in merged.columns else 0
    covered = int(has_market_now.sum())
    print(f"Rows total: {len(merged)}")
    print(f"Rows with market fields now: {covered}")
    print(f"odds_data_quality=real before={before_quality_real} after={after_quality_real}")

    merged.to_csv(RAW_FILE, index=False)
    print(f"OK: RAW actualizado -> {RAW_FILE}")


if __name__ == "__main__":
    main()
