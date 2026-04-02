Advanced ingestion helpers

Files added under `src/`:

- `ingest_umpire_stats.py` — best-effort download from umpscorecards.com and loader. Saves to `data/mlb/cache/umpire_stats.csv`.
- `ingest_lineup_strength.py` — tries to fetch BaseballMonster lineups CSV and computes a simple `lineup_strength` per team/date; saves to `data/mlb/cache/lineup_strength.csv`.
- `ingest_line_movement.py` — uses The Odds API when `THE_ODDS_API_KEY` or `ODDS_API_KEY` env var is set to fetch historical odds and save `data/mlb/cache/line_movement.csv`.

Usage examples:

```powershell
python -m src.ingest_umpire_stats
python -m src.ingest_lineup_strength
set THE_ODDS_API_KEY=your_key
python -m src.ingest_line_movement
```

If automated downloads fail, place vendor CSVs in `src/data/mlb/cache/` with the filenames above.

Next steps to integrate into pipeline:
- Ensure `src/feature_engineering_mlb.py` merges the CSVs if present (it already contains merge hooks).
- Provide player stats CSV (wOBA/wRC+) if you want `lineup_strength` computed from player metrics.
