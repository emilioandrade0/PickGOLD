# MLB Market Odds Backfill Strategy (2026-04-14)

## Objective
Increase historical odds coverage for MLB markets (moneyline, totals, line movement) so totals/run_line models are trained with real market signals instead of zeros.

## Current measured state
- Raw history rows: 6,554.
- `odds_over_under` missing or zero: 6,326 rows (96.5%).
- Moneyline missing (`home_moneyline_odds` or `away_moneyline_odds`): 6,311 rows (96.3%).
- Missing OU dates: 519 unique dates (`2024-02-22` to `2026-04-12`).
- `line_movement.csv`: 84 rows only.
- `line_movement_history.csv`: 191 snapshots, only `2026-04-01` to `2026-04-13`.
- Historical daily cache files: 0.

See structured metrics in `src/data/mlb/reports/mlb_market_odds_gap_2026-04-14.json`.

## Existing plumbing already available
- Historical odds fetch exists in `src/sports/mlb/ingest_line_movement.py` via `THEODDS_MLB_HISTORICAL_BACKFILL=1`.
- Raw history enrichment exists in `src/sports/mlb/data_ingest_mlb.py` via `apply_theodds_line_backfill()`.
- Feature engineering and training already consume these columns once present.
- Historical daily payloads are now written atomically (temp file + replace), so interruptions do not leave half-written cache files.
- Backfill progress is persisted in `src/data/mlb/cache/theodds_historical_daily/_backfill_progress.json` after each processed date.

## Phase plan

### Phase 1: Build historical odds cache (The Odds API)
Run in date windows to control API credits and allow restart from cache.

PowerShell template:

```powershell
$env:THE_ODDS_API_KEY = "<your_key>"
$env:THEODDS_MLB_HISTORICAL_BACKFILL = "1"
$env:THEODDS_MLB_HIST_MAX_DATES = "45"
$env:THEODDS_MLB_HIST_START_DATE = "2025-01-01"
$env:THEODDS_MLB_HIST_END_DATE = "2025-12-31"
$env:THEODDS_MLB_MARKETS = "h2h,totals"
$env:THEODDS_MLB_REGIONS = "us"
$env:THEODDS_MLB_MAX_CREDITS_PER_RUN = "8000"
$env:THEODDS_MLB_ALLOW_REDOWNLOAD_ON_CACHE_ERROR = "0"
& "c:/Users/andra/Desktop/NBA GOLD/.venv/Scripts/python.exe" src/sports/mlb/ingest_line_movement.py

Remove-Item Env:THEODDS_MLB_HISTORICAL_BACKFILL -ErrorAction SilentlyContinue
Remove-Item Env:THEODDS_MLB_HIST_MAX_DATES -ErrorAction SilentlyContinue
Remove-Item Env:THEODDS_MLB_HIST_START_DATE -ErrorAction SilentlyContinue
Remove-Item Env:THEODDS_MLB_HIST_END_DATE -ErrorAction SilentlyContinue
Remove-Item Env:THEODDS_MLB_MARKETS -ErrorAction SilentlyContinue
Remove-Item Env:THEODDS_MLB_REGIONS -ErrorAction SilentlyContinue
Remove-Item Env:THEODDS_MLB_MAX_CREDITS_PER_RUN -ErrorAction SilentlyContinue
Remove-Item Env:THEODDS_MLB_ALLOW_REDOWNLOAD_ON_CACHE_ERROR -ErrorAction SilentlyContinue
```

Recommended window order:
1. `2026-02-20` to current date (highest immediate value).
2. Full 2025 season.
3. Full 2024 season.

### Phase 2: Materialize backfilled odds into raw MLB history
After each Phase 1 batch, run ingest refresh so TheOdds cache is merged into raw files.

```powershell
& "c:/Users/andra/Desktop/NBA GOLD/.venv/Scripts/python.exe" src/sports/mlb/data_ingest_mlb.py
```

Expected effect:
- Fill `home_moneyline_odds`, `away_moneyline_odds`, `closing_moneyline_odds`, `odds_over_under` where raw fields were missing.
- Improve `odds_data_quality` and market feature availability before feature engineering.

### Phase 3: Rebuild model inputs and retrain

```powershell
& "c:/Users/andra/Desktop/NBA GOLD/.venv/Scripts/python.exe" src/sports/mlb/feature_engineering_mlb_core.py
& "c:/Users/andra/Desktop/NBA GOLD/.venv/Scripts/python.exe" src/sports/mlb/train_models_mlb.py
& "c:/Users/andra/Desktop/NBA GOLD/.venv/Scripts/python.exe" src/sports/mlb/historical_predictions_mlb_walkforward.py
```

### Phase 4: Validate uplift and guardrails
Track before/after:
- `odds_over_under` zero/missing rate in `mlb_advanced_history.csv`.
- `market_missing` and `market_micro_missing` rates in `model_ready_features_mlb.csv`.
- Walk-forward metrics for `totals` and `run_line` (`accuracy`, `published_accuracy`, `coverage`, `mae`, `rmse`).

Go/no-go checks:
- Keep change if market coverage rises and totals/run_line does not regress materially.
- If coverage rises but metrics drop, tune thresholds/calibration rather than reverting cache population.

## Operational notes
- The historical endpoint consumes API credits per date request. Keep batch windows explicit.
- Daily files under `src/data/mlb/cache/theodds_historical_daily/` are reusable cache; do not delete between runs.
- If no API key is present, backfill cannot execute; current session confirmed key was not set in terminal env.
- Credit formula for historical endpoint: `10 x markets x regions` per API call.
- If `THEODDS_MLB_MAX_CREDITS_PER_RUN` is set, ingestion stops before exceeding that run cap.
- If `THEODDS_MLB_ALLOW_REDOWNLOAD_ON_CACHE_ERROR=0` (default), ingestion stops on corrupt cache to protect credits.

## Fast execution checklist
1. Set `THE_ODDS_API_KEY`.
2. Run Phase 1 for one window.
3. Run Phase 2 (`data_ingest_mlb.py`).
4. Recompute gap report and confirm missing-rate drop.
5. Run Phase 3 and compare totals/run_line walk-forward.
