External Odds Overrides

File: closing_odds_overrides.csv

Purpose:
- Provide alternative closing prices by game when default feeds do not expose odds.
- Used by Best Picks and unified audit via src/external_odds_overrides.py.

Required key columns:
- sport
- date (YYYY-MM-DD)
- game_id

Supported price columns:
- closing_moneyline_odds
- home_moneyline_odds
- away_moneyline_odds
- closing_spread_odds
- closing_total_odds
- closing_q1_odds
- closing_f5_odds
- closing_home_over_odds
- closing_corners_odds
- closing_btts_odds

Notes:
- American odds preferred (e.g. -110, +135), decimal odds also accepted.
- Zero/blank values are ignored.
- Existing non-empty event odds are preserved; overrides fill missing fields.

Bulk capture workflow (recommended):
1) Build editable sheet with all events and priority context:
	- `python src/build_odds_bulk_entry_sheet.py`
	- Output: `src/data/odds_provider/closing_odds_bulk_entry.csv`
2) Fill odds columns in that file (especially `closing_spread_odds` + market odds needed by each game).
3) Import filled values into official overrides file:
	- `python src/import_odds_bulk_entry.py`
4) Re-run validation/audit:
	- `python src/validate_odds_overrides.py`
	- `python src/audit_unified_backtest.py`

One-command automation:
- Run all generation/priority/validation steps in one command:
	- `python src/run_odds_automation.py`
- Windows shortcut:
	- `AUTO_ODDS_PIPELINE.bat`

Automated scraper (odds + lines):
- `python src/scrape_espn_odds_lines_all_sports.py`
- Pulls sportsbook prices and line fields from ESPN core odds endpoints.
- Auto-fills missing values in `closing_odds_overrides.csv` for:
	- `closing_moneyline_odds`, `home_moneyline_odds`, `away_moneyline_odds`
	- `closing_spread_odds`, `closing_total_odds`
	- `closing_spread_line`, `closing_total_line`, `odds_over_under`
- Raw scrape export:
	- `src/data/odds_provider/espn_all_lines_raw.csv`

TheOddsAPI scraper (optional, recommended):
- Set key as environment variable (PowerShell):
	- `$env:THEODDSAPI_KEY="YOUR_KEY"`
- Run scraper only:
	- `python src/scrape_theoddsapi_lines_all_sports.py`
- Raw scrape export:
	- `src/data/odds_provider/theoddsapi_all_lines_raw.csv`
- Included automatically in `python src/run_odds_automation.py` when `THEODDSAPI_KEY` is present.

Entry sheet helper columns:
- `expected_markets`, `missing_expected_markets`, `priority_score`: help prioritize what to fill first.
- `spread_market`, `odds_over_under`, `corners_line`: context lines from prediction events to speed manual entry.

Sport coverage status:
- Ingest-level normalized market odds are wired for: NBA, MLB, KBO, NHL, Liga MX, LaLiga.
- Overrides application (`closing_odds_overrides.csv`) is applied in Best Picks + unified audit across all sports loaded in those flows.

Scraper coverage details:
- Fully automated via ESPN odds endpoints: NBA, MLB, NHL, Liga MX, LaLiga, KBO.
- EuroLeague and NCAA Baseball do not have the same ESPN odds endpoint coverage in this pipeline; those remain through manual/CSV override load.
- TheOddsAPI path attempts extra coverage for supported keys and can improve fills where ESPN is sparse.
