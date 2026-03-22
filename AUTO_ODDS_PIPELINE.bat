@echo off
setlocal

echo [RUN] Odds automation pipeline...
if "%THEODDSAPI_KEY%"=="" (
  echo [INFO] THEODDSAPI_KEY is not set. TheOddsAPI step will be skipped.
) else (
  echo [INFO] THEODDSAPI_KEY detected. TheOddsAPI step enabled.
)
python src\run_odds_automation.py
if errorlevel 1 (
  echo [ERROR] Odds automation failed.
  exit /b 1
)

echo [OK] Odds automation finished.
echo Next optional step: python src\audit_unified_backtest.py
exit /b 0
