@echo off
setlocal EnableDelayedExpansion
title NBA GOLD Full Refresh + Launch

set "ROOT=C:\Users\andra\Desktop\NBA GOLD"
set "PY=python"
set "ACTIVATE=C:\Users\andra\miniconda3\Scripts\activate.bat"
set "UPDATED_ANY=0"

cd /d "%ROOT%"

echo ==========================================================
echo      NBA GOLD - ACTUALIZACION INTELIGENTE + LAUNCH
echo ==========================================================
echo.

call "%ACTIVATE%" nba
if errorlevel 1 (
    echo ERROR: No se pudo activar entorno conda nba.
    pause
    exit /b 1
)

echo Entorno conda nba activado.
echo.

call :run_sport "NBA" "%ROOT%\src\data_ingest.py" "%ROOT%\src\data\raw\nba_advanced_history.csv" "%ROOT%\src\feature_engineering.py" "%ROOT%\src\train_models.py" "%ROOT%\src\predict_today.py"
if errorlevel 1 goto :fail

call :run_sport "MLB" "%ROOT%\src\data_ingest_mlb.py" "%ROOT%\src\data\mlb\raw\mlb_advanced_history.csv" "%ROOT%\src\feature_engineering_mlb.py" "%ROOT%\src\train_models_mlb.py" "%ROOT%\src\predict_today_mlb.py"
if errorlevel 1 goto :fail

call :run_sport "KBO" "%ROOT%\src\data_ingest_kbo.py" "%ROOT%\src\data\kbo\raw\kbo_advanced_history.csv" "%ROOT%\src\feature_engineering_kbo.py" "%ROOT%\src\train_models_kbo.py" "%ROOT%\src\predict_today_kbo.py"
if errorlevel 1 goto :fail

call :run_sport "NHL" "%ROOT%\src\data_ingest_nhl.py" "%ROOT%\src\data\nhl\raw\nhl_advanced_history.csv" "%ROOT%\src\feature_engineering_nhl.py" "%ROOT%\src\train_models_nhl.py" "%ROOT%\src\predict_today_nhl.py"
if errorlevel 1 goto :fail

call :run_sport "Liga MX" "%ROOT%\src\data_ingest_liga_mx.py" "%ROOT%\src\data\liga_mx\raw\liga_mx_advanced_history.csv" "%ROOT%\src\feature_engineering_liga_mx.py" "%ROOT%\src\train_models_liga_mx.py" "%ROOT%\src\predict_today_liga_mx.py"
if errorlevel 1 goto :fail

call :run_sport "LaLiga" "%ROOT%\src\data_ingest_laliga.py" "%ROOT%\src\data\laliga\raw\laliga_advanced_history.csv" "%ROOT%\src\feature_engineering_laliga.py" "%ROOT%\src\train_models_laliga.py" "%ROOT%\src\predict_today_laliga.py"
if errorlevel 1 goto :fail

call :run_sport "EuroLeague" "%ROOT%\src\data_ingest_euroleague.py" "%ROOT%\src\data\euroleague\raw\euroleague_advanced_history.csv" "%ROOT%\src\feature_engineering_euroleague.py" "%ROOT%\src\train_models_euroleague.py" "%ROOT%\src\predict_today_euroleague.py"
if errorlevel 1 goto :fail

call :run_sport "NCAA Baseball" "%ROOT%\src\data_ingest_ncaa_baseball.py" "%ROOT%\src\data\ncaa_baseball\raw\ncaa_baseball_advanced_history.csv" "%ROOT%\src\feature_engineering_ncaa_baseball.py" "%ROOT%\src\train_models_ncaa_baseball.py" "%ROOT%\src\predict_today_ncaa_baseball.py"
if errorlevel 1 goto :fail

echo.
echo [ODDS] Ejecutando automatizacion de odds/spreads/lines...
echo [ODDS] run_odds_automation.py
%PY% "%ROOT%\src\run_odds_automation.py"
if errorlevel 1 goto :fail

echo.
echo [HIST] Generando historicos para todos los deportes...
echo [HIST] NBA - historical_predictions.py
%PY% "%ROOT%\src\historical_predictions.py"
if errorlevel 1 goto :fail

echo [HIST] MLB - historical_predictions_mlb.py
%PY% "%ROOT%\src\historical_predictions_mlb.py"
if errorlevel 1 goto :fail

echo [HIST] KBO - historical_predictions_kbo.py
%PY% "%ROOT%\src\historical_predictions_kbo.py"
if errorlevel 1 goto :fail

echo [HIST] NHL - historical_predictions_nhl.py
%PY% "%ROOT%\src\historical_predictions_nhl.py"
if errorlevel 1 goto :fail

echo [HIST] Liga MX - historical_predictions_liga_mx.py
%PY% "%ROOT%\src\historical_predictions_liga_mx.py"
if errorlevel 1 goto :fail

echo [HIST] LaLiga - historical_predictions_laliga.py
%PY% "%ROOT%\src\historical_predictions_laliga.py"
if errorlevel 1 goto :fail

echo [HIST] EuroLeague - historical_predictions_euroleague.py
%PY% "%ROOT%\src\historical_predictions_euroleague.py"
if errorlevel 1 goto :fail

echo [HIST] NCAA Baseball - historical_predictions_ncaa_baseball.py
%PY% "%ROOT%\src\historical_predictions_ncaa_baseball.py"
if errorlevel 1 goto :fail

echo.
echo [META] Reentrenando meta-modelo Best Picks...
echo [META] train_best_picks_meta_model.py
%PY% "%ROOT%\src\train_best_picks_meta_model.py"
if errorlevel 1 goto :fail

echo.
echo Entrenamiento ejecutado para todos los deportes.
echo Ejecutando auditoria y recalibracion global...

echo [GLOBAL] Auditoria unificada - audit_unified_backtest.py
%PY% "%ROOT%\src\audit_unified_backtest.py"
if errorlevel 1 goto :fail

echo [GLOBAL] Generar calibracion - generate_calibration_params.py
%PY% "%ROOT%\src\generate_calibration_params.py"
if errorlevel 1 goto :fail

echo [GLOBAL] Rebuild predicciones con calibracion nueva
%PY% "%ROOT%\src\predict_today.py"
if errorlevel 1 goto :fail
%PY% "%ROOT%\src\predict_today_mlb.py"
if errorlevel 1 goto :fail
%PY% "%ROOT%\src\predict_today_kbo.py"
if errorlevel 1 goto :fail
%PY% "%ROOT%\src\predict_today_nhl.py"
if errorlevel 1 goto :fail
%PY% "%ROOT%\src\predict_today_liga_mx.py"
if errorlevel 1 goto :fail
%PY% "%ROOT%\src\predict_today_laliga.py"
if errorlevel 1 goto :fail
%PY% "%ROOT%\src\predict_today_euroleague.py"
if errorlevel 1 goto :fail
%PY% "%ROOT%\src\predict_today_ncaa_baseball.py"
if errorlevel 1 goto :fail

echo.
echo Iniciando API en nueva ventana...
start "NBA GOLD API" cmd /k "cd /d %ROOT% && call %ACTIVATE% nba && python -m uvicorn src.api:app --host 127.0.0.1 --port 8000"

echo Iniciando UI en nueva ventana...
start "NBA GOLD UI" cmd /k "set PATH=C:\Program Files\nodejs;C:\Users\andra\miniconda3;C:\Users\andra\miniconda3\Scripts;%PATH% && cd /d %ROOT%\ui && npm run dev -- --host 127.0.0.1 --port 5173"

echo Esperando servidores...
timeout /t 10 /nobreak >nul

echo.
echo Verificando endpoints (deportes + insights + best picks)...

set "CHECK_PS=powershell -NoProfile -Command"

echo [CHECK] NBA
%CHECK_PS% "for($i=0;$i -lt 5;$i++){try{$r=Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/api/nba/predictions/today' -TimeoutSec 8; if($r.StatusCode -eq 200){exit 0}}catch{}; Start-Sleep -Seconds 2}; exit 1"
if errorlevel 1 goto :fail

echo [CHECK] MLB
%CHECK_PS% "for($i=0;$i -lt 5;$i++){try{$r=Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/api/mlb/predictions/today' -TimeoutSec 8; if($r.StatusCode -eq 200){exit 0}}catch{}; Start-Sleep -Seconds 2}; exit 1"
if errorlevel 1 goto :fail

echo [CHECK] KBO
%CHECK_PS% "for($i=0;$i -lt 5;$i++){try{$r=Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/api/kbo/predictions/today' -TimeoutSec 8; if($r.StatusCode -eq 200){exit 0}}catch{}; Start-Sleep -Seconds 2}; exit 1"
if errorlevel 1 goto :fail

echo [CHECK] NHL
%CHECK_PS% "for($i=0;$i -lt 5;$i++){try{$r=Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/api/nhl/predictions/today' -TimeoutSec 8; if($r.StatusCode -eq 200){exit 0}}catch{}; Start-Sleep -Seconds 2}; exit 1"
if errorlevel 1 goto :fail

echo [CHECK] Liga MX
%CHECK_PS% "for($i=0;$i -lt 5;$i++){try{$r=Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/api/liga_mx/predictions/today' -TimeoutSec 8; if($r.StatusCode -eq 200){exit 0}}catch{}; Start-Sleep -Seconds 2}; exit 1"
if errorlevel 1 goto :fail

echo [CHECK] LaLiga
%CHECK_PS% "for($i=0;$i -lt 5;$i++){try{$r=Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/api/laliga/predictions/today' -TimeoutSec 8; if($r.StatusCode -eq 200){exit 0}}catch{}; Start-Sleep -Seconds 2}; exit 1"
if errorlevel 1 goto :fail

echo [CHECK] EuroLeague
%CHECK_PS% "for($i=0;$i -lt 5;$i++){try{$r=Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/api/euroleague/predictions/today' -TimeoutSec 8; if($r.StatusCode -eq 200){exit 0}}catch{}; Start-Sleep -Seconds 2}; exit 1"
if errorlevel 1 goto :fail

echo [CHECK] NCAA Baseball
%CHECK_PS% "for($i=0;$i -lt 5;$i++){try{$r=Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/api/ncaa_baseball/predictions/today' -TimeoutSec 8; if($r.StatusCode -eq 200){exit 0}}catch{}; Start-Sleep -Seconds 2}; exit 1"
if errorlevel 1 goto :fail

echo [CHECK] Insights Summary
%CHECK_PS% "for($i=0;$i -lt 5;$i++){try{$r=Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/api/insights/summary' -TimeoutSec 10; if($r.StatusCode -eq 200){exit 0}}catch{}; Start-Sleep -Seconds 2}; exit 1"
if errorlevel 1 goto :fail

echo [CHECK] Insights Weekday Scoring
%CHECK_PS% "for($i=0;$i -lt 5;$i++){try{$r=Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/api/insights/weekday-scoring' -TimeoutSec 10; if($r.StatusCode -eq 200){exit 0}}catch{}; Start-Sleep -Seconds 2}; exit 1"
if errorlevel 1 goto :fail

echo [CHECK] Insights Tier Performance
%CHECK_PS% "for($i=0;$i -lt 5;$i++){try{$r=Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/api/insights/tier-performance' -TimeoutSec 10; if($r.StatusCode -eq 200){exit 0}}catch{}; Start-Sleep -Seconds 2}; exit 1"
if errorlevel 1 goto :fail

echo [CHECK] Best Picks
%CHECK_PS% "for($i=0;$i -lt 5;$i++){try{$r=Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/api/insights/best-picks/today?top_n=12' -TimeoutSec 10; if($r.StatusCode -eq 200){exit 0}}catch{}; Start-Sleep -Seconds 2}; exit 1"
if errorlevel 1 goto :fail

echo [OK] Endpoints validados.

start http://127.0.0.1:5173/

echo.
echo ==========================================================
echo OK: Actualizacion completada y app iniciada.
echo UI:  http://127.0.0.1:5173/
echo API: http://127.0.0.1:8000/docs
echo ==========================================================
pause
exit /b 0

:run_sport
set "SPORT=%~1"
set "INGEST=%~2"
set "RAW_FILE=%~3"
set "FE_SCRIPT=%~4"
set "TR_SCRIPT=%~5"
set "PR_SCRIPT=%~6"

echo.
echo ------------------------------------------------------
echo %SPORT%
echo ------------------------------------------------------

echo [INGEST] %SPORT%
%PY% "%INGEST%"
if errorlevel 1 exit /b 1

echo [TRAIN] %SPORT% ejecutando feature_engineering + train_models...
%PY% "%FE_SCRIPT%"
if errorlevel 1 exit /b 1
%PY% "%TR_SCRIPT%"
if errorlevel 1 exit /b 1
set "UPDATED_ANY=1"

echo [PREDICT] %SPORT%
%PY% "%PR_SCRIPT%"
if errorlevel 1 exit /b 1

exit /b 0

:hash_file
set "HFILE=%~1"
set "OUTVAR=%~2"
set "HASH=__MISSING__"
if exist "%HFILE%" (
    for /f "delims=" %%H in ('certutil -hashfile "%HFILE%" SHA256 ^| findstr /R "^[0-9A-F][0-9A-F]"') do set "HASH=%%H"
)
set "%OUTVAR%=%HASH%"
exit /b 0

:fail
echo.
echo ERROR: El pipeline se detuvo en el paso anterior.
pause
exit /b 1
