@echo off
setlocal EnableExtensions
title NBA GOLD - FULL AUDIT (Optional Scripts)

set "ROOT=C:\Users\andra\Desktop\NBA GOLD"
set "PY=python"
set "ACTIVATE=C:\Users\andra\miniconda3\Scripts\activate.bat"
set /a STEP=0
set "TOTAL=15"

cd /d "%ROOT%"

echo ==========================================================
echo   NBA GOLD - FULL AUDIT (scripts opcionales + checks)
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

call :run_optional "backtest_picks.py" "%ROOT%\src\backtest_picks.py"
call :run_optional "backtest_vs_vegas.py" "%ROOT%\src\backtest_vs_vegas.py"
call :run_optional "event_adjustments_laliga.py" "%ROOT%\src\event_adjustments_laliga.py"
call :run_optional "event_adjustments_liga_mx.py" "%ROOT%\src\event_adjustments_liga_mx.py"
call :run_optional "historical_predictions.py" "%ROOT%\src\historical_predictions.py"
call :run_optional "historical_predictions_kbo.py" "%ROOT%\src\historical_predictions_kbo.py"
call :run_optional "historical_predictions_laliga.py" "%ROOT%\src\historical_predictions_laliga.py"
call :run_optional "historical_predictions_liga_mx.py" "%ROOT%\src\historical_predictions_liga_mx.py"
call :run_optional "historical_predictions_mlb.py" "%ROOT%\src\historical_predictions_mlb.py"
call :run_optional "historical_predictions_nhl.py" "%ROOT%\src\historical_predictions_nhl.py"
call :run_optional "data_ingest_euroleague.py" "%ROOT%\src\data_ingest_euroleague.py"
call :run_optional "feature_engineering_euroleague.py" "%ROOT%\src\feature_engineering_euroleague.py"
call :run_optional "train_models_euroleague.py" "%ROOT%\src\train_models_euroleague.py"
call :run_optional "predict_today_euroleague.py" "%ROOT%\src\predict_today_euroleague.py"
call :run_optional "train_lgbm.py" "%ROOT%\src\train_lgbm.py"

set /a STEP+=1
echo [%STEP%/%TOTAL%] Verificando modulos clave (api.py, weekday scoring, best picks)...
%PY% -c "import py_compile; py_compile.compile(r'%ROOT%\src\api.py', doraise=True); py_compile.compile(r'%ROOT%\src\weekday_insights\scoring.py', doraise=True); py_compile.compile(r'%ROOT%\src\best_picks\daily.py', doraise=True); print('OK')"
if errorlevel 1 goto :fail

echo.
echo ==========================================================
echo OK: Full audit completado sin errores.
echo ==========================================================
pause
exit /b 0

:run_optional
set /a STEP+=1
echo [%STEP%/%TOTAL%] %~1
if not exist "%~2" (
    echo    SKIP: no existe %~2
    goto :eof
)
%PY% "%~2"
if errorlevel 1 goto :fail
goto :eof

:fail
echo.
echo ERROR: Full audit se detuvo en el paso anterior.
pause
exit /b 1
