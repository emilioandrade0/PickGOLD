@echo off
setlocal EnableDelayedExpansion
title NBA GOLD - Update Data Only

set "ROOT=C:\Users\andra\Desktop\NBA GOLD"
set "PY=python"
set "ACTIVATE=C:\Users\andra\miniconda3\Scripts\activate.bat"
set "UPDATED_ANY=0"

cd /d "%ROOT%"

echo ======================================================
echo   NBA GOLD - ACTUALIZAR SOLO DATOS (sin levantar app)
echo ======================================================
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

echo [EXTRA] EuroLeague - historical_predictions_euroleague.py
%PY% "%ROOT%\src\historical_predictions_euroleague.py"
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

echo [CHECK] Verificacion rapida API source files (sin levantar servidor)
%PY% -c "import src.api as _; print('API import OK')"
if errorlevel 1 goto :fail

echo.
echo ======================================================
echo OK: Actualizacion de datos completada.
echo Puedes levantar la app con: NBA GOLD START APP.bat
echo ======================================================
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
echo ERROR: La actualizacion se detuvo en el paso anterior.
pause
exit /b 1
