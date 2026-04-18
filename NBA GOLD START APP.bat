@echo off
setlocal
title NBA GOLD - Start App Only

set "ROOT=C:\Users\andra\Desktop\NBA GOLD"
set "ACTIVATE=C:\Users\andra\miniconda3\Scripts\activate.bat"

cd /d "%ROOT%"

if exist "%ROOT%\tools\mlb_apply_profile.bat" (
    call "%ROOT%\tools\mlb_apply_profile.bat"
)

echo ==========================================
echo   NBA GOLD - LEVANTAR SOLO APP (API + UI)
echo ==========================================
echo.

call "%ACTIVATE%" nba
if errorlevel 1 (
    echo ERROR: No se pudo activar entorno conda nba.
    pause
    exit /b 1
)

if exist "%ROOT%\tools\mlb_daily_semaforo.py" (
    python "%ROOT%\tools\mlb_daily_semaforo.py" --apply-profile --quiet >nul 2>&1
)

if exist "%ROOT%\tools\mlb_apply_profile.bat" (
    call "%ROOT%\tools\mlb_apply_profile.bat"
)

echo Cerrando proceso previo en puerto 8000 (si existe)...
powershell -NoProfile -Command "$c=Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue; if($c){Stop-Process -Id $c.OwningProcess -Force}" >nul 2>&1

echo Iniciando API en nueva ventana...
start "NBA GOLD API" cmd /k "cd /d %ROOT% && call %ACTIVATE% nba && python -m uvicorn src.api:app --host 127.0.0.1 --port 8000"

echo Iniciando UI en nueva ventana...
start "NBA GOLD UI" cmd /k "set PATH=C:\Program Files\nodejs;C:\Users\andra\miniconda3;C:\Users\andra\miniconda3\Scripts;%PATH% && cd /d %ROOT%\ui && npm run dev -- --host 127.0.0.1 --port 5173"

echo Esperando servicios...
timeout /t 8 /nobreak >nul

echo Verificando API base...
powershell -NoProfile -Command "for($i=0;$i -lt 8;$i++){try{$r=Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/api/sports' -TimeoutSec 6; if($r.StatusCode -eq 200){exit 0}}catch{}; Start-Sleep -Seconds 1}; exit 1"
if errorlevel 1 (
    echo ERROR: API no respondio a tiempo.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo OK: App levantada correctamente.
echo UI:  http://127.0.0.1:5173/
echo API: http://127.0.0.1:8000/docs
echo ==========================================
start http://127.0.0.1:5173/
pause
exit /b 0
