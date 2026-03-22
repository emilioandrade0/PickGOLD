@echo off
setlocal EnableExtensions
title NBA GOLD - Share via ngrok

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "PORT=%~1"

if not defined PORT (
  for /f %%P in ('powershell -NoProfile -Command "$ports = 5173,5174,5175; foreach($p in $ports){ if(Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue){ Write-Output $p; break } }"') do set "PORT=%%P"
)

if not defined PORT (
  echo [ERROR] No encontre Vite escuchando en 5173/5174/5175.
  echo         Inicia tu UI primero y vuelve a ejecutar este archivo.
  pause
  exit /b 1
)

where ngrok >nul 2>&1
if errorlevel 1 (
  echo [ERROR] ngrok no esta disponible en PATH.
  echo         Instala ngrok o agrega su ruta al PATH.
  pause
  exit /b 1
)

echo ==========================================================
echo  Compartiendo NBA GOLD por ngrok
echo  UI local detectada en: http://127.0.0.1:%PORT%
echo ==========================================================
echo.

start "NBA GOLD ngrok" cmd /k "cd /d %ROOT% && ngrok http http://127.0.0.1:%PORT%"

rem Espera breve para que ngrok inicialice su API local.
timeout /t 3 /nobreak >nul

set "PUBLIC_URL="
for /f "usebackq delims=" %%U in (`powershell -NoProfile -Command "$u=''; try { $u=(Invoke-RestMethod -Uri 'http://127.0.0.1:4040/api/tunnels' -TimeoutSec 5).tunnels | Where-Object { $_.proto -eq 'https' } | Select-Object -First 1 -ExpandProperty public_url } catch {}; if($u){$u}"`) do set "PUBLIC_URL=%%U"

if defined PUBLIC_URL (
  echo [OK] Link publico:
  echo %PUBLIC_URL%
  powershell -NoProfile -Command "Set-Clipboard -Value '%PUBLIC_URL%'" >nul 2>&1
  echo [OK] Link copiado al portapapeles.
  start "" "%PUBLIC_URL%"
) else (
  echo [INFO] No pude leer el link automaticamente.
  echo        Revisa la ventana "NBA GOLD ngrok" para copiar el Forwarding URL.
)

echo.
echo Consejo: si quieres forzar puerto manual, ejecuta:
echo   SHARE_NBA_GOLD.bat 5174
echo.
pause
