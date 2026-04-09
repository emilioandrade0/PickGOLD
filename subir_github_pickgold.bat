@echo off
setlocal
cd /d "%~dp0"

echo.
echo ======================================
echo   PickGOLD - Subida automatica GitHub
echo ======================================
echo.

where git >nul 2>nul
if errorlevel 1 (
  echo ERROR: Git no esta instalado o no esta en PATH.
  pause
  exit /b 1
)

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format \"yyyy-MM-dd HH:mm:ss\""') do set TS=%%i
set MSG=chore: auto sync project %TS%

echo [1/4] Agregando todos los cambios...
git add -A
if errorlevel 1 goto :fail

echo [2/4] Creando commit...
git commit -m "%MSG%"
if errorlevel 1 (
  echo No hubo cambios nuevos para commitear o el commit fallo.
)

echo [3/4] Actualizando referencias remotas...
git fetch origin
if errorlevel 1 goto :fail

echo [4/4] Subiendo a GitHub con force-with-lease...
git push origin main --force-with-lease
if errorlevel 1 goto :fail

echo.
echo Listo. Proyecto subido correctamente a GitHub.
pause
exit /b 0

:fail
echo.
echo Ocurrio un error durante la subida.
pause
exit /b 1
