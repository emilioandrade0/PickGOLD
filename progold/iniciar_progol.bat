@echo off
setlocal

cd /d "%~dp0"

echo ========================================================
echo   Progol Contrarian Lab - Inicio Automatico
echo ========================================================
echo.

if exist ".venv\Scripts\python.exe" (
    goto :venv_ready
)

set "PYTHON_CMD="
where python >nul 2>nul && set "PYTHON_CMD=python"
if not defined PYTHON_CMD (
    where py >nul 2>nul && set "PYTHON_CMD=py -3"
)

if not defined PYTHON_CMD (
    echo [ERROR] No se encontro Python en PATH.
    echo Instala Python 3.10+ y vuelve a ejecutar este archivo.
    echo.
    pause
    exit /b 1
)

echo [INFO] No existe .venv. Creando entorno virtual...
call %PYTHON_CMD% -m venv .venv
if errorlevel 1 (
    echo [ERROR] No se pudo crear el entorno virtual.
    echo.
    pause
    exit /b 1
)

:venv_ready
echo [INFO] Instalando o validando dependencias...
call ".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Fallo la instalacion de dependencias.
    echo.
    pause
    exit /b 1
)

if /I "%~1"=="--solo-setup" (
    echo [INFO] Setup completado. No se inicio Streamlit por bandera --solo-setup.
    echo.
    pause
    exit /b 0
)

echo.
echo [INFO] Iniciando app en http://localhost:8501
echo [INFO] Para detenerla: Ctrl + C
echo.
call ".venv\Scripts\python.exe" -m streamlit run app.py

echo.
echo [INFO] App cerrada.
pause
