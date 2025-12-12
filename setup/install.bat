@echo off
REM Adaptive installation script for rppg2ppg (Windows)
REM Usage: setup\install.bat [--cpu] [--check]
REM
REM Note: Activate your virtual environment first!
REM   conda activate customenv
REM   .\venv\Scripts\activate

setlocal enabledelayedexpansion

echo ==============================================
echo   rppg2ppg Installer (Windows)
echo ==============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [!] Python not found. Please install Python 3.10+
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VER=%%i
echo [OK] Python %PYTHON_VER%

REM Pass arguments to Python installer
set ARGS=
if "%1"=="--cpu" set ARGS=--cpu
if "%1"=="--check" set ARGS=--check
if "%2"=="--cpu" set ARGS=%ARGS% --cpu
if "%2"=="--check" set ARGS=%ARGS% --check

echo.
echo Running Python installer...
echo.

python "%~dp0install.py" %ARGS%

if errorlevel 1 (
    echo.
    echo [!] Installation failed
    exit /b 1
)

echo.
echo [OK] Installation complete!
endlocal
