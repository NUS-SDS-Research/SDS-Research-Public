@echo off
setlocal enabledelayedexpansion

:: 1. Get the Git root directory
for /f "tokens=*" %%i in ('git rev-parse --show-toplevel 2^>nul') do set "GIT_ROOT=%%i"

:: Fallback if not in a git repo
if "%GIT_ROOT%"=="" (
    echo [FLaas] Error: Not in a git repository.
    exit /b 1
)

:: 2. Set the Virtual Environment directory path (now venv)
set "VENV_DIR=%GIT_ROOT%\federated_learning\venv"
set "REQS_FILE=%GIT_ROOT%\federated_learning\requirements.txt"

:: 3. Check if venv exists; if not, create and install
if not exist "%VENV_DIR%" (
    echo [FLaas] Creating new virtual environment at %VENV_DIR%...
    python -m venv "%VENV_DIR%"
    
    if exist "%REQS_FILE%" (
        echo [FLaas] Installing requirements...
        "%VENV_DIR%\Scripts\python.exe" -m pip install -r "%REQS_FILE%" -q
    ) else (
        echo [FLaas] Warning: requirements.txt not found. Skipping install.
    )
)

:: 4. Activate the environment
:: Use 'call' so the script continues and stays active in your CMD session
call "%VENV_DIR%\Scripts\activate.bat"

:: 5. Output status
for /f "tokens=*" %%v in ('python --version') do set "PY_VER=%%v"
echo [FLaas] venv active ^<- !PY_VER!