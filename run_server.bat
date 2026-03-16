@echo off
setlocal
cd /d "%~dp0"

echo --- Tohshin Keiba Local Server Starter ---

rem Check if index.html exists
if not exist "index.html" (
    echo "[ERROR] index.html not found in %cd%"
    echo "Please make sure you are running this from the correct folder."
    pause
    exit /b 1
)

rem Check if port 8000 is already in use and kill it
echo [INFO] Checking for existing processes on port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000') do (
    echo [INFO] Found process %%a using port 8000. Terminating...
    taskkill /f /pid %%a >nul 2>&1
)

rem Define Python check function
set "PYTHON_CMD="

where python >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_CMD=python"
) else (
    where py >nul 2>&1
    if %errorlevel% equ 0 (
        set "PYTHON_CMD=py"
    ) else (
        where python3 >nul 2>&1
        if %errorlevel% equ 0 (
            set "PYTHON_CMD=python3"
        )
    )
)

if "%PYTHON_CMD%"=="" (
    echo [ERROR] Python not found.
    echo Please make sure Python is installed and added to your PATH.
    pause
    exit /b 1
)

echo [INFO] Starting server with %PYTHON_CMD%...
echo [INFO] Access the site at: http://localhost:8000
echo [INFO] Press Ctrl+C in this window to stop the server.

start http://localhost:8000
"%PYTHON_CMD%" -m http.server 8000

:end
endlocal
