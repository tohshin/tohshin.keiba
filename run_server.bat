@echo off
setlocal

cd /d "%~dp0"

echo --- Tohshin Keiba Local Server Starter (with auto-generate) ---

rem Check if index.html and generate_html.py exist
if not exist "index.html" (
    echo [ERROR] index.html not found in %cd%
    echo Please make sure you are running this from the correct folder.
    pause
    exit /b 1
)

if not exist "generate_html.py" (
    echo [ERROR] generate_html.py not found in %cd%
    echo Cannot auto-generate race data without generate_html.py.
    pause
    exit /b 1
)

rem Check if port 8000 is already in use and kill it
echo [INFO] Checking for existing processes on port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000') do (
    echo [INFO] Found process %%a using port 8000. Terminating...
    taskkill /f /pid %%a >nul 2>&1
)

rem Generate latest data first (conda new environment, same pattern as gits.bat)
echo [INFO] Generating latest HTML/JSON via generate_html.py...
call C:\Users\kyoui\anaconda3\Scripts\activate.bat C:\Users\kyoui\anaconda3
call conda activate new
python generate_html.py
if errorlevel 1 (
    echo [WARN] generate_html.py exited with an error. Server will still start.
) else (
    echo [INFO] generate_html.py completed successfully.
)

rem Make sure port 8000 is free before starting the server
echo [INFO] Ensuring port 8000 is free again before server start...
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

rem --- TEMP TEST STOP: pause here to inspect output before server blocks the window ---
echo [TEST] Before starting http.server 8000. Current dir: %cd%
echo [TEST] PYTHON_CMD=%PYTHON_CMD%
pause

"%PYTHON_CMD%" -m http.server 8000

:end
endlocal
