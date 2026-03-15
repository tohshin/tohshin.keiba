@echo off
setlocal

start http://localhost:8000

python --version >nul 2>&1
if %errorlevel% equ 0 (
    python -m http.server 8000
    goto end
)

py --version >nul 2>&1
if %errorlevel% equ 0 (
    py -m http.server 8000
    goto end
)

python3 --version >nul 2>&1
if %errorlevel% equ 0 (
    python3 -m http.server 8000
    goto end
)

pause
:end
endlocal
