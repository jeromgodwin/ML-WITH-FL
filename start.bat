@echo off
setlocal EnableDelayedExpansion
title FEDSHIELD - SYSTEM INITIALIZATION

:: Set terminal colors (Black background, Green text)
color 0A

:: ASCII Art Logo
echo.
echo    ███████╗███████╗██████╗ ███████╗██╗  ██╗██╗███████╗██╗     ██████╗ 
echo    ██╔════╝██╔════╝██╔══██╗██╔════╝██║  ██║██║██╔════╝██║     ██╔══██╗
echo    █████╗  █████╗  ██║  ██║███████╗███████║██║█████╗  ██║     ██║  ██║
echo    ██╔══╝  ██╔══╝  ██║  ██║╚════██║██╔══██║██║██╔══╝  ██║     ██║  ██║
echo    ██║     ███████╗██████╔╝███████║██║  ██║██║███████╗███████╗██████╔╝
echo    ╚═╝     ╚══════╝╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝╚═════╝ 
echo.
echo    SECURITY INTELLIGENCE CONSOLE - INITIALIZATION SEQUENCE
echo    =======================================================
echo.

cd /d "%~dp0"

:: 1. Dependency Checks
echo [SYSTEM] Verifying environment...

:: Check for .venv
if not exist ".venv\Scripts\python.exe" (
    echo [WARNING] Python virtual environment not found.
    echo [SYSTEM] Creating virtual environment...
    python -m venv .venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment. Ensure Python is installed.
        pause
        exit /b 1
    )
    echo [SYSTEM] Installing backend dependencies...
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements.txt
) else (
    echo [OK] Python virtual environment detected.
)

:: Check for frontend node_modules
if not exist "frontend\node_modules" (
    echo [WARNING] Frontend dependencies not found.
    echo [SYSTEM] Installing frontend dependencies...
    cd frontend
    call npm install
    cd ..
) else (
    echo [OK] Frontend dependencies detected.
)

:: Port clearance removed to prevent startup hangs. Use stop.bat if ports are blocked.
echo [SYSTEM] Ports will be allocated dynamically.

echo.
echo [SYSTEM] Booting sub-systems...
echo.

:: 2. Start Backend
echo [STARTING] FedShield API Backend (Port 8000)...
start "FEDSHIELD-BACKEND" cmd /c "title FEDSHIELD-BACKEND && color 09 && echo [FEDSHIELD API] Online... && .venv\Scripts\python.exe -m uvicorn backend.app:create_app --factory --host 127.0.0.1 --port 8000"
timeout /t 2 >nul

:: 3. Start Frontend
echo [STARTING] FedShield React Dashboard (Port 3000)...
start "FEDSHIELD-FRONTEND" cmd /c "title FEDSHIELD-FRONTEND && color 0B && cd frontend && echo [FEDSHIELD DASHBOARD] Online... && npm run dev"
timeout /t 3 >nul

:: 4. Start Monitor
echo [STARTING] FedShield Live File Monitor...
start "FEDSHIELD-MONITOR" cmd /c "title FEDSHIELD-MONITOR && color 0C && set PYTHONPATH=%CD% && echo [FEDSHIELD MONITOR] Active... && .venv\Scripts\python.exe scripts/run_monitor.py"
timeout /t 2 >nul

:: 5. Launch Browser
echo [SYSTEM] All systems online.
echo [SYSTEM] Launching dashboard in default browser...
start http://localhost:3000

echo.
echo    =======================================================
echo    FEDSHIELD ACTIVE
echo    =======================================================
echo    [ Dashboard ] http://localhost:3000
echo    [ API ]       http://127.0.0.1:8000/health
echo    =======================================================
echo.
echo    Keep this window open to maintain the system.
echo    Press any key to initiate shutdown sequence.
echo.
pause >nul

:: Stop services if this window is closed/key is pressed
call stop.bat --no-pause
exit
