@echo off
REM FedShield — single-click start (backend 8000 + frontend 3000 + live monitor)
cd /d "%~dp0"
echo Starting FedShield...
start "FedShield Backend" cmd /c ".venv\Scripts\python.exe -m uvicorn backend.app:create_app --factory --host 127.0.0.1 --port 8000"
timeout /t 2 >nul
start "FedShield Frontend" cmd /c "cd frontend && npm run dev"
timeout /t 2 >nul
start "FedShield Monitor" cmd /c "set PYTHONPATH=%CD% && .venv\Scripts\python.exe scripts/run_monitor.py"
timeout /t 2 >nul
echo.
echo FedShield running:
echo   Dashboard  http://localhost:3000
echo   API        http://127.0.0.1:8000/health
echo   Monitor    D:/Telegram  D:/Downloads
echo.
echo Stop with stop.bat or scripts/stop_fedshield.py
pause
