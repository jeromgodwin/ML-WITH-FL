@echo off
REM FedShield — single-click stop
echo Stopping FedShield...
for %%P in (8000 3000) do (
  for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%%P') do taskkill /F /PID %%a >nul 2>&1
)
taskkill /F /FI "WINDOWTITLE eq FedShield Monitor*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq FedShield Backend*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq FedShield Frontend*" >nul 2>&1
echo Stopped.
pause
