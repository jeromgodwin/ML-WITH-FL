@echo off
setlocal EnableDelayedExpansion
title FEDSHIELD - SHUTDOWN
color 0C

echo.
echo    =======================================================
echo    FEDSHIELD SYSTEM SHUTDOWN
echo    =======================================================
echo.

echo [SYSTEM] Terminating active connections...
for %%P in (8000 3000) do (
  for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%%P') do (
      taskkill /F /PID %%a >nul 2>&1
  )
)

echo [SYSTEM] Terminating background processes...
taskkill /F /FI "WINDOWTITLE eq FEDSHIELD-MONITOR*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq FEDSHIELD-BACKEND*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq FEDSHIELD-FRONTEND*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq FedShield Monitor*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq FedShield Backend*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq FedShield Frontend*" >nul 2>&1

echo [SYSTEM] Shutdown complete.
echo.

if "%1"=="--no-pause" goto :EOF

pause
