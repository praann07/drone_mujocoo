@echo off
REM Double-click this file AFTER you've flown at least once in the demo
REM window (RUN_EVERYTHING.bat). Opens fresh 2D + 3D graphs of the flight
REM you just did - the demo logs every command you fly to a file these
REM read automatically, so this always reflects your MOST RECENT flight,
REM not an old test run.
cd /d "%~dp0"

echo ============================================================
echo   Voice-Controlled Quadrotor - SHOW RESULTS (step 2 of 2)
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    pause
    exit /b 1
)

echo Opening 2 windows with graphs of your most recent flight:
echo   1. 2D Telemetry Plot  - position/attitude error, latency
echo   2. 3D Trajectory Plot - the flight path over the city
echo.
echo If you haven't flown yet, these will say "no logged flight data
echo found" - fly first in the demo window (RUN_EVERYTHING.bat), then
echo come back and run this again.
echo.

start "2D Telemetry Plot" cmd /k ".venv\Scripts\python.exe 05_voice_interface\plot_2d_telemetry.py"
timeout /t 2 >nul
start "3D Trajectory Plot" cmd /k ".venv\Scripts\python.exe 05_voice_interface\plot_3d_trajectory.py"

echo Both windows launched. This launcher window can be closed now.
echo.
pause >nul
