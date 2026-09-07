@echo off
title Quadrotor Keyboard / Text Control
cd /d "%~dp0"

echo ======================================================================
echo    QUADROTOR NAVIGATION (KEYBOARD / TEXT MODE)
echo ======================================================================
echo.
echo Starting Live 3D Simulation + Mission-Control Telemetry HUD...
echo Type commands in this window and press ENTER:
echo   - forward
echo   - back
echo   - left
echo   - right
echo   - hover
echo   - stop
echo.

.venv\Scripts\python.exe 05_voice_interface\demo.py --source text

pause
