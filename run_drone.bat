@echo off
title Quadrotor Flight System Launcher
cd /d "%~dp0"

:MENU
cls
echo ======================================================================
echo       VOICE-CONTROLLED QUADROTOR NAVIGATION (MUJOCO + SINDy)
echo ======================================================================
echo.
echo   [1] Launch Live Voice Control (Microphone)
echo   [2] Launch Keyboard / Text Control (Type commands)
echo   [3] Run Full Validation Suite ^& 24+5 Flight Trials
echo   [4] Run Pytest Test Suite (28 Unit Tests)
echo   [5] Exit
echo.
set /p choice="Enter choice [1-5]: "

if "%choice%"=="1" (
    cls
    call run_voice_control.bat
    goto MENU
)
if "%choice%"=="2" (
    cls
    call run_text_control.bat
    goto MENU
)
if "%choice%"=="3" (
    cls
    echo Running Stage E Isolated (24/24) and Chained (5/5) Trials...
    .venv\Scripts\python.exe 05_voice_interface\run_trials.py
    echo.
    pause
    goto MENU
)
if "%choice%"=="4" (
    cls
    echo Running full pytest suite...
    .venv\Scripts\python.exe -m pytest tests\
    echo.
    pause
    goto MENU
)
if "%choice%"=="5" (
    exit /b 0
)

echo Invalid selection. Please try again.
pause
goto MENU
