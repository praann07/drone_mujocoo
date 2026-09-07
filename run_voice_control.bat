@echo off
title Quadrotor Live Voice Control
cd /d "%~dp0"

echo ======================================================================
echo    VOICE-CONTROLLED QUADROTOR NAVIGATION (MUJOCO + SINDy)
echo ======================================================================
echo.
echo Checking virtual environment...
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    echo Please run 'python -m venv .venv' and install requirements first.
    pause
    exit /b 1
)

echo Starting Live 3D Simulation + Mission-Control Telemetry HUD...
echo Speak commands: "forward", "back", "left", "right", "hover", "stop"
echo.

.venv\Scripts\python.exe 05_voice_interface\demo.py --source vosk --model-dir .\vosk-model-small-en-us-0.15

if errorlevel 1 (
    echo.
    echo ======================================================================
    echo [NOTE] If you saw a PortAudio / Microphone error:
    echo   1. Make sure your microphone or Bluetooth earbuds are CONNECTED.
    echo   2. You can also test without a mic by running 'run_text_control.bat'
    echo ======================================================================
    echo.
    pause
)
