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

echo Checking voice model...
if not exist "vosk-model-small-en-us-0.15" (
    echo [ERROR] vosk-model-small-en-us-0.15 folder not found in this project.
    echo Download it from https://alphacephei.com/vosk/models and unzip it
    echo into this same folder, or run 'run_text_control.bat' instead ^(no
    echo mic needed - type commands into the window^).
    pause
    exit /b 1
)

echo Starting Live 3D Simulation + Telemetry Dashboard...
echo.
echo Speak: forward / back / left / right / up / down / hover / stop
echo   - Every word you say shows up on screen, recognized or not.
echo   - Click TOGGLE CAMERA in the dashboard window for bird's-eye view.
echo   - Click any command button in the dashboard as a backup if a word
echo     doesn't get recognized.
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
