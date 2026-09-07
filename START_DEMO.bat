@echo off
REM Double-click this file to launch the live 3D drone demo.
REM Opens two windows: the MuJoCo 3D viewer, and the telemetry dashboard
REM (stats charts + click-to-fly buttons). Control the drone by:
REM   - clicking a button in the dashboard window, OR
REM   - typing a command word into THIS console window and pressing Enter
REM     (forward / back / left / right / hover / stop), OR
REM   - choosing option 2 below for real microphone voice control.
cd /d "%~dp0"

echo ============================================================
echo   Voice-Controlled Quadrotor - Live 3D Demo
echo ============================================================
echo.
echo   1 = Type + Buttons  (type commands here, AND click buttons - both live)
echo   2 = Voice + Buttons (speak into mic, AND click buttons - both live)
echo   3 = Run full test suite
echo   4 = Run headless trial gate (24 isolated + 5 chained flights)
echo   5 = LAUNCH EVERYTHING (demo + both plots + test suite, all at once)
echo.
set /p MODE="Choose 1, 2, 3, 4, or 5: "

if "%MODE%"=="1" (
    echo.
    echo Launching text/button demo. Two windows will open: the 3D
    echo viewer and the telemetry dashboard. Type a command below and
    echo press Enter, or click a button in the dashboard window.
    echo.
    .venv\Scripts\python.exe 05_voice_interface\demo.py --source text
) else if "%MODE%"=="2" (
    echo.
    echo Launching voice demo. Speak "forward", "back", "left", "right",
    echo "hover", or "stop" into your microphone - the dashboard's click
    echo buttons are ALSO live at the same time, as a backup if a word
    echo doesn't get recognized.
    echo.
    if not exist "vosk-model-small-en-us-0.15" (
        echo ERROR: vosk-model-small-en-us-0.15 folder not found.
        echo Download it from https://alphacephei.com/vosk/models and
        echo unzip it into this project folder first.
        pause
        exit /b 1
    )
    .venv\Scripts\python.exe 05_voice_interface\demo.py --source vosk --model-dir vosk-model-small-en-us-0.15
) else if "%MODE%"=="3" (
    echo.
    .venv\Scripts\python.exe -m pytest tests\ -v
) else if "%MODE%"=="4" (
    echo.
    .venv\Scripts\python.exe 05_voice_interface\run_trials.py
) else if "%MODE%"=="5" (
    echo.
    echo Launching everything - each opens in its OWN window so they run
    echo at the same time, not one after another:
    echo   1. Live 3D demo + dashboard  (type commands in ITS window)
    echo   2. 2D telemetry plot          (shows most recent logged flight)
    echo   3. 3D trajectory plot         (shows most recent chained flight)
    echo   4. Full test suite            (runs in the background, closes when done)
    echo.
    echo NOTE: the two plot windows read the LAST logged flight. If you
    echo haven't flown yet, they'll say "no logged flight data found" -
    echo fly with the demo window first, then rerun them via the commands
    echo in README.md if you want fresh plots.
    echo.
    start "Drone 3D Demo (type commands HERE)" cmd /k ".venv\Scripts\python.exe 05_voice_interface\demo.py --source text"
    timeout /t 2 >nul
    start "2D Telemetry Plot" cmd /k ".venv\Scripts\python.exe 05_voice_interface\plot_2d_telemetry.py"
    start "3D Trajectory Plot" cmd /k ".venv\Scripts\python.exe 05_voice_interface\plot_3d_trajectory.py"
    start "Test Suite" cmd /k ".venv\Scripts\python.exe -m pytest tests\ -v"
    echo All four windows launched. This launcher window can be closed now.
) else (
    echo Invalid choice.
)

echo.
echo ============================================================
echo   Done. Press any key to close this window.
echo ============================================================
pause >nul
