@echo off
REM Double-click this file. No menu, no typing. STEP 1 of the presentation:
REM opens the commands guide, the live 3D demo, and the test suite - the
REM 3 things you need WHILE you're flying. After you land, double-click
REM SHOW_RESULTS.bat for a fresh graph of what you just did.
cd /d "%~dp0"

echo ============================================================
echo   Voice-Controlled Quadrotor - RUN EVERYTHING (step 1 of 2)
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    echo Run 'python -m venv .venv' and 'pip install -r requirements.txt' first.
    pause
    exit /b 1
)

set DEMO_CMD=.venv\Scripts\python.exe 05_voice_interface\demo.py --source text
if exist "vosk-model-small-en-us-0.15" (
    set DEMO_CMD=.venv\Scripts\python.exe 05_voice_interface\demo.py --source vosk --model-dir vosk-model-small-en-us-0.15
    echo Voice model found - launching with microphone control.
) else (
    echo [NOTE] vosk-model-small-en-us-0.15 not found - launching in TYPE
    echo mode instead - type commands into the demo window. Download the
    echo model from https://alphacephei.com/vosk/models for voice control.
)
echo.

echo Opening 3 windows - give each a second to appear:
echo   1. Commands Guide  (stays open - keep this visible while presenting)
echo   2. Drone 3D Demo   (the live sim - speak/type/click here)
echo   3. Test Suite      (pytest, proves everything still passes)
echo.
echo When you're done flying: double-click SHOW_RESULTS.bat for a fresh
echo graph of the flight you just did.
echo.

start "Commands Guide" cmd /k ".venv\Scripts\python.exe 05_voice_interface\print_commands_guide.py"
timeout /t 2 >nul
start "Drone 3D Demo (speak/type/click HERE)" cmd /k "%DEMO_CMD%"
timeout /t 2 >nul
start "Test Suite" cmd /k ".venv\Scripts\python.exe -m pytest tests\ -v"

echo All 3 windows launched. This launcher window can be closed now.
echo.
pause >nul
