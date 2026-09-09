@echo off
REM Double-click this to regenerate EVERYTHING under results/ from scratch -
REM the full Stage A-E pipeline, the 67-maneuver gauntlet, then collects
REM every plot/CSV/GIF into results/ (matching README.md's gallery and
REM results/RESULTS.md). Takes several minutes - this is a "rebuild the
REM evidence" utility, not something you run before every presentation;
REM RUN_EVERYTHING.bat / SHOW_RESULTS.bat are for that.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    echo Run 'python -m venv .venv' and 'pip install -r requirements.txt' first.
    pause
    exit /b 1
)

echo ============================================================
echo   Rebuilding ALL results - this takes several minutes
echo ============================================================
echo Each step's output is saved to results\logs\ as it runs.
echo.

if not exist "results\logs" mkdir "results\logs"
set PY=.venv\Scripts\python.exe

echo [1/10] Stage A - excitation dataset + gate plots...
%PY% 01_simulation\run_excitation.py > results\logs\excitation.log 2>&1
if errorlevel 1 (echo   FAILED - see results\logs\excitation.log & goto :error)

echo [2/10] Stage B - SINDy/DMDc identification...
%PY% 02_identification\run_identification.py > results\logs\identification.log 2>&1
if errorlevel 1 (echo   FAILED - see results\logs\identification.log & goto :error)

echo [3/10] Stage C - rollouts, ablations, model freeze...
%PY% 03_validation\run_validation.py > results\logs\validation.log 2>&1
if errorlevel 1 (echo   FAILED - see results\logs\validation.log & goto :error)

echo [4/10] Stage D - inner attitude loop...
%PY% 04_control\run_inner_loop.py > results\logs\inner_loop.log 2>&1
if errorlevel 1 (echo   FAILED - see results\logs\inner_loop.log & goto :error)

echo [5/10] Stage D - outer cascade...
%PY% 04_control\run_cascade.py > results\logs\cascade.log 2>&1
if errorlevel 1 (echo   FAILED - see results\logs\cascade.log & goto :error)

echo [6/10] Stage E - isolated (24) + chained (5) trials...
%PY% 05_voice_interface\run_trials.py > results\logs\trials.log 2>&1
if errorlevel 1 (echo   FAILED - see results\logs\trials.log & goto :error)

echo [7/10] Stage E - navigation + preemption GIFs...
%PY% 05_voice_interface\render_offscreen.py > results\logs\render_nav.log 2>&1
%PY% 05_voice_interface\render_offscreen.py --preempt > results\logs\render_preempt.log 2>&1
%PY% 05_voice_interface\run_preemption.py > results\logs\preemption.log 2>&1

echo [8/10] Maneuver gauntlet - 67 maneuvers (this is the slow one)...
%PY% run_gauntlet.py > results\logs\gauntlet.log 2>&1
if errorlevel 1 (echo   FAILED - see results\logs\gauntlet.log & goto :error)

echo [9/10] Collecting everything into results\...
%PY% save_plots.py > results\logs\save_plots.log 2>&1
type results\logs\save_plots.log

echo [10/10] Running the full test suite...
%PY% -m pytest tests\ -v > results\logs\pytest.log 2>&1
if errorlevel 1 (echo   Some tests FAILED - see results\logs\pytest.log) else (echo   All tests passed.)

echo.
echo ============================================================
echo   Done. Check results\RESULTS.md and results\logs\ for details.
echo ============================================================
pause
exit /b 0

:error
echo.
echo Rebuild stopped early - fix the error above and rerun.
pause
exit /b 1
