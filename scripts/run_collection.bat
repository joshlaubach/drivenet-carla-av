@echo off
REM ================================================================
REM  Auto-restart wrapper for CARLA data collection.
REM  Relaunches CARLA + runs the notebook collection script in a loop
REM  until all 324 conditions are complete. Relies on the notebook's
REM  checkpoint system to resume from where it left off each time.
REM ================================================================

set DXGI_GPU_PREFERENCE=2
set CARLA_EXE=%~dp0..\CARLA_0.9.16\CarlaUE4.exe
set COLLECT_SCRIPT=%~dp0..\scripts\collect.py
set CHECKPOINT=%~dp0..\data\checkpoint.json

:LOOP
echo.
echo ============================================
echo  Starting CARLA server...
echo ============================================
start "" "%CARLA_EXE%" ^
    -dx12 ^
    -quality-level=Low ^
    -fps=20 ^
    -benchmark ^
    -windowed ^
    -ResX=800 ^
    -ResY=600 ^
    -nosound

REM Wait for CARLA to finish loading
echo Waiting 30s for CARLA to start...
timeout /t 30 /nobreak >nul

echo Running data collection...
"%~dp0..\.venv\Scripts\python.exe" "%COLLECT_SCRIPT%"

echo.
echo CARLA or collection script exited.

REM Check if all conditions are done
if exist "%CHECKPOINT%" (
    for /f "tokens=2 delims=:," %%a in ('findstr "conditions_completed" "%CHECKPOINT%"') do (
        set /a DONE=%%a
    )
    echo Completed %DONE%/324 conditions so far.
    if %DONE% GEQ 324 (
        echo All conditions complete!
        goto END
    )
)

echo Killing leftover CARLA processes...
taskkill /F /IM CarlaUE4-Win64-Shipping.exe >nul 2>&1
taskkill /F /IM CarlaUE4.exe >nul 2>&1
timeout /t 5 /nobreak >nul

goto LOOP

:END
echo.
echo ============================================
echo  Data collection finished!
echo ============================================
pause
