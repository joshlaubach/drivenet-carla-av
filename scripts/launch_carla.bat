@echo off
REM ================================================================
REM  CARLA 0.9.16 Launch Script for DriveNet Data Collection
REM ================================================================
REM  -quality-level=Low   Reduces GPU render load for the viewport
REM                        (camera sensor output is unaffected --
REM                        it renders to its own buffer at full quality)
REM  -fps=20              Caps server frame rate to match notebook FPS=20,
REM                        preventing runaway GPU usage between ticks
REM  -benchmark           Enables fixed-timestep mode for stable
REM                        synchronous data collection
REM  -windowed            Runs in a window instead of fullscreen
REM  -ResX=800 -ResY=600  Small viewport to save GPU memory
REM  -dx12                 Use DirectX 12 to avoid camera sensor deadlocks
REM                        observed with -dx11 on RTX 5080 (Blackwell).
REM                        Keep all scripts on the same RHI flag.
REM ================================================================

set DXGI_GPU_PREFERENCE=2

set CARLA_EXE=%~dp0..\CARLA_0.9.16\CarlaUE4.exe
set CARLA_RHI=-dx12
set CARLA_QUALITY=Low
set CARLA_FPS=20
set CARLA_RES_X=800
set CARLA_RES_Y=600

"%CARLA_EXE%" ^
    %CARLA_RHI% ^
    -quality-level=%CARLA_QUALITY% ^
    -fps=%CARLA_FPS% ^
    -benchmark ^
    -windowed ^
    -ResX=%CARLA_RES_X% ^
    -ResY=%CARLA_RES_Y% ^
    -nosound ^
    -NoSplash
