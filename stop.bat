@echo off
setlocal EnableExtensions

REM Stop Singularity backend (:8000) and frontend (:3000) if listening.
cd /d "%~dp0"

echo.
echo  Stopping Singularity servers...
echo.

call :kill_port 8000 "Backend (uvicorn)"
call :kill_port 3000 "Frontend (Vite)"

echo.
echo  Done. If Ollama was started by run.bat, close its window separately.
echo.
pause
exit /b 0

:kill_port
set "PORT=%~1"
set "LABEL=%~2"
set "FOUND=0"

for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%PORT% " ^| findstr LISTENING') do (
    set "FOUND=1"
    set "PID=%%P"
    for /f "tokens=1" %%N in ('tasklist /FI "PID eq %%P" /FO LIST ^| findstr "Image Name:"') do set "IMG=%%N"
    echo  [%LABEL%] port %PORT% in use by PID %%P - terminating...
    taskkill /PID %%P /F >nul 2>&1
    if errorlevel 1 (
        echo  [WARN] Could not stop PID %%P. Try closing the server window manually.
    ) else (
        echo  [OK] Stopped PID %%P
    )
)

if "%FOUND%"=="0" echo  [%LABEL%] nothing listening on port %PORT%
exit /b 0
