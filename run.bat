@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Project Singularity - start Ollama, backend :8000, frontend :3000.
REM Run from repo root: double-click run.bat or:  run.bat

cd /d "%~dp0"
set "ROOT=%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ and ensure it is on PATH.
    pause
    exit /b 1
)

where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Install Node.js 18+ and ensure it is on PATH.
    pause
    exit /b 1
)

REM --- Ollama local Gemma -----------------------------------------------------
set "OLLAMA_MODEL=gemma4:latest"
if exist "backend\.env" (
    for /f "usebackq tokens=1,* delims==" %%A in (`findstr /i /b "OLLAMA_MODEL=" "backend\.env"`) do (
        set "OLLAMA_MODEL=%%B"
    )
)
set "OLLAMA_MODEL=!OLLAMA_MODEL:"=!"
for /f "tokens=* delims= " %%A in ("!OLLAMA_MODEL!") do set "OLLAMA_MODEL=%%A"

set "OLLAMA_OK=0"
curl -s --max-time 3 http://localhost:11434/api/tags >nul 2>&1
if not errorlevel 1 set "OLLAMA_OK=1"

if "!OLLAMA_OK!"=="0" (
    where ollama >nul 2>&1
    if errorlevel 1 (
        echo [WARN] Ollama CLI not found on PATH.
        echo        Install from https://ollama.com/download then re-run run.bat
        echo.
    ) else (
        echo [INFO] Ollama not responding - starting ollama serve
        start "Singularity Ollama" cmd /k ollama serve
        call :wait_for_ollama
        if "!OLLAMA_OK!"=="1" echo [OK] Ollama is reachable on :11434
    )
) else (
    echo [OK] Ollama is reachable on :11434
)
goto after_ollama_wait

:wait_for_ollama
set "WAIT=0"
:wait_ollama_loop
timeout /t 2 /nobreak >nul
curl -s --max-time 3 http://localhost:11434/api/tags >nul 2>&1
if not errorlevel 1 (
    set "OLLAMA_OK=1"
    exit /b 0
)
set /a WAIT+=2
if !WAIT! LSS 30 goto wait_ollama_loop
echo [WARN] Ollama did not become ready within 30s. Check the Ollama window.
echo.
exit /b 1

:after_ollama_wait

if "!OLLAMA_OK!"=="1" (
    where ollama >nul 2>&1
    if not errorlevel 1 (
        ollama list 2>nul | findstr /i /c:"!OLLAMA_MODEL!" >nul 2>&1
        if errorlevel 1 (
            echo [INFO] Model !OLLAMA_MODEL! not found locally - pulling now, first run may take a while...
            ollama pull !OLLAMA_MODEL!
            if errorlevel 1 (
                echo [WARN] ollama pull !OLLAMA_MODEL! failed. Edit backend\.env OLLAMA_MODEL or pull manually.
            ) else (
                echo [OK] Model !OLLAMA_MODEL! ready.
            )
        ) else (
            echo [OK] Model !OLLAMA_MODEL! is available.
        )
    )
)

if not exist "backend\main.py" (
    echo [ERROR] backend\main.py not found. Run this script from the repo root.
    pause
    exit /b 1
)

if not exist "frontend\package.json" (
    echo [ERROR] frontend\package.json not found. Run this script from the repo root.
    pause
    exit /b 1
)

echo.
echo  Starting Project Singularity...
echo    Ollama   - http://localhost:11434  local Gemma / !OLLAMA_MODEL!
echo    Backend  - http://localhost:8000  API + SSE
echo    Frontend - http://localhost:3000  dashboard
echo.
echo  First time setup:
echo    cd backend  ^&^&  python -m pip install -r requirements.txt
echo    cd frontend ^&^&  npm install
echo.

set "START_BACKEND=1"
set "START_FRONTEND=1"

call :port_in_use 8000
if not errorlevel 1 (
    echo [WARN] Port 8000 is already in use - skipping a new backend start.
    echo        Reusing the existing server, or run stop.bat then run.bat again.
    set "START_BACKEND=0"
)

call :port_in_use 3000
if not errorlevel 1 (
    echo [WARN] Port 3000 is already in use - skipping a new frontend start.
    echo        Reusing the existing dev server, or run stop.bat then run.bat again.
    set "START_FRONTEND=0"
)

if "!START_BACKEND!"=="1" (
    start "Singularity Backend" cmd /k cd /d "%ROOT%backend" ^&^& python -m uvicorn main:app --host 0.0.0.0 --port 8000
) else (
    echo [INFO] Backend already running on http://localhost:8000
)

if "!START_FRONTEND!"=="1" (
    start "Singularity Frontend" cmd /k cd /d "%ROOT%frontend" ^&^& timeout /t 2 /nobreak ^>nul ^&^& npm run dev
) else (
    echo [INFO] Frontend already running on http://localhost:3000
)

echo.
echo  Open the dashboard: http://localhost:3000
echo  To stop servers: close their windows, or run stop.bat
echo.
pause
exit /b 0

:port_in_use
REM Returns errorlevel 0 if the port is listening, 1 if free.
netstat -ano | findstr ":%~1 " | findstr LISTENING >nul 2>&1
exit /b %errorlevel%
