@echo off
setlocal EnableExtensions

REM Project Singularity — start backend (FastAPI :8000) and frontend (Vite :3000).
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
echo    Backend  - http://localhost:8000  (API + SSE)
echo    Frontend - http://localhost:3000  (dashboard; proxies /api to backend)
echo.
echo  Prerequisites: Ollama with your model pulled (see backend\.env OLLAMA_MODEL).
echo  First time: install deps with setup.bat or manually:
echo    cd backend  ^&^&  python -m pip install -r requirements.txt
echo    cd frontend ^&^&  npm install
echo.

REM Backend — uvicorn from backend/ so imports and .env resolve correctly.
start "Singularity Backend" cmd /k cd /d "%ROOT%backend" ^&^& python -m uvicorn main:app --host 0.0.0.0 --port 8000

REM Brief delay so the API is up before the dev server opens.
start "Singularity Frontend" cmd /k cd /d "%ROOT%frontend" ^&^& timeout /t 2 /nobreak ^>nul ^&^& npm run dev

echo  Two terminal windows were opened. Close them to stop the servers.
echo  Open the dashboard: http://localhost:3000
echo.
pause
