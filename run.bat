@echo off
setlocal

REM ProTrader AI - dev launcher
REM Starts the FastAPI backend (inside the Python 3.11 .venv) and the Next.js frontend.
REM The frontend at http://localhost:3000 is the primary app for testing/use.

set "ROOT=%~dp0"

REM --- Make sure the Python 3.11 venv exists -------------------------------
if not exist "%ROOT%.venv\Scripts\activate.bat" (
    echo [ERROR] Python venv not found at "%ROOT%.venv"
    echo.
    echo Create it once with:
    echo     "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" -m venv .venv
    echo     .venv\Scripts\python -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo Starting backend  (FastAPI  -^> http://localhost:8000) ...
start "ProTrader Backend" cmd /k "cd /d "%ROOT%" && call ".venv\Scripts\activate.bat" && uvicorn api.main:app --reload --port 8000"

echo Starting frontend (Next.js  -^> http://localhost:3000) ...
start "ProTrader Frontend" cmd /k "cd /d "%ROOT%frontend" && npm run dev"

echo.
echo Both started in separate windows. Close those windows to stop the servers.
endlocal
