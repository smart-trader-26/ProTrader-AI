@echo off

echo Starting backend...
start cmd /k uvicorn api.main:app --reload --port 8000

echo Starting frontend...
cd frontend
start cmd /k npm run dev