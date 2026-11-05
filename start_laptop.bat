@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run install_laptop.bat first.
  pause
  exit /b 1
)
echo Open http://127.0.0.1:5000 after the server starts.
.venv\Scripts\python.exe run_laptop.py --profile balanced %*
pause
