@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run install_laptop.bat first.
  pause
  exit /b 1
)
echo Long-range profile uses the larger general model and image slicing.
echo It does NOT certify detection at 150m or 90 percent precision/recall.
.venv\Scripts\python.exe run_laptop.py --profile long-range %*
pause
