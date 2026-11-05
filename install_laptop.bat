@echo off
cd /d "%~dp0"
where py >nul 2>&1
if errorlevel 1 (
  echo Install Python 3.11 or 3.12 from python.org first.
  pause
  exit /b 1
)
if not exist .venv\Scripts\python.exe (
  py -3.11 -m venv .venv
  if errorlevel 1 py -3.12 -m venv .venv
  if errorlevel 1 exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe verify_models.py
if errorlevel 1 exit /b 1
echo CPU installation complete. Run start_laptop.bat.
echo Optional NVIDIA GPU setup: run install_nvidia.bat.
pause
