@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run install_laptop.bat first.
  pause
  exit /b 1
)
echo Aerial-trained model: intended for overhead/drone views, not validated on your camera.
.venv\Scripts\python.exe run_laptop.py --profile aerial %*
pause
