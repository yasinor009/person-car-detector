@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run install_laptop.bat first.
  pause
  exit /b 1
)
where nvidia-smi >nul 2>&1
if errorlevel 1 (
  echo NVIDIA GPU/driver not detected. Use CPU mode or install the correct NVIDIA driver.
  pause
  exit /b 1
)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
.venv\Scripts\python.exe -m pip uninstall -y onnxruntime
.venv\Scripts\python.exe -m pip install "onnxruntime-gpu[cuda,cudnn]>=1.23,<1.32"
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe diagnose_laptop.py
echo GPU package installation does not prove model compatibility or FPS.
echo Check that the dashboard reports CUDAExecutionProvider after starting a model.
pause
