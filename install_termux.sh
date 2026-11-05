#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p models

if ! command -v pkg >/dev/null 2>&1; then
  echo "ERROR: This installer must be run inside Termux on Android." >&2
  exit 1
fi

echo "[1/3] Installing Termux packages..."
pkg update
pkg install -y python python-numpy opencv python-onnxruntime curl coreutils

echo "[2/3] Checking Python modules..."
python - <<'PY'
import cv2, numpy, onnxruntime
print('OpenCV:', cv2.__version__)
print('NumPy:', numpy.__version__)
print('ONNX Runtime:', onnxruntime.__version__)
PY

echo "[3/3] Preparing mobile ground-camera model..."
python download_models.py --profile mobile
python - <<'CHECK'
from laptop_detector import LaptopDetector
model = LaptopDetector('models/person_car_yolo26n.onnx',imgsz=640,device='cpu',include_trucks=True)
if model.model_status != 'ready':
    raise SystemExit(model.model_message)
print(model.model_message)
CHECK

echo "Install complete. Start: python main.py --browser-camera"
echo "Long-range (slower): python main.py --browser-camera --profile long-range"
echo "Open http://127.0.0.1:5000 on this phone and allow camera access."
