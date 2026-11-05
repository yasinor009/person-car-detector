"""Runtime configuration. Hardware sources and model path can be overridden with env vars."""
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent

# The dashboard intentionally binds to localhost by default.
WEB_HOST = os.getenv("PERSON_CAR_WEB_HOST", os.getenv("DRONE_WEB_HOST", "127.0.0.1"))
WEB_PORT = int(os.getenv("PERSON_CAR_WEB_PORT", os.getenv("DRONE_WEB_PORT", "5000")))

# Use "0" for the first local camera, or an RTSP/HTTP URL for a network camera.
VISIBLE_CAMERA_SOURCE = os.getenv("PERSON_CAR_VISIBLE_SOURCE", os.getenv("DRONE_VISIBLE_SOURCE", "0"))
# Leave empty if there is no separate thermal stream.
THERMAL_CAMERA_SOURCE = os.getenv("PERSON_CAR_THERMAL_SOURCE", os.getenv("DRONE_THERMAL_SOURCE", ""))

# COCO YOLO11 raw ONNX; decoder permits only person=0 and car=2.
MODEL_PATH = os.getenv("PERSON_CAR_MODEL_PATH", os.getenv("DRONE_MODEL_PATH", str(BASE_DIR / "models" / "person_car_yolo11n.onnx")))
TARGET_CLASS_NAMES = ("person", "car")
CONFIDENCE_THRESHOLD = float(os.getenv("PERSON_CAR_CONFIDENCE", os.getenv("DRONE_CONFIDENCE", "0.60")))
# SAHI-style optional full image + four overlapping slices (up to 5 inferences).
# Default off on phones. Enable only after measuring speed and false positives.
PERSON_CAR_TILED_INFERENCE = os.getenv("PERSON_CAR_TILED", "0").strip().lower() in ("1", "true", "yes", "on")
DETECTION_INTERVAL_SECONDS = 0.10
MAX_FRAME_WIDTH = 1280

# Legacy thermal helper settings; thermal candidates are excluded from live detections.
THERMAL_ENABLED = False
THERMAL_THRESHOLD_STD = 1.5
THERMAL_MIN_AREA_PX = 12
THERMAL_MAX_AREA_RATIO = 0.15
THERMAL_MIN_ASPECT_RATIO = 0.25
THERMAL_MAX_ASPECT_RATIO = 4.0

# Distance is disabled until the camera is calibrated and the target's real width is known.
DEFAULT_TARGET_WIDTH_M = None
DEFAULT_FOCAL_LENGTH_PX = None

ALARM_ENABLED = True
ALARM_COOLDOWN_SECONDS = 10.0
ALARM_DURATION_SECONDS = 3.0
ALARM_SOUND_FILE = str(BASE_DIR / "alarm_sound.wav")

SAVE_DETECTION_IMAGES = True
DETECTION_LOG_DIR = BASE_DIR / "detections"
LOG_DIR = BASE_DIR / "logs"
SETTINGS_FILE = BASE_DIR / "data" / "settings.json"
