"""Minimal, thread-safe camera source manager for visible and optional thermal streams."""
from __future__ import annotations
import logging
import threading
import cv2
import config


def parse_camera_source(value):
    """Convert numeric camera IDs to int; preserve URLs and device paths as strings."""
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return text


class CameraInterface:
    def __init__(self, visible_source=None, thermal_source=None):
        self.visible_source = parse_camera_source(
            config.VISIBLE_CAMERA_SOURCE if visible_source is None else visible_source
        )
        self.thermal_source = parse_camera_source(
            config.THERMAL_CAMERA_SOURCE if thermal_source is None else thermal_source
        )
        self.visible_cap = None
        self.thermal_cap = None
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _open(source):
        if source is None:
            return None
        cap = cv2.VideoCapture(source)
        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def connect(self):
        with self.lock:
            self.disconnect()
            self.visible_cap = self._open(self.visible_source)
            self.thermal_cap = self._open(self.thermal_source)
            visible_ok = self.visible_cap is not None
            thermal_ok = self.thermal_cap is not None
            if not visible_ok and not thermal_ok:
                self.logger.error("No camera source opened. Check DRONE_VISIBLE_SOURCE / DRONE_THERMAL_SOURCE.")
                return False
            self.logger.info("Camera sources opened: visible=%s thermal=%s", visible_ok, thermal_ok)
            return True

    def read_frames(self):
        with self.lock:
            visible = self._read_one(self.visible_cap)
            thermal = self._read_one(self.thermal_cap)
            return visible, thermal

    def _read_one(self, cap):
        if cap is None:
            return None
        try:
            ok, frame = cap.read()
            return frame if ok and frame is not None and getattr(frame, "size", 0) else None
        except Exception as exc:
            self.logger.warning("Camera frame read failed: %s", exc)
            return None

    def status(self):
        with self.lock:
            return {
                "visible_configured": self.visible_source is not None,
                "thermal_configured": self.thermal_source is not None,
                "visible_open": self.visible_cap is not None and self.visible_cap.isOpened(),
                "thermal_open": self.thermal_cap is not None and self.thermal_cap.isOpened(),
                "visible_source": "local camera" if isinstance(self.visible_source, int) else ("configured" if self.visible_source else None),
                "thermal_source": "configured" if self.thermal_source else None,
            }

    def disconnect(self):
        with self.lock:
            for attr in ("visible_cap", "thermal_cap"):
                cap = getattr(self, attr, None)
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    setattr(self, attr, None)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.disconnect()
