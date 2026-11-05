"""Thermal image enhancement and hot-region candidate extraction."""
from __future__ import annotations
import logging
import cv2
import numpy as np
import config


class ThermalProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.background_model = None

    def preprocess(self, frame):
        if frame is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if gray.size:
            gray = cv2.bilateralFilter(gray, 5, 35, 35)
        return gray

    def enhance(self, frame):
        gray = self.preprocess(frame)
        if gray is None:
            return None
        if int(gray.max()) == int(gray.min()):
            normalized = np.zeros_like(gray)
        else:
            normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return cv2.applyColorMap(clahe.apply(normalized), cv2.COLORMAP_INFERNO)

    def detect_hot_objects(self, frame):
        """Return heat/anomaly candidates; these are not person/car classifications."""
        gray = self.preprocess(frame)
        if gray is None or gray.size == 0:
            return []
        mean = float(np.mean(gray))
        std = float(np.std(gray))
        if std < 1e-6:
            return []
        threshold = min(255.0, mean + config.THERMAL_THRESHOLD_STD * std)
        binary = (gray >= threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = gray.shape[:2]
        max_area = max(config.THERMAL_MIN_AREA_PX, int(h * w * config.THERMAL_MAX_AREA_RATIO))
        results = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < config.THERMAL_MIN_AREA_PX or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if bh <= 0:
                continue
            aspect = bw / bh
            if not config.THERMAL_MIN_ASPECT_RATIO <= aspect <= config.THERMAL_MAX_ASPECT_RATIO:
                continue
            roi = gray[y:y + bh, x:x + bw]
            # Confidence is an anomaly score, not a person/car probability.
            anomaly = max(0.0, (float(np.mean(roi)) - mean) / max(std, 1e-6))
            confidence = min(0.99, anomaly / 4.0)
            results.append({
                "bbox": (int(x), int(y), int(bw), int(bh)),
                "confidence": round(confidence, 3),
                "class": "thermal_candidate",
                "source": "thermal",
                "area_px": round(area, 1),
                "mean_intensity": round(float(np.mean(roi)), 1),
                "classification_note": "Thermal anomaly only; not confirmed as person or car.",
            })
        return sorted(results, key=lambda item: item["confidence"], reverse=True)
