"""Local web dashboard and processing service for drone detection and metric range estimation.

Run with: python main.py --demo
The live mode uses a camera source configured by environment variables and a drone-trained model.
"""
from __future__ import annotations
import argparse
import json
from collections import deque
import logging
import math
import os
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

import config
from alarm_system import AlarmSystem
from camera_interface import CameraInterface
from distance_estimator import DistanceEstimator
from camera_geometry import LensRectifier
from drone_detector import DroneDetector
from motion_detector import MotionDetector
from thermal_processor import ThermalProcessor


LOG = logging.getLogger("drone_distance")


def _load_json(path: Path, fallback: dict) -> dict:
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        LOG.exception("Could not read settings file %s", path)
    return dict(fallback)


class DroneDistanceService:
    def __init__(self, demo: bool = False, browser_camera: bool = False, visible_source=None, thermal_source=None,
                 model_path=None, camera=None, detector=None, alarm=None):
        self.demo = bool(demo)
        self.browser_camera = bool(browser_camera) and not self.demo
        self.camera = camera or CameraInterface(visible_source, thermal_source)
        self.browser_frame = None
        self.browser_frame_received_at = 0.0
        self.detector = detector or DroneDetector(model_path=model_path)
        self.rectifier = LensRectifier(os.getenv("CAMERA_LENS_CALIBRATION") or None)
        from metric_range import GroundCalibration, RangeBank
        self.metric_ranges = RangeBank()
        ground_path = os.getenv("CAMERA_GROUND_CALIBRATION")
        self.ground_calibration = GroundCalibration(**json.loads(Path(ground_path).read_text())) if ground_path else None
        if self.ground_calibration:
            if self.rectifier.calibration is None:
                raise ValueError('Ground ranging requires CAMERA_LENS_CALIBRATION')
            k, _, size = self.rectifier.calibration
            if list(size) != self.ground_calibration.image_size or not np.allclose(k, self.ground_calibration.k):
                raise ValueError('Ground and lens calibration must have identical K and resolution')
        # Generic motion segmentation complements the drone-only model. It finds moving regions,
        # not semantic classes such as person/car, and works best with a steady camera.
        self.motion_detector = MotionDetector()
        self.thermal_processor = ThermalProcessor()
        self.alarm = alarm or AlarmSystem()
        self.lock = threading.RLock()
        self.running = False
        self.worker = None
        self.shutdown_event = threading.Event()
        self.started_at = time.time()
        self.frame_count = 0
        self.last_error = None
        self.latest_visible = None
        self.latest_raw_visible = None
        self.latest_frame_size = None
        self.latest_thermal = None
        self.latest_detections = []
        # User-selected object: a tap selects a detected box; dragging selects an arbitrary ROI.
        self.selected_target = None
        self.selected_target_width_m = None
        self._selected_width_samples = deque(maxlen=12)
        self._tracking_prev_gray = None
        self._tracking_points = None
        self._tracking_template = None
        self._tracking_frames = 0
        self._tracking_misses = 0
        # Automatic single-target mode: suppress scattered motion blobs and acquire
        # one persistent candidate before drawing a box. Manual tap/drag selection
        # still overrides this target.
        self._auto_candidate_bbox = None
        self._auto_candidate_streak = 0
        self._last_snapshot_at = 0.0
        self.last_detection_at = None
        self.last_frame_at = None
        self.settings = _load_json(config.SETTINGS_FILE, {
            "confidence_threshold": config.CONFIDENCE_THRESHOLD,
            "target_width_m": config.DEFAULT_TARGET_WIDTH_M,
            "focal_length_px": config.DEFAULT_FOCAL_LENGTH_PX,
            "thermal_enabled": config.THERMAL_ENABLED,
            "alarm_enabled": config.ALARM_ENABLED,
        })
        # v2.4 fixed YOLOv5 raw-output decoding; v2.5 adds vectorized inference and
        # runtime diagnostics. Align only the old untouched default
        # (0.45) with the model publisher's recommended starting threshold (0.25).
        if "model_decoder_version" not in self.settings:
            try:
                old_conf = float(self.settings.get("confidence_threshold", config.CONFIDENCE_THRESHOLD))
                if abs(old_conf - 0.45) < 1e-6:
                    self.settings["confidence_threshold"] = config.CONFIDENCE_THRESHOLD
            except (TypeError, ValueError):
                self.settings["confidence_threshold"] = config.CONFIDENCE_THRESHOLD
        if isinstance(self.settings.get("alarm_enabled"), bool):
            self.alarm.set_enabled(self.settings["alarm_enabled"])
        self.distance = DistanceEstimator(
            focal_length_px=self.settings.get("focal_length_px"),
            target_width_m=self.settings.get("target_width_m"),
            calibration_frame_width_px=self.settings.get("calibration_frame_width_px"),
        )
        try:
            self.detector.set_confidence(float(self.settings.get("confidence_threshold", config.CONFIDENCE_THRESHOLD)))
        except Exception:
            self.detector.set_confidence(config.CONFIDENCE_THRESHOLD)
        self._save_settings()

    def _save_settings(self):
        config.SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "confidence_threshold": self.detector.confidence,
            "target_width_m": self.distance.target_width_m,
            "focal_length_px": self.distance.focal_length_px,
            "calibration_frame_width_px": self.distance.calibration_frame_width_px,
            "thermal_enabled": bool(self.settings.get("thermal_enabled", True)),
            "alarm_enabled": bool(self.alarm.enabled),
            "model_decoder_version": 3,
        }
        tmp = config.SETTINGS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(config.SETTINGS_FILE)
        self.settings = payload

    def update_settings(self, data: dict):
        if not isinstance(data, dict):
            raise ValueError("Settings payload must be a JSON object.")
        if "confidence_threshold" in data:
            self.detector.set_confidence(float(data["confidence_threshold"]))
        if "target_width_m" in data:
            self.distance.set_target_width(data["target_width_m"])
        if "thermal_enabled" in data:
            if not isinstance(data["thermal_enabled"], bool):
                raise ValueError("thermal_enabled must be true or false.")
            self.settings["thermal_enabled"] = data["thermal_enabled"]
        if "alarm_enabled" in data:
            if not isinstance(data["alarm_enabled"], bool):
                raise ValueError("alarm_enabled must be true or false.")
            self.alarm.set_enabled(data["alarm_enabled"])
        self._save_settings()
        return self.status()

    def calibrate_distance(self, known_width_m, known_distance_m, observed_width_px, frame_width_px=None):
        if frame_width_px in (None, "") and self.latest_frame_size:
            frame_width_px = self.latest_frame_size.get("width")
        focal = self.distance.calibrate(
            known_width_m, known_distance_m, observed_width_px, frame_width_px=frame_width_px
        )
        self._save_settings()
        return {"focal_length_px": round(focal, 3), "distance": self.distance.to_dict()}

    def reset_calibration(self):
        self.distance = DistanceEstimator(target_width_m=self.distance.target_width_m)
        self._save_settings()
        self._update_selected_distance()
        return self.distance.to_dict()

    def start_detection(self):
        with self.lock:
            if self.running:
                return True, "Detection is already running."
            if not self.demo and not self.browser_camera:
                if not self.camera.connect():
                    self.last_error = "No camera stream opened. Check the configured source and camera availability."
                    return False, self.last_error
            self.last_error = None
            self.shutdown_event.clear()
            self.running = True
            target = self._demo_loop if self.demo else (self._browser_loop if self.browser_camera else self._live_loop)
            self.worker = threading.Thread(target=target, name="detection-loop", daemon=True)
            self.worker.start()
            return True, "Demo stream started." if self.demo else "Detection started."

    def stop_detection(self):
        with self.lock:
            if not self.running:
                if not self.demo and not self.browser_camera:
                    self.camera.disconnect()
                return True, "Detection is already stopped."
            self.running = False
            self.shutdown_event.set()
            worker = self.worker
        if worker and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=3)
        if not self.demo and not self.browser_camera:
            self.camera.disconnect()
        return True, "Detection stopped."

    def shutdown(self):
        self.stop_detection()
        self.alarm.stop_alarm()

    def _attach_distance(self, detection):
        """Attach distance only when the object's physical width is meaningful.

        Generic motion blobs have unknown physical dimensions, so their range is
        intentionally left unset until the user selects and sizes that object.
        """
        _, _, width, _ = detection.get("bbox", (0, 0, 0, 0))
        if self.demo:
            # The synthetic demo is intentionally allowed to use its synthetic target size.
            estimate = self.distance.estimate(width).to_dict()
        else:
            # In live mode, only the explicitly selected object receives a metric estimate.
            # Reusing one object's physical width for every detection would be misleading.
            estimate = self.distance.estimate(width, target_width_m=0).to_dict()
        detection["distance"] = estimate
        detection["distance_m"] = estimate["distance_m"]
        return detection

    @staticmethod
    def _clamp_bbox(bbox, frame_shape):
        x, y, w, h = [int(round(float(v))) for v in bbox]
        height, width = frame_shape[:2]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        return [x, y, w, h]

    def _initialize_selected_tracker(self, frame, bbox):
        """Initialize sparse optical-flow tracking inside a selected ROI."""
        if frame is None or frame.size == 0:
            self._tracking_prev_gray = None
            self._tracking_points = None
            self._tracking_template = None
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = self._clamp_bbox(bbox, frame.shape)
        roi = gray[y:y + h, x:x + w]
        # Keep a local appearance template as a fallback when optical flow has too few
        # corners (common on small, smooth or low-light targets).
        self._tracking_template = roi.copy() if roi.size and w >= 8 and h >= 8 else None
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[y:y + h, x:x + w] = 255
        points = cv2.goodFeaturesToTrack(
            gray, maxCorners=80, qualityLevel=0.01, minDistance=4,
            blockSize=5, mask=mask
        )
        self._tracking_prev_gray = gray
        self._tracking_points = points
        self._tracking_frames = 0

    def _template_match_bbox(self, frame, previous_bbox):
        """Fallback appearance tracking for low-texture selected regions."""
        template = self._tracking_template
        if frame is None or frame.size == 0 or template is None or template.size == 0:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        th, tw = template.shape[:2]
        if tw < 8 or th < 8 or tw >= gray.shape[1] or th >= gray.shape[0]:
            return None
        x, y, w, h = [int(v) for v in previous_bbox]
        margin_x = max(24, int(w * 1.25))
        margin_y = max(24, int(h * 1.25))
        sx = max(0, x - margin_x)
        sy = max(0, y - margin_y)
        ex = min(gray.shape[1], x + w + margin_x)
        ey = min(gray.shape[0], y + h + margin_y)
        search = gray[sy:ey, sx:ex]
        if search.shape[0] < th or search.shape[1] < tw:
            return None
        try:
            scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, max_score, _, max_loc = cv2.minMaxLoc(scores)
        except cv2.error:
            return None
        if not np.isfinite(max_score) or max_score < 0.50:
            return None
        matched = self._clamp_bbox((sx + max_loc[0], sy + max_loc[1], tw, th), frame.shape)
        # Reject implausibly large jumps; this is a local tracker, not a global detector.
        old_cx, old_cy = x + w / 2.0, y + h / 2.0
        new_cx, new_cy = matched[0] + matched[2] / 2.0, matched[1] + matched[3] / 2.0
        if np.hypot(new_cx - old_cx, new_cy - old_cy) > max(120.0, 2.5 * np.hypot(w, h)):
            return None
        return matched

    def _fallback_selected_track(self, frame, previous_bbox):
        matched = self._template_match_bbox(frame, previous_bbox)
        if matched is not None:
            self._initialize_selected_tracker(frame, matched)
            return matched
        self._initialize_selected_tracker(frame, previous_bbox)
        return None

    def _optical_flow_bbox(self, frame, previous_bbox):
        """Track a selected ROI with optical flow, then template matching as fallback."""
        if frame is None or frame.size == 0:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        old_gray = self._tracking_prev_gray
        points = self._tracking_points
        if old_gray is None or points is None or len(points) < 3:
            return self._fallback_selected_track(frame, previous_bbox)
        try:
            next_points, status, error = cv2.calcOpticalFlowPyrLK(
                old_gray, gray, points, None, winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
            )
        except cv2.error:
            return self._fallback_selected_track(frame, previous_bbox)
        if next_points is None or status is None:
            return self._fallback_selected_track(frame, previous_bbox)
        valid = status.reshape(-1) == 1
        if error is not None:
            valid &= error.reshape(-1) < 35.0
        old_good = points.reshape(-1, 2)[valid]
        new_good = next_points.reshape(-1, 2)[valid]
        if len(new_good) < 3:
            return self._fallback_selected_track(frame, previous_bbox)
        old_center_pts = np.median(old_good, axis=0)
        new_center_pts = np.median(new_good, axis=0)
        displacements = new_good - old_good
        delta = np.median(displacements, axis=0)
        residual = float(np.median(np.linalg.norm(displacements - delta, axis=1)))
        box_diag = max(1.0, float(np.hypot(previous_bbox[2], previous_bbox[3])))
        # Reject incoherent optical flow. Without this check, a few background
        # corners can drag the selected ROI to a totally unrelated part of frame.
        if residual > max(4.0, box_diag * 0.18) or float(np.linalg.norm(delta)) > max(90.0, box_diag * 0.8):
            return self._fallback_selected_track(frame, previous_bbox)
        # Estimate apparent scale change from tracked-feature spread. This lets the
        # selected ROI adapt as the object approaches/recedes instead of keeping a
        # fixed-size box and reporting a stale range.
        old_radius = float(np.median(np.linalg.norm(old_good - old_center_pts, axis=1)))
        new_radius = float(np.median(np.linalg.norm(new_good - new_center_pts, axis=1)))
        scale_factor = 1.0
        if old_radius >= 3.0 and new_radius >= 3.0:
            candidate_scale = new_radius / old_radius
            if 0.70 <= candidate_scale <= 1.43:
                scale_factor = float(np.clip(candidate_scale, 0.75, 1.33))
        x, y, w, h = previous_bbox
        old_cx, old_cy = x + w / 2.0, y + h / 2.0
        new_w, new_h = w * scale_factor, h * scale_factor
        moved = self._clamp_bbox((old_cx + delta[0] - new_w / 2.0,
                                  old_cy + delta[1] - new_h / 2.0,
                                  new_w, new_h), frame.shape)
        nx, ny, nw, nh = moved
        # Keep only tracked points that remain inside the selected region.
        inside = (
            (new_good[:, 0] >= nx) & (new_good[:, 0] <= nx + nw) &
            (new_good[:, 1] >= ny) & (new_good[:, 1] <= ny + nh)
        )
        kept = new_good[inside]
        self._tracking_prev_gray = gray
        self._tracking_points = kept.reshape(-1, 1, 2).astype(np.float32) if len(kept) >= 3 else None
        self._tracking_frames += 1
        if self._tracking_points is None or self._tracking_frames % 8 == 0:
            self._initialize_selected_tracker(frame, moved)
        return moved

    def select_target(self, x=None, y=None, bbox=None):
        """Select a target by tapping an existing detection or drawing a rectangle."""
        with self.lock:
            frame = self.latest_raw_visible.copy() if self.latest_raw_visible is not None else None
            detections = [dict(item) for item in self.latest_detections]
        if frame is None:
            raise ValueError("اول دوربین پشتی را فعال کن و صبر کن تصویر پردازش شود.")
        height, width = frame.shape[:2]
        if bbox is not None:
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError("کادر انتخاب باید شامل x، y، عرض و ارتفاع باشد.")
            x0, y0, bw, bh = [float(v) for v in bbox]
            if not all(np.isfinite(v) for v in (x0, y0, bw, bh)) or bw < 8 or bh < 8:
                raise ValueError("کادر انتخاب خیلی کوچک یا نامعتبر است؛ دور جسم یک کادر بکش.")
            chosen_bbox = self._clamp_bbox((x0, y0, bw, bh), frame.shape)
            candidates = [d for d in detections if (self.demo or d.get("class") in ("person", "car"))
                          and self._box_iou(chosen_bbox, d.get("bbox", (0,0,0,0))) >= 0.5]
            if not candidates:
                raise ValueError("کادر باید روی انسان یا خودروی تأییدشده باشد.")
            chosen = dict(max(candidates, key=lambda d: self._box_iou(chosen_bbox, d["bbox"])))
            chosen_bbox = chosen["bbox"]
        else:
            try:
                px, py = float(x), float(y)
            except (TypeError, ValueError):
                raise ValueError("مختصات انتخاب نامعتبر است.")
            if not np.isfinite(px) or not np.isfinite(py) or not (0 <= px < width and 0 <= py < height):
                raise ValueError("نقطه انتخاب خارج از تصویر است.")
            candidates = [d for d in detections if d.get("source") != "thermal"]
            def point_score(item):
                bx, by, bw, bh = item.get("bbox", (0, 0, 0, 0))
                inside = bx <= px <= bx + bw and by <= py <= by + bh
                cx, cy = bx + bw / 2, by + bh / 2
                return (0 if inside else 1, (cx - px) ** 2 + (cy - py) ** 2)
            candidates.sort(key=point_score)
            if not candidates:
                raise ValueError("هنوز انسان یا خودرویی تأیید نشده است؛ روی کادر تأییدشده بزن.")
            first = candidates[0]
            bx, by, bw, bh = first.get("bbox", (0, 0, 0, 0))
            inside = bx <= px <= bx + bw and by <= py <= by + bh
            cx, cy = bx + bw / 2, by + bh / 2
            if not inside and (cx - px) ** 2 + (cy - py) ** 2 > max(55, min(width, height) * 0.10) ** 2:
                raise ValueError("روی کادر رنگی هدف ضربه بزن، یا با کشیدن انگشت دور جسم کادر بکش.")
            chosen = dict(first)
            chosen_bbox = self._clamp_bbox(chosen.get("bbox", (0, 0, 0, 0)), frame.shape)
        # A newly selected object may have a different physical width from the
        # calibration reference. Never silently reuse the previous object's width.
        self.selected_target_width_m = None
        self._selected_width_samples.clear()
        self._selected_width_samples.append(float(chosen_bbox[2]))
        self._tracking_misses = 0
        selected = {
            "bbox": chosen_bbox,
            "class": chosen.get("class", "selected_object"),
            "source": chosen.get("source", "manual_roi"),
            "confidence": float(chosen.get("confidence", 1.0)),
            "tracking_status": "tracking",
            "distance_m": None,
            "distance": {},
            "width_px": chosen_bbox[2],
            "calibration_samples": 1,
            "auto_selected": False,
            "track_id": chosen.get("track_id"),
            "truncated": chosen.get("truncated", False),
        }
        with self.lock:
            self.selected_target = selected
            self._initialize_selected_tracker(frame, chosen_bbox)
            self._update_selected_distance()
        return self._selected_target_payload()

    def _selected_target_payload(self):
        with self.lock:
            if self.selected_target is None:
                return None
            payload = dict(self.selected_target)
            payload["bbox"] = list(payload.get("bbox", (0, 0, 0, 0)))
            payload["distance"] = dict(payload.get("distance", {}))
            payload["target_width_m"] = self.selected_target_width_m
            payload["calibration_samples"] = len(self._selected_width_samples)
            return payload

    def clear_selected_target(self):
        with self.lock:
            self.selected_target = None
            self.selected_target_width_m = None
            self._selected_width_samples.clear()
            self._tracking_prev_gray = None
            self._tracking_points = None
            self._tracking_template = None
            self._tracking_misses = 0
            self._auto_candidate_bbox = None
            self._auto_candidate_streak = 0
        return {"selected_target": None}

    def set_selected_target_width(self, width_m):
        if self.selected_target is None:
            raise ValueError("ابتدا یک جسم را روی تصویر انتخاب کن.")
        if width_m in (None, ""):
            raise ValueError("عرض واقعی جسم باید برحسب متر وارد شود.")
        try:
            width = float(width_m)
        except (TypeError, ValueError):
            raise ValueError("عرض واقعی جسم باید عددی باشد.")
        if not np.isfinite(width) or width <= 0 or width > 10000:
            raise ValueError("عرض واقعی جسم باید عددی مثبت و برحسب متر باشد.")
        self.selected_target_width_m = width
        self._update_selected_distance()
        return self._selected_target_payload()

    def _update_selected_distance(self):
        with self.lock:
            target = self.selected_target
            if target is None:
                return
            if self.ground_calibration is not None:
                # Never replace a ground-contact range with the legacy width
                # formula when the user changes selection/settings between frames.
                match = next((d for d in self.latest_detections
                              if d.get('track_id') == target.get('track_id')
                              and d.get('class') == target.get('class')
                              and d.get('distance', {}).get('method') == 'calibrated_ground_contact'), None)
                if target.get('tracking_status') == 'tracking' and match:
                    target.update(distance=dict(match['distance']), distance_m=match['distance_m'])
                else:
                    estimate = dict(target.get('distance', {}))
                    estimate.update(distance_m=None, filtered_distance_m=None, valid=False,
                                    quality='invalid', status='awaiting_fresh_ground_measurement',
                                    reason='awaiting_fresh_ground_measurement', method='calibrated_ground_contact')
                    target.update(distance=estimate, distance_m=None)
                return
            width_px = float(target.get("bbox", (0, 0, 0, 0))[2])
            known_width = self.selected_target_width_m
            frame_width_px = self.latest_frame_size.get("width") if self.latest_frame_size else None

            # Stabilize only small frame-to-frame box jitter. If the apparent width
            # changes materially (e.g. the object approaches), use the current box
            # immediately so smoothing does not hide real scale changes.
            recent_widths = [float(v) for v in list(self._selected_width_samples)[-3:]
                             if np.isfinite(float(v)) and float(v) >= 12]
            range_width_px = width_px
            if len(recent_widths) >= 3:
                sample_median = float(np.median(recent_widths))
                sample_spread = (max(recent_widths) - min(recent_widths)) / max(1.0, sample_median)
                if sample_spread <= 0.08:
                    range_width_px = sample_median
            if known_width is None:
                estimate = self.distance.estimate(range_width_px, target_width_m=0, frame_width_px=frame_width_px).to_dict()
            else:
                estimate = self.distance.estimate(range_width_px, target_width_m=known_width, frame_width_px=frame_width_px).to_dict()
            tracking_status = target.get("tracking_status")
            # Do not erase a valid calibrated range merely because tracking briefly
            # loses features. During reacquisition it is the last-box estimate; when
            # lost, preserve it explicitly as stale instead of presenting it as current.
            if estimate.get("distance_m") is not None and tracking_status == "reacquiring":
                estimate["status"] = "tracking_unstable_last_estimate"
                estimate["is_stale"] = True
            elif estimate.get("distance_m") is not None and tracking_status == "lost":
                estimate["status"] = "target_lost_last_estimate"
                estimate["is_stale"] = True
            if tracking_status in ("lost", "reacquiring") or target.get("truncated") or width_px < 12:
                estimate["distance_m"] = None
                estimate["status"] = "unreliable_target_measurement"
                estimate["sensitivity"] = {}
            if self.latest_frame_size:
                bx, by, bw, bh = target.get("bbox", (0,0,0,0))
                if bx <= 1 or by <= 1 or bx+bw >= self.latest_frame_size["width"]-1 or by+bh >= self.latest_frame_size["height"]-1:
                    estimate["distance_m"] = None
                    estimate["status"] = "truncated_target"
                    estimate["sensitivity"] = {}
            target["distance"] = estimate
            target["distance_m"] = estimate["distance_m"]
            target["width_px"] = width_px
            target["distance_bbox_width_px"] = round(range_width_px, 2)

    def calibrate_selected_target(self, known_width_m, known_distance_m):
        """Calibrate from a full-object box measured at one known distance.

        Never calibrate from a background-subtraction blob: those boxes usually
        cover only changed pixels, not the object's silhouette. Also use only a
        short, recent width window and reject unstable boxes instead of silently
        mixing measurements from different distances.
        """
        target = self._selected_target_payload()
        if target is None:
            raise ValueError("ابتدا روی تصویر یک جسم را انتخاب کن.")

        if target.get("truncated"):
            raise ValueError("هدف کامل در تصویر نیست؛ کالیبراسیون نامعتبر است.")
        source = str(target.get("source", ""))
        if source == "motion" or source == "thermal" or source.startswith("thermal"):
            raise ValueError(
                "این کادر فقط ناحیه حرکت/حرارت را نشان می‌دهد و لزوماً کل جسم نیست. "
                "با کشیدن انگشت، یک کادر دستی دور تمام جسم بکش و سپس کالیبره کن."
            )

        tracking_status = str(target.get("tracking_status", "tracking"))
        if tracking_status in ("lost", "reacquiring"):
            raise ValueError(
                "ردیابی در حال بازیابی یا قطع است. گوشی و جسم را ثابت نگه دار تا کادر پایدار شود، "
                "یا هدف را دوباره انتخاب کن و سپس کالیبره کن."
            )

        width = DistanceEstimator._positive_or_none(known_width_m)
        distance = DistanceEstimator._positive_or_none(known_distance_m)
        if width is None or distance is None:
            raise ValueError("عرض واقعی و فاصله واقعی باید عددهای مثبت باشند.")

        current_px = float(target.get("bbox", (0, 0, 0, 0))[2])
        if not np.isfinite(current_px) or current_px < 12:
            raise ValueError("عرض کادر هدف کمتر از ۱۲ پیکسل است؛ نزدیک‌تر شو یا دور تمام جسم کادر بکش.")

        # Only the most recent five frames belong to this calibration moment.
        # A rolling 12-frame median can mix frames recorded at different ranges,
        # producing a completely wrong focal length for a moving target.
        recent = [float(v) for v in list(self._selected_width_samples)[-5:]
                  if np.isfinite(float(v)) and float(v) >= 12]
        if not recent or abs(recent[-1] - current_px) > max(3.0, current_px * 0.08):
            recent = (recent + [current_px])[-5:]
        else:
            # The current selected box is the latest measurement; append it only
            # when it is not already represented by the last tracker sample.
            if abs(recent[-1] - current_px) > 0.5:
                recent = (recent + [current_px])[-5:]

        sample_array = np.asarray(recent, dtype=np.float64)
        median_px = float(np.median(sample_array)) if len(sample_array) else current_px
        p10, p90 = np.percentile(sample_array, [10, 90]) if len(sample_array) else (current_px, current_px)
        spread_pct = float((p90 - p10) / max(1.0, median_px) * 100.0)
        if len(sample_array) >= 3 and spread_pct > 12.0:
            raise ValueError(
                f"عرض کادر هنگام کالیبراسیون ناپایدار است (تغییر حدود {spread_pct:.1f}٪). "
                "جسم را در فاصله معلوم ثابت نگه دار، دور کل جسم کادر بکش و دوباره امتحان کن. "
                "کالیبراسیون قبلی تغییر نکرده است."
            )

        observed_px = median_px if len(sample_array) >= 3 else current_px
        frame_width_px = self.latest_frame_size.get("width") if self.latest_frame_size else None
        focal = self.distance.calibrate(width, distance, observed_px, frame_width_px=frame_width_px)
        self.selected_target_width_m = width
        self._save_settings()
        self._update_selected_distance()

        warning = None
        if len(sample_array) < 3:
            warning = "کالیبراسیون ذخیره شد، اما نمونه‌های پایدار کم بود؛ فاصله تقریبی است. جسم را ثابت نگه دار و کالیبراسیون را دوباره انجام بده."
        elif spread_pct > 8.0:
            warning = "کالیبراسیون ذخیره شد؛ نوسان عرض کادر قابل توجه است، بنابراین فاصله تقریبی خواهد بود."
        elif target.get("auto_selected"):
            warning = "هدف به‌صورت خودکار انتخاب شده است؛ مطمئن شو کادر تمام جسم را می‌پوشاند، نه فقط ناحیه متحرک را."

        return {
            "focal_length_px": round(focal, 3),
            "observed_width_px": round(observed_px, 2),
            "width_samples": int(len(sample_array)),
            "width_spread_pct": round(spread_pct, 1),
            "quality_warning": warning,
            "distance": self.distance.to_dict(),
            "selected_target": self._selected_target_payload(),
        }

    def _create_selected_target(self, frame, bbox, class_name="moving_object",
                                source="motion", confidence=1.0, auto_selected=False):
        """Create/reset one tracked target without borrowing another object's width."""
        bbox = self._clamp_bbox(bbox, frame.shape)
        with self.lock:
            self.selected_target_width_m = None
            self._selected_width_samples.clear()
            self._selected_width_samples.append(float(bbox[2]))
            self._tracking_misses = 0
            self.selected_target = {
                "bbox": list(bbox),
                "class": str(class_name or "moving_object"),
                "source": str(source or "motion"),
                "confidence": float(confidence if confidence is not None else 1.0),
                "tracking_status": "tracking",
                "distance_m": None,
                "distance": {},
                "width_px": int(bbox[2]),
                "calibration_samples": 1,
                "auto_selected": bool(auto_selected),
            }
            self._initialize_selected_tracker(frame, bbox)
            self._update_selected_distance()

    def _maybe_auto_select_target(self, detections, frame):
        """Acquire only one persistent moving target; ignore transient speckle boxes."""
        if frame is None or frame.size == 0:
            return
        with self.lock:
            target = self.selected_target
            if target is not None:
                if target.get("auto_selected") and target.get("tracking_status") == "lost":
                    # Automatically reacquire after the previous automatic target is gone.
                    self.selected_target = None
                    self.selected_target_width_m = None
                    self._selected_width_samples.clear()
                    self._tracking_prev_gray = None
                    self._tracking_points = None
                    self._tracking_template = None
                    self._tracking_misses = 0
                else:
                    return

        frame_area = int(frame.shape[0] * frame.shape[1])
        min_area = max(150, int(frame_area * 0.00020))
        candidates = []
        for item in detections:
            if item.get("source") != "motion":
                continue
            x, y, w, h = item.get("bbox", (0, 0, 0, 0))
            area = int(item.get("area_px", max(0, w * h)))
            if area < min_area or w < 12 or h < 12:
                continue
            candidates.append(item)
        if not candidates:
            self._auto_candidate_bbox = None
            self._auto_candidate_streak = 0
            return

        previous = self._auto_candidate_bbox
        chosen = None
        if previous is not None:
            px, py, pw, ph = previous
            pcx, pcy = px + pw / 2.0, py + ph / 2.0
            pdiag = max(1.0, float(np.hypot(pw, ph)))
            ranked = []
            for item in candidates:
                bx, by, bw, bh = item.get("bbox", (0, 0, 0, 0))
                iou = self._box_iou(previous, (bx, by, bw, bh))
                center_dist = float(np.hypot(bx + bw / 2.0 - pcx, by + bh / 2.0 - pcy))
                if iou < 0.02 and center_dist > max(45.0, pdiag * 1.2):
                    continue
                score = 2.0 * iou - center_dist / pdiag + 0.05 * np.log1p(max(0, item.get("area_px", 0)))
                ranked.append((score, item))
            if ranked:
                chosen = max(ranked, key=lambda pair: pair[0])[1]
        if chosen is None:
            chosen = max(candidates, key=lambda item: int(item.get("area_px", 0)))
            self._auto_candidate_streak = 1
        else:
            self._auto_candidate_streak += 1
        self._auto_candidate_bbox = list(chosen.get("bbox", (0, 0, 0, 0)))
        if self._auto_candidate_streak < 2:
            return

        # If a confirmed drone-model box overlaps this moving region, use its full
        # semantic box instead of the smaller changed-pixel mask.
        chosen_box = chosen.get("bbox", (0, 0, 0, 0))
        best_visual = None
        best_overlap = 0.0
        for item in detections:
            if not str(item.get("source", "")).startswith("visual_model"):
                continue
            overlap = self._box_iou(chosen_box, item.get("bbox", (0, 0, 0, 0)))
            if overlap > best_overlap:
                best_visual, best_overlap = item, overlap
        if best_visual is not None and best_overlap >= 0.05:
            chosen_box = best_visual.get("bbox", chosen_box)
            class_name = best_visual.get("class", "drone")
            source = best_visual.get("source", "visual_model")
            confidence = best_visual.get("confidence", 1.0)
        else:
            class_name, source, confidence = "moving_object", "motion", chosen.get("confidence", 1.0)
        self._create_selected_target(frame, chosen_box, class_name, source, confidence, auto_selected=True)
        self._auto_candidate_bbox = None
        self._auto_candidate_streak = 0

    def _single_target_detection(self):
        """Return one display record only; never expose every motion contour as a box."""
        target = self._selected_target_payload()
        if target is None:
            return []
        return [{
            "bbox": list(target.get("bbox", (0, 0, 0, 0))),
            "class": target.get("class", "moving_object"),
            "source": target.get("last_detection_source", target.get("source", "motion")),
            "confidence": target.get("confidence", 1.0),
            "selected": True,
            "distance_m": target.get("distance_m"),
            "distance": dict(target.get("distance", {})),
            "tracking_status": target.get("tracking_status", "tracking"),
        }]

    @staticmethod
    def _box_iou(a, b):
        ax, ay, aw, ah = [float(v) for v in a]
        bx, by, bw, bh = [float(v) for v in b]
        left, top = max(ax, bx), max(ay, by)
        right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - intersection
        return intersection / union if union > 0 else 0.0

    def _update_selected_target(self, detections, frame):
        for item in detections:
            item.pop("selected", None)
        with self.lock:
            if self.selected_target is None or frame is None:
                return detections
            previous = list(self.selected_target.get("bbox", (0, 0, 0, 0)))
            auto_motion_target = bool(self.selected_target.get("auto_selected")) and not str(self.selected_target.get("source", "")).startswith("visual_model")
            # Motion masks are not object silhouettes; optical flow initialized on
            # a thin changed-pixel strip can latch onto background texture. In auto
            # mode, use the temporally matched motion region as the position cue.
            flow_bbox = None if auto_motion_target else self._optical_flow_bbox(frame, previous)
            anchor = flow_bbox or previous
            candidates = [d for d in detections if d.get("source") != "thermal"]
            match = None
            best_score = -1e9
            ax, ay, aw, ah = anchor
            acx, acy = ax + aw / 2, ay + ah / 2
            diagonal = max(1.0, float(np.hypot(aw, ah)))
            for item in candidates:
                bx, by, bw, bh = item.get("bbox", (0, 0, 0, 0))
                iou = self._box_iou(anchor, (bx, by, bw, bh))
                distance = float(np.hypot((bx + bw / 2) - acx, (by + bh / 2) - acy))
                if iou < 0.08 and distance > max(32.0, diagonal * 0.85):
                    continue
                score = iou * 2.0 - distance / diagonal
                if score > best_score:
                    best_score, match = score, item
            if match is not None:
                new_bbox = self._clamp_bbox(match.get("bbox", anchor), frame.shape)
                source = match.get("source")
                # Motion masks are often only the changed edge/ghost of an object, not
                # its full silhouette. Never let such a mask replace a manually selected
                # ROI's width/height: doing so makes monocular range jump unpredictably.
                is_auto_motion = bool(self.selected_target.get("auto_selected")) and not str(source or "").startswith("visual_model")
                if is_auto_motion:
                    # Motion masks often show the leading/trailing changed pixels,
                    # not the object's silhouette. Use their center as a position
                    # cue but preserve the tracked box dimensions and reject optical
                    # flow jumps for this automatic mode.
                    mx, my, mw, mh = new_bbox
                    px, py, pw, ph = previous
                    old_cx, old_cy = px + pw / 2.0, py + ph / 2.0
                    motion_cx, motion_cy = mx + mw / 2.0, my + mh / 2.0
                    cx = 0.35 * old_cx + 0.65 * motion_cx
                    cy = 0.35 * old_cy + 0.65 * motion_cy
                    anchor = self._clamp_bbox((cx - pw / 2.0, cy - ph / 2.0, pw, ph), frame.shape)
                    self._initialize_selected_tracker(frame, anchor)
                    match["selected"] = True
                    self._selected_width_samples.append(float(anchor[2]))
                    self._tracking_misses = 0
                    tracking_status = "tracking_motion_hint"
                elif not str(source or "").startswith("visual_model"):
                    if flow_bbox is None:
                        mx, my, mw, mh = new_bbox
                        ax, ay, aw, ah = anchor
                        old_cx, old_cy = ax + aw / 2.0, ay + ah / 2.0
                        motion_cx, motion_cy = mx + mw / 2.0, my + mh / 2.0
                        # Use motion only as a weak position hint while preserving ROI size.
                        cx = 0.65 * old_cx + 0.35 * motion_cx
                        cy = 0.65 * old_cy + 0.35 * motion_cy
                        anchor = self._clamp_bbox((cx - aw / 2.0, cy - ah / 2.0, aw, ah), frame.shape)
                        self._initialize_selected_tracker(frame, anchor)
                    else:
                        anchor = flow_bbox
                    match["selected"] = True
                    self._selected_width_samples.append(float(anchor[2]))
                    self._tracking_misses = 0
                    tracking_status = "tracking_optical_flow" if flow_bbox else "tracking_motion_hint"
                else:
                    # Semantic model boxes are more appropriate than motion masks for
                    # updating the full object geometry, provided scale changes are plausible.
                    old_area = max(1, anchor[2] * anchor[3])
                    new_area = max(1, new_bbox[2] * new_bbox[3])
                    if 0.22 <= new_area / old_area <= 4.5:
                        anchor = new_bbox
                        match["selected"] = True
                        self._selected_width_samples.append(float(new_bbox[2]))
                        self._tracking_misses = 0
                        self._initialize_selected_tracker(frame, anchor)
                        tracking_status = "tracking_detection"
                    else:
                        tracking_status = "tracking_optical_flow" if flow_bbox else "tracking"
            elif flow_bbox is not None:
                self._tracking_misses = 0
                self._selected_width_samples.append(float(anchor[2]))
                tracking_status = "tracking_optical_flow"
            else:
                self._tracking_misses += 1
                tracking_status = "reacquiring" if self._tracking_misses <= 3 else "lost"
            self.selected_target["bbox"] = list(self._clamp_bbox(anchor, frame.shape))
            self.selected_target["width_px"] = self.selected_target["bbox"][2]
            self.selected_target["tracking_status"] = tracking_status
            if match is not None:
                self.selected_target["last_detection_source"] = match.get("source")
            self._update_selected_distance()
        return detections

    def _process(self, visible, thermal):
        visible = self.rectifier.correct(visible)
        detections = []
        thermal_detections = []
        visual = self.detector.detect(visible) if visible is not None else []
        for item in visual:
            detections.append(self._attach_distance(dict(item)))
        # Motion/thermal blobs cannot establish semantic identity.

        if visual:
            self.last_detection_at = datetime.now().isoformat(timespec="seconds")
            self.alarm.trigger_alarm({"count": len(visual), "max_confidence": max(d["confidence"] for d in visual)})
        if visible is not None:
            with self.lock:
                self.latest_raw_visible = visible.copy()
                self.latest_frame_size = {"width": int(visible.shape[1]), "height": int(visible.shape[0])}
            if self.selected_target is not None:
                # A selected target must have a fresh confirmed semantic match.
                previous = self.selected_target.get("bbox", (0, 0, 0, 0))
                matches = [d for d in detections
                           if d["class"] == self.selected_target.get("class")
                           and (self.selected_target.get("track_id") is None or
                                d.get("track_id") == self.selected_target.get("track_id"))
                           and (self.selected_target.get("track_id") is not None
                                or self._box_iou(previous, d["bbox"]) >= 0.3)]
                if matches:
                    match = max(matches, key=lambda d: self._box_iou(previous, d["bbox"]))
                    self.selected_target.update(bbox=list(match["bbox"]), tracking_status="tracking",
                                                track_id=match.get("track_id"), truncated=match.get("truncated", False))
                    self._selected_width_samples.append(match["bbox"][2])
                    self._update_selected_distance()
                    match.update(selected=True, distance=self.selected_target["distance"],
                                 distance_m=self.selected_target["distance_m"])
                else:
                    self.selected_target["tracking_status"] = "lost"
                    self._selected_width_samples.clear()
                    self._update_selected_distance()
        else:
            detections = []
        if visible is not None and self.ground_calibration is not None:
            detections = self.metric_ranges.update(detections, time.monotonic(), self.ground_calibration, visible.shape)
            for item in detections:
                item['distance']['status'] = item['distance']['reason']
            if self.selected_target is not None:
                match = next((d for d in detections if d.get('track_id') == self.selected_target.get('track_id')), None)
                if match:
                    self.selected_target.update(distance=match['distance'], distance_m=match['distance_m'])
        annotated_visible = self._annotate(visible, detections, source="visual") if visible is not None else None
        annotated_thermal = self._annotate(self.thermal_processor.enhance(thermal), thermal_detections, source="thermal") if thermal is not None else None
        with self.lock:
            self.latest_visible = annotated_visible
            self.latest_thermal = annotated_thermal
            # Keep thermal candidates available to diagnostics/API, but the UI's
            # moving-target table and visible overlay show only the selected target.
            self.latest_detections = detections[:50]
            self.frame_count += 1
            self.last_frame_at = datetime.now().isoformat(timespec="seconds")

        if (visual and config.SAVE_DETECTION_IMAGES and annotated_visible is not None
                and time.monotonic() - self._last_snapshot_at >= 10.0):
            try:
                config.DETECTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
                filename = config.DETECTION_LOG_DIR / f"drone_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                if cv2.imwrite(str(filename), annotated_visible):
                    self._last_snapshot_at = time.monotonic()
            except Exception:
                LOG.exception("Could not save detection image")

    @staticmethod
    def _annotate(frame, detections, source):
        output = frame.copy()
        for item in detections:
            if source == "visual" and item.get("source") == "thermal":
                continue
            if source == "thermal" and item.get("source") != "thermal":
                continue
            x, y, w, h = [int(v) for v in item.get("bbox", (0, 0, 0, 0))]
            if w <= 0 or h <= 0:
                continue
            color = (40, 70, 255) if item.get("selected") else ((0, 220, 80) if str(item.get("source", "")).startswith("visual_model") else (0, 180, 255))
            if item.get("selected"):
                label = "هدف متحرک"
            elif item.get("source") == "motion":
                label = "MOTION REGION"
            else:
                label = f"{item.get('class', 'target')} {item.get('confidence', 0):.2f}"
            dist = item.get("distance_m")
            if dist is not None:
                label += f" | {dist:.2f} متر"
            elif item.get("selected"):
                label += " | فاصله نامشخص"
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            cv2.putText(output, label, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        return output

    def receive_browser_frame(self, image_bytes: bytes):
        """Decode a JPEG posted by the dashboard's browser-based rear-camera capture."""
        if not self.browser_camera:
            raise ValueError("Browser camera mode is not enabled. Start with --browser-camera.")
        if not image_bytes:
            raise ValueError("Empty camera frame.")
        frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            raise ValueError("Could not decode camera frame as JPEG.")
        height, width = frame.shape[:2]
        if height * width > 16_777_216:
            raise ValueError("Camera frame exceeds the 16-megapixel limit.")
        if width > config.MAX_FRAME_WIDTH:
            scale = config.MAX_FRAME_WIDTH / float(width)
            frame = cv2.resize(frame, (config.MAX_FRAME_WIDTH, max(1, int(height * scale))))
        with self.lock:
            self.browser_frame = frame
            self.browser_frame_received_at = time.time()
        return {"width": int(frame.shape[1]), "height": int(frame.shape[0])}

    def _invalidate_live_observations(self):
        """Remove observations after stream failure; never retain a current range."""
        with self.lock:
            self.metric_ranges.states.clear()
            self.latest_detections = []
            if self.latest_raw_visible is not None:
                self.latest_visible = self.latest_raw_visible.copy()
            if self.selected_target is not None:
                self.selected_target["tracking_status"] = "lost"
                self._update_selected_distance()
            if hasattr(self.detector, "reset_tracking"):
                self.detector.reset_tracking()

    def _browser_loop(self):
        """Process frames sent by the browser; this avoids relying on cv2.VideoCapture for Android cameras."""
        last_seen = 0.0
        while self.running and not self.shutdown_event.is_set():
            with self.lock:
                received_at = self.browser_frame_received_at
                frame = self.browser_frame.copy() if self.browser_frame is not None and received_at > last_seen else None
            if frame is None:
                if time.time() - received_at > 3.0:
                    self._invalidate_live_observations()
                    self.last_error = "Waiting for frames from the phone's rear camera. Allow camera permission in the browser."
                self.shutdown_event.wait(0.05)
                continue
            last_seen = received_at
            self.last_error = None
            try:
                self._process(frame, None)
            except Exception as exc:
                self.last_error = str(exc)
                self._invalidate_live_observations()
                LOG.exception("Browser-camera processing loop error")
                self.shutdown_event.wait(0.2)
            self.shutdown_event.wait(max(0.0, config.DETECTION_INTERVAL_SECONDS))

    def _live_loop(self):
        while self.running and not self.shutdown_event.is_set():
            started = time.monotonic()
            try:
                visible, thermal = self.camera.read_frames()
                if visible is None and thermal is None:
                    self._invalidate_live_observations()
                    self.last_error = "Camera connected but no frames are arriving."
                    self.shutdown_event.wait(0.2)
                    continue
                self.last_error = None
                self._process(visible, thermal)
            except Exception as exc:
                self.last_error = str(exc)
                self._invalidate_live_observations()
                LOG.exception("Processing loop error")
                self.shutdown_event.wait(0.2)
            elapsed = time.monotonic() - started
            self.shutdown_event.wait(max(0.0, config.DETECTION_INTERVAL_SECONDS - elapsed))

    def _demo_loop(self):
        """Generate clearly marked synthetic frames for interface and pipeline acceptance tests."""
        while self.running and not self.shutdown_event.is_set():
            now = time.monotonic()
            frame = np.full((480, 640, 3), 28, dtype=np.uint8)
            # Simple sky-like synthetic background; this is not a camera image.
            for y in range(480):
                shade = 30 + int(20 * y / 480)
                frame[y, :, :] = (shade, shade + 4, shade + 8)
            width = 54 + int(10 * (1 + math.sin(now * 0.7)))
            x = 270 + int(70 * math.sin(now * 0.35))
            y = 180 + int(25 * math.cos(now * 0.4))
            cv2.ellipse(frame, (x + width // 2, y + 18), (width // 2, 9), 0, 0, 360, (210, 210, 210), -1)
            cv2.line(frame, (x + 8, y + 5), (x + width - 8, y + 31), (220, 220, 220), 2)
            cv2.line(frame, (x + width - 8, y + 5), (x + 8, y + 31), (220, 220, 220), 2)
            cv2.putText(frame, "DEMO MODE - SYNTHETIC TARGET", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2)
            detection = {
                "bbox": (x, y + 5, width, 27),
                "confidence": 0.99,
                "class": "synthetic_demo_target_not_a_real_detection",
                "source": "demo",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            self._attach_distance(detection)
            with self.lock:
                self.latest_raw_visible = frame.copy()
                self.latest_frame_size = {"width": int(frame.shape[1]), "height": int(frame.shape[0])}
            demo_detections = self._update_selected_target([detection], frame)
            with self.lock:
                self.latest_visible = self._annotate(frame, demo_detections, source="visual")
                self.latest_thermal = None
                self.latest_detections = demo_detections
                self.frame_count += 1
                self.last_frame_at = datetime.now().isoformat(timespec="seconds")
            self.shutdown_event.wait(0.1)

    def get_frame(self, kind: str):
        with self.lock:
            frame = self.latest_visible if kind == "visible" else self.latest_thermal
            return None if frame is None else frame.copy()

    def status(self):
        with self.lock:
            detections = [dict(item) for item in self.latest_detections]
            running = self.running
            frame_count = self.frame_count
            last_error = self.last_error
            last_frame_at = self.last_frame_at
            last_detection_at = self.last_detection_at
            frame_size = dict(self.latest_frame_size) if self.latest_frame_size else None
            selected_target = self._selected_target_payload()
        if self.demo:
            camera_status = {"demo": True}
            mode = "demo"
        elif self.browser_camera:
            camera_status = {
                "source": "browser rear camera",
                "visible_configured": True,
                "visible_open": bool(self.browser_frame is not None and time.time() - self.browser_frame_received_at < 3.0),
                "thermal_configured": False,
                "thermal_open": False,
            }
            mode = "browser_camera"
        else:
            camera_status = self.camera.status()
            mode = "live"
        distance_status = self.distance.to_dict()
        if selected_target is not None:
            # The calibrated reference width is not necessarily the width of the
            # currently selected object. Report readiness for this target, not the
            # previous calibration object.
            distance_status["target_width_m"] = self.selected_target_width_m
            distance_status["ready"] = (
                self.distance.focal_length_px is not None and self.selected_target_width_m is not None
            )
        return {
            "ok": True,
            "mode": mode,
            "demo_warning": "Synthetic data only; this mode does not detect real drones." if self.demo else None,
            "detection_running": running,
            "camera": camera_status,
            "detector": self.detector.get_stats(),
            "motion": self.motion_detector.get_stats(),
            "distance": distance_status,
            "alarm": self.alarm.get_status(),
            "settings": {
                "confidence_threshold": self.detector.confidence,
                "target_width_m": self.distance.target_width_m,
                "thermal_enabled": bool(self.settings.get("thermal_enabled", True)),
            },
            "frame_count": frame_count,
            "last_frame_at": last_frame_at,
            "last_detection_at": last_detection_at,
            "last_error": last_error,
            "detections": detections,
            "frame_size": frame_size,
            "selected_target": selected_target,
            "uptime_s": round(time.time() - self.started_at, 1),
        }


def make_handler(service: DroneDistanceService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "PersonCarDistanceLocal/4.0"

        def log_message(self, fmt, *args):
            LOG.info("%s - %s", self.address_string(), fmt % args)

        def _json(self, status_code, payload):
            data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _body(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise ValueError("Invalid Content-Length")
            if length < 0:
                raise ValueError("Invalid Content-Length")
            if length > 65536:
                raise ValueError("Request body is too large")
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                raise ValueError("Body must be valid JSON") from exc
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object")
            return data

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                index = config.BASE_DIR / "templates" / "index.html"
                if not index.is_file():
                    return self._json(500, {"ok": False, "error": "Dashboard template is missing."})
                data = index.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
            if path == "/health":
                return self._json(200, {"ok": True, "service": "drone-distance", "version": "4.0-laptop-preview"})
            if path == "/api/status":
                return self._json(200, service.status())
            if path in ("/video/visible", "/video/thermal"):
                return self._mjpeg("visible" if path.endswith("visible") else "thermal")
            return self._json(404, {"ok": False, "error": "Not found"})

        def _mjpeg(self, kind):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                frame = service.get_frame(kind)
                if frame is None:
                    frame = np.zeros((360, 640, 3), dtype=np.uint8)
                    label = "Waiting for demo/camera stream" if not service.running else "No frame available"
                    cv2.putText(frame, label, (35, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok:
                    try:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(buffer)).encode() + b"\r\n\r\n" + buffer.tobytes() + b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return
                time.sleep(0.1)

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                if path == "/api/camera-frame":
                    if not service.browser_camera:
                        return self._json(409, {"ok": False, "error": "Start the server with --browser-camera to accept phone camera frames."})
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                    except ValueError:
                        raise ValueError("Invalid Content-Length")
                    if length <= 0 or length > 12 * 1024 * 1024:
                        return self._json(413, {"ok": False, "error": "JPEG frame must be between 1 byte and 12 MiB."})
                    raw = self.rfile.read(length)
                    info = service.receive_browser_frame(raw)
                    return self._json(200, {"ok": True, **info})
                data = self._body()
                if path == "/api/start":
                    success, message = service.start_detection()
                    return self._json(200 if success else 409, {"ok": success, "message": message, "status": service.status()})
                if path == "/api/stop":
                    success, message = service.stop_detection()
                    return self._json(200, {"ok": success, "message": message, "status": service.status()})
                if path == "/api/settings":
                    status = service.update_settings(data)
                    return self._json(200, {"ok": True, "status": status})
                if path == "/api/target/select":
                    if "bbox" in data:
                        result = service.select_target(bbox=data.get("bbox"))
                    else:
                        result = service.select_target(data.get("x"), data.get("y"))
                    return self._json(200, {"ok": True, "selected_target": result})
                if path == "/api/target/clear":
                    return self._json(200, {"ok": True, **service.clear_selected_target()})
                if path == "/api/target/width":
                    result = service.set_selected_target_width(data.get("width_m"))
                    return self._json(200, {"ok": True, "selected_target": result})
                if path == "/api/calibrate-selected":
                    result = service.calibrate_selected_target(data.get("known_width_m"), data.get("known_distance_m"))
                    return self._json(200, {"ok": True, **result})
                if path == "/api/calibrate":
                    result = service.calibrate_distance(
                        data.get("known_width_m"), data.get("known_distance_m"), data.get("observed_width_px"),
                        frame_width_px=data.get("frame_width_px")
                    )
                    return self._json(200, {"ok": True, **result})
                if path == "/api/calibration/reset":
                    return self._json(200, {"ok": True, "distance": service.reset_calibration()})
                if path == "/api/alarm/test":
                    result = service.alarm.test_alarm()
                    return self._json(200, {"ok": bool(result), "triggered": bool(result), "alarm": service.alarm.get_status()})
                if path == "/api/alarm/enable":
                    if not isinstance(data.get("enabled"), bool):
                        raise ValueError("enabled must be true or false")
                    enabled = service.alarm.set_enabled(data["enabled"])
                    service._save_settings()
                    return self._json(200, {"ok": True, "enabled": enabled, "alarm": service.alarm.get_status()})
                return self._json(404, {"ok": False, "error": "Not found"})
            except (ValueError, TypeError) as exc:
                return self._json(400, {"ok": False, "error": str(exc)})
            except Exception as exc:
                LOG.exception("API error on %s", path)
                return self._json(500, {"ok": False, "error": "Internal server error", "detail": str(exc)})

    return Handler


def run_server(host=None, port=None, demo=False, visible_source=None, thermal_source=None, model_path=None, browser_camera=False, detector=None, camera=None):
    host = host or config.WEB_HOST
    port = int(port or config.WEB_PORT)
    # Keep the unauthenticated dashboard local-only. Do not expose it on a LAN interface.
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("For safety, the dashboard binds to localhost only. Use a secured reverse proxy for remote access.")
    service = DroneDistanceService(demo=demo, browser_camera=browser_camera, visible_source=visible_source,
                                   thermal_source=thermal_source, model_path=model_path, detector=detector, camera=camera)
    server = ThreadingHTTPServer((host, port), make_handler(service))
    server.daemon_threads = True
    mode = "demo" if demo else ("browser_camera" if browser_camera else "live")
    LOG.info("Dashboard listening on http://%s:%s (mode=%s)", host, port, mode)
    print(f"Dashboard: http://{host}:{port}")
    if demo:
        print("DEMO MODE: synthetic target only; no real drone detection is performed.")
    if browser_camera:
        print("BROWSER CAMERA MODE: open the dashboard in your phone browser and allow rear-camera access.")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        service.shutdown()
        server.shutdown()
        server.server_close()
    return service


def main(argv=None):
    parser = argparse.ArgumentParser(description="Drone detection and calibrated distance estimation dashboard")
    parser.add_argument("--demo", action="store_true", help="Run a synthetic demo without camera/model hardware")
    parser.add_argument("--browser-camera", action="store_true", help="Use the phone browser's rear camera instead of cv2.VideoCapture")
    parser.add_argument("--host", default=config.WEB_HOST, help="Must be localhost for the built-in server")
    parser.add_argument("--port", type=int, default=config.WEB_PORT)
    parser.add_argument("--visible-source", default=None, help="Camera index or RTSP/HTTP URL")
    parser.add_argument("--thermal-source", default=None, help="Optional thermal camera index or RTSP/HTTP URL")
    parser.add_argument("--model", default=None, help="Path to a COCO YOLO11 raw ONNX model (person/car only)")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    config.DETECTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return run_server(args.host, args.port, args.demo, args.visible_source, args.thermal_source, args.model,
                      browser_camera=args.browser_camera)


if __name__ == "__main__":
    main()
