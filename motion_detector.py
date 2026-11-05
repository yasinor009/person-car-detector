"""Motion-region detector with camera-shake suppression for handheld phone video.

It reports moving regions only, never semantic object classes. A stable camera
still produces the best results; strong global camera motion is suppressed to
avoid drawing giant false-positive boxes around the whole scene.
"""
from __future__ import annotations

from datetime import datetime
import cv2
import numpy as np


class MotionDetector:
    def __init__(self, min_area_px: int = 80, max_area_ratio: float = 0.25,
                 max_objects: int = 12, warmup_frames: int = 5,
                 camera_motion_threshold_px: float = 3.0):
        self.min_area_px = int(min_area_px)
        self.max_area_ratio = float(max_area_ratio)
        self.max_objects = int(max_objects)
        self.warmup_frames = max(0, int(warmup_frames))
        self.camera_motion_threshold_px = float(camera_motion_threshold_px)
        self.background = cv2.createBackgroundSubtractorMOG2(
            history=180, varThreshold=28, detectShadows=False
        )
        self.frames_seen = 0
        self.previous_gray_small = None
        self.camera_moving = False
        self.camera_motion_px = 0.0
        self.suppressed_shake_frames = 0
        self.last_candidate_count = 0

    def _global_camera_shift(self, gray):
        """Estimate translation/rotation/zoom using robust affine optical flow.

        RANSAC makes camera-motion suppression less sensitive to one independently
        moving object than a plain median displacement test.
        """
        small_w = 320
        h, w = gray.shape[:2]
        small_h = max(1, int(round(h * small_w / max(1, w))))
        small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
        previous = self.previous_gray_small
        self.previous_gray_small = small
        if previous is None:
            return 0.0, False
        points = cv2.goodFeaturesToTrack(previous, maxCorners=150, qualityLevel=0.01,
                                         minDistance=6, blockSize=5)
        if points is None or len(points) < 12:
            return 0.0, False
        try:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                previous, small, points, None, winSize=(21, 21), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03)
            )
        except cv2.error:
            return 0.0, False
        if nxt is None or status is None:
            return 0.0, False
        valid = status.reshape(-1) == 1
        old_pts = points.reshape(-1, 2)[valid]
        new_pts = nxt.reshape(-1, 2)[valid]
        if len(new_pts) < 10:
            return 0.0, False

        try:
            affine, inlier_mask = cv2.estimateAffinePartial2D(
                old_pts, new_pts, method=cv2.RANSAC, ransacReprojThreshold=2.5,
                maxIters=200, confidence=0.95, refineIters=10
            )
        except cv2.error:
            affine, inlier_mask = None, None
        if affine is not None and inlier_mask is not None:
            inliers = inlier_mask.reshape(-1).astype(bool)
            inlier_ratio = float(np.mean(inliers)) if len(inliers) else 0.0
            transformed = old_pts @ affine[:, :2].T + affine[:, 2]
            residual = float(np.median(np.linalg.norm(transformed - new_pts, axis=1)))
            a, b = float(affine[0, 0]), float(affine[1, 0])
            scale = float(np.sqrt(abs(np.linalg.det(affine[:, :2]))))
            angle = abs(float(np.arctan2(b, a)))
            center = np.array([small_w / 2.0, small_h / 2.0], dtype=np.float32)
            moved_center = affine[:, :2] @ center + affine[:, 2]
            translation = float(np.linalg.norm(moved_center - center))
            rotation_shift = angle * max(small_w, small_h) / 2.0
            scale_shift = abs(scale - 1.0) * max(small_w, small_h) / 2.0
            magnitude = max(translation, rotation_shift, scale_shift)
            coherent = inlier_ratio >= 0.45 and residual < 2.8
            return magnitude, bool(magnitude >= self.camera_motion_threshold_px and coherent)

        shifts = new_pts - old_pts
        median_shift = np.median(shifts, axis=0)
        residual = np.median(np.linalg.norm(shifts - median_shift, axis=1))
        magnitude = float(np.linalg.norm(median_shift))
        coherent = residual < max(1.8, magnitude * 0.35)
        return magnitude, bool(magnitude >= self.camera_motion_threshold_px and coherent)

    def get_stats(self):
        return {
            "frames_seen": self.frames_seen,
            "camera_moving": self.camera_moving,
            "camera_motion_px": round(self.camera_motion_px, 2),
            "suppressed_shake_frames": self.suppressed_shake_frames,
            "last_candidate_count": self.last_candidate_count,
        }

    def detect(self, frame):
        if frame is None or not hasattr(frame, "shape") or frame.size == 0:
            return []
        height, width = frame.shape[:2]
        frame_area = height * width
        if frame_area <= 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        shift, camera_moving = self._global_camera_shift(gray)
        self.camera_motion_px = shift
        self.camera_moving = camera_moving
        self.frames_seen += 1

        if camera_moving:
            # Quickly re-learn the current scene after a pan/shake; do not turn the
            # whole image into a fake moving object.
            self.background.apply(frame, learningRate=0.75)
            self.suppressed_shake_frames += 1
            self.last_candidate_count = 0
            return []

        mask = self.background.apply(frame)
        if self.frames_seen <= self.warmup_frames:
            self.last_candidate_count = 0
            return []
        mask = cv2.medianBlur(mask, 5)
        # A 3x3 kernel preserves smaller targets in the 640px browser feed; a
        # 5x5 opening often erased small or distant objects entirely.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        max_area = frame_area * self.max_area_ratio
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area_px or area > max_area:
                continue
            x, y, box_w, box_h = cv2.boundingRect(contour)
            if box_w < 8 or box_h < 8:
                continue
            if box_w >= width * 0.88 and box_h >= height * 0.70:
                continue
            aspect = box_w / max(1.0, float(box_h))
            if aspect > 12 or aspect < 1 / 12:
                continue
            fill = min(1.0, area / max(1.0, float(box_w * box_h)))
            candidates.append({
                "bbox": (int(x), int(y), int(box_w), int(box_h)),
                # This is a region-shape score, not a probability that the region is an object.
                "confidence": round(max(0.30, min(0.85, 0.35 + 0.45 * fill)), 3),
                "class": "moving_object",
                "source": "motion",
                "area_px": int(area),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "classification_note": "Motion region only; object type is not classified.",
            })
        candidates.sort(key=lambda item: item["area_px"], reverse=True)
        kept = []
        for candidate in candidates:
            x, y, w, h = candidate["bbox"]
            duplicate = False
            for existing in kept:
                ex, ey, ew, eh = existing["bbox"]
                ix1, iy1, ix2, iy2 = max(x, ex), max(y, ey), min(x + w, ex + ew), min(y + h, ey + eh)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = w * h + ew * eh - inter
                if union and inter / union > 0.55:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
            if len(kept) >= self.max_objects:
                break
        self.last_candidate_count = len(kept)
        return kept
