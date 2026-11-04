"""Monocular pinhole-model distance estimation using a calibrated camera and known target width."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class DistanceEstimate:
    distance_m: Optional[float]
    status: str
    focal_length_px: Optional[float] = None
    bbox_width_px: Optional[float] = None
    target_width_m: Optional[float] = None
    method: str = "monocular_pinhole_known_width"
    sensitivity: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "distance_m": self.distance_m,
            "status": self.status,
            "focal_length_px": self.focal_length_px,
            "bbox_width_px": self.bbox_width_px,
            "target_width_m": self.target_width_m,
            "method": self.method,
            "quantity": "optical_axis_depth",
            "sensitivity": self.sensitivity,
        }


class DistanceEstimator:
    """Estimate range as distance = focal_length_px * known_width_m / bbox_width_px.

    The focal length must be calibrated with a known object width at a known distance.
    The target width supplied for inference must correspond to the object's dimension
    measured by the detector. Results are approximate and are not valid for unknown
    target sizes, severe perspective, or boxes that do not cover the same dimension.
    """

    def __init__(self, focal_length_px: Optional[float] = None,
                 target_width_m: Optional[float] = None,
                 calibration_frame_width_px: Optional[float] = None):
        self.focal_length_px = self._positive_or_none(focal_length_px)
        self.target_width_m = self._positive_or_none(target_width_m)
        self.calibration_frame_width_px = self._positive_or_none(calibration_frame_width_px)

    @staticmethod
    def _positive_or_none(value):
        if value is None or value == "":
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) and result > 0 else None

    def calibrate(self, known_width_m: float, known_distance_m: float,
                  observed_width_px: float, frame_width_px: Optional[float] = None) -> float:
        width = self._positive_or_none(known_width_m)
        distance = self._positive_or_none(known_distance_m)
        pixels = self._positive_or_none(observed_width_px)
        if width is None or distance is None or pixels is None:
            raise ValueError("Known width, known distance, and measured pixel width must all be positive numbers.")
        frame_width = self._positive_or_none(frame_width_px)
        if frame_width_px not in (None, "") and frame_width is None:
            raise ValueError("Calibration frame width must be a positive number of pixels.")
        focal = pixels * distance / width
        if not math.isfinite(focal) or focal <= 0:
            raise ValueError("Calibration produced an invalid focal length.")
        self.focal_length_px = focal
        # Focal length in pixels scales with the processed frame width. Recording
        # the calibration resolution prevents portrait/landscape or browser stream
        # size changes from silently changing the estimated range.
        self.calibration_frame_width_px = frame_width
        self.target_width_m = width
        return focal

    def set_target_width(self, width_m: Optional[float]) -> Optional[float]:
        if width_m in (None, ""):
            self.target_width_m = None
        else:
            width = self._positive_or_none(width_m)
            if width is None:
                raise ValueError("Target width must be a positive number in meters.")
            self.target_width_m = width
        return self.target_width_m

    def estimate(self, bbox_width_px: float,
                 target_width_m: Optional[float] = None,
                 frame_width_px: Optional[float] = None) -> DistanceEstimate:
        try:
            pixels = float(bbox_width_px)
        except (TypeError, ValueError):
            pixels = 0.0
        width = self._positive_or_none(target_width_m) if target_width_m not in (None, "") else self.target_width_m

        if not math.isfinite(pixels) or pixels <= 0:
            return DistanceEstimate(None, "invalid_bbox_width", self.focal_length_px, pixels, width)
        if self.focal_length_px is None:
            return DistanceEstimate(None, "camera_not_calibrated", None, pixels, width)
        if width is None:
            return DistanceEstimate(None, "target_width_not_set", self.focal_length_px, pixels, None)

        effective_focal = self.focal_length_px
        current_frame_width = self._positive_or_none(frame_width_px)
        if current_frame_width is not None and self.calibration_frame_width_px is not None:
            effective_focal *= current_frame_width / self.calibration_frame_width_px
        distance = effective_focal * width / pixels
        if not math.isfinite(distance) or distance <= 0:
            return DistanceEstimate(None, "invalid_estimate", effective_focal, pixels, width)
        # Exact perturbation bounds, NOT a confidence interval. Width/focal/pose
        # are held fixed; real total error may be substantially larger.
        sensitivity = {
            "assumed_bbox_error_px": 2.0,
            "lower_m": effective_focal * width / (pixels + 2.0),
            "upper_m": effective_focal * width / (pixels - 2.0) if pixels > 2.0 else None,
            "relative_pixel_error": 2.0 / pixels,
            "scope": "bbox_width_only_fixed_focal_and_target_size_not_total_uncertainty",
        }
        return DistanceEstimate(round(distance, 2), "ok", effective_focal, pixels, width,
                                sensitivity=sensitivity)

    def to_dict(self) -> dict:
        return {
            "focal_length_px": self.focal_length_px,
            "target_width_m": self.target_width_m,
            "calibration_frame_width_px": self.calibration_frame_width_px,
            "resolution_aware": self.calibration_frame_width_px is not None,
            "calibrated": self.focal_length_px is not None,
            "ready": self.focal_length_px is not None and self.target_width_m is not None,
        }
