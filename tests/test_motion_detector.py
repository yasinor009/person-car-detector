import cv2
import numpy as np

from motion_detector import MotionDetector


def test_motion_detector_reports_a_moving_region():
    detector = MotionDetector(min_area_px=120, max_area_ratio=0.5)
    background = np.zeros((240, 320, 3), dtype=np.uint8)
    # Warm up background model on repeated static frames.
    for _ in range(8):
        detector.detect(background)
    first = background.copy()
    cv2.rectangle(first, (40, 70), (100, 140), (255, 255, 255), -1)
    detector.detect(first)
    second = background.copy()
    cv2.rectangle(second, (80, 70), (140, 140), (255, 255, 255), -1)
    found = detector.detect(second)
    assert found
    assert any(item["source"] == "motion" and item["class"] == "moving_object" for item in found)
    x, y, w, h = found[0]["bbox"]
    assert x >= 0 and y >= 0 and w > 0 and h > 0


def test_motion_detector_keeps_small_moving_region():
    detector = MotionDetector(min_area_px=40, warmup_frames=3)
    background = np.zeros((240, 320, 3), dtype=np.uint8)
    for _ in range(5):
        detector.detect(background)
    first = background.copy()
    cv2.rectangle(first, (40, 70), (53, 83), (255, 255, 255), -1)
    detector.detect(first)
    second = background.copy()
    cv2.rectangle(second, (52, 70), (65, 83), (255, 255, 255), -1)
    found = detector.detect(second)
    assert any(item["source"] == "motion" for item in found)


def test_global_camera_shift_detects_translated_textured_scene():
    detector = MotionDetector(camera_motion_threshold_px=2.0)
    rng = np.random.default_rng(42)
    base = rng.integers(0, 256, size=(240, 320), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (3, 3), 0)
    shifted = cv2.warpAffine(base, np.float32([[1, 0, 10], [0, 1, 0]]), (320, 240), borderMode=cv2.BORDER_REFLECT)
    detector._global_camera_shift(base)
    magnitude, moving = detector._global_camera_shift(shifted)
    assert magnitude >= 2.0
    assert moving is True
