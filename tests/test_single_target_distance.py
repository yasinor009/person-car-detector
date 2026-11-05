import numpy as np
import pytest

from app import DroneDistanceService


def test_auto_target_requires_persistent_motion_and_exposes_one_box():
    service = DroneDistanceService(demo=True)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    candidate = {
        "bbox": (40, 50, 40, 36),
        "class": "moving_object",
        "source": "motion",
        "confidence": 0.6,
        "area_px": 900,
    }
    try:
        service._maybe_auto_select_target([dict(candidate)], frame)
        assert service.selected_target is None  # single-frame noise is ignored
        service._maybe_auto_select_target([dict(candidate)], frame)
        assert service.selected_target is not None
        assert service.selected_target["auto_selected"] is True
        assert len(service._single_target_detection()) == 1
        assert service.selected_target_width_m is None
    finally:
        service.shutdown()


def test_calibration_uses_recent_stable_widths_not_old_range_history(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config, "DETECTION_LOG_DIR", tmp_path / "detections")
    service = DroneDistanceService(demo=True)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    try:
        service._create_selected_target(frame, (50, 50, 200, 80), "selected_object", "manual_roi")
        # Older widths came from another apparent range. The most recent five
        # frames are stable around 200 px and must be the only calibration basis.
        service._selected_width_samples.clear()
        service._selected_width_samples.extend([20, 40, 80, 100, 110, 120, 130, 198, 201, 199, 200, 200])
        result = service.calibrate_selected_target(0.30, 2.0)
        assert result["observed_width_px"] == pytest.approx(200.0, abs=1.0)
        assert result["focal_length_px"] == pytest.approx(200.0 * 2.0 / 0.30, rel=0.01)
        assert result["selected_target"]["distance_m"] == pytest.approx(2.0, abs=0.05)
        assert result["quality_warning"] is None
        assert service.status()["distance"]["ready"] is True
        # A different selected object must not inherit the previous object's width.
        service._create_selected_target(frame, (70, 60, 80, 70), "moving_object", "motion", auto_selected=True)
        assert service.status()["distance"]["ready"] is False
        assert service.selected_target_width_m is None
    finally:
        service.shutdown()


def test_calibration_rejects_unstable_recent_widths_without_changing_previous_calibration(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config, "DETECTION_LOG_DIR", tmp_path / "detections")
    service = DroneDistanceService(demo=True)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    try:
        service._create_selected_target(frame, (50, 50, 150, 80), "selected_object", "manual_roi")
        service._selected_width_samples.clear()
        service._selected_width_samples.extend([100, 120, 145, 160, 180])
        with pytest.raises(ValueError, match="ناپایدار"):
            service.calibrate_selected_target(0.30, 2.0)
        assert service.distance.focal_length_px is None
        assert service.selected_target_width_m is None
    finally:
        service.shutdown()


def test_calibration_rejects_motion_mask_and_requires_full_object_roi(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config, "DETECTION_LOG_DIR", tmp_path / "detections")
    service = DroneDistanceService(demo=True)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    try:
        service._create_selected_target(frame, (50, 50, 100, 80), "moving_object", "motion", auto_selected=True)
        with pytest.raises(ValueError, match="کادر فقط ناحیه حرکت"):
            service.calibrate_selected_target(0.30, 2.0)
        assert service.distance.focal_length_px is None
    finally:
        service.shutdown()


def test_selected_distance_scales_for_current_camera_frame_width(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config, "DETECTION_LOG_DIR", tmp_path / "detections")
    service = DroneDistanceService(demo=False)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    try:
        service.distance.calibrate(0.30, 2.0, 100, frame_width_px=960)
        service.latest_frame_size = {"width": 1920, "height": 1080}
        service._create_selected_target(frame, (50, 50, 200, 80), "selected_object", "manual_roi")
        service.selected_target_width_m = 0.30
        service._update_selected_distance()
        target = service._selected_target_payload()
        assert target["distance_m"] == pytest.approx(2.0, abs=0.05)
        assert target["distance"]["focal_length_px"] == pytest.approx(2.0 * 100 / 0.30 * 2)
    finally:
        service.shutdown()
