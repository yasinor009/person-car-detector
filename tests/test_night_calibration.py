"""Regression coverage for configurable recall and detector-independent calibration."""
import json
import numpy as np
import pytest
from app import DroneDistanceService
from drone_detector import DroneDetector


def test_user_threshold_controls_decode_and_track_confirmation(tmp_path):
    d = DroneDetector(model_path=str(tmp_path / 'absent.onnx'), confidence=.25)
    output = np.zeros((1, 84, 1), np.float32)
    output[0, :4, 0] = [200, 200, 80, 60]
    output[0, 6, 0] = .4  # COCO car, below the old hidden .55 floor
    items = d._decode(output, (640, 640, 3), 1, 1, 0, 0)
    assert len(items) == 1
    assert d._confirm(items, 1., (640, 640, 3)) == []
    assert d._confirm(items, 1.1, (640, 640, 3)) == []
    assert d._confirm(items, 1.2, (640, 640, 3))[0]['class'] == 'car'
    d.set_confidence(.6)
    assert d._decode(output, (640, 640, 3), 1, 1, 0, 0) == []
    for t in (2., 2.1, 2.2):
        assert d._confirm(items, t, (640, 640, 3)) == []
    assert d.get_stats()['effective_confidence_threshold'] == .6


def test_manual_reference_calibrates_without_semantic_detection_and_scales(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, 'SETTINGS_FILE', tmp_path / 'settings.json')
    d = DroneDetector(model_path=str(tmp_path / 'absent.onnx'))
    s = DroneDistanceService(browser_camera=True, detector=d)
    try:
        assert s.latest_detections == [] and s.selected_target is None
        # Frozen portrait reference, 100px object, measured width 2m at 20m.
        result = s.calibrate_distance(2., 20., 100., frame_width_px=720)
        assert result['focal_length_px'] == 1000
        estimate = s.distance.estimate(150., target_width_m=2., frame_width_px=1080)
        assert estimate.distance_m == pytest.approx(20.)
        s.update_settings({'confidence_threshold': .3, 'tiled_inference': True})
        saved = json.loads(config.SETTINGS_FILE.read_text())
        assert saved['tiled_inference'] is True
        assert saved['confidence_threshold'] == .3
        restored = DroneDistanceService(browser_camera=True, detector=DroneDetector(model_path=str(tmp_path / 'absent.onnx')))
        try:
            assert restored.detector.tiled_inference is True
            assert restored.distance.focal_length_px == 1000
        finally:
            restored.shutdown()
    finally:
        s.shutdown()
