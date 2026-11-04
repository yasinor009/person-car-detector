import numpy as np
import pytest

from distance_estimator import DistanceEstimator
from thermal_processor import ThermalProcessor
from camera_interface import parse_camera_source
from drone_detector import DroneDetector


def test_distance_estimator_requires_calibration_and_known_width():
    estimator = DistanceEstimator()
    assert estimator.estimate(50).status == "camera_not_calibrated"
    estimator.focal_length_px = 1200
    assert estimator.estimate(50).status == "target_width_not_set"


def test_distance_calibration_and_estimation_formula():
    estimator = DistanceEstimator()
    focal = estimator.calibrate(known_width_m=0.4, known_distance_m=10, observed_width_px=80)
    assert focal == pytest.approx(2000.0)
    result = estimator.estimate(40)
    assert result.status == "ok"
    assert result.distance_m == pytest.approx(20.0)
    assert result.method == "monocular_pinhole_known_width"


def test_distance_rejects_invalid_calibration_inputs():
    estimator = DistanceEstimator()
    with pytest.raises(ValueError):
        estimator.calibrate(0, 10, 80)
    with pytest.raises(ValueError):
        estimator.set_target_width(-0.5)
    assert estimator.estimate(0).status == "invalid_bbox_width"


def test_thermal_processor_finds_hot_region_as_candidate_not_drone():
    frame = np.full((120, 160), 30, dtype=np.uint8)
    frame[40:62, 55:83] = 245
    found = ThermalProcessor().detect_hot_objects(frame)
    assert found
    best = max(found, key=lambda item: item["area_px"])
    assert best["source"] == "thermal"
    assert best["class"] == "thermal_candidate"
    assert "not confirmed" in best["classification_note"].lower()
    x, y, w, h = best["bbox"]
    assert w > 0 and h > 0 and x >= 0 and y >= 0


def test_thermal_processor_handles_empty_and_constant_frames():
    processor = ThermalProcessor()
    assert processor.detect_hot_objects(None) == []
    assert processor.detect_hot_objects(np.full((40, 40), 100, dtype=np.uint8)) == []
    assert processor.enhance(None) is None


def test_camera_source_parser():
    assert parse_camera_source("0") == 0
    assert parse_camera_source("12") == 12
    assert parse_camera_source("rtsp://camera/stream") == "rtsp://camera/stream"
    assert parse_camera_source("") is None


def test_detector_does_not_fake_detections_without_model(tmp_path):
    detector = DroneDetector(model_path=str(tmp_path / "absent.pt"))
    assert detector.model_status == "missing_model"
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detector.detect(frame) == []
    assert "not found" in detector.model_message.lower()




def test_thermal_candidate_never_receives_drone_distance(tmp_path, monkeypatch):
    import config
    from app import DroneDistanceService

    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config, "DETECTION_LOG_DIR", tmp_path / "detections")
    service = DroneDistanceService(demo=True)
    service.distance.calibrate(0.3, 10, 80)
    frame = np.full((120, 160), 30, dtype=np.uint8)
    frame[40:62, 55:83] = 245
    service._process(None, frame)
    candidates = [d for d in service.latest_detections if d["source"] == "thermal"]
    assert candidates == []
    assert service.latest_detections == []
    service.shutdown()




def _onnx_detector_for_output(output, confidence=0.25):
    from collections import deque
    import logging

    class FakeSession:
        def run(self, _outputs, feed):
            assert feed["images"].shape == (1, 3, 416, 416)
            return [output]

    detector = DroneDetector.__new__(DroneDetector)
    detector.session = FakeSession()
    detector.input_name = "images"
    detector.input_width = 416
    detector.input_height = 416
    detector.confidence = confidence
    detector.model_status = "ready"
    detector.backend = "onnx"
    detector.model_names = {0: "drone"}
    detector.drone_class_ids = {0}
    detector.logger = logging.getLogger("test-onnx-layout")
    detector.history = deque(maxlen=1000)
    return detector










def test_distance_estimate_scales_focal_length_when_frame_width_changes():
    estimator = DistanceEstimator()
    estimator.calibrate(known_width_m=0.30, known_distance_m=2.0,
                        observed_width_px=100, frame_width_px=960)
    # If the browser switches to a frame twice as wide, both the focal length in
    # pixels and the object's apparent width should double for the same distance.
    result = estimator.estimate(200, frame_width_px=1920)
    assert result.status == "ok"
    assert result.distance_m == pytest.approx(2.0)
    assert estimator.to_dict()["resolution_aware"] is True
