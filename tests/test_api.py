import json
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

import pytest
import numpy as np

import config
from app import DroneDistanceService, make_handler, run_server


@pytest.fixture
def running_demo_server(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config, "DETECTION_LOG_DIR", tmp_path / "detections")
    service = DroneDistanceService(demo=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield service, base
    finally:
        service.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request_json(base, path, method="GET", payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(base + path, data=data, method=method,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=3) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def test_health_and_dashboard_routes(running_demo_server):
    _, base = running_demo_server
    status, health = request_json(base, "/health")
    assert status == 200 and health["ok"] is True and health["version"] == "4.0-laptop-preview"
    # The dashboard is HTML, so inspect it separately.
    with urlopen(base + "/") as response:
        assert response.status == 200
        html = response.read().decode("utf-8")
    assert "تشخیص اجسام متحرک" in html
    assert "facingMode: { exact: 'environment' }" in html
    assert "modelDiagnostics" in html
    assert "motionDiagnostics" in html


def test_demo_pipeline_and_metric_distance_api(running_demo_server):
    _, base = running_demo_server
    status, calibrated = request_json(base, "/api/calibrate", "POST", {
        "known_width_m": 0.3,
        "known_distance_m": 10,
        "observed_width_px": 80,
    })
    assert status == 200
    assert calibrated["ok"] is True
    assert calibrated["focal_length_px"] == pytest.approx(2666.667, abs=0.01)

    status, started = request_json(base, "/api/start", "POST", {})
    assert status == 200 and started["ok"] is True
    time.sleep(0.25)
    status, result = request_json(base, "/api/status")
    assert status == 200
    assert "motion" in result and "decode_layout" in result["detector"]
    assert result["mode"] == "demo"
    assert result["detection_running"] is True
    assert result["frame_count"] > 0
    assert result["detections"]
    assert result["detections"][0]["distance_m"] is not None
    assert "synthetic" in result["detections"][0]["class"]

    status, stopped = request_json(base, "/api/stop", "POST", {})
    assert status == 200 and stopped["ok"] is True


def test_invalid_calibration_returns_client_error(running_demo_server):
    _, base = running_demo_server
    request = Request(base + "/api/calibrate", data=json.dumps({
        "known_width_m": 0,
        "known_distance_m": 10,
        "observed_width_px": 80,
    }).encode(), method="POST", headers={"Content-Type": "application/json"})
    try:
        urlopen(request, timeout=3)
        assert False, "Expected HTTP 400"
    except Exception as exc:
        assert getattr(exc, "code", None) == 400
        payload = json.loads(exc.read().decode())
        assert payload["ok"] is False


def test_settings_api_validates_confidence(running_demo_server):
    _, base = running_demo_server
    request = Request(base + "/api/settings", data=json.dumps({"confidence_threshold": 1.5}).encode(),
                      method="POST", headers={"Content-Type": "application/json"})
    try:
        urlopen(request, timeout=3)
        assert False, "Expected HTTP 400"
    except Exception as exc:
        assert getattr(exc, "code", None) == 400
        payload = json.loads(exc.read().decode())
        assert payload["ok"] is False


def test_server_refuses_unauthenticated_remote_bind():
    with pytest.raises(ValueError, match="localhost only"):
        run_server(host="0.0.0.0", port=5000, demo=True)


def test_browser_camera_mode_accepts_jpeg_and_processes(tmp_path, monkeypatch):
    import cv2
    import numpy as np

    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config, "DETECTION_LOG_DIR", tmp_path / "detections")
    service = DroneDistanceService(browser_camera=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        frame = np.full((240, 320, 3), 80, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", frame)
        assert ok
        request = Request(base + "/api/camera-frame", data=encoded.tobytes(), method="POST",
                          headers={"Content-Type": "image/jpeg"})
        with urlopen(request, timeout=3) as response:
            payload = json.loads(response.read().decode())
            assert response.status == 200 and payload["ok"] is True
            assert payload["width"] == 320 and payload["height"] == 240

        status, started = request_json(base, "/api/start", "POST", {})
        assert status == 200 and started["ok"] is True
        deadline = time.time() + 2
        while time.time() < deadline and service.frame_count == 0:
            time.sleep(0.05)
        current = service.status()
        assert current["mode"] == "browser_camera"
        assert current["frame_count"] > 0
        assert current["camera"]["visible_open"] is True
        assert service.get_frame("visible") is not None
    finally:
        service.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_selected_target_calibration_and_width_are_automatic_from_bbox(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config, "DETECTION_LOG_DIR", tmp_path / "detections")
    service = DroneDistanceService(demo=True)
    try:
        ok, _ = service.start_detection()
        assert ok
        deadline = time.time() + 2
        while time.time() < deadline and service.latest_raw_visible is None:
            time.sleep(0.05)
        status = service.status()
        assert status["frame_size"]["width"] == 640
        bbox = status["detections"][0]["bbox"]
        selected = service.select_target(bbox=bbox)
        assert selected["width_px"] == bbox[2]
        calibrated = service.calibrate_selected_target(known_width_m=0.30, known_distance_m=5.0)
        assert calibrated["observed_width_px"] > 0
        assert calibrated["focal_length_px"] == pytest.approx(calibrated["observed_width_px"] * 5.0 / 0.30)
        assert calibrated["selected_target"]["distance_m"] is not None
        updated = service.set_selected_target_width(0.60)
        assert updated["target_width_m"] == pytest.approx(0.60)
        assert updated["distance_m"] is not None
        service.clear_selected_target()
        assert service.status()["selected_target"] is None
    finally:
        service.shutdown()


def test_target_selection_and_selected_calibration_api(running_demo_server):
    _, base = running_demo_server
    status, started = request_json(base, "/api/start", "POST", {})
    assert status == 200 and started["ok"] is True
    deadline = time.time() + 2
    current = {}
    while time.time() < deadline:
        _, current = request_json(base, "/api/status")
        if current.get("detections") and current.get("frame_size"):
            break
        time.sleep(0.05)
    assert current.get("detections")
    bbox = current["detections"][0]["bbox"]
    status, selected = request_json(base, "/api/target/select", "POST", {"bbox": bbox})
    assert status == 200 and selected["selected_target"]["bbox"][2] > 0
    status, calibrated = request_json(base, "/api/calibrate-selected", "POST", {
        "known_width_m": 0.3, "known_distance_m": 5.0
    })
    assert status == 200 and calibrated["ok"] is True
    assert calibrated["observed_width_px"] > 0
    assert calibrated["selected_target"]["distance_m"] is not None


def test_motion_blob_does_not_get_distance_from_unrelated_global_width(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config, "DETECTION_LOG_DIR", tmp_path / "detections")
    service = DroneDistanceService(demo=False)
    try:
        service.distance.calibrate(known_width_m=0.3, known_distance_m=10, observed_width_px=80)
        item = service._attach_distance({"bbox": (10, 10, 40, 20), "source": "motion"})
        assert item["distance_m"] is None
        assert item["distance"]["status"] == "target_width_not_set"
    finally:
        service.shutdown()


def test_manual_roi_does_not_shrink_to_motion_blob(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config, "DETECTION_LOG_DIR", tmp_path / "detections")
    service = DroneDistanceService(demo=False)
    try:
        service.selected_target = {
            "bbox": [100, 80, 90, 70], "class": "selected_object", "source": "manual_roi",
            "tracking_status": "tracking", "distance_m": None, "distance": {}, "width_px": 90,
        }
        service.selected_target_width_m = 0.30
        service._selected_width_samples.clear()
        service._selected_width_samples.append(90.0)
        service._optical_flow_bbox = lambda frame, previous: [102, 82, 88, 68]
        service._initialize_selected_tracker = lambda frame, bbox: None
        detections = [{"bbox": (108, 90, 18, 12), "source": "motion", "class": "moving_object", "confidence": 0.6}]
        service._update_selected_target(detections, np.zeros((240, 320, 3), dtype=np.uint8))
        x, y, w, h = service.selected_target["bbox"]
        assert w >= 80 and h >= 60
        assert service.selected_target["tracking_status"] == "tracking_optical_flow"
        assert service.selected_target["width_px"] == w
    finally:
        service.shutdown()


def test_detector_reports_inference_errors():
    import logging
    from collections import deque
    from drone_detector import DroneDetector

    class BrokenSession:
        def run(self, _outputs, _feed):
            raise RuntimeError("synthetic inference failure")

    detector = DroneDetector(model_path="/nonexistent/test-model.onnx")
    detector.model = None
    detector.session = BrokenSession()
    detector.input_name = "images"
    detector.input_width = detector.input_height = 416
    detector.input_layout = "NCHW"
    detector.output_names = []
    detector.output_shapes = []
    detector.input_shape = None
    detector.backend = "onnx"
    detector.model_names = {0: "drone"}
    detector.drone_class_ids = {0}
    detector.model_status = "ready"
    detector.model_message = "ready"
    detector.confidence = 0.25
    detector.history = deque(maxlen=10)
    detector.inference_ms = None
    detector.last_raw_candidates = detector.last_decoded_count = detector.last_detection_count = 0
    detector.last_decode_layout = "unknown"
    detector.last_inference_error = None
    detector.inference_failures = 0
    detector.last_tiled_fallback_used = False
    detector.tiled_inference = False
    detector.logger = logging.getLogger("test-broken-detector")
    detections = detector.detect(np.zeros((80, 80, 3), dtype=np.uint8))
    assert detections == []
    assert "synthetic inference failure" in detector.last_inference_error
    assert detector.inference_failures == 1


def test_unknown_onnx_output_contract_is_rejected():
    from drone_detector import DroneDetector
    with pytest.raises(ValueError, match="Expected YOLO11"):
        DroneDetector._to_rows(np.zeros((1, 25200, 6), dtype=np.float32))


def test_calibrated_distance_is_preserved_with_explicit_stale_status_during_reacquisition(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config, "DETECTION_LOG_DIR", tmp_path / "detections")
    service = DroneDistanceService(demo=False)
    try:
        service.distance.calibrate(known_width_m=0.30, known_distance_m=5.0, observed_width_px=60.0)
        service.selected_target = {
            "bbox": [20, 20, 60, 40], "class": "selected_object", "source": "manual_roi",
            "tracking_status": "reacquiring", "distance_m": None, "distance": {}, "width_px": 60.0,
        }
        service.selected_target_width_m = 0.30
        service._update_selected_distance()
        payload = service._selected_target_payload()
        assert payload["distance_m"] is None
        assert payload["distance"]["status"] == "unreliable_target_measurement"
        assert payload["distance"]["is_stale"] is True

        service.selected_target["tracking_status"] = "lost"
        service._update_selected_distance()
        payload = service._selected_target_payload()
        assert payload["distance_m"] is None
        assert payload["distance"]["status"] == "unreliable_target_measurement"
        assert payload["distance"]["is_stale"] is True
    finally:
        service.shutdown()


def test_selected_calibration_rejects_reacquiring_target(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config, "DETECTION_LOG_DIR", tmp_path / "detections")
    service = DroneDistanceService(demo=False)
    try:
        service.selected_target = {
            "bbox": [20, 20, 60, 40], "class": "selected_object", "source": "manual_roi",
            "tracking_status": "reacquiring", "distance_m": None, "distance": {}, "width_px": 60.0,
        }
        service._selected_width_samples.clear()
        service._selected_width_samples.extend([60.0, 61.0, 60.0])
        with pytest.raises(ValueError, match="ردیابی در حال بازیابی"):
            service.calibrate_selected_target(known_width_m=0.30, known_distance_m=5.0)
        assert service.distance.focal_length_px is None
    finally:
        service.shutdown()


def test_template_tracker_fallback_can_follow_low_texture_roi(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config, "DETECTION_LOG_DIR", tmp_path / "detections")
    service = DroneDistanceService(demo=False)
    try:
        rng = np.random.default_rng(12)
        frame1 = np.zeros((180, 240, 3), dtype=np.uint8)
        patch = rng.integers(20, 240, size=(28, 36, 3), dtype=np.uint8)
        frame1[55:83, 60:96] = patch
        service._initialize_selected_tracker(frame1, [60, 55, 36, 28])
        frame2 = np.zeros_like(frame1)
        frame2[59:87, 66:102] = patch
        service._tracking_points = None  # force the appearance fallback
        result = service._optical_flow_bbox(frame2, [60, 55, 36, 28])
        assert result is not None
        assert abs(result[0] - 66) <= 2
        assert abs(result[1] - 59) <= 2
    finally:
        service.shutdown()
