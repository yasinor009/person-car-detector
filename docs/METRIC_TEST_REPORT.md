# Verification of experimental range runtime

Date: 2026-11-04. Linux x86_64 CPU, ONNX Runtime 1.31.1.

- Full suite after API integration: 86 passed, 1 pytest plugin deprecation warning, 4.74 s.
- New tests use synthetic geometry, not real measured ranges.
- Actual ONNX smoke: verified visdrone26m_1280.onnx; four MJPEG frames repeating
  https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg.
- Per-frame processing: 2517.2, 2194.0, 2176.7, 2422.5 ms; source decoding and
  model initialization are excluded from these individual timings.
- Zero confirmed tracks at confidence .25 in this smoke. This aerial model is
  unsuitable evidence of ground-camera detection quality; successful execution
  must not be represented as successful detection or accuracy validation.
- Calibration absent in smoke; no numeric range was generated. The null-range
  contract for observed objects is tested separately in test_metric_range.py.
- No measured distance labels, training, field acceptance, phone FPS, sustained
  thermal test, RAM benchmark or learned-depth comparison was performed.

Research snapshot only. Outstanding items are explicit in METRIC_RUNTIME_FA.md.
