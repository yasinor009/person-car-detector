"""Offline reproducible video or local live camera: two-class tracking + range.
No model/camera calibration is inferred from metadata or guessed object sizes.
"""
import argparse
import json
import time
from pathlib import Path
import cv2
from laptop_detector import LaptopDetector
from temporal_tracker import TemporalTracker
from camera_geometry import LensRectifier
from metric_range import GroundCalibration, RangeBank


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--ground', help='GroundCalibration JSON, measured in rectified camera coordinates')
    p.add_argument('--lens', help='OpenCV lens calibration JSON; required with --ground')
    p.add_argument('--output', required=True, help='Fresh JSONL output path')
    p.add_argument('--device', default='cpu')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--confidence', type=float, default=.25)
    p.add_argument('--tiled', action='store_true')
    p.add_argument('--show', action='store_true')
    p.add_argument('--max-frames', type=int, default=0)
    args = p.parse_args()
    calibration = GroundCalibration(**json.loads(Path(args.ground).read_text())) if args.ground else None
    if calibration and not args.lens:
        p.error('--ground requires --lens (zero distortion is allowed if measured)')
    rectifier = LensRectifier(args.lens)
    if calibration:
        import numpy as np
        k, _, size = rectifier.calibration
        if list(size) != calibration.image_size or not np.allclose(k, calibration.k):
            p.error('Ground and lens calibration must use identical K and resolution')
    detector = LaptopDetector(args.model, confidence=args.confidence, device=args.device,
                              imgsz=args.imgsz, tiled=args.tiled)
    if detector.model_status != 'ready':
        raise SystemExit(detector.model_message)
    source = int(args.source) if args.source.isdigit() else args.source
    video_file = isinstance(source, str) and Path(source).is_file()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit('Cannot open source')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if video_file and not 0 < fps <= 1000:
        cap.release()
        raise SystemExit('Video requires valid FPS/timebase')
    tracker, ranges = TemporalTracker(), RangeBank()
    start = time.monotonic()
    count = 0
    try:
        with open(args.output, 'x', encoding='utf-8') as log:
            while args.max_frames <= 0 or count < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    if not video_file:
                        raise RuntimeError('Live camera stopped delivering frames')
                    break
                timestamp = count/fps if video_file else time.monotonic()-start
                t = time.perf_counter()
                frame = rectifier.correct(frame)
                detections = detector._detect_frame(frame, min(.25, args.confidence*.5))
                tracks = tracker.update(detections, timestamp, frame.shape, args.confidence)
                objects = ranges.update(tracks, timestamp, calibration, frame.shape)
                record = dict(frame=count, timestamp_s=timestamp, objects=objects,
                              processing_ms=(time.perf_counter()-t)*1000,
                              source_mode='offline_all_frames' if video_file else 'live',
                              field_validated=False)
                log.write(json.dumps(record, allow_nan=False)+'\n')
                log.flush()
                count += 1
                if args.show:
                    for d in objects:
                        x, y, w, h = map(int, d['bbox'])
                        distance = d['distance_m']
                        label = f"{d['class']} #{d['track_id']} {d['confidence']:.2f} "
                        label += 'range invalid' if distance is None else f"{distance:.1f}m +/-{d['distance']['uncertainty_std_m']:.1f} (1sd)"
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 220, 80), 2)
                        cv2.putText(frame, label, (x, max(18, y-5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 220, 80), 1)
                    cv2.imshow('Person / Car metric research', frame)
                    if cv2.waitKey(1) & 255 == ord('q'):
                        break
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
    print(json.dumps(dict(frames=count, elapsed_s=time.monotonic()-start, field_validated=False)))


if __name__ == '__main__':
    main()
