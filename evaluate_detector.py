#!/usr/bin/env python3
"""Evaluate a person/car detector on a labelled YOLO-format test set.

Expected layout:
  dataset_root/images/**/*.jpg (or png/webp)
  dataset_root/labels/**/*.txt
Each label line: class_id x_center y_center width height (normalized 0..1).
COCO class IDs: 0=person, 2=car; other classes are ignored. Negative images must have an
empty label file so unlabelled positives are not silently counted as negatives.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from drone_detector import DroneDetector

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1, x2, y2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def read_yolo_labels(path: Path, image_w: int, image_h: int):
    boxes = []
    if not path.exists():
        raise FileNotFoundError(f"Missing label file: {path} (use an empty .txt for a negative image)")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid label at {path}:{line_number}")
        cls, xc, yc, bw, bh = map(float, parts[:5])
        if cls not in (0, 2):
            continue
        if not all(np.isfinite(v) for v in (cls, xc, yc, bw, bh)) or bw <= 0 or bh <= 0:
            raise ValueError(f"Invalid normalized box at {path}:{line_number}")
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError(f"Label values must be normalized to 0..1 at {path}:{line_number}")
        x1 = (xc - bw / 2.0) * image_w
        y1 = (yc - bh / 2.0) * image_h
        x2 = (xc + bw / 2.0) * image_w
        y2 = (yc + bh / 2.0) * image_h
        boxes.append({"box": (x1, y1, x2, y2), "class_id": int(cls)})
    return boxes


def score_predictions(cached, threshold: float, iou_threshold: float, include_per_class=True):
    tp = fp = fn = 0
    image_rows = []
    for row in cached:
        gt_boxes = row["ground_truth"]
        preds = sorted((p for p in row["predictions"] if p["confidence"] >= threshold),
                       key=lambda item: item["confidence"], reverse=True)
        matched = set()
        local_tp = local_fp = 0
        for pred in preds:
            candidates = [(iou(pred["box"], gt["box"]), idx) for idx, gt in enumerate(gt_boxes) if idx not in matched and pred["class_id"] == gt["class_id"]]
            best_iou, best_idx = max(candidates, default=(0.0, -1))
            if best_iou >= iou_threshold and best_idx >= 0:
                matched.add(best_idx)
                local_tp += 1
            else:
                local_fp += 1
        local_fn = len(gt_boxes) - len(matched)
        tp += local_tp
        fp += local_fp
        fn += local_fn
        image_rows.append({
            "image": row["image"], "ground_truth": len(gt_boxes), "predictions": len(preds),
            "tp": local_tp, "fp": local_fp, "fn": local_fn,
        })
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    report = {
        "confidence_threshold": float(threshold), "iou_threshold": float(iou_threshold),
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "precision_percent": round(100 * precision, 2), "recall_percent": round(100 * recall, 2),
        "f1_percent": round(100 * f1, 2),
        "false_positives_per_image": round(fp / len(cached), 4) if cached else 0.0,
        "per_image": image_rows,
    }

    negatives=[r for r in image_rows if r['ground_truth']==0]
    negative_fp=sum(r['fp'] for r in negatives)
    negative_alarms=sum(r['fp']>0 for r in negatives)
    trials=tp+fp
    if trials:
        z=1.96;p=tp/trials;denom=1+z*z/trials
        center=(p+z*z/(2*trials))/denom
        half=z*((p*(1-p)/trials+z*z/(4*trials*trials))**.5)/denom
        interval=[max(0.,center-half),min(1.,center+half)]
    else:interval=None
    report.update(negative_images=len(negatives),negative_images_with_false_positives=negative_alarms,
                  negative_image_false_alarm_rate=negative_alarms/len(negatives) if negatives else None,
                  negative_false_positives=negative_fp,precision_wilson95=interval)
    if include_per_class:
        report['per_class']={}
        for cls,name in ((0,'person'),(2,'car')):
            rows=[dict(image=r['image'],ground_truth=[g for g in r['ground_truth'] if g['class_id']==cls],
                       predictions=[p for p in r['predictions'] if p['class_id']==cls]) for r in cached]
            one=score_predictions(rows,threshold,iou_threshold,include_per_class=False)
            one.pop('per_image')
            report['per_class'][name]=one
    return report


def select_operating_point(results, objective='precision', minimum_recall=.5):
    """Choose on validation data only; test data must remain held out."""
    feasible=[r for r in results if r['recall']>=minimum_recall and r['true_positives']>0]
    if not feasible:raise ValueError('No threshold satisfies minimum recall; collect data or change the target explicitly')
    if objective=='precision':
        return max(feasible,key=lambda r:(r['precision_wilson95'][0],r['precision'],r['recall']))
    if objective=='f1':return max(feasible,key=lambda r:(r['f1'],r['precision'],r['recall']))
    raise ValueError('Unknown operating point objective')


def main():
    parser = argparse.ArgumentParser(description="Measure person/car model precision/recall on labelled images.")
    parser.add_argument("--model", default=None, help="COCO YOLO11 ONNX path")
    parser.add_argument("--tiled", action="store_true", help="Evaluate optional full image + four slices")
    parser.add_argument("--objective", choices=("precision", "f1"), default="precision")
    parser.add_argument("--min-recall", type=float, default=.5, help="Required validation recall for threshold selection")
    parser.add_argument("--dataset", required=True, help="Directory containing images/ and labels/ subdirectories")
    parser.add_argument("--conf", type=float, default=0.60, help="Confidence threshold (minimum live threshold 0.55)")
    parser.add_argument("--iou", type=float, default=0.50, help="IoU match threshold, default 0.50")
    parser.add_argument("--output", default="evaluation_report.json", help="Output JSON report path")
    parser.add_argument("--sweep", action="store_true", help="Evaluate thresholds 0.55..0.95 and select a validation operating threshold")
    args = parser.parse_args()

    root = Path(args.dataset).expanduser().resolve()
    image_root, label_root = root / "images", root / "labels"
    if not image_root.is_dir() or not label_root.is_dir():
        parser.error("Dataset must contain both images/ and labels/ folders.")
    images = sorted(p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not images:
        parser.error("No supported images found under images/.")
    if not 0.55 <= args.conf <= 0.99 or not 0.1 <= args.iou <= 0.95 or not 0 <= args.min_recall <= 1:
        parser.error("Use confidence 0.55..0.99, IoU 0.10..0.95, and minimum recall 0..1.")

    detector = DroneDetector(model_path=args.model)
    detector.tiled_inference = args.tiled
    if detector.model_status != "ready":
        print(f"Model is not ready: {detector.model_message}", file=sys.stderr)
        return 2
    # Cache one low-threshold inference per image, then score several thresholds.
    # This avoids rerunning the neural network for each threshold candidate.
    detector.set_confidence(0.55 if args.sweep else args.conf)
    cached = []
    for image_path in images:
        relative = image_path.relative_to(image_root)
        label_path = (label_root / relative).with_suffix(".txt")
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        h, w = frame.shape[:2]
        gt = read_yolo_labels(label_path, w, h)
        detections = detector.detect(frame, temporal=False)
        cached.append({
            "image": str(relative), "ground_truth": gt,
            "predictions": [
                {"box": (int(d["bbox"][0]), int(d["bbox"][1]),
                         int(d["bbox"][0] + d["bbox"][2]), int(d["bbox"][1] + d["bbox"][3])),
                 "confidence": float(d["confidence"]), "class_id": d["class_id"]} for d in detections
            ],
        })

    thresholds = [round(float(v), 2) for v in np.arange(0.55, 0.951, 0.05)] if args.sweep else [args.conf]
    results = [score_predictions(cached, threshold, args.iou) for threshold in thresholds]
    best = select_operating_point(results, args.objective, args.min_recall) if args.sweep else results[0]
    report = {
        "model_path": detector.model_path, "images": len(images),
        "evaluation_mode": "confidence_sweep" if args.sweep else "single_threshold",
        "selected_threshold": best["confidence_threshold"],
        "selection_objective": args.objective if args.sweep else None,
        "minimum_recall": args.min_recall if args.sweep else None,
        "tiled_inference": args.tiled,
        "selected_f1_percent": best["f1_percent"],
        "selected_precision_percent": best["precision_percent"],
        "selected_recall_percent": best["recall_percent"],
        "threshold_results": results,
        "note": "Results apply only to this labelled dataset. Wilson intervals assume independent detections and can be optimistic for correlated video frames. The sweep reuses predictions made at confidence 0.55 after class-aware NMS, so it is a threshold-selection aid; final reporting should use a separate held-out test set.",
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "threshold_results"}, ensure_ascii=False, indent=2))
    if args.sweep:
        print("Threshold sweep:")
        for row in results:
            print(f"  conf={row['confidence_threshold']:.2f} P={row['precision_percent']:.2f}% R={row['recall_percent']:.2f}% F1={row['f1_percent']:.2f}%")
    print(f"Full report saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
