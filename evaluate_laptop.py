"""Evaluate laptop inference on independent images; optionally verify measured far scenes.

Dataset: images/, labels/ (YOLO), optional ranges.json:
{"verified_minimum_range_m":150,"method":"laser/rangefinder or surveyed scene",
 "images":["scene1.jpg", "scene2.jpg"]}
Only list scenes whose targets are ALL independently verified >=150m. Do not use
estimated model distances to populate this manifest.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from laptop_detector import LaptopDetector
from evaluate_detector import score_predictions
from acceptance import acceptance_report


def dataset_rows(root,schema='coco'):
    root=Path(root);images=root/'images';labels=root/'labels'
    ignore_path=root/'ignore_regions.json'
    ignored_map=json.loads(ignore_path.read_text()) if ignore_path.exists() else {}
    paths=sorted(p for p in images.rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png','.webp','.bmp'))
    if not paths:raise ValueError('No dataset images')
    for p in paths:
        frame=cv2.imread(str(p))
        if frame is None:raise ValueError(f'Unreadable image: {p}')
        h,w=frame.shape[:2];relative=p.relative_to(images);lab=(labels/relative).with_suffix('.txt')
        if not lab.is_file():raise ValueError(f'Missing label file (negatives need empty files): {lab}')
        gt=[]
        for line_no,line in enumerate(lab.read_text().splitlines(),1):
            if not line.strip():continue
            values=list(map(float,line.split()))
            if len(values)!=5 or not all(np.isfinite(values)):raise ValueError(f'Invalid YOLO row: {lab}:{line_no}')
            cls,xc,yc,bw,bh=values
            if cls!=int(cls) or cls<0 or cls>=(80 if schema=='coco' else 2):raise ValueError(f'Invalid class: {lab}:{line_no}')
            if not (0<=xc<=1 and 0<=yc<=1 and 0<bw<=1 and 0<bh<=1):raise ValueError(f'Invalid normalized geometry: {lab}:{line_no}')
            mapping={0:0,2:2} if schema=='coco' else {0:0,1:2}
            if int(cls) not in mapping:continue
            gt.append(dict(box=[(xc-bw/2)*w,(yc-bh/2)*h,(xc+bw/2)*w,(yc+bh/2)*h],class_id=mapping[int(cls)],area=bw*w*bh*h))
        ignores=ignored_map.get(relative.as_posix(),[])
        for box in ignores:
            if len(box)!=4 or not all(np.isfinite(box)) or box[2]<=box[0] or box[3]<=box[1]:
                raise ValueError('Invalid ignored region')
        yield dict(image=relative.as_posix(),ground_truth=gt,ignored_boxes=ignores,image_sha256=hashlib.sha256(p.read_bytes()).hexdigest()),frame


def filter_ignored(predictions,ground_truth,ignored_boxes):
    from evaluate_detector import iou
    result=[]
    for d in predictions:
        if any(g['class_id']==d['class_id'] and iou(d['box'],g['box'])>=.5 for g in ground_truth):
            result.append(d);continue
        x,y,X,Y=d['box'];area=max(1e-9,(X-x)*(Y-y))
        ignored=any(max(0,min(X,b[2])-max(x,b[0]))*max(0,min(Y,b[3])-max(y,b[1]))/area>.5 for b in ignored_boxes)
        if not ignored:result.append(d)
    return result


def recall_by_pixel_height(cached):
    """Recall strata only. Pixel height is not measured physical distance."""
    from evaluate_detector import iou
    result={name:{key:{'objects':0,'matched':0} for key in ('under_16px','16_to_32px','at_least_32px')}
            for name in ('person','car')}
    for row in cached:
        matched=set()
        for pred in sorted(row['predictions'],key=lambda d:d['confidence'],reverse=True):
            choices=[(iou(pred['box'],gt['box']),i) for i,gt in enumerate(row['ground_truth'])
                     if i not in matched and gt['class_id']==pred['class_id']]
            overlap,index=max(choices,default=(0.,-1))
            if overlap>=.5:matched.add(index)
        for i,gt in enumerate(row['ground_truth']):
            height=gt['box'][3]-gt['box'][1]
            key='under_16px' if height<16 else ('16_to_32px' if height<32 else 'at_least_32px')
            name='person' if gt['class_id']==0 else 'car'
            result[name][key]['objects']+=1;result[name][key]['matched']+=i in matched
    for bins in result.values():
        for row in bins.values():row['recall']=row['matched']/row['objects'] if row['objects'] else None
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--model',required=True)
    p.add_argument('--split',choices=('development','validation','test'),required=True)
    p.add_argument('--label-schema',choices=('coco','two-class'),default='coco')
    p.add_argument('--device',default='auto');p.add_argument('--confidence',type=float,default=.25)
    p.add_argument('--thresholds');p.add_argument('--imgsz',type=int,default=960)
    p.add_argument('--tiled',action='store_true');p.add_argument('--tile-size',type=int,default=960)
    p.add_argument('--batch',type=int,default=1);p.add_argument('--max-tiles',type=int,default=24)
    p.add_argument('--min-fps',type=float,default=10);p.add_argument('--min-objects',type=int,default=50)
    p.add_argument('--output',default='validation/laptop_evaluation.json')
    args=p.parse_args(argv)
    thresholds=json.loads(Path(args.thresholds).read_text()) if args.thresholds else None
    if thresholds and 'class_thresholds' in thresholds:thresholds=thresholds['class_thresholds']
    model=LaptopDetector(args.model,args.confidence,args.device,args.imgsz,args.tiled,args.tile_size,
                         max_tiles=args.max_tiles,batch_size=args.batch,class_thresholds=thresholds)
    if model.model_status!='ready':raise SystemExit(model.model_message)
    cached=[];latencies=[]
    for row,frame in dataset_rows(args.dataset,args.label_schema):
        # Exclude model startup and one explicit warm-up from the steady-state timer.
        if not cached:model._detect_frame(frame,.05)
        start=time.perf_counter();detections=model.detect(frame,temporal=False);latencies.append(time.perf_counter()-start)
        if model.last_inference_error:raise RuntimeError(model.last_inference_error)
        row['predictions']=[dict(box=[d['bbox'][0],d['bbox'][1],d['bbox'][0]+d['bbox'][2],d['bbox'][1]+d['bbox'][3]],
                                 confidence=d['confidence'],class_id=d['class_id']) for d in detections]
        row['predictions']=filter_ignored(row['predictions'],row['ground_truth'],row['ignored_boxes'])
        cached.append(row)
        if len(cached)%20==0:print(f'Evaluated {len(cached)} images',flush=True)
    # Per-class deployment thresholds were already applied by the detector.
    metrics=score_predictions(cached,0.,.5)
    ranges_file=Path(args.dataset)/'ranges.json';far=None;verified_range=None
    if ranges_file.exists():
        data=json.loads(ranges_file.read_text());names=set(data['images']);verified_range=float(data['verified_minimum_range_m'])
        if not data.get('method') or not names:raise ValueError('Range evidence method and image names are required')
        missing=names-{r['image'] for r in cached}
        if missing:raise ValueError(f'Unknown range-manifest images: {sorted(missing)[:3]}')
        far=score_predictions([r for r in cached if r['image'] in names],0.,.5)
    fps=len(latencies)/sum(latencies)
    report=dict(model=model.get_stats(),split=args.split,metrics=metrics,long_range_metrics=far,recall_by_pixel_height=recall_by_pixel_height(cached),
                mean_inference_ms=float(np.mean(latencies)*1000),p95_inference_ms=float(np.percentile(latencies,95)*1000),
                acceptance=acceptance_report(metrics,args.split,far,verified_range,fps,args.min_fps,min_objects=args.min_objects),
                images=cached,note='Single-image detector evaluation, not video track metrics. No ground-truth size/distance exclusions. Do not tune on the test split.')
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report['acceptance'],indent=2))
    # A failed acceptance gate must be visible to automation, not silently pass.
    return 0 if report['acceptance']['passed'] else 2

if __name__=='__main__':raise SystemExit(main())
