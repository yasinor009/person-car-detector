"""Tune each class on validation data; refuse to label a failed 90/90 target as met."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from laptop_detector import LaptopDetector
from evaluate_laptop import dataset_rows,filter_ignored
from evaluate_detector import score_predictions


def choose_class_thresholds(cached,target=.9,min_objects=50):
    rows=[score_predictions(cached,float(t),.5) for t in np.arange(.05,.951,.025)]
    chosen={};status={}
    for name in ('person','car'):
        def satisfies(row):
            m=row['per_class'][name];tp=m['true_positives'];fp=m['false_positives'];fn=m['false_negatives']
            return tp+fn>=min_objects and tp+fp>0 and tp/(tp+fp)>=target and tp/(tp+fn)>=target
        feasible=[r for r in rows if satisfies(r)]
        if feasible:
            best=max(feasible,key=lambda r:(min(r['per_class'][name]['precision'],r['per_class'][name]['recall']),r['per_class'][name]['f1']))
        else:
            best=max(rows,key=lambda r:(min(r['per_class'][name]['precision'],r['per_class'][name]['recall']),r['per_class'][name]['f1']))
        chosen[name]=best['confidence_threshold']
        status[name]=dict(target_met=bool(feasible),metrics=best['per_class'][name])
    return dict(target_met=all(v['target_met'] for v in status.values()),class_thresholds=chosen,classes=status,
                target_precision=target,target_recall=target,scope='Validation threshold tuning only. Freeze thresholds before separate test and measured >=150m evaluation.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--model',required=True)
    p.add_argument('--split',choices=['validation'],required=True)
    p.add_argument('--label-schema',choices=('coco','two-class'),default='two-class')
    p.add_argument('--device',default='auto');p.add_argument('--imgsz',type=int,default=1280)
    p.add_argument('--tiled',action='store_true');p.add_argument('--tile-size',type=int,default=960)
    p.add_argument('--output',default='validation/threshold_tuning.json');p.add_argument('--min-objects',type=int,default=50)
    args=p.parse_args()
    model=LaptopDetector(args.model,.01,args.device,args.imgsz,args.tiled,args.tile_size)
    if model.model_status!='ready':raise SystemExit(model.model_message)
    cached=[]
    for row,frame in dataset_rows(args.dataset,args.label_schema):
        ds=model.detect(frame,temporal=False)
        if model.last_inference_error:raise RuntimeError(model.last_inference_error)
        row['predictions']=[dict(box=[d['bbox'][0],d['bbox'][1],d['bbox'][0]+d['bbox'][2],d['bbox'][1]+d['bbox'][3]],class_id=d['class_id'],confidence=d['confidence']) for d in ds]
        row["predictions"]=filter_ignored(row["predictions"],row["ground_truth"],row["ignored_boxes"])
        cached.append(row)
        if len(cached)%20==0:print(f'Validation images: {len(cached)}',flush=True)
    result=choose_class_thresholds(cached,min_objects=args.min_objects)
    result.update(model_path=model.model_path,images=len(cached),device=model.device)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if not result['target_met']:
        print('90/90 target NOT met. Proposed thresholds are diagnostic only; do not claim acceptance.')
    return 0 if result['target_met'] else 2
if __name__=='__main__':raise SystemExit(main())
