"""Portable multi-model diagnostic benchmark on a fixed prepared development set.

Configuration JSON is a list of {name, model, backend, imgsz, tiled}. Backend:
legacy = original v3 ONNX pipeline, laptop = new runtime, rtdetr = Ultralytics
RT-DETR reference (optional dependency; not a YOLO runtime compatibility claim).
Never run threshold/model selection on the acceptance test set.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from evaluate_laptop import dataset_rows,filter_ignored
from evaluate_detector import score_predictions
from laptop_detector import LaptopDetector,class_contract


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--config',required=True)
    p.add_argument('--label-schema',choices=('two-class','coco'),default='two-class')
    p.add_argument('--device',default='cpu');p.add_argument('--output',default='validation/model_comparison.json')
    args=p.parse_args();root=Path(__file__).resolve().parent
    entries=json.loads(Path(args.config).read_text());reports=[]
    for entry in entries:
        path=Path(entry['model']);path=path if path.is_absolute() else root/path
        if not path.is_file():raise ValueError(f'Missing model for benchmark: {path}')
        backend=entry.get('backend','laptop');size=int(entry.get('imgsz',960))
        if backend=='legacy':
            from drone_detector import DroneDetector
            model=DroneDetector(str(path))
        elif backend=='laptop':
            model=LaptopDetector(path,.05,args.device,size,bool(entry.get('tiled',False)),int(entry.get('tile_size',960)))
        elif backend=='rtdetr':
            from ultralytics import RTDETR
            model=RTDETR(str(path));_,mapping=class_contract(model.names)
        else:raise ValueError(f'Unknown benchmark backend: {backend}')
        if backend!='rtdetr' and model.model_status!='ready':raise ValueError(model.model_message)
        cached=[];latency=[]
        for row,frame in dataset_rows(args.dataset,args.label_schema):
            if backend=='rtdetr':
                def infer():
                    r=model.predict(frame,imgsz=size,conf=.05,device=args.device,max_det=1000,verbose=False)[0]
                    return [dict(box=b[:4].tolist(),confidence=float(b[4]),class_id=0 if mapping[int(b[5])]=='person' else 2)
                            for b in r.boxes.data.cpu().numpy() if int(b[5]) in mapping]
            else:
                def infer():
                    if backend=='legacy':ds=model._detect_frame(frame,.05)
                    else:ds=model.detect(frame,temporal=False)
                    if model.last_inference_error:raise RuntimeError(model.last_inference_error)
                    return [dict(box=[d['bbox'][0],d['bbox'][1],d['bbox'][0]+d['bbox'][2],d['bbox'][1]+d['bbox'][3]],
                                 class_id=d['class_id'],confidence=d['confidence']) for d in ds]
            if not cached:infer()
            start=time.perf_counter();predictions=infer();latency.append((time.perf_counter()-start)*1000)
            row['predictions']=filter_ignored(predictions,row['ground_truth'],row['ignored_boxes']);cached.append(row)
            if len(cached)%16==0:print(entry['name'],len(cached),'images',flush=True)
        reports.append(dict(name=entry['name'],model=str(path),backend=backend,imgsz=size,images=len(cached),
                            device=(args.device if backend=='rtdetr' else getattr(model,'device','cpu')),
                            device_note='Legacy backend is CPU-only; compare devices explicitly, not as a pure architecture speed test',
                            mean_ms=float(np.mean(latency)),p95_ms=float(np.percentile(latency,95)),
                            thresholds=[score_predictions(cached,t,.5) for t in (.05,.1,.15,.2,.25,.35,.45,.55,.65,.75,.85,.9)],
                            image_manifest=[dict(image=r['image'],sha256=r['image_sha256']) for r in cached]))
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(scope='Development diagnostics only; not acceptance, official VisDrone AP or physical-range evidence',results=reports),indent=2))
    print('Saved:',output)
if __name__=='__main__':main()
