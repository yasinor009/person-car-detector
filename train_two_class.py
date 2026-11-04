"""Fine-tune on the actual camera domain. Training completion is NOT acceptance.

Required structure:
  dataset/{train,val,test}/{images,labels}; labels use 0=person, 1=car.
Split by recording/location, not adjacent video frames. Include labelled hard
negatives (empty .txt) from walls, shadows, reflections and posters.
"""
import argparse
import hashlib
import json
from pathlib import Path
from evaluate_laptop import dataset_rows


def audit_splits(root):
    root=Path(root);hashes={};summary={}
    for split in ('train','val','test'):
        count=0;objects={0:0,2:0};negatives=0
        for row,frame in dataset_rows(root/split,'two-class'):
            digest=row['image_sha256']
            if digest in hashes:
                raise ValueError(f'Duplicate image across/in splits: {split}/{row["image"]}; previous={hashes[digest]}')
            hashes[digest]=f'{split}/{row["image"]}'
            count+=1;negatives+=not row['ground_truth']
            for g in row['ground_truth']:objects[g['class_id']]+=1
        if min(objects.values())<20:raise ValueError(f'{split}: insufficient examples of one class')
        summary[split]=dict(images=count,person_objects=objects[0],car_objects=objects[2],negative_images=negatives)
    # Hash checking catches exact duplicates, not near-identical frames. The
    # operator must separately enforce scene/video-separated splits.
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--model',default='yolo26s.pt')
    p.add_argument('--imgsz',type=int,default=1280);p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--device',default='0');p.add_argument('--batch',type=int,default=1)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--audit-only',action='store_true')
    p.add_argument('--output',default='runs/person_car_camera')
    args=p.parse_args()
    root=Path(args.dataset).resolve();audit=audit_splits(root)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    (output/'dataset_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
    if args.audit_only:return
    import torch
    from ultralytics import YOLO
    if args.device!='cpu' and not torch.cuda.is_available():
        raise SystemExit('CUDA training is not available. Install compatible PyTorch or explicitly choose --device cpu. No training was claimed.')
    yaml=output/'dataset.yaml'
    yaml.write_text(f'path: {json.dumps(str(root))}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: person\n  1: car\n')
    model=YOLO(args.model)
    model.train(data=str(yaml.resolve()),imgsz=args.imgsz,epochs=args.epochs,batch=args.batch,
                device=args.device,workers=args.workers,amp=True,cache=False,seed=42,
                close_mosaic=10,patience=20,project=str(output.resolve()),name='fit',exist_ok=False)
    best=output/'fit/weights/best.pt'
    if not best.is_file():raise SystemExit('Training did not produce a best checkpoint')
    trained=YOLO(str(best))
    # Validation is used for development; independent test acceptance is performed
    # later with evaluate_laptop.py using frozen thresholds and measured ranges.
    exported=trained.export(format='onnx',imgsz=args.imgsz,opset=17,dynamic=True,simplify=False,nms=False,end2end=False)
    print('Exported:',exported)
    print('No >=90% or >=150m claim follows from training. Run the independent acceptance evaluation.')
if __name__=='__main__':main()
