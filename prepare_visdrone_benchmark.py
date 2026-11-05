"""Prepare fixed VisDrone development/test subsets. No physical range labels exist.

Download the official val and test-dev archives separately from the documented
Ultralytics mirror. This script accepts an extracted VisDrone2019-DET-{val,test-dev}
root; originals are never modified. Label IDs become 0=person, 1=car.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,default=96)
    p.add_argument('--split',choices=('development','test'),required=True)
    args=p.parse_args();source=Path(args.source);out=Path(args.output)
    if (out/'images').exists() and any((out/'images').iterdir()):
        raise ValueError('Output image folder is not empty; choose a fresh folder to avoid mixing splits')
    images=sorted((source/'images').glob('*.jpg'))
    if not images or not 1<=args.count<=len(images):raise ValueError('Invalid source/count')
    selected=[images[i] for i in np.linspace(0,len(images)-1,args.count,dtype=int)]
    (out/'images').mkdir(parents=True,exist_ok=True);(out/'labels').mkdir(exist_ok=True)
    ignore={};manifest=[]
    for image in selected:
        frame=cv2.imread(str(image))
        if frame is None:raise ValueError(f'Unreadable image {image}')
        h,w=frame.shape[:2];rows=[];regions=[]
        for line in (source/'annotations'/image.with_suffix('.txt').name).read_text().splitlines():
            if not line.strip():continue
            x,y,bw,bh,score,cls,*_=map(int,line.strip(',').split(','))
            if score==0 or cls in (0,11):regions.append([x,y,x+bw,y+bh]);continue
            if cls not in (1,2,4):continue
            canonical=0 if cls in (1,2) else 1
            rows.append(f'{canonical} {(x+bw/2)/w:.9f} {(y+bh/2)/h:.9f} {bw/w:.9f} {bh/h:.9f}')
        shutil.copy2(image,out/'images'/image.name)
        (out/'labels'/image.with_suffix('.txt').name).write_text('\n'.join(rows)+'\n' if rows else '')
        ignore[image.name]=regions
        manifest.append(dict(image=image.name,sha256=hashlib.sha256(image.read_bytes()).hexdigest()))
    (out/'ignore_regions.json').write_text(json.dumps(ignore))
    (out/'dataset_manifest.json').write_text(json.dumps(dict(source=str(source),split=args.split,selection='uniform sorted indices before inference',
        images=manifest,range_evidence=None,limitations='Public benchmark subset, not a measured 150m dataset or user-camera acceptance test'),indent=2))
    print(f'Prepared {len(selected)} images under {out}')
if __name__=='__main__':main()
