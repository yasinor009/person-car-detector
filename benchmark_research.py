"""Reproducible synthetic ablations. These are NOT camera accuracy benchmarks."""
import argparse
import json
from pathlib import Path
from statistics import median
import math
import numpy as np
from temporal_tracker import TemporalTracker,box_iou
from calibrate_multiview import fit


def old_median_fit(samples):
    values=[s['observed_width_px']*s['known_distance_m']/s['known_width_m']*640/s['frame_width_px'] for s in samples]
    center=median(values)
    gate=max(.02*center,3*1.4826*median(abs(f-center) for f in values))
    accepted=[v for v in values if abs(v-center)<=gate]
    if len(accepted)<5 or len(accepted)<.7*len(values):raise ValueError('Inconsistent samples')
    result=median(accepted)
    residual=math.sqrt(sum((f/result-1)**2 for f in accepted)/len(accepted))
    if residual>.08:raise ValueError('High residual')
    return result


def tracking_ablation():
    tracker=TemporalTracker()
    old_tracks=[];old_id=1;old_observed=[];new_observed=[];rows=[]
    # Same clean observed boxes for both. One missing observation represents an
    # occlusion, not a fabricated neural-model error distribution.
    for frame_index,x in enumerate((100,105,110,None,120,125,130),1):
        detections=[] if x is None else [dict(bbox=[x,100,80,120],confidence=.9,**{'class':'person'})]
        old_next=[];shown=[]
        for d in detections:
            previous=max(old_tracks,key=lambda t:box_iou(t['bbox'],d['bbox']),default=None)
            if previous is None or box_iou(previous['bbox'],d['bbox'])<.3:
                t=dict(d,track_id=old_id,hits=1);old_id+=1
            else:t=dict(d,track_id=previous['track_id'],hits=previous['hits']+1)
            old_next.append(t)
            if t['hits']>=3:shown.append(t)
        old_tracks=old_next
        newer=tracker.update(detections,frame_index*.1,(480,640,3))
        old_observed.extend(d['track_id'] for d in shown)
        new_observed.extend(d['track_id'] for d in newer)
        rows.append(dict(frame=frame_index,has_observation=x is not None,baseline_ids=[d['track_id'] for d in shown],
                         upgraded_ids=[d['track_id'] for d in newer]))
    return dict(scenario='one short occlusion after initial confirmation',frames=rows,
                baseline_visible_identity_count=len(set(old_observed)),upgraded_visible_identity_count=len(set(new_observed)),
                ghost_boxes_on_missing_frame=sum(len(r['upgraded_ids']) for r in rows if not r['has_observation']))


def calibration_ablation(trials=250):
    paired_old=[];paired_new=[];old_rejected=new_rejected=0
    for seed in range(trials):
        rng=np.random.default_rng(seed)
        samples=[dict(known_width_m=.5,known_distance_m=d,observed_width_px=900*.5/d+rng.normal(0,2),frame_width_px=640)
                 for d in (2,3,4,6,9,12,15,20,25)]
        try:old=old_median_fit(samples)
        except ValueError:old_rejected+=1;old=None
        try:new=fit(samples)['focal_length_px']
        except ValueError:new_rejected+=1;new=None
        if old is not None and new is not None:
            paired_old.append(abs(old-900)/900);paired_new.append(abs(new-900)/900)
    return dict(trials=trials,paired_accepted_trials=len(paired_new),baseline_rejected=old_rejected,upgraded_rejected=new_rejected,
                baseline_mean_abs_relative_focal_error=float(np.mean(paired_old)),
                upgraded_mean_abs_relative_focal_error=float(np.mean(paired_new)),
                assumptions='True focal=900px, width=.5m, exact known ranges [2,3,4,6,9,12,15,20,25]m; independent Gaussian 2px width noise; fixed pose; no detector or reference-size bias')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='validation/research_ablation.json')
    args=p.parse_args()
    report=dict(scope='Synthetic software ablation only; not measured field accuracy or phone performance',
                tracking=tracking_ablation(),calibration=calibration_ablation())
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
