"""Robust multi-distance focal calibration. Run with camera stopped.

JSON input: [{"known_width_m":0.5,"known_distance_m":3,
              "observed_width_px":100,"frame_width_px":640}, ...]
All observations must use the same camera, crop, zoom and object orientation.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import median
import config

def fit(samples, reference_width=640):
    """MAD screening + Huber IRLS in pixel space for y = focal * W/Z.

    With similar pixel noise, a 20px distant target carries less focal information
    than a 200px near target. Fitting pixel residuals respects that geometry.
    Standard deviations default to an explicit assumption of 2px; they are not
    measured physical uncertainty and exclude pose, width and distance errors.
    """
    if not math.isfinite(float(reference_width)) or reference_width <= 0:
        raise ValueError('Reference width must be finite and positive')
    if len(samples) < 5: raise ValueError('At least five independent measurements required')
    observations=[]
    for sample in samples:
        values=[float(sample[k]) for k in ('known_width_m','known_distance_m','observed_width_px','frame_width_px')]
        if not all(math.isfinite(v) and v>0 for v in values): raise ValueError('All values must be finite and positive')
        width,distance,pixels,resolution=values
        sigma=float(sample.get('observed_width_std_px',2.0))
        if not math.isfinite(sigma) or sigma<=0:raise ValueError('Pixel standard deviation must be positive')
        if pixels < 12 or pixels > resolution: raise ValueError('Invalid/too-small observed width')
        x=width/distance
        y=pixels*reference_width/resolution
        observations.append((x,y,sigma*reference_width/resolution,y/x))
    center=median(o[3] for o in observations)
    mad=median(abs(o[3]-center) for o in observations)
    gate=max(.02*center,3*1.4826*mad)
    accepted=[o for o in observations if abs(o[3]-center)<=gate]
    if len(accepted)<5 or len(accepted)<.7*len(observations): raise ValueError('Insufficient consistent calibration samples')
    focal=center
    information=0.
    for _ in range(30):
        residuals=[(y-focal*x)/sigma for x,y,sigma,_ in accepted]
        scale=max(1.,1.4826*median(abs(r-median(residuals)) for r in residuals))
        weights=[min(1.,1.345*scale/max(abs(r),1e-12))/sigma**2
                 for r,(_,_,sigma,_) in zip(residuals,accepted)]
        information=sum(w*x*x for w,(x,_,_,_) in zip(weights,accepted))
        fitted=sum(w*x*y for w,(x,y,_,_) in zip(weights,accepted))/information
        if abs(fitted-focal)<1e-8*max(1.,focal):focal=fitted;break
        focal=fitted
    residual=math.sqrt(sum((f/focal-1)**2 for _,_,_,f in accepted)/len(accepted))
    if not math.isfinite(focal) or focal<=0 or residual>.08:
        raise ValueError('Calibration spread exceeds 8%; fix pose, camera or measurement')
    pixel_rms=math.sqrt(sum((y-focal*x)**2 for x,y,_,_ in accepted)/len(accepted))
    return dict(focal_length_px=focal,calibration_frame_width_px=reference_width,
                accepted_samples=len(accepted),rejected_samples=len(observations)-len(accepted),
                relative_calibration_rms=residual,pixel_residual_rms=pixel_rms,
                conditional_focal_std_px=math.sqrt(1/information),
                uncertainty_basis='pixel_noise_only_excludes_reference_size_distance_pose_and_model_bias',
                pixel_noise_source='provided' if all('observed_width_std_px' in s for s in samples) else 'includes_assumed_2px',
                calibration_method='mad_screened_huber_pixel_regression')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('samples')
    parser.add_argument('--apply',action='store_true',help='Write camera calibration to data/settings.json with app stopped')
    args=parser.parse_args()
    result=fit(json.loads(Path(args.samples).read_text()))
    print(json.dumps(result,indent=2))
    if args.apply:
        path=config.SETTINGS_FILE
        settings=json.loads(path.read_text()) if path.exists() else {}
        settings.update(result)
        settings['target_width_m']=None  # camera calibration does not determine target size
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(settings,indent=2));tmp.replace(path)

if __name__=='__main__':main()
