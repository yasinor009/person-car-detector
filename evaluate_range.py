"""Evaluate independently matched range observations, never infer ground truth.

JSON document: {split, reference_method, quantity, rows:[{sequence, frame,
object_id, class, truth_m, prediction_m, valid}]}. Include every eligible ground
truth target, even missed detections (valid=false, prediction_m=null).
Matching must be established with independent annotations, not predicted IDs.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np

STATIONS = (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)


def evaluate(data):
    if data.get('split') not in ('validation', 'test') or not data.get('reference_method'):
        raise ValueError('Independent reference method and explicit split required')
    if data.get('quantity') not in ('euclidean_ground_contact_range', 'optical_axis_depth', 'euclidean_surface_range'):
        raise ValueError('Specify the same physical quantity for predictions and truth')
    seen = set()
    for r in data['rows']:
        key = (r['sequence'], r['frame'], r['object_id'])
        if key in seen:
            raise ValueError('Duplicate target/frame')
        seen.add(key)
        if r['class'] not in ('person', 'car') or type(r['valid']) is not bool:
            raise ValueError('Invalid class or validity')
        if not math.isfinite(r['truth_m']) or not 0 < r['truth_m'] <= 100:
            raise ValueError('Reference range must be in (0,100]m')
        prediction = r.get('prediction_m')
        if prediction is not None and (not math.isfinite(prediction) or prediction <= 0):
            raise ValueError('Prediction must be finite and positive or null')
        if r['valid'] and prediction is None:
            raise ValueError('Valid predictions cannot be null')

    def metrics(rows):
        valid = [r for r in rows if r['valid']]
        n, m = len(rows), len(valid)
        out = dict(eligible=n, valid=m, invalid=n-m, coverage=m/n if n else None,
                   mae_m=None, rmse_m=None, signed_relative_error=None, abs_relative_error=None,
                   median_relative_error=None, within_5pct=None, within_10pct=None,
                   within_15pct=None, good_within_10pct_over_all=None)
        if not m:
            if n:
                out['good_within_10pct_over_all'] = 0.
            return out
        truth = np.array([r['truth_m'] for r in valid])
        error = np.array([r['prediction_m'] for r in valid])-truth
        rel = np.abs(error)/truth
        out.update(mae_m=float(np.mean(np.abs(error))), rmse_m=float(np.sqrt(np.mean(error**2))),
                   signed_relative_error=float(np.mean(error/truth)), abs_relative_error=float(np.mean(rel)),
                   median_relative_error=float(np.median(rel)))
        for percent in (5, 10, 15):
            out[f'within_{percent}pct'] = float(np.mean(rel <= percent/100+1e-12))
        out['good_within_10pct_over_all'] = float(np.sum(rel <= .10+1e-12)/n)
        return out

    per_class = {}
    for name in ('person', 'car'):
        rows = [r for r in data['rows'] if r['class'] == name]
        # Bins are explicitly upper edges, not invented exact measurement stations.
        previous = 0
        bins = {}
        for edge in STATIONS:
            bins[f'({previous},{edge}]m'] = metrics([r for r in rows if previous < r['truth_m'] <= edge])
            previous = edge
        per_class[name] = dict(overall=metrics(rows), distance_bins=bins)
    return dict(split=data['split'], reference_method=data['reference_method'], quantity=data['quantity'],
                per_class=per_class, accepted=False,
                note='Metrics only; acceptance also needs detection, independent scene splits, sample-size/coverage criteria and target-device tests. Correlated video frames are not independent trials.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input'); p.add_argument('--output', required=True)
    args = p.parse_args()
    result = evaluate(json.loads(Path(args.input).read_text()))
    with open(args.output, 'x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
