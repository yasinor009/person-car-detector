import math
import numpy as np
import pytest
from metric_range import GroundCalibration, RangeBank
from evaluate_range import evaluate


def calibration(**changes):
    values = dict(camera_matrix=[[1000, 0, 640], [0, 1000, 360], [0, 0, 1]],
                  image_size=[1280, 720], normal=[0, 1, 0], height_m=1.5,
                  height_std_m=.005, angle_std_deg=.01, pixel_std=.1,
                  calibration_id='synthetic-test-only', fixed_camera=True,
                  planar_ground=True, systematic_relative_std=.01)
    values.update(changes)
    return GroundCalibration(**values)


def detection(track=1, bottom=435):
    return dict(bbox=[620., bottom-100., 40., 100.], track_id=track,
                confidence=.9, **{'class': 'person'})


def test_geometry_matches_analytic_ground_point():
    c = calibration()
    m = c.measure(detection(), (720, 1280, 3))
    assert m['valid']
    assert m['raw_distance_m'] == pytest.approx(math.hypot(20, 1.5))
    assert m['uncertainty_std_m'] >= m['raw_distance_m']*.01


def test_horizon_truncation_resolution_and_handheld_rejected():
    c = calibration()
    assert not c.measure(detection(bottom=360), (720,1280,3))['valid']
    assert not c.measure(dict(detection(), truncated=True), (720,1280,3))['valid']
    assert not c.measure(detection(), (360,640,3))['valid']
    with pytest.raises(ValueError):
        calibration(fixed_camera=False)


def test_independent_tracks_outlier_expiry_and_systematic_floor():
    c, bank = calibration(), RangeBank()
    for i in range(20):
        output = bank.update([detection(), detection(2, 510)], i*.1, c, (720,1280,3))
    assert output[0]['distance_m'] > output[1]['distance_m']*1.8
    assert output[0]['distance']['uncertainty_std_m'] >= output[0]['distance_m']*.01
    rejected = bank.update([detection(bottom=510)], 2., c, (720,1280,3))[0]
    assert rejected['distance_m'] is None
    assert rejected['distance']['reason'] == 'temporal_outlier'
    assert bank.update([], 4., c, (720,1280,3)) == []
    assert not bank.states


def test_missing_calibration_never_returns_range():
    row = RangeBank().update([detection()], 0., None, (720,1280,3))[0]
    assert row['distance_m'] is None
    assert not row['distance']['valid']


def test_range_evaluator_counts_missing_and_exact_boundary():
    rows = [dict(sequence='test',frame=i,object_id=1,**{'class':'person'},
                 truth_m=100.,prediction_m=p,valid=p is not None) for i,p in enumerate([110.,90.,None])]
    data = dict(rows=rows,split='test',reference_method='synthetic unit test',quantity='euclidean_ground_contact_range')
    report = evaluate(data)
    m = report['per_class']['person']['overall']
    assert m['within_10pct'] == 1
    assert m['coverage'] == pytest.approx(2/3)
    assert m['mae_m'] == 10
    assert report['per_class']['car']['overall']['mae_m'] is None
    assert not report['accepted']
    with pytest.raises(ValueError):
        evaluate(dict(data, rows=rows+rows[:1]))
