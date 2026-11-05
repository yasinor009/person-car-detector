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


def test_changed_calibration_with_same_label_resets_old_range():
    bank = RangeBank()
    c = calibration()
    for i in range(20):
        bank.update([detection()], i*.1, c, (720,1280,3))
    updated = calibration(height_m=2.5)  # same human-readable calibration_id
    result = bank.update([detection()], 2., updated, (720,1280,3))[0]['distance']
    assert result['valid']
    assert result['filtered_distance_m'] == pytest.approx(result['raw_distance_m'])


def test_invalid_frame_geometry_clears_filter_and_duplicate_id_rejected():
    bank, c = RangeBank(), calibration()
    bank.update([detection()], 0., c, (720,1280,3))
    bank.update([detection()], .1, c, (360,640,3))
    assert not bank.states
    with pytest.raises(ValueError, match='Duplicate'):
        bank.update([detection(), detection()], .2, c, (720,1280,3))


@pytest.mark.parametrize('box', [None, 'bad', [1,2], [1,2,float('nan'),4]])
def test_bad_box_does_not_abort_other_targets(box):
    result = RangeBank().update([dict(detection(), bbox=box), detection(2)], 0., calibration(), (720,1280,3))
    assert not result[0]['distance']['valid']
    assert result[1]['distance']['valid']


@pytest.mark.parametrize('distance', [5,10,20,30,40,50,60,70,80,90,100])
def test_analytic_range_and_pixel_uncertainty_at_requested_stations(distance):
    c = calibration()
    # Independently project the ground point (X=0,Y=height,Z>0) onto image.
    z = math.sqrt(distance**2-c.height_m**2)
    v = 360+1000*c.height_m/z
    row = c.measure(detection(bottom=v), (720,1280,3))
    assert row['raw_distance_m'] == pytest.approx(distance)
    # Ground range sqrt(h²+(fy*h/(v-cy))²), exact vertical derivative.
    derivative = (1000*c.height_m)**2/((v-360)**3*distance)
    assert row['random_std_m'] == pytest.approx(max(.01, derivative*c.pixel_std))


def test_selected_ground_range_survives_width_setting_and_invalidates_on_loss(tmp_path, monkeypatch):
    import config
    from app import DroneDistanceService
    from drone_detector import DroneDetector
    monkeypatch.setattr(config, 'SETTINGS_FILE', tmp_path/'settings.json')
    service = DroneDistanceService(detector=DroneDetector(str(tmp_path/'missing.onnx')))
    service.ground_calibration = calibration()
    rows = RangeBank().update([detection()], 0., service.ground_calibration, (720,1280,3))
    service.latest_detections = rows
    service.selected_target = dict(rows[0], tracking_status='tracking')
    expected = rows[0]['distance_m']
    service.set_selected_target_width(9.)
    assert service.selected_target['distance_m'] == expected
    assert service.selected_target['distance']['method'] == 'calibrated_ground_contact'
    service._invalidate_live_observations()
    assert service.selected_target['distance_m'] is None
    assert service.selected_target['distance']['valid'] is False
    service.shutdown()
