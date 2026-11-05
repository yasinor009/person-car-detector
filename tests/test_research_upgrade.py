from itertools import permutations
import numpy as np
import pytest
from temporal_tracker import TemporalTracker, BoxMotion, linear_assignment, associate


def detection(x=100.,confidence=.9,cls='person',width=80.):
    return dict(bbox=[x,100.,width,120.],confidence=confidence,**{'class':cls})


def test_hungarian_matches_exhaustive_optimum():
    rng=np.random.default_rng(11)
    for n,m in ((1,3),(3,3),(3,5),(4,2)):
        for _ in range(10):
            cost=rng.random((n,m))
            got=sum(cost[i,j] for i,j in linear_assignment(cost))
            if n<=m:
                expected=min(sum(cost[i,j] for i,j in enumerate(p)) for p in permutations(range(m),n))
            else:
                expected=min(sum(cost[i,j] for j,i in enumerate(p)) for p in permutations(range(n),m))
            assert got==pytest.approx(expected)


def test_assignment_avoids_greedy_trap():
    # Greedy takes cost .1 then .9; optimum is .2 + .15.
    cost=np.array([[.1,.2],[.15,.9]])
    assert set(linear_assignment(cost))=={(0,1),(1,0)}


def test_invalid_assignment_cannot_consume_other_valid_pair():
    tracks=[dict(motion=BoxMotion([100,100,80,120]),**{'class':'person'}),
            dict(motion=BoxMotion([400,100,80,120]),**{'class':'car'})]
    matches=associate(tracks,[detection(400,cls='car')])
    assert matches==[(1,0)]


def confirm(tracker):
    results=[]
    for t in (1.,1.1,1.2):results=tracker.update([detection()],t,(480,640,3))
    return results[0]['track_id']


def test_short_occlusion_preserves_id_without_ghost_box():
    tracker=TemporalTracker();original=confirm(tracker)
    assert tracker.update([],1.3,(480,640,3))==[]
    recovered=tracker.update([detection()],1.4,(480,640,3))
    assert recovered[0]['track_id']==original
    assert recovered[0]['bbox']==[100.,100.,80.,120.]


def test_weak_boxes_never_create_or_display_tracks():
    tracker=TemporalTracker()
    for i in range(10):assert tracker.update([detection(confidence=.4)],i*.1,(480,640,3))==[]
    assert tracker.tracks==[]
    original=confirm(tracker)
    assert tracker.update([detection(confidence=.4)],1.3,(480,640,3))==[]
    assert tracker.last_weak_updates==1
    assert tracker.update([detection()],1.4,(480,640,3))[0]['track_id']==original


def test_weak_boxes_cannot_complete_initial_confirmation():
    tracker=TemporalTracker()
    for t in (1.,1.1):assert tracker.update([detection()],t,(480,640,3))==[]
    assert tracker.update([detection(confidence=.4)],1.2,(480,640,3))==[]
    assert tracker.update([detection()],1.3,(480,640,3))==[]


def test_lost_identity_expires_and_is_never_recycled():
    tracker=TemporalTracker();original=confirm(tracker)
    for t in (1.3,1.4,1.5):assert tracker.update([],t,(480,640,3))==[]
    assert tracker.tracks==[]
    for t in (1.6,1.7,1.8):result=tracker.update([detection()],t,(480,640,3))
    assert result[0]['track_id']!=original
    tracker.reset()
    assert confirm(tracker)!=original


def test_class_and_time_gap_prevent_identity_reuse():
    tracker=TemporalTracker();original=confirm(tracker)
    assert tracker.update([detection(cls='car')],1.3,(480,640,3))==[]
    assert tracker.update([detection()],5.,(480,640,3))==[]
    assert tracker.tracks[0]['track_id']!=original


def test_kalman_predictions_do_not_replace_measured_width():
    tracker=TemporalTracker();confirm(tracker)
    result=tracker.update([detection(width=90)],1.3,(480,640,3))
    assert result[0]['bbox'][2]==90
    assert tracker.tracks[0]['motion'].bbox()[2]!=90


def test_kalman_covariance_remains_positive_with_variable_cadence():
    m=BoxMotion([100,100,80,120])
    for i in range(100):
        m.predict(.05+(i%4)*.1);m.update([100+i,100,80+i*.1,120],weak=i%3==0)
        assert np.isfinite(m.x).all()
        assert np.min(np.linalg.eigvalsh(m.p))>=-1e-8


def test_pixel_sensitivity_bounds_are_not_total_error():
    from distance_estimator import DistanceEstimator
    e=DistanceEstimator(1000,.5).estimate(100).to_dict()
    assert e['distance_m']==5
    assert e['sensitivity']['lower_m']==pytest.approx(500/102)
    assert e['sensitivity']['upper_m']==pytest.approx(500/98)
    assert 'not_total_uncertainty' in e['sensitivity']['scope']


def lens_views(corrupt=False):
    import cv2
    rng=np.random.default_rng(10)
    k=np.array([[800.,0,320],[0,810,240],[0,0,1]])
    obj=np.zeros((54,3),np.float32)
    obj[:,:2]=np.mgrid[:9,:6].T.reshape(-1,2)*.025
    obj[:,:2]-=[.1,.0625]
    objects=[];images=[]
    for i in range(18):
        r=rng.uniform(-.4,.4,3);t=np.array([rng.uniform(-.14,.14),rng.uniform(-.09,.09),rng.uniform(.5,.8)])
        points,_=cv2.projectPoints(obj,r,t,k,np.array([-.08,.02,0,0,0]))
        points+=rng.normal(0,3 if corrupt and i==17 else .08,points.shape)
        objects.append(obj.copy());images.append(points.astype(np.float32))
    return objects,images,k


def test_lens_refits_after_rejecting_corrupted_view():
    from calibrate_lens import fit_lens
    objects,images,k=lens_views(corrupt=True)
    fitted=fit_lens(objects,images,(640,480))
    assert fitted['rejected_images']>=1
    assert 17 not in fitted['accepted_view_indices']
    assert fitted['reprojection_rms_px']<.2
    assert fitted['camera_matrix'][0][0]==pytest.approx(k[0,0],rel=.03)


def test_lens_requires_diverse_views():
    from calibrate_lens import fit_lens
    objects,images,k=lens_views()
    with pytest.raises(ValueError):fit_lens([objects[0]]*12,[images[0]]*12,(640,480))


def test_slice_merge_preserves_classes_coordinates_and_rejects_partial_box(tmp_path):
    from drone_detector import DroneDetector
    m=DroneDetector(model_path=str(tmp_path/'missing.onnx'))
    m.tiled_inference=True
    calls=[]
    def fake(frame,score_floor=None):
        calls.append(frame.shape)
        m.last_raw_candidates=10;m.last_decoded_count=2
        if len(calls)==1:return []
        # This interior observation is valid; the left-cut one is invalid for
        # right-hand slices but still valid/truncated on a global frame edge.
        return [dict(detection(30,width=30),bbox=[30,30,30,50],class_id=0,truncated=False),
                dict(detection(0,cls='car'),bbox=[0,50,40,40],class_id=2,truncated=True)]
    m._detect_onnx_single=fake
    result=m._detect_frame(np.zeros((480,640,3),np.uint8),.25)
    assert len(calls)==5 and m.last_tile_passes==4
    assert m.last_raw_candidates==50
    assert all(d['class'] in ('person','car') for d in result)
    assert sum(d['class']=='car' for d in result)==2
    assert any(d['bbox'][0]>250 for d in result)
    assert all(d['source']=='visual_model_tiled' for d in result)


def test_precision_selection_uses_evidence_and_recall_constraint():
    from evaluate_detector import score_predictions,select_operating_point
    data=[dict(image='positive',ground_truth=[dict(box=(0,0,100,100),class_id=0)],
               predictions=[dict(box=(0,0,100,100),class_id=0,confidence=.9),
                            dict(box=(200,200,250,250),class_id=2,confidence=.6)]),
          dict(image='wall',ground_truth=[],predictions=[dict(box=(0,0,100,100),class_id=0,confidence=.65)])]
    lower=score_predictions(data,.55,.5);higher=score_predictions(data,.8,.5)
    assert lower['negative_image_false_alarm_rate']==1
    assert higher['negative_image_false_alarm_rate']==0
    assert lower['per_class']['person']['true_positives']==1
    assert select_operating_point([lower,higher])['confidence_threshold']==.8
    with pytest.raises(ValueError):select_operating_point([dict(higher,recall=0)])
