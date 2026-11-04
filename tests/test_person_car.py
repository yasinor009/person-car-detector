import numpy as np
import pytest
from drone_detector import DroneDetector
from calibrate_multiview import fit

@pytest.fixture
def detector(tmp_path):
    return DroneDetector(model_path=str(tmp_path/'missing.onnx'))

def output(candidates):
    a=np.zeros((1,84,len(candidates)),dtype=np.float32)
    for i,(box,probs) in enumerate(candidates):
        a[0,:4,i]=box
        for cls,score in probs.items():a[0,4+cls,i]=score
    return a

def decode(d,candidates):
    return d._decode(output(candidates),(640,640,3),1,1,0,0)

def test_only_person_and_car_with_class_aware_nms(detector):
    result=decode(detector,[([200,200,100,160],{0:.9}),([200,200,100,160],{2:.88}),
                            ([201,201,100,160],{0:.8}),([400,400,40,50],{5:.99})])
    assert {d['class'] for d in result}=={'person','car'}
    assert len(result)==2

def test_secondary_person_score_does_not_relabel_bus(detector):
    assert decode(detector,[([200,200,100,160],{0:.8,5:.99})])==[]

def test_nonfinite_and_tiny_boxes_rejected(detector):
    assert decode(detector,[([np.nan,100,50,60],{0:.9}),([100,100,2,2],{2:.9}),
                            ([100,100,20,40],{0:np.inf})])==[]

def test_letterbox_coordinates_and_truncation(detector):
    a=output([([320,320,100,100],{0:.9})])
    d=detector._decode(a,(480,640,3),1,1,0,80)[0]
    assert d['bbox']==pytest.approx([270,190,100,100])
    assert decode(detector,[([0,100,100,100],{2:.9})])[0]['truncated']

def detection(cls='person',box=(100,100,80,120)):
    return dict(bbox=box,confidence=.9,source='visual_model',**{'class':cls})

def test_requires_three_consecutive_frames_and_no_ghost(detector):
    assert detector._confirm([detection()],1,(480,640,3))==[]
    assert detector._confirm([detection()],1.1,(480,640,3))==[]
    assert len(detector._confirm([detection()],1.2,(480,640,3)))==1
    assert detector._confirm([],1.3,(480,640,3))==[]
    recovered = detector._confirm([detection()],1.4,(480,640,3))
    assert len(recovered) == 1
    assert recovered[0]["track_id"] == 1

def test_class_change_gap_and_resize_reset_confirmation(detector):
    for t in (1,1.1,1.2):detector._confirm([detection()],t,(480,640,3))
    assert detector._confirm([detection('car')],1.3,(480,640,3))==[]
    assert detector._confirm([detection('car')],3,(480,640,3))==[]
    assert detector._confirm([detection('car')],3.1,(640,480,3))==[]

def test_matching_is_one_to_one(detector):
    detector._confirm([detection()],1,(480,640,3))
    detector._confirm([detection(),detection(box=(105,100,80,120))],1.1,(480,640,3))
    assert sorted(t['hits'] for t in detector._tracks)==[1,2]

def test_service_never_uses_motion_or_thermal_as_detection(tmp_path,monkeypatch,detector):
    import config
    from app import DroneDistanceService
    monkeypatch.setattr(config,'SETTINGS_FILE',tmp_path/'settings.json')
    service=DroneDistanceService(detector=detector)
    def forbidden(*a):raise AssertionError('nonsemantic detector was called')
    service.motion_detector.detect=forbidden
    service.thermal_processor.detect_hot_objects=forbidden
    service._process(np.zeros((480,640,3),np.uint8),np.zeros((480,640),np.uint8))
    assert service.latest_detections==[]
    service.shutdown()

def test_multipoint_calibration_rejects_outlier_and_scales_resolution():
    samples=[dict(known_width_m=.5,known_distance_m=d,observed_width_px=1000*.5/d,
                  frame_width_px=640) for d in (2,3,4,5,6)]
    samples.append(dict(known_width_m=.5,known_distance_m=3,observed_width_px=2000*.5/3,frame_width_px=1280))
    samples.append(dict(known_width_m=.5,known_distance_m=10,observed_width_px=250,frame_width_px=640))
    result=fit(samples)
    assert result['focal_length_px']==pytest.approx(1000)
    assert result['rejected_samples']==1
    assert result['accepted_samples']==6

def test_class_mismatch_is_false_positive_and_false_negative():
    from evaluate_detector import score_predictions
    rows=[dict(image='x',ground_truth=[dict(box=(0,0,100,100),class_id=0)],
               predictions=[dict(box=(0,0,100,100),class_id=2,confidence=.9)])]
    score=score_predictions(rows,.6,.5)
    assert (score['true_positives'],score['false_positives'],score['false_negatives'])==(0,1,1)

def test_rectification_identity_and_aspect_guard(tmp_path):
    import json
    from camera_geometry import LensRectifier
    p=tmp_path/'lens.json'
    p.write_text(json.dumps(dict(camera_matrix=[[500,0,320],[0,500,240],[0,0,1]],
                                distortion=[0,0,0,0,0],image_size=[640,480])))
    r=LensRectifier(p)
    frame=np.random.default_rng(42).integers(0,256,(480,640,3),dtype=np.uint8)
    assert np.array_equal(r.correct(frame),frame)
    with pytest.raises(ValueError,match='aspect ratio'):r.correct(np.zeros((640,480,3),np.uint8))

def test_missing_selected_detection_invalidates_distance(tmp_path,monkeypatch,detector):
    import config
    from app import DroneDistanceService
    monkeypatch.setattr(config,'SETTINGS_FILE',tmp_path/'settings.json')
    monkeypatch.setattr(config,'SAVE_DETECTION_IMAGES',False)
    s=DroneDistanceService(detector=detector)
    s.latest_frame_size={'width':640,'height':480}
    s.selected_target=dict(bbox=[100,100,80,120],track_id=1,**{'class':'person'},tracking_status='tracking')
    s.selected_target_width_m=.5
    s.distance.calibrate(.5,5,80,640)
    s._update_selected_distance()
    assert s.selected_target['distance_m']==5
    s._process(np.zeros((480,640,3),np.uint8),None)
    assert s.selected_target['distance_m'] is None
    assert s.latest_detections==[]
    s.shutdown()
