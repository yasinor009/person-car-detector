import json
from pathlib import Path
import numpy as np
import pytest
from laptop_detector import LaptopDetector,class_contract,tile_plan,canonical_nms
from acceptance import acceptance_report,metric_gate
from optics_check import pixel_budget


def bare(tmp_path,names=None):
    m=LaptopDetector(tmp_path/'absent.onnx',confidence=.25,imgsz=640)
    m.model_names,m.target_ids=class_contract(names or {0:'person',1:'car'})
    m.input_width=m.input_height=640;m.dtype=np.float32;m.onnx_layout='raw'
    return m


def raw(m,boxes):
    a=np.zeros((1,len(m.model_names)+4,len(boxes)),np.float32)
    for i,(box,scores) in enumerate(boxes):
        a[0,:4,i]=box
        for cls,score in scores.items():a[0,4+cls,i]=score
    return a


def test_custom_two_class_and_visdrone_names_are_explicit():
    names,mapping=class_contract(['pedestrian','people','bicycle','car','van','truck'])
    assert mapping=={0:'person',1:'person',3:'car'}
    assert class_contract('{0:"person",1:"car"}')[1]=={0:'person',1:'car'}
    with pytest.raises(ValueError):class_contract({0:'drone',1:'bird'})
    with pytest.raises(ValueError):class_contract({0:'person',3:'car'})


def test_normalized_boxes_are_converted_only_by_explicit_contract(tmp_path):
    m=bare(tmp_path)
    a=raw(m,[([.5,.5,.2,.4],{0:.9})])
    assert m._decode_rows(a,640,640,(1,1,0,0))==[]
    m.output_normalized=True
    d=m._decode_rows(a,640,640,(1,1,0,0))[0]
    assert d['bbox']==pytest.approx([256,192,128,256])


def test_raw_and_end_to_end_decoders_preserve_classes_and_truncation(tmp_path):
    m=bare(tmp_path)
    result=m._decode_rows(raw(m,[([50,100,60,100],{0:.9}),([5,5,20,20],{1:.8})]),640,640,(1,1,0,0))
    assert {d['class'] for d in result}=={'person','car'}
    assert next(d for d in result if d['class']=='car')['truncated']
    m.onnx_layout='end2end'
    ds=m._decode_rows(np.array([[[20,50,80,150,.9,0],[10,10,50,50,.8,1.5]]],np.float32),640,640,(1,1,0,0))
    assert len(ds)==1 and ds[0]['bbox']==pytest.approx([20,50,60,100])


def test_argmax_across_all_classes_does_not_relabel_non_target(tmp_path):
    m=bare(tmp_path,{0:'person',1:'bus',2:'car'})
    assert m._decode_rows(raw(m,[([100,100,50,100],{0:.8,1:.95})]),640,640,(1,1,0,0))==[]


def test_native_pixel_tiles_cover_edges_and_refuse_partial_coverage():
    plan=tile_plan(3840,2160,960,.25,24)
    assert len(plan)==15
    assert max(x+w for x,y,w,h in plan)==3840
    assert max(y+h for x,y,w,h in plan)==2160
    coverage=np.zeros((2160,3840),np.uint8)
    for x,y,w,h in plan:coverage[y:y+h,x:x+w]=1
    assert coverage.all()
    with pytest.raises(ValueError,match='exceeding'):tile_plan(3840,2160,320,.25,8)


def test_tile_mapping_rejects_internal_partial_widths(tmp_path):
    m=bare(tmp_path);m.tiled_inference=True;m.tile_size=640;m.max_tiles=24
    calls=[]
    def infer(frames,floor):
        calls.append([f.shape for f in frames])
        return [[dict(bbox=[0,30,40,60],confidence=.9,**{'class':'car'},class_id=2),
                 dict(bbox=[100,100,30,60],confidence=.8,**{'class':'person'},class_id=0)] for frame in frames]
    m._infer=infer
    ds=m._detect_frame(np.zeros((640,1280,3),np.uint8))
    assert m.last_tile_passes==3
    assert sum(d['class']=='car' for d in ds)==1
    assert any(d['bbox'][0]>500 for d in ds if d['class']=='person')


def test_per_class_threshold_and_no_display_of_low_score_recovery(tmp_path):
    m=bare(tmp_path);m.model_status='ready';m.class_thresholds={'person':.2,'car':.6}
    m._detect_frame=lambda f,floor:[dict(bbox=[100,100,40,100],confidence=.3,**{'class':'person'},class_id=0),
                                   dict(bbox=[300,100,80,50],confidence=.4,**{'class':'car'},class_id=2)]
    f=np.zeros((480,640,3),np.uint8)
    assert m.detect(f)==[];assert m.detect(f)==[]
    assert [d['class'] for d in m.detect(f)]==['person']


def sample_metrics(tp=90,fp=10,fn=10):
    return {'per_class':{c:dict(true_positives=tp,false_positives=fp,false_negatives=fn,
                               precision=tp/(tp+fp),recall=tp/(tp+fn)) for c in ('person','car')}}


def test_acceptance_requires_independent_far_data_and_both_classes():
    metrics=sample_metrics()
    assert not acceptance_report(metrics,split='validation',long_range_metrics=metrics,verified_range_m=150,inference_fps=20)['passed']
    assert not acceptance_report(metrics,split='test',inference_fps=20)['passed']
    assert acceptance_report(metrics,split='test',long_range_metrics=metrics,verified_range_m=150,inference_fps=20)['passed']
    metrics['per_class']['person']['false_negatives']=100
    assert not metric_gate(metrics)['passed']


def test_rounded_90_percent_does_not_pass_if_counts_are_below_target():
    metrics=sample_metrics(89996,10004,10004)
    for row in metrics['per_class'].values():row['precision']=row['recall']=.9
    assert not metric_gate(metrics)['passed']


def test_optics_exposes_150m_pixel_budget_not_detection_guarantee():
    b=pixel_budget(1920,1080,60,150)
    assert b['source_object_height_px']==pytest.approx(18.8447,rel=1e-4)
    assert b['full_frame_network_height_px']<7
    assert 'no 150m detection' in b['verification']
    with pytest.raises(ValueError):pixel_budget(1920,1080,0,150)


def test_browser_upload_preserves_laptop_source_resolution(tmp_path,monkeypatch):
    import config,cv2
    from app import DroneDistanceService
    monkeypatch.setattr(config,'SETTINGS_FILE',tmp_path/'settings.json')
    monkeypatch.setattr(config,'MAX_FRAME_WIDTH',3840)
    service=DroneDistanceService(browser_camera=True,detector=bare(tmp_path))
    try:
        ok,encoded=cv2.imencode('.jpg',np.zeros((1440,2560,3),np.uint8))
        assert ok
        assert service.receive_browser_frame(encoded.tobytes())=={'width':2560,'height':1440}
    finally:service.shutdown()


def test_cpu_only_machine_refuses_explicit_cuda_request(tmp_path,monkeypatch):
    import onnxruntime as ort
    monkeypatch.setattr(ort,'get_available_providers',lambda:['CPUExecutionProvider'])
    p=tmp_path/'fake.onnx';p.write_bytes(b'not read because GPU unavailable')
    m=LaptopDetector(p,device='cuda:0')
    assert m.model_status=='load_error'
    assert 'CUDA' in m.model_message


def test_acceptance_report_serializes_numpy_benchmark_scalar():
    report=acceptance_report(sample_metrics(),inference_fps=np.float64(12.))
    assert json.loads(json.dumps(report))['speed_passed'] is True


def test_latest_camera_discards_old_frames_without_repeating_eof(monkeypatch):
    from laptop_camera import LatestFrameCamera
    class Cap:
        def __init__(self):self.n=0;self.released=False
        def get(self,prop):return 0
        def read(self):
            self.n+=1
            return (True,np.full((32,32,3),self.n,np.uint8)) if self.n<=3 else (False,None)
        def isOpened(self):return not self.released
        def release(self):self.released=True
    cap=Cap();camera=LatestFrameCamera('0');camera._file=True
    monkeypatch.setattr(camera,'_open',lambda source:cap)
    try:
        assert camera.connect()
        with camera._condition:assert camera._condition.wait_for(lambda:camera._eof,timeout=1.)
        frame,_=camera.read_frames()
        assert int(frame[0,0,0])==3
        assert camera.read_frames()==(None,None)
        assert camera.status()['dropped_before_inference']==2
    finally:camera.disconnect()
    assert cap.released


def test_threshold_tuning_cannot_hide_missed_class():
    from tune_laptop import choose_class_thresholds
    cached=[dict(image='a',ground_truth=[dict(box=[0,0,100,100],class_id=0),dict(box=[200,0,300,100],class_id=2)],
                 predictions=[dict(box=[0,0,100,100],class_id=0,confidence=.95)])]
    result=choose_class_thresholds(cached,min_objects=1)
    assert result['classes']['person']['target_met']
    assert not result['classes']['car']['target_met']
    assert not result['target_met']


def test_size_strata_do_not_claim_measured_distance():
    from evaluate_laptop import recall_by_pixel_height
    rows=[dict(image='a',ground_truth=[dict(box=[0,0,10,10],class_id=0),dict(box=[20,0,60,50],class_id=2)],
               predictions=[dict(box=[20,0,60,50],class_id=2,confidence=.9)])]
    r=recall_by_pixel_height(rows)
    assert r['person']['under_16px']['recall']==0
    assert r['car']['at_least_32px']['recall']==1


def test_ignored_region_never_suppresses_match_to_valid_ground_truth():
    from evaluate_laptop import filter_ignored
    gt=[dict(box=[10,10,50,80],class_id=0)]
    predictions=[dict(box=[10,10,50,80],class_id=0,confidence=.9),
                 dict(box=[100,100,120,130],class_id=2,confidence=.8)]
    result=filter_ignored(predictions,gt,[[0,0,200,200]])
    assert result==[predictions[0]]


def test_high_quality_1080p_jpeg_is_not_rejected_by_old_upload_limit(tmp_path,monkeypatch):
    import cv2,config,threading
    from http.server import ThreadingHTTPServer
    from urllib.request import Request,urlopen
    from app import DroneDistanceService,make_handler
    monkeypatch.setattr(config,'SETTINGS_FILE',tmp_path/'settings.json')
    monkeypatch.setattr(config,'MAX_FRAME_WIDTH',3840)
    service=DroneDistanceService(browser_camera=True,detector=bare(tmp_path))
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(service))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        frame=np.random.default_rng(3).integers(0,256,(1080,1920,3),np.uint8)
        ok,jpeg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,95])
        assert ok and len(jpeg)>1_500_000
        req=Request(f'http://127.0.0.1:{server.server_address[1]}/api/camera-frame',data=jpeg.tobytes(),method='POST',headers={'Content-Type':'image/jpeg'})
        with urlopen(req,timeout=5) as response:
            result=json.loads(response.read())
            assert response.status==200
            assert result['width']==1920 and result['height']==1080
    finally:
        service.shutdown();server.shutdown();server.server_close();thread.join(timeout=2)


def test_optical_budget_uses_long_edge_for_square_letterbox_in_portrait():
    b=pixel_budget(1080,1920,60,150)
    assert b['full_frame_network_height_px']/b['source_object_height_px']==pytest.approx(640/1920)
