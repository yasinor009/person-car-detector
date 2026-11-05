"""Laptop entry point. The mobile entry point remains available as main.py."""
from pathlib import Path
import argparse
import json
import logging
import config
from laptop_detector import LaptopDetector


def main(argv=None):
    p=argparse.ArgumentParser(description='Person/car laptop detection; measured acceptance is separate from confidence')
    p.add_argument('--profile',choices=('balanced','long-range','aerial','accuracy-aerial','cpu'),default='balanced')
    p.add_argument('--model',default=None)
    p.add_argument('--device',choices=('auto','cpu','cuda:0'),default='auto')
    p.add_argument('--source',default='0',help='Camera index, RTSP URL or video path')
    p.add_argument('--browser-camera',action='store_true')
    p.add_argument('--capture-width',type=int,default=None)
    p.add_argument('--capture-height',type=int,default=None)
    p.add_argument('--imgsz',type=int,default=None)
    p.add_argument('--confidence',type=float,default=.25)
    p.add_argument('--thresholds',help='JSON with person/car thresholds tuned on separate validation data')
    p.add_argument('--tile-size',type=int,default=960)
    p.add_argument('--overlap',type=float,default=.25)
    p.add_argument('--max-tiles',type=int,default=24)
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--confirm-frames',type=int,choices=(2,3,4,5),default=3)
    p.add_argument('--port',type=int,default=5000)
    args=p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    root=Path(__file__).resolve().parent
    defaults={'balanced':root/'models'/'person_car_yolo26s.onnx',
              'cpu':root/'models'/'person_car_yolo26n.onnx',
              'long-range':root/'models'/'person_car_yolo26s.onnx',
              'aerial':root/'models'/'visdrone11s_1280.onnx',
              'accuracy-aerial':root/'models'/'visdrone26m_1280.onnx'}
    model=args.model or str(defaults[args.profile])
    thresholds=json.loads(Path(args.thresholds).read_text()) if args.thresholds else None
    if thresholds and 'class_thresholds' in thresholds:thresholds=thresholds['class_thresholds']
    size=args.imgsz or (640 if args.profile=='cpu' else (1280 if args.profile in ('long-range','aerial','accuracy-aerial') else 960))
    detector=LaptopDetector(model,args.confidence,device='cpu' if args.profile=='cpu' else args.device,
                            imgsz=size,tiled=args.profile in ('long-range','aerial','accuracy-aerial'),tile_size=args.tile_size,
                            overlap=args.overlap,max_tiles=args.max_tiles,batch_size=args.batch,
                            confirmation_frames=args.confirm_frames,class_thresholds=thresholds)
    if detector.model_status!='ready':raise SystemExit(detector.model_message)
    print(detector.model_message)
    print('A high confidence threshold is not measured precision or recall.')
    if detector.device=='cpu' and args.profile in ('long-range','aerial','accuracy-aerial'):
        print('Long-range slicing on CPU may be slow; check measured frame latency in the dashboard.')
    capture_width=args.capture_width or (3840 if args.profile in ('long-range','aerial','accuracy-aerial') else 1920)
    capture_height=args.capture_height or (2160 if args.profile in ('long-range','aerial','accuracy-aerial') else 1080)
    config.CONFIDENCE_THRESHOLD=args.confidence
    config.MAX_FRAME_WIDTH=3840
    config.DETECTION_INTERVAL_SECONDS=0.0
    # Runtime setting files are model/profile-specific. Calibration is not reused
    # across cameras, model changes, crop, zoom or lens-correction changes.
    import hashlib,os
    model_hash=hashlib.sha256(Path(model).read_bytes()).hexdigest()
    lens_path=os.getenv('CAMERA_LENS_CALIBRATION','')
    lens_hash=hashlib.sha256(Path(lens_path).read_bytes()).hexdigest() if lens_path else ''
    signature=hashlib.sha256((model_hash+args.profile+args.source+str(args.browser_camera)+lens_hash+
                             json.dumps(thresholds,sort_keys=True)+f'{capture_width}x{capture_height}').encode()).hexdigest()[:12]
    config.SETTINGS_FILE=root/'data'/f'laptop_{signature}.json'
    config.SAVE_DETECTION_IMAGES=False
    from app import run_server
    from laptop_camera import LatestFrameCamera
    camera=LatestFrameCamera(args.source,capture_width,capture_height) if not args.browser_camera else None
    return run_server(port=args.port,visible_source=args.source,browser_camera=args.browser_camera,detector=detector,camera=camera)

if __name__=='__main__':main()
