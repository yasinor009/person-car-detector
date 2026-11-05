"""CPU ONNX YOLO11 detector. Only person and car can enter the live pipeline.

The historical class name is retained for dashboard/API compatibility.
Supported export contract: batch-one float32 NCHW, raw COCO [1,84,N].
"""
from __future__ import annotations
import ast
import logging
import os
import time
from collections import deque
from datetime import datetime
import config
from temporal_tracker import TemporalTracker

class DroneDetector:
    def __init__(self, model_path=None, confidence=None):
        self.model_path = model_path or config.MODEL_PATH
        self.confidence = float(config.CONFIDENCE_THRESHOLD if confidence is None else confidence)
        self.logger = logging.getLogger(__name__)
        self.session = None
        self.backend = 'onnx'
        self.model_status = 'not_loaded'
        self.model_message = ''
        self.model_names = {0: 'person', 2: 'car'}
        self.drone_class_ids = {0, 2}  # legacy diagnostic attribute
        self.history = deque(maxlen=1000)
        self.inference_ms = None
        self.last_inference_error = None
        self.inference_failures = 0
        self.last_raw_candidates = self.last_decoded_count = self.last_detection_count = 0
        self.input_shape = self.output_shapes = None
        self._tracker = TemporalTracker()
        self.tiled_inference = bool(getattr(config, "PERSON_CAR_TILED_INFERENCE", False))
        self.last_tile_passes = 0
        self._load_model()

    def _load_model(self):
        if not os.path.isfile(self.model_path):
            self.model_status = 'missing_model'
            self.model_message = 'Person/car model not found; run install_termux.sh. Motion fallback is disabled.'
            return
        try:
            import onnxruntime as ort
            import numpy as np
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 2))
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(self.model_path, sess_options=options, providers=['CPUExecutionProvider'])
            meta = session.get_inputs()[0]
            self.input_shape = list(meta.shape)
            if meta.type != 'tensor(float)' or len(meta.shape) != 4 or meta.shape[:2] != [1, 3]:
                raise ValueError('Require batch-one float32 NCHW export')
            self.input_height, self.input_width = meta.shape[2:]
            if not all(isinstance(v, int) and v >= 32 for v in (self.input_height, self.input_width)):
                raise ValueError('Static input size required')
            names = ast.literal_eval(session.get_modelmeta().custom_metadata_map.get('names', '{}'))
            if isinstance(names, list): names = dict(enumerate(names))
            names = {int(k): str(v) for k, v in names.items()}
            if names.get(0) != 'person' or names.get(2) != 'car' or len(names) != 80:
                raise ValueError('Model metadata must identify COCO 80 classes (0=person, 2=car)')
            probe = session.run(None, {meta.name: np.zeros(meta.shape, np.float32)})
            self._to_rows(probe[0])
            self.output_shapes = [list(v.shape) for v in probe]
            self.input_name, self.session = meta.name, session
            self.model_status, self.model_message = 'ready', 'YOLO11 COCO ONNX CPU; allowed classes: person, car'
        except Exception as exc:
            self.session = None
            self.model_status, self.model_message = 'load_error', str(exc)
            self.logger.exception('Could not load person/car model')

    @staticmethod
    def _to_rows(output):
        import numpy as np
        a = np.asarray(output)
        if a.ndim != 3 or a.shape[0] != 1 or a.shape[1] != 84:
            raise ValueError(f'Expected YOLO11 raw [1,84,N], received {a.shape}')
        return a[0].T

    def set_confidence(self, value):
        import math
        value = float(value)
        if not math.isfinite(value) or not 0.05 <= value <= 0.99:
            raise ValueError('Confidence threshold must be between 0.05 and 0.99.')
        if value != self.confidence: self.reset_tracking()
        self.confidence = value

    @staticmethod
    def _iou(a, b):
        x,y,w,h = a; X,Y,W,H = b
        inter = max(0,min(x+w,X+W)-max(x,X))*max(0,min(y+h,Y+H)-max(y,Y))
        return inter / max(1e-9,w*h+W*H-inter)

    @property
    def _tracks(self):
        # Legacy diagnostic accessor; tracks include private motion state.
        return self._tracker.tracks

    def reset_tracking(self):
        self._tracker.reset()

    def _confirm(self, detections, now, shape):
        return self._tracker.update(detections, now, shape,
                                    high_threshold=self.confidence,
                                    max_gap_seconds=max(1., 3*(self.inference_ms or 0)/1000))

    def _decode(self, output, frame_shape, scale_x, scale_y, left, top, score_floor=None):
        import numpy as np
        import cv2
        floor = self.confidence if score_floor is None else float(score_floor)
        rows = self._to_rows(output)
        self.last_raw_candidates = len(rows)
        probs = rows[:,4:]
        valid = np.isfinite(rows).all(axis=1)
        ids = np.argmax(np.where(np.isfinite(probs),probs,-np.inf),axis=1)
        scores = probs[np.arange(len(rows)),ids]
        # Argmax over ALL 80 classes before filtering avoids relabeling another
        # COCO object as person/car just because it has a secondary class score.
        valid &= np.isin(ids,[0,2]) & (scores >= floor) & (scores <= 1)
        valid &= (rows[:,2] > 0) & (rows[:,3] > 0)
        rows, scores, ids = rows[valid], scores[valid], ids[valid]
        h,w = frame_shape[:2]
        found = []
        for row,score,cls in zip(rows,scores,ids):
            cx,cy,bw,bh = row[:4]
            raw = ((cx-bw/2-left)/scale_x,(cy-bh/2-top)/scale_y,
                   (cx+bw/2-left)/scale_x,(cy+bh/2-top)/scale_y)
            x1,y1,x2,y2 = raw
            truncated = x1 <= 1 or y1 <= 1 or x2 >= w-1 or y2 >= h-1
            x1,x2 = np.clip([x1,x2],0,w); y1,y2 = np.clip([y1,y2],0,h)
            box = [float(x1),float(y1),float(x2-x1),float(y2-y1)]
            if min(box[2:]) < 8: continue
            found.append({'bbox':box,'confidence':round(float(score),4),'class':self.model_names[int(cls)],
                          'class_id':int(cls),'truncated':bool(truncated),'source':'visual_model',
                          'timestamp':datetime.now().isoformat(timespec='seconds')})
        self.last_decoded_count = len(found)
        return self._nms(found, floor)

    @staticmethod
    def _nms(found, floor):
        import numpy as np
        import cv2
        keep = []
        for cls in ('person','car'):
            group = [d for d in found if d['class']==cls]
            if not group: continue
            indices = cv2.dnn.NMSBoxes([d['bbox'] for d in group], [d['confidence'] for d in group], floor, .45)
            if indices is None: continue
            for i in np.asarray(indices).reshape(-1): keep.append(group[int(i)])
        return sorted(keep,key=lambda d:d['confidence'],reverse=True)[:100]

    def _detect_onnx_single(self, frame, score_floor=None):
        import cv2
        import numpy as np
        h,w = frame.shape[:2]
        scale = min(self.input_width/w,self.input_height/h)
        rw,rh = max(1,round(w*scale)),max(1,round(h*scale))
        left,top = (self.input_width-rw)//2,(self.input_height-rh)//2
        canvas = np.full((self.input_height,self.input_width,3),114,np.uint8)
        canvas[top:top+rh,left:left+rw] = cv2.resize(frame,(rw,rh))
        tensor = np.ascontiguousarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB).transpose(2,0,1)[None],dtype=np.float32)/255
        output = self.session.run(None,{self.input_name:tensor})[0]
        return self._decode(output,frame.shape,rw/w,rh/h,left,top,score_floor)

    def _detect_frame(self, frame, score_floor):
        """Optional bounded SAHI-style full image + four overlapping slices."""
        found = self._detect_onnx_single(frame, score_floor)
        raw, decoded = self.last_raw_candidates, self.last_decoded_count
        self.last_tile_passes = 0
        h,w = frame.shape[:2]
        if self.tiled_inference and min(h,w) >= 192:
            th,tw = int(round(h*.58)),int(round(w*.58))
            for top in (0,h-th):
                for left in (0,w-tw):
                    crop = frame[top:top+th,left:left+tw]
                    items = self._detect_onnx_single(crop, score_floor)
                    raw += self.last_raw_candidates; decoded += self.last_decoded_count
                    self.last_tile_passes += 1
                    for d in items:
                        x,y,bw,bh = d['bbox']
                        # Reject boxes clipped at an internal slice boundary:
                        # their partial widths would corrupt metric distance.
                        if ((left>0 and x<=1) or (top>0 and y<=1)
                                or (left+tw<w and x+bw>=tw-1)
                                or (top+th<h and y+bh>=th-1)):
                            continue
                        d = dict(d, bbox=[x+left,y+top,bw,bh],source='visual_model_tiled')
                        found.append(d)
            found = self._nms(found, score_floor)
        self.last_raw_candidates, self.last_decoded_count = raw, decoded
        return found

    def detect(self, frame, temporal=True):
        if frame is None or self.model_status != 'ready': return []
        start = time.perf_counter()
        try:
            # Keep low-score observations for association; only the user threshold
            # may create/confirm a visible track. Never impose a hidden .55 floor.
            detections = self._detect_frame(frame, min(.25, self.confidence * .5) if temporal else self.confidence)
            now = time.monotonic()
            detections = self._confirm(detections,now,frame.shape) if temporal else detections
            self.last_inference_error = None
            self.last_detection_count = len(detections)
            if detections: self.history.append({'time':time.time(),'count':len(detections)})
            return detections
        except Exception as exc:
            self.reset_tracking()
            self.last_detection_count = 0
            self.inference_failures += 1
            self.last_inference_error = str(exc)
            self.logger.exception('Inference failed')
            return []
        finally: self.inference_ms = round((time.perf_counter()-start)*1000,2)

    def get_stats(self):
        return dict(model_status=self.model_status,model_message=self.model_message,model_path=self.model_path,
                    backend=self.backend,input_shape=self.input_shape,output_shapes=self.output_shapes,
                    inference_ms=self.inference_ms,last_inference_error=self.last_inference_error,
                    inference_failures=self.inference_failures,raw_candidates=self.last_raw_candidates,
                    decoded_candidates=self.last_decoded_count,last_detection_count=self.last_detection_count,
                    confidence_threshold=self.confidence,effective_confidence_threshold=self.confidence,
                    drone_classes=['person','car'],allowed_classes=['person','car'],confirmation_frames=3,
                    tracker='xywh_kalman_hungarian_two_stage',
                    retained_tracks=len(self._tracks), weak_updates=self._tracker.last_weak_updates,
                    tiled_inference=self.tiled_inference,tile_passes=self.last_tile_passes,
                    decode_layout='yolo11_raw_class',detections_last_60s=sum(d['count'] for d in self.history if time.time()-d['time']<60))
