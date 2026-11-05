"""Laptop detector: CUDA-aware ONNX/PyTorch, explicit class contracts, native-pixel slices.

Pretrained COCO and VisDrone names map to canonical person/car. A custom model
must declare both targets in metadata. No guessed class-ID mappings are used.
"""
from __future__ import annotations
import ast
import json
import logging
import math
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from temporal_tracker import TemporalTracker

ALIASES={'person':'person','pedestrian':'person','people':'person','human':'person',
         'car':'car','automobile':'car','passenger car':'car'}
CANONICAL_IDS={'person':0,'car':2}


def class_contract(names):
    if isinstance(names,str):
        try:names=json.loads(names)
        except json.JSONDecodeError:names=ast.literal_eval(names)
    if isinstance(names,list):names=dict(enumerate(names))
    if not isinstance(names,dict) or not names:raise ValueError('Model class names are required')
    normalized={int(k):str(v).strip().lower().replace('_',' ') for k,v in names.items()}
    if set(normalized)!=set(range(len(normalized))):raise ValueError('Class IDs must be contiguous starting at zero')
    targets={i:ALIASES[name] for i,name in normalized.items() if name in ALIASES}
    if set(targets.values())!={'person','car'}:raise ValueError('Model must explicitly contain both person and car classes')
    return normalized,targets


def tile_plan(width,height,tile_size=960,overlap=.25,max_tiles=24):
    """All pixels covered, including right/bottom edges; no silent truncation."""
    if tile_size<128 or not 0<=overlap<.75 or max_tiles<1:raise ValueError('Invalid tile configuration')
    def starts(length):
        if length<=tile_size:return [0]
        step=max(1,int(tile_size*(1-overlap)))
        positions=list(range(0,length-tile_size+1,step))
        if positions[-1]!=length-tile_size:positions.append(length-tile_size)
        return positions
    result=[(x,y,min(tile_size,width-x),min(tile_size,height-y)) for y in starts(height) for x in starts(width)]
    if len(result)>max_tiles:
        raise ValueError(f'Frame needs {len(result)} tiles, exceeding max_tiles={max_tiles}; increase tile size or tile limit')
    return result


def canonical_nms(items,iou=.5,max_detections=500):
    keep=[]
    for name in ('person','car'):
        group=[d for d in items if d['class']==name]
        if not group:continue
        indices=cv2.dnn.NMSBoxes([d['bbox'] for d in group],[d['confidence'] for d in group],0.,iou)
        if indices is not None:
            keep.extend(group[int(i)] for i in np.asarray(indices).reshape(-1))
    return sorted(keep,key=lambda d:d['confidence'],reverse=True)[:max_detections]


class LaptopDetector:
    def __init__(self,model_path,confidence=.25,device='auto',imgsz=960,tiled=False,
                 tile_size=960,overlap=.25,max_tiles=24,batch_size=1,confirmation_frames=3,
                 nms_iou=.5,onnx_layout='auto',class_thresholds=None):
        self.model_path=str(Path(model_path).resolve())
        self.confidence=float(confidence)
        self.class_thresholds=class_thresholds or {}
        self.device_requested=device
        self.device='cpu';self.provider=None;self.backend=None
        self.imgsz=int(imgsz);self.tiled_inference=bool(tiled)
        self.tile_size=int(tile_size);self.overlap=float(overlap);self.max_tiles=int(max_tiles)
        self.batch_size=int(batch_size);self.nms_iou=float(nms_iou);self.onnx_layout=onnx_layout
        if not .01<=self.confidence<=.99 or self.imgsz<128 or self.imgsz%32 or self.batch_size<1:
            raise ValueError('Invalid confidence/image size/batch size')
        if not 0<self.nms_iou<1 or onnx_layout not in ('auto','raw','end2end'):raise ValueError('Invalid decoder configuration')
        tile_plan(128,128,self.tile_size,self.overlap,self.max_tiles)
        for k,v in self.class_thresholds.items():
            if k not in CANONICAL_IDS or not .01<=float(v)<=.99:raise ValueError('Invalid per-class threshold')
        self._tracker=TemporalTracker(confirmation_frames=confirmation_frames)
        self.history=deque(maxlen=1000)
        self.session=None;self.model=None;self.model_status='not_loaded';self.model_message=''
        self.model_names={};self.target_ids={};self.output_shapes=[];self.input_shape=None
        self.output_normalized=False
        self.inference_ms=None;self.last_inference_error=None;self.inference_failures=0
        self.last_raw_candidates=0;self.last_decoded_count=0;self.last_detection_count=0
        self.last_tile_passes=0;self.last_batches=0
        self.logger=logging.getLogger(__name__)
        self._load_model()

    def _load_model(self):
        if not Path(self.model_path).is_file():
            self.model_status='missing_model';self.model_message=f'Model file not found: {self.model_path}';return
        try:
            suffix=Path(self.model_path).suffix.lower()
            if suffix=='.onnx':self._load_onnx()
            elif suffix=='.pt':self._load_torch()
            else:raise ValueError('Use a verified .onnx or .pt detection model')
            self.model_status='ready'
            self.model_message=f'{Path(self.model_path).name}: {self.backend}, {self.device}; person/car only'
        except Exception as exc:
            self.session=None;self.model=None;self.model_status='load_error';self.model_message=str(exc)
            self.logger.exception('Laptop model load failed')

    def _load_onnx(self):
        import onnxruntime as ort
        providers=ort.get_available_providers()
        requested=self.device_requested
        cuda=('CUDAExecutionProvider' in providers)
        if cuda and requested!='cpu' and hasattr(ort,'preload_dlls'):
            ort.preload_dlls()
        if requested.startswith('cuda') and not cuda:raise ValueError('CUDA ONNX Runtime unavailable; install the GPU runtime or choose --device cpu')
        gpu=requested!='cpu' and cuda
        if requested not in ('auto','cpu','cuda','cuda:0'):raise ValueError('Supported devices: auto, cpu, cuda:0')
        chosen=['CUDAExecutionProvider','CPUExecutionProvider'] if gpu else ['CPUExecutionProvider']
        options=ort.SessionOptions();options.intra_op_num_threads=min(4,os.cpu_count() or 2);options.inter_op_num_threads=1
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session=ort.InferenceSession(self.model_path,sess_options=options,providers=chosen)
        self.provider=self.session.get_providers()[0]
        if gpu and self.provider!='CUDAExecutionProvider':raise ValueError('CUDA initialization failed; refusing to silently claim GPU inference')
        self.device='cuda:0' if self.provider=='CUDAExecutionProvider' else 'cpu'
        self.backend='onnx'
        if len(self.session.get_inputs())!=1:raise ValueError('Require a single image input')
        meta=self.session.get_inputs()[0];self.input_name=meta.name;self.input_shape=list(meta.shape)
        if len(meta.shape)!=4 or meta.shape[1]!=3 or meta.type not in ('tensor(float)','tensor(float16)'):
            raise ValueError('Require float32/float16 NCHW image input')
        self.dtype=np.float16 if meta.type=='tensor(float16)' else np.float32
        self.input_height=meta.shape[2] if isinstance(meta.shape[2],int) else self.imgsz
        self.input_width=meta.shape[3] if isinstance(meta.shape[3],int) else self.imgsz
        if not (128<=self.input_height<=4096 and 128<=self.input_width<=4096):raise ValueError('Unsupported model input size')
        self.static_batch=meta.shape[0] if isinstance(meta.shape[0],int) else None
        if self.static_batch not in (None,1):raise ValueError('Use batch-one or dynamic-batch ONNX exports')
        metadata=self.session.get_modelmeta().custom_metadata_map
        if 'edgefirst' in metadata:
            edge=json.loads(metadata['edgefirst'])
            outputs=edge.get('outputs',[])
            if len(outputs)!=1 or outputs[0].get('encoding')!='direct':
                raise ValueError('Unsupported EdgeFirst decoder contract')
            self.output_normalized=bool(outputs[0].get('normalized',False))
        names=metadata.get('names',metadata.get('labels'))
        self.model_names,self.target_ids=class_contract(names)
        if self.onnx_layout=='auto':self.onnx_layout='end2end' if metadata.get('end2end','').lower()=='true' else 'raw'
        probe=self.session.run(None,{self.input_name:np.zeros((1,3,self.input_height,self.input_width),dtype=self.dtype)})
        if len(probe)!=1:raise ValueError('Require one YOLO output tensor; RT-DETR multi-output models need a different decoder')
        self.output_shapes=[list(v.shape) for v in probe]
        self._decode_rows(probe[0],self.input_width,self.input_height,(1.,1.,0,0))

    def _load_torch(self):
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(min(4,os.cpu_count() or 2))
        if self.device_requested.startswith('cuda') and not torch.cuda.is_available():raise ValueError('CUDA PyTorch unavailable')
        if self.device_requested not in ('auto','cpu','cuda','cuda:0'):raise ValueError('Use auto, cpu or cuda:0')
        self.device='cuda:0' if self.device_requested!='cpu' and torch.cuda.is_available() else 'cpu'
        self.model=YOLO(self.model_path,task='detect')
        self.model_names,self.target_ids=class_contract(self.model.names)
        self.backend='pytorch';self.provider=self.device
        self.input_width=self.input_height=self.imgsz
        self.input_shape=[None,3,self.imgsz,self.imgsz]

    def threshold(self,name):return float(self.class_thresholds.get(name,self.confidence))

    def set_confidence(self,value):
        value=float(value)
        if not math.isfinite(value) or not .01<=value<=.99:raise ValueError('Confidence must be between .01 and .99')
        if value!=self.confidence:
            self.reset_tracking();self.class_thresholds={}
        self.confidence=value

    def reset_tracking(self):self._tracker.reset()

    @property
    def _tracks(self):return self._tracker.tracks

    def _preprocess(self,frame):
        h,w=frame.shape[:2];scale=min(self.input_width/w,self.input_height/h)
        rw,rh=max(1,round(w*scale)),max(1,round(h*scale))
        left,top=(self.input_width-rw)//2,(self.input_height-rh)//2
        canvas=np.full((self.input_height,self.input_width,3),114,np.uint8)
        canvas[top:top+rh,left:left+rw]=cv2.resize(frame,(rw,rh))
        tensor=np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1),dtype=self.dtype)/self.dtype(255.)
        return tensor,(rw/w,rh/h,left,top)

    def _decode_rows(self,output,frame_w,frame_h,transform,floor=.05):
        a=np.asarray(output)
        if a.ndim!=3 or a.shape[0]!=1:raise ValueError(f'Expected batch-one output, got {a.shape}')
        a=a[0];nc=len(self.model_names)
        if self.onnx_layout=='raw':
            if a.shape[0]==nc+4:a=a.T
            elif a.shape[1]!=nc+4:raise ValueError(f'Raw output must have {nc+4} features, got {a.shape}')
            self.last_raw_candidates+=len(a)
            valid=np.isfinite(a).all(axis=1)&(a[:,2]>0)&(a[:,3]>0)
            probs=np.where(np.isfinite(a[:,4:]),a[:,4:],-np.inf)
            classes=probs.argmax(axis=1);scores=probs[np.arange(len(a)),classes]
            valid&=np.isin(classes,list(self.target_ids))&(scores>=floor)&(scores<=1.)
            rows=a[valid];scores=scores[valid];classes=classes[valid]
            # Bound postprocessing work without clipping confidence ranking.
            order=np.argsort(-scores)[:4000];rows=rows[order];scores=scores[order];classes=classes[order]
            coords=rows[:,:4].astype(np.float64,copy=True)
            xyxy=np.column_stack((coords[:,0]-coords[:,2]/2,coords[:,1]-coords[:,3]/2,
                                  coords[:,0]+coords[:,2]/2,coords[:,1]+coords[:,3]/2))
        else:
            if a.shape[1]!=6:raise ValueError('End-to-end layout must be [1,N,6] in xyxy,score,class order')
            self.last_raw_candidates+=len(a)
            valid=np.isfinite(a).all(axis=1)
            valid&=(a[:,5]>=0)&(a[:,5]<nc)&(np.abs(a[:,5]-np.rint(a[:,5]))<1e-5)
            safe_classes=np.where(np.isfinite(a[:,5]),np.rint(a[:,5]),-1).astype(int)
            valid&=np.isin(safe_classes,list(self.target_ids))&(a[:,4]>=floor)&(a[:,4]<=1.)
            rows=a[valid];xyxy=rows[:,:4].astype(np.float64,copy=True);scores=rows[:,4];classes=safe_classes[valid]
        if self.output_normalized:
            xyxy[:,[0,2]]*=self.input_width
            xyxy[:,[1,3]]*=self.input_height
        sx,sy,left,top=transform
        xyxy[:,[0,2]]=(xyxy[:,[0,2]]-left)/sx;xyxy[:,[1,3]]=(xyxy[:,[1,3]]-top)/sy
        # Vectorize clipping/filtering; dense aerial scenes can have thousands
        # of candidates and per-box np.clip calls waste CPU time.
        truncated=(xyxy[:,0]<=1)|(xyxy[:,1]<=1)|(xyxy[:,2]>=frame_w-1)|(xyxy[:,3]>=frame_h-1)
        xyxy[:,[0,2]]=np.clip(xyxy[:,[0,2]],0,frame_w)
        xyxy[:,[1,3]]=np.clip(xyxy[:,[1,3]],0,frame_h)
        widths=xyxy[:,2]-xyxy[:,0];heights=xyxy[:,3]-xyxy[:,1]
        good=(widths>=2)&(heights>=2)
        result=[]
        stamp=datetime.now().isoformat(timespec='seconds')
        for box,bw,bh,score,cls,cut in zip(xyxy[good],widths[good],heights[good],scores[good],classes[good],truncated[good]):
            name=self.target_ids[int(cls)]
            result.append(dict(bbox=[float(box[0]),float(box[1]),float(bw),float(bh)],confidence=float(score),
                               **{'class':name},class_id=CANONICAL_IDS[name],model_class_id=int(cls),
                               truncated=bool(cut),source='visual_model',timestamp=stamp))
        self.last_decoded_count+=len(result)
        return canonical_nms(result,self.nms_iou)

    def _infer(self,frames,floor):
        if self.backend=='pytorch':
            self.last_batches+=1
            results=self.model.predict(frames,imgsz=self.imgsz,conf=floor,iou=self.nms_iou,
                                       classes=None,max_det=1000,device=self.device,half=self.device.startswith('cuda'),verbose=False)
            outputs=[]
            for frame,r in zip(frames,results):
                h,w=frame.shape[:2];items=[]
                for box in r.boxes.data.cpu().numpy():
                    x,y,X,Y,score,cls=map(float,box[:6]);cls=int(cls)
                    if cls not in self.target_ids:continue
                    name=self.target_ids[cls]
                    if X-x<2 or Y-y<2:continue
                    items.append(dict(bbox=[x,y,X-x,Y-y],confidence=score,**{'class':name},class_id=CANONICAL_IDS[name],
                                      model_class_id=cls,truncated=x<=1 or y<=1 or X>=w-1 or Y>=h-1,source='visual_model',
                                      timestamp=datetime.now().isoformat(timespec='seconds')))
                self.last_raw_candidates+=len(r.boxes);self.last_decoded_count+=len(items)
                outputs.append(canonical_nms(items,self.nms_iou))
            return outputs
        prepared=[self._preprocess(frame) for frame in frames]
        # Static exports require separate runs; dynamic exports can batch slices.
        if self.static_batch==1 and len(frames)>1:
            return [self._infer([frame],floor)[0] for frame in frames]
        self.last_batches+=1
        tensor=np.stack([x[0] for x in prepared])
        raw=self.session.run(None,{self.input_name:tensor})[0]
        return [self._decode_rows(raw[i:i+1],frame.shape[1],frame.shape[0],prepared[i][1],floor)
                for i,frame in enumerate(frames)]

    def _detect_frame(self,frame,floor=.05):
        self.last_raw_candidates=self.last_decoded_count=self.last_batches=self.last_tile_passes=0
        h,w=frame.shape[:2]
        found=self._infer([frame],floor)[0]
        if not self.tiled_inference or max(h,w)<=self.tile_size:return found
        plan=tile_plan(w,h,self.tile_size,self.overlap,self.max_tiles)
        for start in range(0,len(plan),self.batch_size):
            chunk=plan[start:start+self.batch_size]
            crops=[frame[y:y+th,x:x+tw] for x,y,tw,th in chunk]
            outputs=self._infer(crops,floor)
            for (left,top,tw,th),items in zip(chunk,outputs):
                self.last_tile_passes+=1
                for d in items:
                    x,y,bw,bh=d['bbox']
                    if ((left>0 and x<=1) or (top>0 and y<=1)
                            or (left+tw<w and x+bw>=tw-1) or (top+th<h and y+bh>=th-1)):continue
                    found.append(dict(d,bbox=[x+left,y+top,bw,bh],source='visual_model_tiled'))
        return canonical_nms(found,self.nms_iou)

    def detect(self,frame,temporal=True):
        if frame is None or self.model_status!='ready':return []
        start=time.perf_counter()
        try:
            minimum=min(self.threshold(name) for name in CANONICAL_IDS)
            floor=min(.25,minimum*.5) if temporal else minimum
            items=self._detect_frame(frame,floor)
            if temporal:
                thresholds={k:self.threshold(k) for k in CANONICAL_IDS}
                items=self._tracker.update(items,time.monotonic(),frame.shape,thresholds,
                                           max_gap_seconds=max(1.,3*(self.inference_ms or 0)/1000))
            else:items=[d for d in items if d['confidence']>=self.threshold(d['class'])]
            self.last_inference_error=None;self.last_detection_count=len(items)
            if items:self.history.append(dict(time=time.time(),count=len(items)))
            return items
        except Exception as exc:
            self.reset_tracking();self.last_detection_count=0;self.inference_failures+=1
            self.last_inference_error=f'{type(exc).__name__}: {exc}'
            self.logger.exception('Laptop inference failed');return []
        finally:self.inference_ms=round((time.perf_counter()-start)*1000,2)

    def get_stats(self):
        return dict(model_status=self.model_status,model_message=self.model_message,model_path=self.model_path,
                    backend=self.backend,device=self.device,provider=self.provider,input_shape=self.input_shape,
                    output_shapes=self.output_shapes,input_layout='NCHW',decode_layout=self.onnx_layout,
                    confidence_threshold=self.confidence,effective_confidence_threshold=self.confidence,
                    class_thresholds={k:self.threshold(k) for k in CANONICAL_IDS},allowed_classes=['person','car'],
                    model_class_names=self.model_names,tiled_inference=self.tiled_inference,tile_passes=self.last_tile_passes,
                    batches=self.last_batches,inference_ms=self.inference_ms,inference_failures=self.inference_failures,
                    last_inference_error=self.last_inference_error,raw_candidates=self.last_raw_candidates,
                    decoded_candidates=self.last_decoded_count,last_detection_count=self.last_detection_count,
                    confirmation_frames=self._tracker.confirmation_frames,retained_tracks=len(self._tracks),
                    detections_last_60s=sum(d['count'] for d in self.history if time.time()-d['time']<60))
