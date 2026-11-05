"""Optional calibrated lens correction with cached OpenCV remap tables."""
import json
from pathlib import Path
import numpy as np
import cv2

class LensRectifier:
    def __init__(self, path=None):
        self.maps = {}
        self.calibration = None
        if path:
            data=json.loads(Path(path).read_text())
            k=np.asarray(data['camera_matrix'],dtype=np.float64)
            d=np.asarray(data['distortion'],dtype=np.float64).reshape(-1)
            size=tuple(data['image_size'])
            if k.shape!=(3,3) or d.size not in (4,5,8,12,14) or len(size)!=2:
                raise ValueError('Invalid camera calibration shapes')
            if not np.isfinite(k).all() or not np.isfinite(d).all() or k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1]):
                raise ValueError('Invalid camera calibration values')
            if any(not isinstance(v,int) or v<=0 for v in size):raise ValueError('Invalid image_size')
            self.calibration=(k,d,size)
    def correct(self, frame):
        if frame is None or self.calibration is None:return frame
        h,w=frame.shape[:2]
        k,d,(cw,ch)=self.calibration
        if abs((w/h)/(cw/ch)-1)>.01:
            raise ValueError('Lens calibration aspect ratio differs; recalibrate for this camera mode')
        key=(w,h)
        if key not in self.maps:
            scaled=k.copy();scaled[0]*=w/cw;scaled[1]*=h/ch
            # Same K for remap and range calibration; no crop or hidden zoom.
            self.maps.clear()
            self.maps[key]=cv2.initUndistortRectifyMap(k if key==(cw,ch) else scaled,d,None,scaled,(w,h),cv2.CV_16SC2)
        return cv2.remap(frame,*self.maps[key],interpolation=cv2.INTER_LINEAR)
