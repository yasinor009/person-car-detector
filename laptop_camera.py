"""Latest-frame camera reader: one slot, no unbounded stale-frame queue."""
from pathlib import Path
import threading
import time
import cv2
from camera_interface import CameraInterface


class LatestFrameCamera(CameraInterface):
    def __init__(self,visible_source=None,width=1920,height=1080):
        super().__init__(visible_source=visible_source,thermal_source='')
        self.requested_width=int(width);self.requested_height=int(height)
        if self.requested_width<1 or self.requested_height<1:raise ValueError('Invalid capture dimensions')
        self._condition=threading.Condition()
        self._stop=threading.Event();self._thread=None
        self._latest=None;self._sequence=0;self._delivered=0;self._dropped=0
        self._captured_at=None;self._eof=False
        self._file=isinstance(self.visible_source,str) and Path(self.visible_source).is_file()

    def _open(self,source):
        cap=super()._open(source)
        if cap is not None and isinstance(source,int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,self.requested_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,self.requested_height)
        return cap

    def connect(self):
        self.disconnect()
        if self._thread is not None and self._thread.is_alive():
            self.logger.error("Previous camera backend is still stopping; retry after its read returns")
            return False
        self.visible_cap=self._open(self.visible_source)
        if self.visible_cap is None:return False
        self._stop.clear();self._eof=False;self._latest=None
        self._sequence=self._delivered=self._dropped=0
        self._thread=threading.Thread(target=self._capture,name='latest-frame-camera',daemon=True)
        self._thread.start();return True

    def _capture(self):
        cap=self.visible_cap
        fps=cap.get(cv2.CAP_PROP_FPS) if self._file else 0
        period=1/fps if fps and 0<fps<=240 else 0
        next_tick=time.monotonic()
        try:
            while not self._stop.is_set():
                ok,frame=cap.read()
                if self._stop.is_set():break
                if not ok or frame is None:
                    with self._condition:self._eof=True;self._condition.notify_all()
                    if self._file:break
                    self._stop.wait(.1);continue
                with self._condition:
                    if self._sequence>self._delivered:self._dropped+=1
                    self._latest=frame;self._sequence+=1;self._captured_at=time.monotonic();self._eof=False
                    self._condition.notify_all()
                if period:
                    next_tick+=period
                    self._stop.wait(max(0,next_tick-time.monotonic()))
        finally:
            # The reader owns release. Never release a VideoCapture from another
            # thread while its backend may still be blocked inside read().
            cap.release()
            with self._condition:self._eof=True;self._condition.notify_all()

    def read_frames(self):
        with self._condition:
            self._condition.wait_for(lambda:self._sequence>self._delivered or self._stop.is_set() or self._eof,timeout=1.)
            if self._sequence<=self._delivered:return None,None
            self._delivered=self._sequence
            return self._latest.copy(),None

    def status(self):
        status=super().status()
        with self._condition:
            status.update(captured_frames=self._sequence,dropped_before_inference=self._dropped,
                          latest_frame_age_ms=(time.monotonic()-self._captured_at)*1000 if self._captured_at else None,
                          queue_policy='latest_only',end_of_stream=self._eof,
                          requested_capture_width=self.requested_width,requested_capture_height=self.requested_height)
        return status

    def disconnect(self):
        self._stop.set()
        with self._condition:self._condition.notify_all()
        thread=self._thread
        if thread and thread is not threading.current_thread():thread.join(timeout=2.)
        if thread and thread.is_alive():
            self.logger.warning("Camera read has not returned; background reader retains ownership until it exits")
            return
        super().disconnect()
        self._thread=None
