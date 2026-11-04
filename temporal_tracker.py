"""Small CPU tracker informed by SORT, ByteTrack and BoT-SORT.

This is a conservative hybrid, not a reproduction of their full pipelines.
High-score observations confirm tracks. Low-score observations can only update
an already confirmed track internally. Missing/predicted/low-score boxes are
never emitted. Eight-state xywh Kalman motion is used for association only;
metric range always uses the current detector's measured width.
"""
from __future__ import annotations
import numpy as np


def box_iou(a, b):
    x,y,w,h = a; X,Y,W,H = b
    inter=max(0.,min(x+w,X+W)-max(x,X))*max(0.,min(y+h,Y+H)-max(y,Y))
    return inter/max(1e-9,w*h+W*H-inter)


def linear_assignment(cost):
    """Exact rectangular minimum-cost assignment, O(n^2 m), no SciPy needed.

    Shortest augmenting path form of the Hungarian algorithm. Matrices must be
    finite. Callers use private dummy columns for explicit unmatched choices.
    """
    a=np.asarray(cost,dtype=float)
    if a.ndim!=2 or not np.isfinite(a).all():raise ValueError('Cost matrix must be finite and two-dimensional')
    n,m=a.shape
    if not n or not m:return []
    if n>m:return [(j,i) for i,j in linear_assignment(a.T)]
    u=np.zeros(n+1);v=np.zeros(m+1)
    p=np.zeros(m+1,dtype=int);way=np.zeros(m+1,dtype=int)
    for i in range(1,n+1):
        p[0]=i;j0=0;minimum=np.full(m+1,np.inf);used=np.zeros(m+1,dtype=bool)
        while True:
            used[j0]=True;i0=p[j0];delta=np.inf;j1=0
            for j in range(1,m+1):
                if used[j]:continue
                cur=a[i0-1,j-1]-u[i0]-v[j]
                if cur<minimum[j]:minimum[j]=cur;way[j]=j0
                if minimum[j]<delta:delta=minimum[j];j1=j
            u[p[used]]+=delta;v[used]-=delta;minimum[~used]-=delta
            j0=j1
            if p[j0]==0:break
        while True:
            j1=way[j0];p[j0]=p[j1];j0=j1
            if j0==0:break
    return [(int(p[j])-1,j-1) for j in range(1,m+1) if p[j]!=0]


def associate(tracks, detections, minimum_iou=.3):
    if not tracks or not detections:return []
    n,m=len(tracks),len(detections)
    # An invalid real match must be more expensive than choosing no match.
    # A dummy for each track prevents post-hoc thresholding from consuming a
    # valid partner for another track (a known greedy-assignment failure).
    cost=np.full((n,m+n),1.0)
    for i,t in enumerate(tracks):
        for j,d in enumerate(detections):
            overlap=box_iou(t['motion'].bbox(),d['bbox'])
            cost[i,j]=1-overlap if t['class']==d['class'] and overlap>=minimum_iou else 1e6
    return [(i,j) for i,j in linear_assignment(cost) if j<m and cost[i,j]<1.0]


class BoxMotion:
    """Constant-velocity xywh Kalman filter; predictions are not detections."""
    @staticmethod
    def measurement(box):
        x,y,w,h=map(float,box)
        return np.array([x+w/2,y+h/2,w,h])

    def __init__(self, box):
        z=self.measurement(box)
        self.x=np.r_[z,np.zeros(4)]
        size=np.maximum(z[[2,3,2,3]],8.)
        self.p=np.diag(np.r_[(.1*size)**2,np.full(4,250.**2)])

    def predict(self, dt):
        dt=max(.001,min(float(dt),3.))
        f=np.eye(8);f[:4,4:]=np.eye(4)*dt
        size=np.maximum(self.x[[2,3,2,3]],8.)
        # Noise is an engineering starting point, not a camera-calibrated model.
        acceleration=(.3*size)**2
        q=np.zeros((8,8))
        q[:4,:4]=np.diag(acceleration*dt**4/4)
        q[:4,4:]=q[4:,:4]=np.diag(acceleration*dt**3/2)
        q[4:,4:]=np.diag(acceleration*dt**2)
        self.x=f@self.x;self.p=f@self.p@f.T+q+np.eye(8)*1e-6
        self.x[2:4]=np.maximum(self.x[2:4],2.)

    def update(self, box, weak=False):
        z=self.measurement(box)
        size=np.maximum(z[[2,3,2,3]],8.)
        r=np.diag((.05*size)**2)*(4 if weak else 1)
        innovation_cov=self.p[:4,:4]+r
        gain=np.linalg.solve(innovation_cov,self.p[:4,:]).T
        self.x+=gain@(z-self.x[:4])
        # Joseph form avoids loss of symmetry/positive semidefiniteness.
        a=np.eye(8);a[:,:4]-=gain
        self.p=a@self.p@a.T+gain@r@gain.T
        self.p=(self.p+self.p.T)/2
        self.x[2:4]=np.maximum(self.x[2:4],2.)

    def bbox(self):
        cx,cy,w,h=self.x[:4]
        return [float(cx-w/2),float(cy-h/2),float(w),float(h)]


class TemporalTracker:
    def __init__(self, confirmation_frames=3, max_missed_frames=2):
        self.confirmation_frames=confirmation_frames
        self.max_missed_frames=max_missed_frames
        self.tracks=[];self.next_id=1;self.last_time=None;self.shape=None
        self.last_weak_updates=0

    def reset(self):
        self.tracks.clear();self.last_time=None;self.shape=None
        self.last_weak_updates=0
        # Never recycle IDs: an old user selection must not attach to a new object.

    def update(self, detections, now, shape, high_threshold=.6, max_gap_seconds=1.):
        if self.shape!=shape or (self.last_time is not None and (now<=self.last_time or now-self.last_time>max_gap_seconds)):
            self.reset()
        dt=now-self.last_time if self.last_time is not None else .1
        self.last_time=now;self.shape=shape;self.last_weak_updates=0
        self.tracks[:]=[t for t in self.tracks if now-t['last_strong_at']<=max_gap_seconds]
        for t in self.tracks:t['motion'].predict(dt)
        def threshold(d):
            return high_threshold[d['class']] if isinstance(high_threshold,dict) else high_threshold
        high=[d for d in detections if d['class'] in ('person','car') and d['confidence']>=threshold(d)]
        low=[d for d in detections if d['class'] in ('person','car')
             and min(.25,threshold(d)*.5)<=d['confidence']<threshold(d)]
        matches=associate(self.tracks,high)
        matched_tracks={i for i,j in matches};matched_detections={j for i,j in matches}
        result=[]
        for i,j in matches:
            t,d=self.tracks[i],high[j]
            t['motion'].update(d['bbox'])
            t['hits']+=1;t['strong_hits']+=1;t['misses']=0;t['last_strong_at']=now
            t['bbox']=d['bbox'];t['confirmed'] |= t['hits']>=self.confirmation_frames
            if t['confirmed']:result.append(dict(d,track_id=t['track_id'],confirmed_frames=t['strong_hits']))
        remaining=[i for i,t in enumerate(self.tracks) if i not in matched_tracks and t['confirmed']]
        weak_matches=associate([self.tracks[i] for i in remaining],low,minimum_iou=.5)
        weakly_matched=set()
        for local_i,j in weak_matches:
            i=remaining[local_i];t=self.tracks[i]
            t['motion'].update(low[j]['bbox'],weak=True)
            t['hits']=0;t['misses']=0;weakly_matched.add(i)
            self.last_weak_updates+=1
        for i,t in enumerate(self.tracks):
            if i not in matched_tracks and i not in weakly_matched:
                t['hits']=0;t['misses']+=1
        self.tracks[:]=[t for t in self.tracks
                       if t['misses']<=self.max_missed_frames and (t['confirmed'] or t['misses']==0)]
        for j,d in enumerate(high):
            if j in matched_detections:continue
            self.tracks.append(dict(d,motion=BoxMotion(d['bbox']),track_id=self.next_id,hits=1,strong_hits=1,
                                    confirmed=self.confirmation_frames<=1,misses=0,last_strong_at=now))
            if self.confirmation_frames<=1:result.append(dict(d,track_id=self.next_id,confirmed_frames=1))
            self.next_id+=1
        # Bounded metadata even during high false-positive loads.
        if len(self.tracks)>100:
            self.tracks.sort(key=lambda t:(t['confirmed'],t['last_strong_at']),reverse=True)
            del self.tracks[100:]
        return sorted(result,key=lambda d:d['confidence'],reverse=True)
