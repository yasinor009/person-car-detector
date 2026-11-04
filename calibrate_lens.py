"""Calibrate lens distortion from >=10 varied chessboard photos.

Rows and columns count INNER corners, not squares. square-m is square side length.
Capture using the same camera, aspect ratio, zoom and focus as the live stream.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

def fit_lens(objects, images, size):
    """Zhang/OpenCV calibration with per-view robust screening and pose checks.

    Quality thresholds are conservative engineering defaults, not guarantees of
    metrological accuracy. Requires independently captured varied board poses.
    """
    if len(images)<10 or len(objects)!=len(images):
        raise ValueError('Need at least 10 matched calibration views')
    if len(size)!=2 or any(not isinstance(v,int) or v<=0 for v in size):
        raise ValueError('Invalid image size')
    for obj,img in zip(objects,images):
        if len(obj)!=len(img) or not np.isfinite(obj).all() or not np.isfinite(img).all():
            raise ValueError('Invalid calibration points')
    active=list(range(len(images)))
    for attempt in range(4):
        rms,k,d,rvecs,tvecs,std_k,_,view_errors=cv2.calibrateCameraExtended(
            [objects[i] for i in active],[images[i] for i in active],size,None,None)
        errors=np.asarray(view_errors).reshape(-1)
        median=float(np.median(errors));mad=float(np.median(np.abs(errors-median)))
        gate=max(.35,median+3*1.4826*mad)
        keep=[i for i,error in zip(active,errors) if error<=gate]
        if len(keep)<10:raise ValueError('Fewer than 10 good views remain after outlier screening')
        if keep==active:break
        if attempt==3:raise ValueError('Unstable calibration outlier screening; collect better views')
        active=keep
    if not np.isfinite(rms) or rms>1:raise ValueError(f'Calibration reprojection RMS too high: {rms:.3f} px')
    if not np.isfinite(k).all() or not np.isfinite(d).all():raise ValueError('Nonfinite calibration')
    width,height=size
    if not (.2*width<k[0,0]<10*width and .2*height<k[1,1]<10*height
            and -.25*width<k[0,2]<1.25*width and -.25*height<k[1,2]<1.25*height):
        raise ValueError('Implausible camera intrinsics; check board dimensions and varied poses')
    normals=np.array([cv2.Rodrigues(r)[0][:,2] for r in rvecs])
    angles=np.degrees(np.arccos(np.clip(normals@normals.T,-1,1)))
    normal_span=float(np.max(angles))
    if normal_span<10:raise ValueError('Board orientations are too similar; capture tilted boards')
    corners=np.concatenate([images[i].reshape(-1,2) for i in active]).astype(np.float32)
    coverage=float(cv2.contourArea(cv2.convexHull(corners))/(width*height))
    if coverage<.12:raise ValueError('Board corners cover too little of the image; vary position and size')
    std_k=np.asarray(std_k).reshape(-1)
    if not np.isfinite(std_k[:4]).all() or max(std_k[0]/k[0,0],std_k[1]/k[1,1])>.15:
        raise ValueError('Focal parameters too uncertain; collect more diverse calibration views')
    return dict(camera_matrix=k.tolist(),distortion=d.reshape(-1).tolist(),image_size=list(size),
                reprojection_rms_px=float(rms),images=len(active),rejected_images=len(images)-len(active),
                accepted_view_indices=active,per_view_rms_px=errors.tolist(),
                conditional_intrinsic_std=std_k[:4].tolist(),normal_span_degrees=normal_span,
                corner_coverage_ratio=coverage,method='zhang_opencv_with_mad_view_screening',
                quality_note='Reprojection and diversity checks do not bound real-world depth error')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder');p.add_argument('--cols',type=int,default=9);p.add_argument('--rows',type=int,default=6)
    p.add_argument('--square-m',type=float,required=True);p.add_argument('--output',default='data/lens.json')
    args=p.parse_args()
    if args.cols<3 or args.rows<3 or not np.isfinite(args.square_m) or args.square_m<=0:p.error('Invalid board dimensions')
    board=(args.cols,args.rows)
    object_points=np.zeros((board[0]*board[1],3),np.float32)
    object_points[:,:2]=np.mgrid[:board[0],:board[1]].T.reshape(-1,2)*args.square_m
    objects=[];images=[];size=None
    for file in sorted(Path(args.folder).glob('*')):
        frame=cv2.imread(str(file))
        if frame is None:continue
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        current=(gray.shape[1],gray.shape[0])
        if size and current!=size:raise ValueError('All calibration photos must have identical dimensions')
        size=current
        found,corners=cv2.findChessboardCorners(gray,board)
        if not found:continue
        corners=cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,40,.001))
        objects.append(object_points.copy());images.append(corners)
    data=fit_lens(objects,images,size)
    rms=data['reprojection_rms_px']
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(data,indent=2))
    print(f'Saved {output}; RMS={rms:.3f}px. Recalibrate metric range on corrected images.')
if __name__=='__main__':main()
