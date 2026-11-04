"""Pinhole pixel-budget estimate. This is not a probability of detection."""
import argparse
import math
import json


def pixel_budget(image_width,image_height,hfov_deg,distance_m,object_width_m=.5,object_height_m=1.7,network_width=640):
    values=(image_width,image_height,hfov_deg,distance_m,object_width_m,object_height_m,network_width)
    if not all(math.isfinite(float(v)) and float(v)>0 for v in values) or hfov_deg>=179:
        raise ValueError('Use finite positive dimensions, range and 0<HFOV<179 degrees')
    fx=image_width/(2*math.tan(math.radians(hfov_deg)/2))
    width=fx*object_width_m/distance_m;height=fx*object_height_m/distance_m
    resize=min(1.,network_width/max(image_width,image_height))
    return dict(focal_px=fx,source_object_width_px=width,source_object_height_px=height,
                full_frame_network_width_px=width*resize,full_frame_network_height_px=height*resize,
                distance_m=distance_m,assumptions='Pinhole, square pixels, front-facing known-size target, no distortion/crop; actual optics and pose may differ',
                verification='Pixel budget only; no 150m detection or >=90% performance is established')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--width',type=int,default=1920);p.add_argument('--height',type=int,default=1080)
    p.add_argument('--hfov',type=float,required=True);p.add_argument('--distance',type=float,default=150)
    p.add_argument('--object-width',type=float,default=.5);p.add_argument('--object-height',type=float,default=1.7)
    args=p.parse_args();print(json.dumps(pixel_budget(args.width,args.height,args.hfov,args.distance,args.object_width,args.object_height),indent=2))
if __name__=='__main__':main()
