"""Safely test one explicitly supplied camera source; does not scan the network."""
import argparse
import cv2
from camera_interface import parse_camera_source


def main():
    parser = argparse.ArgumentParser(description="Test one camera source (local index, file, HTTP, or RTSP URL)")
    parser.add_argument("source", help="Explicit source, e.g. 0 or rtsp://camera/stream")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to attempt")
    args = parser.parse_args()
    source = parse_camera_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("FAIL: source could not be opened")
        return 2
    got = 0
    shape = None
    try:
        for _ in range(max(1, args.frames)):
            ok, frame = cap.read()
            if ok and frame is not None:
                got += 1
                shape = frame.shape
    finally:
        cap.release()
    if not got:
        print("FAIL: source opened but no frames were read")
        return 1
    print(f"PASS: read {got} frame(s); last frame shape={shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
