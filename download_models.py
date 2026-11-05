"""Download hash-verified ground-camera ONNX weights, no PyTorch on the phone."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

MODELS = {
    'mobile': ('person_car_yolo26n.onnx', 'dddeb2b7f34a57d6ac126a0531ca5957b991a135bf5359af7c3348cfc8e9ef59'),
    'balanced': ('person_car_yolo26s.onnx', 'ce93650c133e381bc13bad8d02baf5bd63c17f3eaa0ad479f2d3ab667f04a80d'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('mobile', 'balanced', 'all'), default='mobile')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent/'models'
    root.mkdir(exist_ok=True)
    for profile in (MODELS if args.profile == 'all' else [args.profile]):
        filename, expected = MODELS[profile]
        path = root/filename
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            print('Verified:', path.name)
            continue
        temporary = path.with_suffix('.onnx.download')
        url = 'https://raw.githubusercontent.com/yasinor009/person-car-detector/main/models/'+filename
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as output:
            while chunk := response.read(1024*1024):
                output.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
            temporary.unlink()
            raise SystemExit('Checksum mismatch; model rejected')
        temporary.replace(path)
        print('Downloaded and verified:', path.name)


if __name__ == '__main__':
    main()
