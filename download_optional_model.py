"""Download an optional large aerial model; verify exact tested SHA-256.

Third-party pretrained weights. This does not train or certify >=90% performance.
"""
import hashlib
from pathlib import Path
import urllib.request

URL='https://huggingface.co/EdgeFirst/visdrone-yolo26-det/resolve/main/1280/onnx/visdrone-yolo26m-det-fp32-1280.onnx'
# Populated from the tested artifact in the release preparation.
SHA256='2b02c222c8dc19e036958d3655d827948e6f06b02c00f1146a88d7a57faba794'

def main():
    path=Path(__file__).resolve().parent/'models/visdrone26m_1280.onnx'
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==SHA256:
        print('Verified existing optional model');return
    temporary=path.with_suffix('.onnx.download')
    print('Downloading optional aerial model (~79 MiB)...')
    urllib.request.urlretrieve(URL,temporary)
    if hashlib.sha256(temporary.read_bytes()).hexdigest()!=SHA256:
        temporary.unlink();raise SystemExit('Checksum mismatch; rejected download')
    temporary.replace(path);print('Verified:',path)
if __name__=='__main__':main()
