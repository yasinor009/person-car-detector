"""Non-secret runtime diagnostics; no environment or credential dump."""
import json
import platform
import sys
import importlib.metadata
import onnxruntime as ort

def main():
    report={'os':platform.platform(),'python':sys.version.split()[0],
            'onnx_providers':ort.get_available_providers(),
            'versions':{k:importlib.metadata.version(k) for k in ('numpy','opencv-python')}}
    for package in ('onnxruntime','onnxruntime-gpu'):
        try:report['versions'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:pass
    try:
        import torch
        report['cuda_available']=torch.cuda.is_available()
        if torch.cuda.is_available():
            props=torch.cuda.get_device_properties(0)
            report['gpu']={'name':props.name,'vram_gib':props.total_memory/1024**3,'compute_capability':f'{props.major}.{props.minor}'}
    except ImportError:report['pytorch']='not installed (optional for ONNX inference)'
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
