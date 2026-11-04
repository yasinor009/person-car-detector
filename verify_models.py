from pathlib import Path
import hashlib

def main():
    root=Path(__file__).resolve().parent
    for line in (root/'models/SHA256SUMS').read_text().splitlines():
        if not line.strip():continue
        expected,name=line.split(maxsplit=1);p=root/name.strip().lstrip('*')
        if not p.is_file():raise SystemExit(f'Missing bundled model: {p.name}')
        actual=hashlib.sha256(p.read_bytes()).hexdigest()
        if actual!=expected:raise SystemExit(f'Model checksum mismatch: {p.name}')
        print(f'OK {p.name}')
if __name__=='__main__':main()
