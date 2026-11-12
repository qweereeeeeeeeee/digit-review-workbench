"""Fetch the pinned official checkpoint; never replace it with random weights."""
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def fetch(manifest,filename):
    folder = ROOT / 'artifacts/objects'
    source = json.loads((folder / manifest).read_text())
    target = folder / filename
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == source['sha256']:
        print('模型已存在，SHA-256 校验通过'); return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=folder, suffix='.part', delete=False) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(source['url'], timeout=90) as response:
                while chunk := response.read(1024 * 1024): output.write(chunk)
        actual = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if actual != source['sha256']: raise RuntimeError('下载文件校验失败，原模型未改动')
        temporary.replace(target)
        print(f'下载完成，SHA-256: {actual}')
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)

if __name__ == '__main__':
    fetch('model-source.json','resnet50-11ad3fa6.pth')
    fetch('detector-source.json','yolo11s.pt')
