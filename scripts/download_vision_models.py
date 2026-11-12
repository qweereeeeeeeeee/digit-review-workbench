"""Download pinned official model bundles; verify every file before replacing it."""
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request
ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'artifacts/vision'

def fetch_bundle(folder,manifest):
    meta=json.loads((ASSETS/manifest).read_text());target=ASSETS/folder;target.mkdir(parents=True,exist_ok=True)
    for entry in meta['files']:
        name=entry['name']
        if Path(name).name!=name:raise ValueError('模型清单含非法文件名')
        path=target/name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']:
            print('已校验',name,flush=True);continue
        url=entry.get('url') or f"https://huggingface.co/{meta['repo_id']}/resolve/{meta['revision']}/{name}"
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(dir=target,suffix='.part',delete=False) as output:
                temporary=Path(output.name)
                print('下载',name,flush=True)
                with urllib.request.urlopen(url,timeout=120) as response:
                    while chunk:=response.read(1024*1024):output.write(chunk)
            if hashlib.sha256(temporary.read_bytes()).hexdigest()!=entry['sha256']:raise RuntimeError('文件校验失败：'+name)
            temporary.replace(path)
        finally:
            if temporary is not None:temporary.unlink(missing_ok=True)

if __name__=='__main__':
    fetch_bundle('florence','florence-source.json')
    fetch_bundle('ppocr','ppocr-source.json')
