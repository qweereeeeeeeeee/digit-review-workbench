"""Re-run every bundled public photograph, including failures; no benchmark claim."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from objects.engine import ObjectEngine

root=Path(__file__).resolve().parents[1]
engine=ObjectEngine()
rows=[]
for path in sorted((root/'artifacts/objects/examples').glob('*.png')):
    result=engine.predict(Image.open(path).convert('RGB'))
    rows.append(dict(name=path.stem,image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**result))
    print(path.stem,[(r['label'],round(r['score'],3)) for r in result['detection']['items']])
(root/'artifacts/objects/detection-smoke.json').write_text(json.dumps(dict(
    purpose='Five public real photographs, integration smoke test only; not a held-out accuracy benchmark.',
    results=rows),ensure_ascii=False,indent=2)+'\n')
