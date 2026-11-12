"""Run full, real inference; private user images/results stay under var/."""
import hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image,ImageOps
import torch
from objects.general import GeneralVision
ROOT=Path(__file__).resolve().parents[1]

def main():
    torch.set_num_threads(2)
    (ROOT/'var/round3').mkdir(parents=True,exist_ok=True)
    model=GeneralVision()
    baseline=ROOT/'var/round3/baseline.json'
    private=[x for x in json.loads(baseline.read_text()) if x['name'].startswith('微信')] if baseline.exists() else []
    public=[dict(name=p.name,path=str(p.relative_to(ROOT))) for p in sorted((ROOT/'artifacts/objects/examples').glob('*.png'))]
    rows=[]
    for item in private+public:
        path=ROOT/item['path'];raw=path.read_bytes();im=ImageOps.exif_transpose(Image.open(path)).convert('RGB');r=model.predict(im)
        row=dict(path=item['path'],image_sha256=hashlib.sha256(raw).hexdigest(),result=r,private=item in private)
        rows.append(row);print(item['name'],{'texts':[(t['text'],t['score']) for t in r['texts']],'regions':[x['label'] for x in r['regions']],'seconds':r['seconds']},flush=True)
        (ROOT/'var/round3/final-results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
        (ROOT/'artifacts/vision/public-results.json').write_text(json.dumps([x for x in rows if not x['private']],ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
